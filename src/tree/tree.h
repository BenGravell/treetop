#pragma once

#include <algorithm>
#include <cassert>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#ifdef PI
#undef PI
#endif
#include "nanoflann.h"
const double PI = nanoflann::pi_const<double>();

#include "core/constants.h"
#include "core/obstacle.h"
#include "core/search_space.h"
#include "core/space.h"
#include "core/trajectory.h"
#include "core/util.h"
#include "tree/node.h"
#include "tree/sampling.h"
#include "tree/steer.h"

// Settings

// Trajectory optimization time is linearly proportional to this number.
// Good results if chosen in proportion to NUM_NODE_ATTEMPTS in planner.h
static constexpr int num_path_candidates = 2;

// Number of steering segments in a trajectory optimization trajectory.
// Integer division is OK because TRAJ_LENGTH_OPT is an integer multiple of TRAJ_LENGTH_STEER.
static constexpr int NUM_STEER_SEGMENTS = TRAJ_LENGTH_OPT / TRAJ_LENGTH_STEER;

// Time index runs from 0 to NUM_STEER_SEGMENTS - 1, endpoints inclusive, leaving one time ix for goal node.
// Time index of the goal node.
static constexpr int TIME_IX_GOAL = NUM_STEER_SEGMENTS;
static constexpr int TIME_IX_MAX = TIME_IX_GOAL - 1;

// Time-averaged acceleration magnitude.
// This is what Loss.totalValue() would be in the absence of obstacle avoidance and smoothness considerations.
// TODO move this to Loss as a special method.
template <int N>
inline double softLoss(const Trajectory<N>& traj) {
    double cost = 0.0;
    for (int i = 0; i < traj.length; ++i) {
        const StateVector& state = traj.stateAt(i);
        const ActionVector& action = traj.actionAt(i);
        const double lon_accel = action(0);
        const double lat_accel = action(1) * square(state(3));
        const double mag_accel = shypot(lon_accel, lat_accel);
        cost += mag_accel;
    }
    return cost / N;
}

struct SteerOutputs {
    Trajectory<TRAJ_LENGTH_STEER> traj;
    double cost;
};

inline SteerOutputs steer(const StateVector& start, const StateVector& goal) {
    const ActionSequence<TRAJ_LENGTH_STEER> action_sequence = steerCubic<TRAJ_LENGTH_STEER>(start, goal, TRAJ_DURATION_STEER);

    Trajectory<TRAJ_LENGTH_STEER> traj;
    rolloutOpenLoopConstrained(action_sequence, start, traj);

    const double cost = softLoss(traj);

    return {traj, cost};
}

using Path = std::array<NodePtr, NUM_STEER_SEGMENTS>;

inline bool checkTargetHit(const StateVector& state, const StateVector& target, const bool relax = false) {
    const StateVector delta = state - target;
    const double dx = delta(0);
    const double dy = delta(1);
    const double dyaw = delta(2);
    const double dv = delta(3);

    // These are the same as problem.h -> makeProblem() -> terminal_state_params
    double x_tol = 0.01;
    double y_tol = 0.01;
    double yaw_tol = 0.02;
    double v_tol = 0.01;
    // NOTE: These are much looser than
    // problem.h -> makeProblem() -> terminal_state_params
    // to encourage choosing a trajectory in the lowest cost homotopy,
    // even if it is not perfectly goal-reaching
    if (relax) {
        x_tol = 2.0;
        y_tol = 2.0;
        yaw_tol = 60.0 * (2.0 * M_PI / 360.0);
        v_tol = 2.0;
    }
    const bool dx_hit = std::abs(dx) < x_tol;
    const bool dy_hit = std::abs(dy) < y_tol;
    const bool dyaw_hit = std::abs(dyaw) < yaw_tol;
    const bool dv_hit = std::abs(dv) < v_tol;

    return dx_hit && dy_hit && dyaw_hit && dv_hit;
}

using Nodes = std::vector<NodePtr>;
using Layers = std::array<Nodes, NUM_STEER_SEGMENTS + 1>;

struct StateCloud {
    std::vector<StateVector> pts;

    inline size_t kdtree_get_point_count() const { return pts.size(); }

    inline double kdtree_get_pt(const size_t idx, const size_t dim) const {
        return pts[idx](dim);
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const { return false; }
};

typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, StateCloud>, StateCloud, 4 /* dimension */> KDTree;

struct NodeAndValue {
    NodePtr node{nullptr};
    double value{std::numeric_limits<double>::infinity()};

    void compareWith(const NodePtr& node_other, const double value_other) {
        if (value_other < value) {
            node = node_other;
            value = value_other;
        }
    }
};

struct Tree {
    Layers layers;

    const NodePtr getNearest(const StateVector& target, const int target_time_ix, const KDTree& zap_kdtree) const {
        // Query point from KDTree index.

        // Distance from [zero-action-point of start] to [target]
        // This is a good proxy for softLoss since acceleration is proportional to
        // distance(zap(start), target) under simplifying kinematic assumptions e.g. @ high speed.

        // TODO calibrate this heuristic using data from steering function and many start-goal pairs.
        // We could define a (nonlinear) transform of the state in which Euclidean distance is well-correlated with the actual loss

        size_t ret_index;
        double out_dist_sqr;
        nanoflann::KNNResultSet<double> resultSet(1);
        resultSet.init(&ret_index, &out_dist_sqr);
        zap_kdtree.findNeighbors(resultSet, target.data());
        return layers[target_time_ix - 1][ret_index];
    }

    void addNode(const NodePtr& node, const int time_ix) {
        layers[time_ix].push_back(node);
    }

    void growRootNode(const StateVector& start, const StateVector& goal) {
        // Root node is the only node in tree.layers[0]
        const bool near_goal = checkTargetHit(start, goal, false);
        const Node root{start, nullptr, std::nullopt, 0.0, 0.0, SampleReason::kCold, near_goal};
        const NodePtr root_ptr = std::make_shared<Node>(root);
        addNode(root_ptr, 0);
    }

    NodePtr getRootNode() const {
        return layers.front().front();
    }

    void growZap(const StateVector& goal) {
        // Add zero action nodes as a fallback.
        // This ensures there is always at least one node in each layer,
        // which is needed later for the final steer to goal node and extractPathToNode call,
        // which expects fully connected parent chain to root.
        // NOTE: this ignores collisions.
        NodePtr parent = getRootNode();
        for (int time_ix = 1; time_ix <= TIME_IX_MAX; ++time_ix) {
            const SteerOutputs steer_outputs = steer(parent->state, rolloutZeroAction(parent->state, TRAJ_DURATION_STEER));
            const Trajectory<TRAJ_LENGTH_STEER>& traj = steer_outputs.traj;
            const double cost = steer_outputs.cost;
            const StateVector& state = traj.stateTerminal();
            const bool near_goal = checkTargetHit(state, goal, false);
            const Node node{state, parent, traj, cost, cost + parent->cost_to_come, SampleReason::kZeroActionPoint, near_goal};
            const NodePtr node_ptr = std::make_shared<Node>(node);
            addNode(node_ptr, time_ix);
            parent = node_ptr;
        }
    }

    void growHot(const Trajectory<TRAJ_LENGTH_OPT>& warm_traj, const StateVector& goal) {
        // Break warm_traj up into several smaller sub-nodes.
        NodePtr parent = getRootNode();
        for (int time_ix = 1; time_ix <= TIME_IX_GOAL; ++time_ix) {
            // Infer the indices into the whole warm_traj for the current sub-node.
            const int ix_offset = (time_ix - 1) * TRAJ_LENGTH_STEER;
            // Form the sub-node.
            Trajectory<TRAJ_LENGTH_STEER> traj;
            for (int stage_ix = 0; stage_ix <= TRAJ_LENGTH_STEER; ++stage_ix) {
                const int ix_in_warm_traj = ix_offset + stage_ix;
                traj.setStateAt(stage_ix, warm_traj.stateAt(ix_in_warm_traj));
                if (stage_ix < TRAJ_LENGTH_STEER) {
                    traj.setActionAt(stage_ix, warm_traj.actionAt(ix_in_warm_traj));
                }
            }
            const double cost = softLoss(traj);
            const StateVector& state = traj.stateTerminal();
            const bool near_goal = checkTargetHit(state, goal, false);

            // Special handling for goal node.
            // We must only add a goal node from the warm traj if it is nearish to the goal,
            // same acceptance criteria as in growGoalNodes(), sans collision check.
            if (time_ix == TIME_IX_GOAL) {
                const bool nearish_goal = checkTargetHit(state, goal, true);
                if (!nearish_goal) {
                    continue;
                }
            }

            const Node node{state, parent, traj, cost, cost + parent->cost_to_come, SampleReason::kHot, near_goal};
            const NodePtr node_ptr = std::make_shared<Node>(node);
            addNode(node_ptr, time_ix);
            parent = node_ptr;
        }
    }

    void growSingleNode(const StateVector& goal, const std::optional<Trajectory<TRAJ_LENGTH_OPT>>& warm_traj, const int time_ix, const KDTree& zap_kdtree, const SamplingSettings& sampling_settings) {
        // Sample a new state.
        StateAndReason state_and_reason = sample(goal, warm_traj, time_ix, sampling_settings);
        StateVector state = state_and_reason.state;
        const SampleReason reason = state_and_reason.reason;

        // Set the parent.
        const NodePtr parent = getNearest(state, time_ix, zap_kdtree);

        // Could not find a parent.
        if (parent == nullptr) {
            return;
        }

        // Sampled state is in collision.
        if (obstaclesCollidesWith(obstacles, state)) {
            return;
        }

        // Steer from parent to child.
        // NOTE: Using projection onto the action constraints is critical to sampling efficiency.
        // Using rejection sampling to honor action constraints leads to very few feasible samples and long runtimes.
        // NOTE: Using projection tends to produce bang-bang trajectories.
        // This might not be good on its own, but using traj opt post-processing mitigates any ill-effects.
        const SteerOutputs steer_outputs = steer(parent->state, state);
        const Trajectory<TRAJ_LENGTH_STEER>& traj = steer_outputs.traj;
        const double cost = steer_outputs.cost;

        // Reset state sample as the terminal state in the trajectory.
        state = traj.stateTerminal();

        // Trajectory is in collision.
        if (obstaclesCollidesWith(obstacles, traj)) {
            return;
        }

        // Create node from sampled state and add to the tree.
        const bool near_goal = checkTargetHit(state, goal, false);
        const Node node{state, parent, traj, cost, cost + parent->cost_to_come, reason, near_goal};
        const NodePtr node_ptr = std::make_shared<Node>(node);
        addNode(node_ptr, time_ix);
    }

    void growSingleLayer(const StateVector& goal, const std::optional<Trajectory<TRAJ_LENGTH_OPT>>& warm_traj, const int time_ix, const int num_node_attempts_per_layer, const SamplingSettings& sampling_settings) {
        // Build the zero-action-point cloud.
        StateCloud zap_cloud;
        const Nodes& prev_nodes = layers[time_ix - 1];
        zap_cloud.pts.reserve(prev_nodes.size());
        std::transform(
            prev_nodes.begin(), prev_nodes.end(),
            std::back_inserter(zap_cloud.pts),
            [](const NodePtr& node) {
                return rolloutZeroAction(node->state, TRAJ_DURATION_STEER);
            });

        // Build the zero-action-point KD tree index.
        KDTree zap_kdtree(4, zap_cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        zap_kdtree.buildIndex();

        // Grow.
        for (int node_attempt_ix = 0; node_attempt_ix < num_node_attempts_per_layer; ++node_attempt_ix) {
            growSingleNode(goal, warm_traj, time_ix, zap_kdtree, sampling_settings);
        }
    }

    void growLayers(const StateVector& goal, const std::optional<Trajectory<TRAJ_LENGTH_OPT>>& warm_traj, const int num_node_attempts, const SamplingSettings& sampling_settings) {
        const int num_node_attempts_per_layer = num_node_attempts / (TIME_IX_MAX + 1);
        for (int time_ix = 1; time_ix <= TIME_IX_MAX; ++time_ix) {
            growSingleLayer(goal, warm_traj, time_ix, num_node_attempts_per_layer, sampling_settings);
        }
    }

    void growGoalNodes(const StateVector& goal) {
        // Add goal nodes for all the collision-free goal-reaching parents.
        for (NodePtr parent : layers[TIME_IX_GOAL - 1]) {
            // Steer from node to target.
            const SteerOutputs steer_outputs = steer(parent->state, goal);
            const Trajectory<TRAJ_LENGTH_STEER>& traj = steer_outputs.traj;
            const StateVector& state = traj.stateTerminal();
            const double cost = steer_outputs.cost;

            if (obstaclesCollidesWith(obstacles, traj)) {
                continue;
            }
            const bool near_goal = checkTargetHit(state, goal, false);
            const bool nearish_goal = checkTargetHit(state, goal, true);
            if (!nearish_goal) {
                continue;
            }

            // Add node.
            const Node node{state, parent, traj, cost, cost + parent->cost_to_come, SampleReason::kGoal, near_goal};
            const NodePtr node_ptr = std::make_shared<Node>(node);
            addNode(node_ptr, TIME_IX_GOAL);
        }

        // Fallback to make sure there is at least one goal node.
        if (layers[TIME_IX_GOAL].empty()) {
            // Build the zero-action-point cloud.
            StateCloud zap_cloud;
            const Nodes& prev_nodes = layers[TIME_IX_GOAL - 1];
            zap_cloud.pts.reserve(prev_nodes.size());
            std::transform(
                prev_nodes.begin(), prev_nodes.end(),
                std::back_inserter(zap_cloud.pts),
                [](const NodePtr& node) {
                    return rolloutZeroAction(node->state, TRAJ_DURATION_STEER);
                });

            // Build the zero-action-point KD tree index.
            KDTree zap_kdtree(4, zap_cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
            zap_kdtree.buildIndex();

            const NodePtr parent = getNearest(goal, TIME_IX_GOAL, zap_kdtree);

            // Steer from node to goal.
            const SteerOutputs steer_outputs = steer(parent->state, goal);

            const Trajectory<TRAJ_LENGTH_STEER>& traj = steer_outputs.traj;
            const double cost = steer_outputs.cost;
            const StateVector& state = traj.stateTerminal();

            // Add node.
            const bool near_goal = checkTargetHit(state, goal, false);
            const Node node{state, parent, traj, cost, cost + parent->cost_to_come, SampleReason::kGoal, near_goal};
            const NodePtr node_ptr = std::make_shared<Node>(node);
            addNode(node_ptr, TIME_IX_GOAL);
        }
    }

    void grow(const StateVector& start, const StateVector& goal, const int num_node_attempts, std::optional<Trajectory<TRAJ_LENGTH_OPT>> warm_traj, const bool use_hot, const SamplingSettings& sampling_settings) {
        // Add root node.
        growRootNode(start, goal);

        // Add zero-action nodes.
        growZap(goal);

        if (warm_traj) {
            // Re-rollout the warm-start actions from the given start.
            // This mutates warm->traj.
            Trajectory<TRAJ_LENGTH_OPT> new_warm_traj;
            rolloutOpenLoopConstrained(warm_traj->action_sequence, start, new_warm_traj);
            warm_traj = new_warm_traj;
        }

        if (warm_traj && use_hot) {
            // Add warm-start nodes, i.e. hot-start the tree.
            growHot(warm_traj.value(), goal);
        }

        // Skip sampling if settings are all disabled.
        if ((sampling_settings.use_cold || sampling_settings.use_warm || sampling_settings.use_goal)) {
            // Add samples for all layers.
            growLayers(goal, warm_traj, num_node_attempts, sampling_settings);
        }

        // Add goal nodes.
        growGoalNodes(goal);
    }

    Path extractPathToGoal(NodePtr node) const {
        // Reconstruct the path by traversing parent pointers.
        Path path;
        for (int time_ix = TIME_IX_GOAL; time_ix >= 1; --time_ix) {
            path[time_ix - 1] = node;
            node = node->parent;
        }
        // Ensure the PARENT of the first node in the path is the root
        assert(path.front()->parent->parent == nullptr);

        return path;
    }

    std::vector<Path> getPathCandidates() const {
        const Nodes& goal_nodes = layers[TIME_IX_GOAL];
        std::vector<Path> candidates;
        candidates.reserve(num_path_candidates);

        // Collect only nodes that are near the goal.
        std::vector<NodePtr> near_goal_nodes;
        near_goal_nodes.reserve(goal_nodes.size());
        for (const NodePtr& node : goal_nodes) {
            if (node->near_goal) {
                near_goal_nodes.push_back(node);
            }
        }

        // Choose the best node among near-goal nodes if available; otherwise, among all.
        const auto& candidates_source = near_goal_nodes.empty() ? goal_nodes : near_goal_nodes;

        auto best_it = std::min_element(
            candidates_source.begin(), candidates_source.end(),
            [](const NodePtr& a, const NodePtr& b) {
                return a->cost_to_come < b->cost_to_come;
            });

        candidates.push_back(extractPathToGoal(*best_it));

        // Randomly select the rest (if available).
        if (goal_nodes.size() > 1) {
            // Create a list of indices excluding the best one.
            std::vector<size_t> indices;
            indices.reserve(goal_nodes.size() - 1);
            for (size_t i = 0; i < goal_nodes.size(); ++i) {
                if (goal_nodes[i] != *best_it) {
                    indices.push_back(i);
                }
            }

            // Shuffle and select random nodes.
            std::shuffle(indices.begin(), indices.end(), rng);

            int num_random = std::min<int>(num_path_candidates - 1, indices.size());
            for (int i = 0; i < num_random; ++i) {
                candidates.push_back(extractPathToGoal(goal_nodes[indices[i]]));
            }
        }

        return candidates;
    }
};
