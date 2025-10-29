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

// Number of steering segments in a trajectory optimization trajectory.
// Integer division is OK because TRAJ_LENGTH_OPT is an integer multiple of TRAJ_LENGTH_STEER.
static constexpr int NUM_STEER_SEGMENTS = TRAJ_LENGTH_OPT / TRAJ_LENGTH_STEER;

// Time index of the goal node.
static constexpr int TIME_IX_GOAL = NUM_STEER_SEGMENTS;
// Time index of the last non-goal node layer.
static constexpr int TIME_IX_MAX = TIME_IX_GOAL - 1;

using Path = std::array<NodePtr, NUM_STEER_SEGMENTS>;

// Time-averaged acceleration magnitude.
// This is what Loss.totalValue() would be in the absence of constraint satisfaction penalties.
// TODO move this to Loss as a special method.
template <int N>
inline double softLoss(const Trajectory<N>& traj) {
    double cost = 0.0;
    for (int i = 0; i < N; ++i) {
        const StateVector& state = traj.stateAt(i);
        const ActionVector& action = traj.actionAt(i);
        const double lon_accel = action(0);
        const double lat_accel = action(1) * square(state(3));
        const double mag_accel = shypot(lon_accel, lat_accel);
        cost += mag_accel;
    }
    return cost / N;
}

using SteerTraj = Trajectory<TRAJ_LENGTH_STEER>;

inline SteerTraj steer(const StateVector& start, const StateVector& goal, const double duration) {
    const ActionSequence<TRAJ_LENGTH_STEER> action_sequence = steerCubic<TRAJ_LENGTH_STEER>(start, goal, duration, DT);
    SteerTraj traj;
    rolloutOpenLoopConstrained(action_sequence, start, traj);
    return traj;
}

// Heuristic distance between states.
// Slightly different from the kd-tree based distance.
inline double stateDistance(const StateVector& state, const StateVector& target) {
    const StateVector delta = state - target;

    const double dx = delta(0);
    const double dy = delta(1);
    const double dyaw = delta(2);
    const double dv = delta(3);

    return shypot(dx, dy) + std::abs(dyaw) + std::abs(dv);
}

inline bool checkTargetHit(const StateVector& state, const StateVector& target) {
    const StateVector delta = state - target;
    const double dx = delta(0);
    const double dy = delta(1);
    const double dyaw = delta(2);
    const double dv = delta(3);

    // These are the same as problem.h -> makeProblem() -> terminal_state_params
    double x_tol = 0.01;
    double y_tol = 0.01;
    double yaw_tol = 0.01;
    double v_tol = 0.01;

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

struct Tree {
    Layers layers;

    const NodePtr getRandomParent(const int target_time_ix) const {
        const Nodes& nodes = layers[target_time_ix - 1];
        if (nodes.empty()) {
            return nullptr;
        }
        std::uniform_int_distribution<size_t> dist(0, nodes.size() - 1);
        return nodes[dist(rng)];
    }

    const NodePtr getNearestParent(const StateVector& target, const int target_time_ix, const KDTree& zap_kdtree) const {
        // Query point from KDTree index.

        // Distance from [zero-action-point of start] to [target]

        // NOTE: This treats distance in (x, y, yaw, velocity) all equally (isotropic)
        // With the choice of units and scenarios used here, this is a decent choice empirically (even if it is not very principled)

        // XY distance is a good proxy for softLoss since acceleration is proportional to
        // distance(zap(start), target) under simplifying kinematic assumptions e.g. @ high speed.

        // TODO calibrate this heuristic using data from steering function and many start-goal pairs.
        // We could define a (nonlinear) transform of the state in which Euclidean distance is well-correlated with the actual loss

        size_t ret_index;
        double out_dist_sqr;
        nanoflann::KNNResultSet<double> resultSet(1);
        resultSet.init(&ret_index, &out_dist_sqr);
        zap_kdtree.findNeighbors(resultSet, target.data());
        // NOTE: zap_kdtree is the tree made from zero-action-points of (forward-propagation from) layers[target_time_ix - 1],
        // so layers[target_time_ix - 1][ret_index] is the corresponding node which is the parent of the zero-action-point at ret_index.
        return layers[target_time_ix - 1][ret_index];
    }

    NodePtr createNode(const NodePtr& parent, const SteerTraj& traj, const int time_ix, const StateVector& goal, const SampleReason reason) {
        const double cost = softLoss(traj);
        const StateVector& state = traj.stateTerminal();
        const double dist_to_goal = stateDistance(state, goal);
        const Node node{state, parent, traj, cost, cost + parent->cost_to_come, reason, dist_to_goal};
        const NodePtr node_ptr = std::make_shared<Node>(node);
        layers[time_ix].push_back(node_ptr);
        return node_ptr;
    }

    void growRootNode(const StateVector& start, const StateVector& goal) {
        // Root node is the only node in tree.layers[0]
        const double dist_to_goal = stateDistance(start, goal);
        const Node root{start, nullptr, std::nullopt, 0.0, 0.0, SampleReason::kCold, dist_to_goal};
        const NodePtr root_ptr = std::make_shared<Node>(root);
        layers.front().push_back(root_ptr);
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
            const SteerTraj traj = steer(parent->state, rolloutZeroAction(parent->state, TRAJ_DURATION_STEER), TRAJ_DURATION_STEER);
            const SampleReason reason = SampleReason::kZeroActionPoint;
            parent = createNode(parent, traj, time_ix, goal, reason);
        }
    }

    void growHot(const Trajectory<TRAJ_LENGTH_OPT>& warm_traj, const StateVector& goal) {
        // Break warm_traj up into several smaller sub-nodes.
        NodePtr parent = getRootNode();
        for (int time_ix = 1; time_ix <= TIME_IX_GOAL; ++time_ix) {
            // Infer the indices into the whole warm_traj for the current sub-node.
            const int ix_offset = (time_ix - 1) * TRAJ_LENGTH_STEER;
            // Form the sub-node.
            SteerTraj traj;
            for (int stage_ix = 0; stage_ix <= TRAJ_LENGTH_STEER; ++stage_ix) {
                const int ix_in_warm_traj = ix_offset + stage_ix;
                traj.setStateAt(stage_ix, warm_traj.stateAt(ix_in_warm_traj));
                if (stage_ix < TRAJ_LENGTH_STEER) {
                    traj.setActionAt(stage_ix, warm_traj.actionAt(ix_in_warm_traj));
                }
            }
            const SampleReason reason = SampleReason::kHot;
            parent = createNode(parent, traj, time_ix, goal, reason);
        }
    }

    void growSingleNode(const StateVector& goal, const std::optional<Trajectory<TRAJ_LENGTH_OPT>>& warm_traj, const int time_ix, const KDTree& zap_kdtree, const SamplingSettings& sampling_settings) {
        // Sample a new state.
        StateAndReason state_and_reason = sample(goal, warm_traj, time_ix, sampling_settings);
        StateVector state = state_and_reason.state;
        const SampleReason reason = state_and_reason.reason;

        // Set the parent.
        const NodePtr parent = (reason == SampleReason::kGoal) ? getRandomParent(time_ix) : getNearestParent(state, time_ix, zap_kdtree);

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
        const double duration = ((reason == SampleReason::kGoal) ? (TIME_IX_GOAL - time_ix) : 1) * TRAJ_DURATION_STEER;
        const SteerTraj traj = steer(parent->state, state, duration);

        if (obstaclesCollidesWith(obstacles, traj)) {
            return;
        }

        createNode(parent, traj, time_ix, goal, reason);
    }

    void growSingleLayer(const StateVector& goal, const std::optional<Trajectory<TRAJ_LENGTH_OPT>>& warm_traj, const int time_ix, const int num_samples, const SamplingSettings& sampling_settings) {
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
        for (int sample_ix = 0; sample_ix < num_samples; ++sample_ix) {
            growSingleNode(goal, warm_traj, time_ix, zap_kdtree, sampling_settings);
        }
    }

    void growLayers(const StateVector& goal, const std::optional<Trajectory<TRAJ_LENGTH_OPT>>& warm_traj, const int num_samples, const SamplingSettings& sampling_settings) {
        const int num_samples_per_layer = num_samples / NUM_STEER_SEGMENTS;
        for (int time_ix = 1; time_ix <= TIME_IX_MAX; ++time_ix) {
            growSingleLayer(goal, warm_traj, time_ix, num_samples_per_layer, sampling_settings);
        }
    }

    void growGoalNodes(const StateVector& goal) {
        static constexpr SampleReason reason = SampleReason::kGoal;

        // Attempt to steer to goal from every possible parent in the penultimate layer.
        for (NodePtr parent : layers[TIME_IX_GOAL - 1]) {
            const SteerTraj traj = steer(parent->state, goal, TRAJ_DURATION_STEER);

            if (obstaclesCollidesWith(obstacles, traj)) {
                continue;
            }

            createNode(parent, traj, TIME_IX_GOAL, goal, reason);
        }

        // Fallback to make sure there is at least one goal node.
        if (layers[TIME_IX_GOAL].empty()) {
            StateCloud zap_cloud;
            const Nodes& prev_nodes = layers[TIME_IX_GOAL - 1];
            zap_cloud.pts.reserve(prev_nodes.size());
            std::transform(
                prev_nodes.begin(), prev_nodes.end(),
                std::back_inserter(zap_cloud.pts),
                [](const NodePtr& node) {
                    return rolloutZeroAction(node->state, TRAJ_DURATION_STEER);
                });

            KDTree zap_kdtree(4, zap_cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
            zap_kdtree.buildIndex();

            const NodePtr parent = getNearestParent(goal, TIME_IX_GOAL, zap_kdtree);

            const SteerTraj traj = steer(parent->state, goal, TRAJ_DURATION_STEER);

            createNode(parent, traj, TIME_IX_GOAL, goal, reason);
        }
    }

    void grow(const StateVector& start, const StateVector& goal, const int num_samples, std::optional<Trajectory<TRAJ_LENGTH_OPT>> warm_traj, const bool use_hot, const SamplingSettings& sampling_settings) {
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
            growLayers(goal, warm_traj, num_samples, sampling_settings);
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

    std::vector<Path> getPathCandidates(const int num_path_candidates) const {
        const Nodes& goal_nodes = layers[TIME_IX_GOAL];
        std::vector<Path> candidates;
        candidates.reserve(num_path_candidates);

        // Collect nodes that are near the goal.
        std::vector<NodePtr> near_goal_nodes;
        near_goal_nodes.reserve(goal_nodes.size());
        for (const NodePtr& node : goal_nodes) {
            static constexpr double d_tol = 0.01;
            if (node->dist_to_goal < d_tol) {
                near_goal_nodes.push_back(node);
            }
        }

        // Choose the best node based on priorities.
        NodePtr best_node = nullptr;

        if (!near_goal_nodes.empty()) {
            auto best_it = std::min_element(
                near_goal_nodes.begin(), near_goal_nodes.end(),
                [](const NodePtr& a, const NodePtr& b) {
                    return a->cost_to_come < b->cost_to_come;
                });
            best_node = *best_it;
        } else {
            auto best_it = std::min_element(
                goal_nodes.begin(), goal_nodes.end(),
                [](const NodePtr& a, const NodePtr& b) {
                    return a->dist_to_goal < b->dist_to_goal;
                });
            best_node = *best_it;
        }

        candidates.push_back(extractPathToGoal(best_node));

        // Randomly select the rest (if available).
        if (goal_nodes.size() > 1) {
            // TODO just directly sample a few random integers without replacement, don't create the entire list then shuffle it...

            // Create a list of indices excluding the best one.
            std::vector<size_t> indices;
            indices.reserve(goal_nodes.size() - 1);
            for (size_t i = 0; i < goal_nodes.size(); ++i) {
                if (goal_nodes[i] != best_node) {
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
