#pragma once

#include <algorithm>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "core/constants.h"
#include "core/obstacle.h"
#include "core/search_space.h"
#include "core/space.h"
#include "core/trajectory.h"
#include "core/util.h"
#include "tree/node.h"
#include "tree/steer.h"

// Define a global random number generator
std::random_device rd;
std::mt19937 gen(42);  // Mersenne Twister engine w/ fixed seed
// std::mt19937 gen(rd());  // Mersenne Twister engine w/ random seed

// Number of steering segments in a trajectory optimization trajectory.
// Integer division is OK because TRAJ_LENGTH_OPT is an integer multiple of TRAJ_LENGTH_STEER.
static constexpr int NUM_STEER_SEGMENTS = TRAJ_LENGTH_OPT / TRAJ_LENGTH_STEER;

// Time index of the goal node.
static constexpr int TIME_IX_GOAL = NUM_STEER_SEGMENTS;

// Time index runs from 0 to NUM_STEER_SEGMENTS - 1, endpoints inclusive, leaving one time ix for goal node.
static constexpr int TIME_IX_MAX = TIME_IX_GOAL - 1;

inline double distance(const StateVector& start, const StateVector& goal) {
    return (goal.head(2) - start.head(2)).norm();
}

// TODO include contribution from speed diff
// TODO calibrate this heuristic using data from steering function and many start-goal pairs
inline double distanceHeuristic(const StateVector& start, const StateVector& goal) {
    // Heuristic: Scale distance by an additional factor due to yaw difference.
    const double yaw1 = start(2);
    const double yaw2 = goal(2);

    // Compute the raw normalized yaw difference in [0, 1].
    const double yaw_diff = std::abs(angularDifference(yaw1, yaw2)) / PI;

    // Compute a factor based on the yaw diff.
    // Apply square() so that small yaw differences << 1 have less impact.
    const double yaw_diff_factor = square(yaw_diff);

    // Compute scaled distance.
    return (1.0 + yaw_diff_factor) * distance(start, goal);
}

// Distance from [zero-action-point of start] to [goal]
// This is a good proxy for softLoss since acceleration is proportional to
// distance(zap(start), goal) under simplifying kinematic assumptions e.g. @ high speed.
inline double zapDistanceHeuristic(const StateVector& start, const StateVector& goal) {
    return distanceHeuristic(rolloutZeroAction(start, TRAJ_DURATION_STEER), goal);
}

// Time-averaged acceleration magnitude.
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

inline SteerOutputs steer(const StateVector& start, const StateVector& goal, const bool constrain) {
    // Cubic polynomial steering.
    const ActionSequence<TRAJ_LENGTH_STEER> action_sequence = steerCubic<TRAJ_LENGTH_STEER>(start, goal, TRAJ_DURATION_STEER);

    // Rollout.
    Trajectory<TRAJ_LENGTH_STEER> traj;
    if (constrain) {
        rolloutOpenLoopConstrained(action_sequence, start, traj);
    } else {
        rolloutOpenLoop(action_sequence, start, traj);
    }

    // Calculate cost.
    const double cost = softLoss(traj);

    return {traj, cost};
}

using Path = std::array<NodePtr, NUM_STEER_SEGMENTS>;

inline double urand() {
    static std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(gen);
}

inline double urand(const double x_min, const double x_max) {
    std::uniform_real_distribution<double> dist(x_min, x_max);
    return dist(gen);
}

inline StateVector sample() {
    const double x = urand(X_MIN, X_MAX);
    const double y = urand(Y_MIN, Y_MAX);
    const double yaw = urand(YAW_MIN, YAW_MAX);
    const double v = urand(V_MIN, V_MAX);
    return {x, y, yaw, v};
}

inline StateVector sampleNear(const StateVector& state, const double perturb_factor = 1.0) {
    const double d = perturb_factor * 0.5;
    const double d_yaw = perturb_factor * 0.2 * 0.5 * PI;
    const double dv = perturb_factor * 1.0;

    double x_min_s = state(0) - d;
    double x_max_s = state(0) + d;
    double y_min_s = state(1) - d;
    double y_max_s = state(1) + d;
    double yaw_min_s = state(2) - d_yaw;
    double yaw_max_s = state(2) + d_yaw;
    double v_min_s = state(3) - dv;
    double v_max_s = state(3) + dv;

    x_min_s = std::clamp(x_min_s, X_MIN, X_MAX);
    x_max_s = std::clamp(x_max_s, X_MIN, X_MAX);
    y_min_s = std::clamp(y_min_s, Y_MIN, Y_MAX);
    y_max_s = std::clamp(y_max_s, Y_MIN, Y_MAX);
    yaw_min_s = std::clamp(yaw_min_s, YAW_MIN, YAW_MAX);
    yaw_max_s = std::clamp(yaw_max_s, YAW_MIN, YAW_MAX);
    v_min_s = std::clamp(v_min_s, V_MIN, V_MAX);
    v_max_s = std::clamp(v_max_s, V_MIN, V_MAX);

    const double x = urand(x_min_s, x_max_s);
    const double y = urand(y_min_s, y_max_s);
    const double yaw = urand(yaw_min_s, yaw_max_s);
    const double v = urand(v_min_s, v_max_s);
    return {x, y, yaw, v};
}

inline StateVector sampleNearWarm(const Trajectory<TRAJ_LENGTH_OPT>& warm_traj, const int time_ix) {
    const int stage_ix = time_ix * TRAJ_LENGTH_STEER;
    const StateVector& state = warm_traj.stateAt(stage_ix);
    static constexpr double perturb_factor = 2.0;
    return sampleNear(state, perturb_factor);
}

inline StateVector sample(const StateVector& goal, const std::optional<Trajectory<TRAJ_LENGTH_OPT>>& warm_traj, const int time_ix) {
    // Sample near the goal sometimes.
    static constexpr double goal_sampling_proba = 0.02;
    const double selector = urand();
    const bool sample_near_goal{selector < goal_sampling_proba};
    const double perturb_factor = 1.0;
    if (sample_near_goal) {
        return sampleNear(goal, perturb_factor);
    }

    // Sample around the warm-start trajectory, if available.
    if (warm_traj) {
        return sampleNearWarm(warm_traj.value(), time_ix);
    }

    return sample();
}

inline bool outsideEnvironment(const StateVector& state) {
    const bool x_ok = (X_MIN <= state(0)) && (state(0) <= X_MAX);
    const bool y_ok = (Y_MIN <= state(1)) && (state(1) <= Y_MAX);
    const bool yaw_ok = (YAW_MIN <= state(2)) && (state(2) <= YAW_MAX);
    const bool v_ok = (V_MIN <= state(3)) && (state(3) <= V_MAX);
    return !(x_ok && y_ok && yaw_ok && v_ok);
}

template <int N>
inline bool outsideEnvironment(const Trajectory<N>& traj) {
    // Iterate in reverse since, heuristically, states at the end of the trajectory
    // are more likely to fail the check and terminate the loop early.
    for (int stage_ix = N; stage_ix >= 0; --stage_ix) {
        if (outsideEnvironment(traj.stateAt(stage_ix))) {
            return true;
        }
    }
    return false;
}

inline bool checkTargetHit(const StateVector& state, const StateVector& target) {
    const StateVector delta = state - target;
    const double dx = delta(0);
    const double dy = delta(1);
    const double dyaw = delta(2);
    const double dv = delta(3);

    // TODO use the TerminalStateParams.
    // Current numbers are hardcoded to match what is in 
    // problem.h -> makeProblem() -> terminal_state_params
    // and set as a factor of those thresholds.
    static constexpr double tol_factor = 1.0;
    const bool dx_hit = std::abs(dx) < (tol_factor * 0.01);
    const bool dy_hit = std::abs(dy) < (tol_factor * 0.01);
    const bool dyaw_hit = std::abs(dyaw) < (tol_factor * 0.02);
    const bool dv_hit = std::abs(dv) < (tol_factor * 0.01);

    return dx_hit && dy_hit && dyaw_hit && dv_hit;
}

using Nodes = std::vector<NodePtr>;
using Layers = std::array<Nodes, NUM_STEER_SEGMENTS + 1>;

struct Tree {
    Layers layers;

    const NodePtr getNearest(const StateVector& target, const int target_time_ix) const {
        double min_cost = std::numeric_limits<double>::max();
        NodePtr nearest_node = nullptr;
        const Nodes& nodes = layers[target_time_ix - 1];
        for (NodePtr node : nodes) {
            const double cost = zapDistanceHeuristic(node->state, target);
            if (cost < min_cost) {
                min_cost = cost;
                nearest_node = node;
            }
        }

        return nearest_node;
    }

    // Get the node which is nearest to the target in terms of achieving the lowest cost to come to the target via the node.
    const NodePtr getNearestCostToCome(const StateVector& target, const int target_time_ix) const {
        double min_cost_to_come = std::numeric_limits<double>::max();
        NodePtr nearest_node = nullptr;
        const Nodes& nodes = layers[target_time_ix - 1];
        for (NodePtr node : nodes) {
            // Steer from node to target.
            const bool constrain = false;
            const auto steer_outputs = steer(node->state, target, constrain);

            // Check if cost-to-come is improved relative to current best.
            const double cost_to_come = node->cost_to_come + steer_outputs.cost;
            const bool cost_improved = cost_to_come < min_cost_to_come;

            if (cost_improved) {
                min_cost_to_come = cost_to_come;
                nearest_node = node;
            }
        }

        if (nearest_node == nullptr) {
            return getNearest(target, target_time_ix);
        }

        return nearest_node;
    }

    void addNode(const NodePtr& node, const int time_ix) {
        layers[time_ix].push_back(node);
    }

    void growZap(const NodePtr& root) {
        // Add zero action nodes as a fallback.
        // This ensures there is always at least one node in each layer,
        // which is needed later for the final steer to goal node and extractPathToNode call,
        // which expects fully connected parent chain to root.
        // NOTE: this ignores collisions and outsideEnvironment constraints.
        NodePtr parent = root;
        for (int time_ix = 1; time_ix <= TIME_IX_MAX; ++time_ix) {
            const bool constrain = true;
            const auto steer_outputs = steer(parent->state, rolloutZeroAction(parent->state, TRAJ_DURATION_STEER), constrain);
            const auto& traj = steer_outputs.traj;
            const double cost = steer_outputs.cost;
            const StateVector& state = traj.stateTerminal();
            const Node node = Node(state, parent, traj, cost, cost + parent->cost_to_come);
            const NodePtr node_ptr = std::make_shared<Node>(node);
            addNode(node_ptr, time_ix);
            parent = node_ptr;
        }
    }

    void growWarm(const NodePtr& root, const Trajectory<TRAJ_LENGTH_OPT>& warm_traj) {
        // Break warm_traj up into several smaller sub-nodes.
        NodePtr parent = root;
        for (int time_ix = 1; time_ix <= TIME_IX_MAX; ++time_ix) {
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
            const Node node = Node(state, parent, traj, cost, cost + parent->cost_to_come);
            const NodePtr node_ptr = std::make_shared<Node>(node);
            addNode(node_ptr, time_ix);
            parent = node_ptr;
        }
    }

    void growSingleNode(const StateVector& goal, const std::optional<Trajectory<TRAJ_LENGTH_OPT>>& warm_traj, const int time_ix) {
        // Sample a new state.
        StateVector state = sample(goal, warm_traj, time_ix);

        // Set the parent.
        const NodePtr parent = getNearest(state, time_ix);

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
        const bool constrain = true;
        const auto steer_outputs = steer(parent->state, state, constrain);
        const auto& traj = steer_outputs.traj;
        const double cost = steer_outputs.cost;

        // Reset state sample as the terminal state in the trajectory.
        state = traj.stateTerminal();

        // Trajectory is in collision.
        if (obstaclesCollidesWith(obstacles, traj)) {
            return;
        }

        // Something, e.g. steering or action constraint projection,
        // caused a trajectory state to go outside the environment.
        if (outsideEnvironment(traj)) {
            return;
        }

        // Create node from sampled state and add to the tree.
        const Node node{state, parent, traj, cost, cost + parent->cost_to_come};
        const NodePtr node_ptr = std::make_shared<Node>(node);
        addNode(node_ptr, time_ix);
    }

    void growSingleStage(const StateVector& goal, const std::optional<Trajectory<TRAJ_LENGTH_OPT>>& warm_traj, const int time_ix, const int num_node_attempts_per_stage) {
        for (int node_attempt_ix = 0; node_attempt_ix < num_node_attempts_per_stage; ++node_attempt_ix) {
            growSingleNode(goal, warm_traj, time_ix);
        }
    }

    void growStages(const StateVector& goal, const std::optional<Trajectory<TRAJ_LENGTH_OPT>>& warm_traj, const int num_node_attempts) {
        const int num_node_attempts_per_stage = num_node_attempts / (TIME_IX_MAX + 1);
        for (int time_ix = 1; time_ix <= TIME_IX_MAX; ++time_ix) {
            growSingleStage(goal, warm_traj, time_ix, num_node_attempts_per_stage);
        }
    }

    const NodePtr getGoalParent(const StateVector& goal) const {
        double min_cost_to_come = std::numeric_limits<double>::max();
        NodePtr goal_parent = nullptr;
        for (NodePtr node : layers[TIME_IX_GOAL - 1]) {
            // Steer from node to target.
            const bool constrain = true;
            const auto steer_outputs = steer(node->state, goal, constrain);

            if (obstaclesCollidesWith(obstacles, steer_outputs.traj)) {
                continue;
            }
            if (outsideEnvironment(steer_outputs.traj)) {
                continue;
            }
            if (!checkTargetHit(node->state, goal)) {
                continue;
            }

            // Check if cost-to-come is improved relative to current best.
            const double cost_to_come = node->cost_to_come + steer_outputs.cost;
            const bool cost_improved = cost_to_come < min_cost_to_come;

            if (cost_improved) {
                min_cost_to_come = cost_to_come;
                goal_parent = node;
            }
        }

        // Fallback in case collision and goal-hit checks discarded everything.
        // NOTE: This should produce non-nullptr node_nearest_goal with full parent chain to root since we added zero-action fallback earlier.
        if (goal_parent == nullptr) {
            goal_parent = getNearestCostToCome(goal, TIME_IX_GOAL);
        }

        assert(goal_parent != nullptr);

        return goal_parent;
    }

    void growGoalNode(const StateVector& goal) {
        // Get the parent.
        const NodePtr parent = getGoalParent(goal);

        // Steer from parent to goal.
        const bool constrain = true;
        const auto steer_outputs = steer(parent->state, goal, constrain);
        const auto& traj = steer_outputs.traj;
        const double cost = steer_outputs.cost;
        const StateVector& state = traj.stateTerminal();

        // Add node.
        // This ensures the tree always has one node with time_ix = TIME_IX_GOAL.
        const Node node{state, parent, traj, cost, cost + parent->cost_to_come};
        const NodePtr node_ptr = std::make_shared<Node>(node);
        addNode(node_ptr, TIME_IX_GOAL);
    }

    void grow(const StateVector& start, const StateVector& goal, const int num_node_attempts, std::optional<Trajectory<TRAJ_LENGTH_OPT>> warm_traj = std::nullopt) {
        // Add root node.
        // Root node is the only node in tree.layers[0]
        const Node root = Node(start, nullptr, std::nullopt, 0.0, 0.0);
        const NodePtr root_ptr = std::make_shared<Node>(root);
        addNode(root_ptr, 0);

        // Add zero-action nodes.
        growZap(root_ptr);

        if (warm_traj) {
            // Re-rollout the warm-start actions from the given start.
            // This mutates warm->traj.
            Trajectory<TRAJ_LENGTH_OPT> new_warm_traj;
            rolloutOpenLoopConstrained(warm_traj->action_sequence, start, new_warm_traj);
            warm_traj = new_warm_traj;

            // Add warm-start nodes.
            growWarm(root_ptr, warm_traj.value());
        }

        // Add samples for all stages.
        growStages(goal, warm_traj, num_node_attempts);

        // Add goal node.
        growGoalNode(goal);
    }

    Path extractPathToGoal() const {
        // Reconstruct the path by traversing parent pointers.
        Path path;
        // There should only be a single node in layers[TIME_IX_GOAL], which is the goal node.
        NodePtr node = layers[TIME_IX_GOAL].front();
        for (int time_ix = TIME_IX_GOAL; time_ix >= 1; --time_ix) {
            path[time_ix - 1] = node;
            node = node->parent;
        }
        // Ensure the PARENT of the first node in the path is the root
        assert(path.front()->parent->parent == nullptr);

        return path;
    }
};
