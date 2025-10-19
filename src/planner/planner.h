#pragma once

#include "core/problem.h"
#include "core/rollout.h"
#include "core/space.h"
#include "core/util.h"
#include "ilqr/solver.h"
#include "ilqr/solver_settings.h"
#include "tree/tree.h"

static constexpr int NUM_NODE_ATTEMPTS = 40000;

struct TimingInfo {
    int tree_exp;  // ms
    int traj_opt;  // ms
};

struct PlannerOutputs {
    Tree tree;
    Path path;
    Solution<TRAJ_LENGTH_OPT> solution;
    Trajectory<TRAJ_LENGTH_OPT> traj_pre_opt;
    double cost_pre_opt;
    TimingInfo timing_info;
};

struct TrajOptOutputs {
    Solution<TRAJ_LENGTH_OPT> solution;
    Trajectory<TRAJ_LENGTH_OPT> traj_pre_opt;
    double cost_pre_opt;
};

struct Planner {
    static Tree expandTree(const StateVector& start, const StateVector& goal, const int num_node_attempts, const std::optional<Solution<TRAJ_LENGTH_OPT>>& warm, const bool use_hot, const SamplingSettings& sampling_settings) {
        Tree tree;
        std::optional<Trajectory<TRAJ_LENGTH_OPT>> warm_traj = warm ? std::optional(warm->traj) : std::nullopt;
        tree.grow(start, goal, num_node_attempts, warm_traj, use_hot, sampling_settings);
        return tree;
    }

    static ActionSequence<TRAJ_LENGTH_OPT> convertPathToActionSequence(const Path& path) {
        // Concatenate actions from all nodes of path.
        ActionSequence<TRAJ_LENGTH_OPT> action_sequence;
        int j = 0;
        for (const NodePtr& node : path) {
            if (!node->traj) {
                continue;
            }
            for (int i = 0; i < node->traj->length; ++i) {
                action_sequence.col(j) = node->traj->actionAt(i);
                j++;
            }
        }
        return action_sequence;
    }

    static TrajOptOutputs optimizeTrajectory(const StateVector& start, const StateVector& goal, const ActionSequence<TRAJ_LENGTH_OPT>& action_sequence) {
        // Define the optimal control problem.
        const Problem problem = makeProblem(start, goal, TRAJ_DURATION_OPT);

        // Get the pre-optimization trajectory for diagnostics later.
        Trajectory<TRAJ_LENGTH_OPT> traj_pre_opt;
        rolloutOpenLoopConstrained(action_sequence, start, traj_pre_opt);
        const double cost_pre_opt = softLoss(traj_pre_opt);

        // Solver settings.
        const SolverSettings settings = SolverSettings();
        settings.validate();

        // Instantiate the solver.
        Solver solver = Solver(std::make_shared<Problem>(problem), std::make_shared<SolverSettings>(settings));

        // Solve the optimal control problem.
        Solution<TRAJ_LENGTH_OPT> solution = solver.solve(action_sequence);

        // Re-rollout the solution action sequence to ensure the solution trajectory is consistent & honors action constraints.
        Trajectory<TRAJ_LENGTH_OPT> traj_post_opt;
        rolloutOpenLoopConstrained(solution.traj.action_sequence, start, traj_post_opt);
        solution.traj = traj_post_opt;

        // Assign the cost using arbitrary loss.
        solution.cost = softLoss(solution.traj);

        return {solution, traj_pre_opt, cost_pre_opt};
    }

    template <int N>
    static void addJitter(ActionSequence<N>& action_sequence, double sigma_a = 0.01, double sigma_k = 0.001) {
        // Create RNG generator.
        static thread_local std::default_random_engine generator(std::random_device{}());

        // Set the distributions.
        std::normal_distribution<double> distribution_a(0.0, sigma_a);
        std::normal_distribution<double> distribution_k(0.0, sigma_k);

        // Generate noise vectors
        Eigen::Matrix<double, 1, N> noise_a;
        Eigen::Matrix<double, 1, N> noise_k;

        for (int i = 0; i < N; ++i) {
            noise_a(0, i) = distribution_a(generator);
            noise_k(0, i) = distribution_k(generator);
        }

        // Add noise in a single vectorized step
        action_sequence.row(0) += noise_a;
        action_sequence.row(1) += noise_k;
    }

    static PlannerOutputs plan(const StateVector& start, const StateVector& goal, const std::optional<Solution<TRAJ_LENGTH_OPT>>& warm, const bool use_hot, const bool use_action_jitter, const SamplingSettings& sampling_settings) {
        // ---- Tree expansion
        const float tree_exp_clock_start = GetTime();
        const Tree tree = expandTree(start, goal, NUM_NODE_ATTEMPTS, warm, use_hot, sampling_settings);
        const float tree_exp_clock_stop = GetTime();
        const int tree_exp_clock_time = static_cast<int>(std::ceil(1e6 * (tree_exp_clock_stop - tree_exp_clock_start)));

        // ---- Path extraction
        const std::vector<Path>& path_candidates = tree.getPathCandidates();

        // ---- Trajectory optimization
        const float traj_opt_clock_start = GetTime();
        double best_post_opt_cost = std::numeric_limits<double>::infinity();
        TrajOptOutputs best_traj_opt_outputs;
        Path best_path;

        double best_post_opt_goal_hit_cost = std::numeric_limits<double>::infinity();
        TrajOptOutputs best_goal_hit_traj_opt_outputs;
        Path best_goal_hit_path;

        bool found_best_goal_hit = false;

        for (const Path& path : path_candidates) {
            auto action_sequence = convertPathToActionSequence(path);

            if (use_action_jitter) {
                // Add jitter on actions just before traj opt to try and jiggle out of bad local minima
                addJitter(action_sequence);
            }

            const TrajOptOutputs traj_opt_outputs = optimizeTrajectory(start, goal, action_sequence);

            if (traj_opt_outputs.solution.cost < best_post_opt_cost) {
                best_post_opt_cost = traj_opt_outputs.solution.cost;
                best_traj_opt_outputs = traj_opt_outputs;
                best_path = path;
            }

            if (!checkTargetHit(traj_opt_outputs.solution.traj.stateTerminal(), goal)) {
                continue;
            }

            if (traj_opt_outputs.solution.cost < best_post_opt_goal_hit_cost) {
                best_post_opt_goal_hit_cost = traj_opt_outputs.solution.cost;
                best_goal_hit_traj_opt_outputs = traj_opt_outputs;
                best_goal_hit_path = path;
                found_best_goal_hit = true;
            }
        }

        const TrajOptOutputs& ret_traj_opt_outputs = found_best_goal_hit ? best_goal_hit_traj_opt_outputs : best_traj_opt_outputs;
        const Path& ret_path = found_best_goal_hit ? best_goal_hit_path : best_path;

        const float traj_opt_clock_stop = GetTime();
        const int traj_opt_clock_time = static_cast<int>(std::ceil(1e6 * (traj_opt_clock_stop - traj_opt_clock_start)));

        // ---- Return planner outputs.
        return {tree, ret_path, ret_traj_opt_outputs.solution, ret_traj_opt_outputs.traj_pre_opt, ret_traj_opt_outputs.cost_pre_opt, {tree_exp_clock_time, traj_opt_clock_time}};
    }
};
