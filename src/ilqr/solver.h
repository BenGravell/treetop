#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "core/loss.h"
#include "core/policy.h"
#include "core/problem.h"
#include "core/trajectory.h"
#include "ilqr/backward_pass.h"
#include "ilqr/solver_settings.h"

enum class SolveStatus : uint8_t {
    kInProgress = 0,
    kConverged = 1,
    kMaxItersExceeded = 2
};

enum class FfgsSearchStatus : uint8_t {
    kSuccessImmediate = 0,
    kSuccessDelayed = 1,
    kFailure = 2
};

struct SolveRecord {
    uint64_t iters{0};
    uint64_t rollouts{0};
};

template <int N>
struct Solution {
    Trajectory<N> traj;
    Policy<N> policy;
    double cost;
    SolveStatus solve_status;
    SolveRecord solve_record;
};

class Solver {
   public:
    // Constructor.
    explicit Solver(const std::shared_ptr<Problem>& problem, const std::shared_ptr<SolverSettings>& settings)
        : problem_(problem),
          settings_(settings),
          backward_pass_runner_(std::make_shared<BackwardPassRunner>(problem_)) {}

    template <int N>
    std::tuple<double, FfgsSearchStatus> feedfrwdGainSearch(
        Trajectory<N>& traj,
        Trajectory<N>& traj_dummy,
        Policy<N>& policy,
        const ExpectedCostChange& expected_cost_change,
        const double cost) {
        for (size_t num_attempts = 1; num_attempts <= settings_->max_feedfrwd_gain_search_attempts; ++num_attempts) {
            // Perform a forward pass to get a new candidate trajectory.
            solve_record_.rollouts += 1;
            rolloutClosedLoop(policy, traj, traj_dummy);

            // Evaluate the cost of the candidate trajectory and see if it is acceptable.
            const double cost_candidate = problem_->loss.totalValue(traj_dummy);
            const double cost_change = cost - cost_candidate;
            const double cost_change_expected = expected_cost_change.evaluate(policy.feedfrwd_gain_scale);
            const double cost_change_min_acceptable = (settings_->cost_change_ratio_min * cost_change_expected) - settings_->cost_change_tolerance;
            const bool success = cost_change > cost_change_min_acceptable;
            if (success) {
                // Mutate traj and return cost_candidate and a successful solve status.
                traj = traj_dummy;
                const FfgsSearchStatus search_status = (num_attempts <= 1) ? FfgsSearchStatus::kSuccessImmediate : FfgsSearchStatus::kSuccessDelayed;
                return std::make_tuple(cost_candidate, search_status);
            }

            // Scale down the feedforward gain and try again.
            policy.feedfrwd_gain_scale *= settings_->feedfrwd_gain_factor_decrease;
        }
        // We have run out of attempts.
        // Return a failed solve status without mutating traj or updating the cost.
        return std::make_tuple(cost, FfgsSearchStatus::kFailure);
    }

    double getRegularizationFactor(const FfgsSearchStatus status) {
        switch (status) {
            case FfgsSearchStatus::kSuccessImmediate:
                return settings_->regularization_factor_decrease;
            case FfgsSearchStatus::kSuccessDelayed:
                return settings_->regularization_factor_increase;
            case FfgsSearchStatus::kFailure:
                return settings_->regularization_factor_surge;
            default:
                return 1.0;
        }
    }

    double updateRegularization(const double reg, const FfgsSearchStatus status) {
        const double factor = getRegularizationFactor(status);
        return std::clamp(reg * factor, settings_->regularization_min, settings_->regularization_max);
    }

    SolveStatus checkConvergence(
        const double cost,
        const double cost_new,
        const FfgsSearchStatus ffgs_status) {
        const bool ffgs_search_failed = ffgs_status == FfgsSearchStatus::kFailure;
        const bool cost_converged = (cost - cost_new) < settings_->cost_change_tolerance;
        return (!ffgs_search_failed && cost_converged) ? SolveStatus::kConverged : SolveStatus::kInProgress;
    }

    template <int N>
    Solution<N> solve(const ActionSequence<N>& action_sequence) {
        // Initialize trajectory, policy, cost, and regularization.
        Trajectory<N> traj;
        Trajectory<N> traj_dummy;
        rolloutOpenLoopConstrained(action_sequence, problem_->initial_state, traj);
        Policy<N> policy;
        double cost = problem_->loss.totalValue(traj);
        double reg = settings_->regularization_init;

        for (size_t iters = 1; iters <= settings_->max_iters; ++iters) {
            // Perform a backward pass.
            const ExpectedCostChange expected_cost_change = backward_pass_runner_->run(traj, reg, policy);

            // Perform a feedforward gain scaling search.
            const auto [cost_new, ffgs_status] = feedfrwdGainSearch(traj, traj_dummy, policy, expected_cost_change, cost);

            // Check for convergence.
            const SolveStatus solve_status = checkConvergence(cost, cost_new, ffgs_status);

            // Update cost.
            cost = cost_new;

            // Update regularization.
            reg = updateRegularization(reg, ffgs_status);

            // Update the solve record.
            solve_record_.iters = iters;

            if (solve_status != SolveStatus::kInProgress) {
                return {traj, policy, cost, solve_status, solve_record_};
            }
        }

        // We have exceeded the max iterations.
        const SolveStatus solve_status = SolveStatus::kMaxItersExceeded;
        return {traj, policy, cost, solve_status, solve_record_};
    }

    // Impl that does not stop to check convergence, always uses max_iters.
    // By skipping convergence checks, we avoid branching and can get a significant per-iteration speedup. But we may run unnecessary iterations.
    template <int N>
    Solution<N> solveFixedIters(const ActionSequence<N>& action_sequence, const uint64_t num_iters = 0) {
        // Initialize trajectory, policy, cost, and regularization.
        Trajectory<N> traj;
        Trajectory<N> traj_dummy;
        rolloutOpenLoopConstrained(action_sequence, problem_->initial_state, traj);
        Policy<N> policy;
        double cost = problem_->loss.totalValue(traj);
        double cost_old = cost;
        double reg = settings_->regularization_init;
        FfgsSearchStatus ffgs_status = FfgsSearchStatus::kFailure;

        // Choose the number of iterations to use in the loop. Use the argument num_iters if greater than zero (default), else use solver settings.
        const uint64_t num_iters_for_loop = num_iters > 0 ? num_iters : settings_->max_iters;

        for (size_t iters = 1; iters <= num_iters_for_loop; ++iters) {
            // Save the old cost.
            cost_old = cost;

            // Perform a backward pass.
            const ExpectedCostChange expected_cost_change = backward_pass_runner_->run(traj, reg, policy);

            // Perform a feedforward gain scaling search.
            const auto [cost_new, ffgs_status_new] = feedfrwdGainSearch(traj, traj_dummy, policy, expected_cost_change, cost);
            cost = cost_new;
            ffgs_status = ffgs_status_new;

            // Update regularization.
            reg = updateRegularization(reg, ffgs_status);
        }

        // Update the solve record.
        solve_record_.iters = num_iters_for_loop;

        // Check for convergence.
        const SolveStatus solve_status = checkConvergence(cost_old, cost, ffgs_status);

        return {traj, policy, cost, solve_status, solve_record_};
    }

   private:
    const std::shared_ptr<Problem> problem_;
    const std::shared_ptr<SolverSettings> settings_;
    std::shared_ptr<BackwardPassRunner> backward_pass_runner_;
    SolveRecord solve_record_;
};
