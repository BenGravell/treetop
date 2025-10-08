#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "core/loss.h"
#include "core/policy.h"
#include "core/problem.h"
#include "core/space.h"
#include "core/trajectory.h"
#include "ilqr/solver_settings.h"

// Struct representing an expected cost change.
struct ExpectedCostChange {
    double term1{0.0};
    double term2{0.0};

    // Evaluate the expected cost change.
    double evaluate(const double ff_gain_scaling) const {
        return -(ff_gain_scaling * term1 + 0.5 * ff_gain_scaling * ff_gain_scaling * term2);
    }

    // Reset the expected cost change.
    void reset() {
        term1 = 0.0;
        term2 = 0.0;
    }
};

// Regularize a matrix by adding reg to the diagonal.
inline ActionActionMatrix regularize(const ActionActionMatrix& X, double reg) {
    ActionActionMatrix Xreg = X;
    Xreg.diagonal().array() += reg;
    return Xreg;
}

class BackwardPassRunner {
   public:
    // Constructor.
    explicit BackwardPassRunner(const std::shared_ptr<Problem>& problem) : problem_(problem), Q_(), V_(), expected_cost_change_() {}

    // Perform a backward pass.
    template <int N>
    ExpectedCostChange run(const Trajectory<N>& traj, const double reg, Policy<N>& policy) {
        // Reset the feedforward gain scaling.
        policy.feedfrwd_gain_scale = 1.0;

        // Reset the expected cost change.
        expected_cost_change_.reset();

        // Initialize the value function with the terminal gradient and Hessian.
        V_ = problem_->loss.terminalGradientAndHessian(traj.stateTerminal());

        // Iterate backwards in time using dynamic programming.
        for (size_t stage_idx_plus = traj.length; stage_idx_plus > 0; --stage_idx_plus) {
            // Cannot use stage_idx directly in the for loop because it is size_t.
            // Use loop dummy var stage_idx_plus = stage_idx + 1 as a workaround.
            const size_t stage_idx = stage_idx_plus - 1;

            // Update Q-function.
            updateQ(traj.stateAt(stage_idx), traj.actionAt(stage_idx));

            // Regularize control Hessian using gradient-norm scaling.
            Eigen::LLT<ActionActionMatrix> llt(regularize(Q_.uu, reg * Q_.u.norm()));
            // NOTE: There is no need to check for success of the factorization,
            // because we know by construction that the argument is strictly positive definite.

            // Solve for policy gains using the Cholesky factorization.
            const ActionVector k = -llt.solve(Q_.u);
            const ActionStateMatrix K = -llt.solve(Q_.xu.transpose());

            // Update value function.
            updateV(k, K);

            // Update expected cost change.
            updateExpectedCostChange(k);

            // Save gains.
            policy.setFeedfrwdGainAt(stage_idx, k);
            policy.setFeedbackGainAt(stage_idx, K);
        }

        return expected_cost_change_;
    }

    // Update the Q function.
    void updateQ(const StateVector& state, const ActionVector& action) {
        // Get loss derivatives.
        const StateActionValueQ l = problem_->loss.gradientAndHessian(state, action);

        // Get dynamics derivatives.
        const Jacobian jac = Dynamics::jacobian(state, action);
        const StateStateMatrix& A = jac.A;
        const StateActionMatrix& B = jac.B;

        // Compute Q-function arrays.
        const StateStateMatrix ATVxx = A.transpose() * V_.xx;
        const ActionStateMatrix BTVxx = B.transpose() * V_.xx;
        Q_.x = l.x + A.transpose() * V_.x;
        Q_.u = l.u + B.transpose() * V_.x;
        Q_.xx = l.xx + ATVxx * A;
        Q_.xu = l.xu + ATVxx * B;
        Q_.uu = l.uu + BTVxx * B;
    }

    // Update the V function.
    void updateV(const ActionVector& k, const ActionStateMatrix& K) {
        const StateActionMatrix KTQuu_plus_Qxu = K.transpose() * Q_.uu + Q_.xu;
        V_.x = Q_.x + KTQuu_plus_Qxu * k + K.transpose() * Q_.u;
        V_.xx = Q_.xx + KTQuu_plus_Qxu * K + K.transpose() * Q_.xu.transpose();
    }

    // Update the expected cost change.
    void updateExpectedCostChange(const ActionVector& k) {
        expected_cost_change_.term1 += k.transpose() * Q_.u;
        expected_cost_change_.term2 += k.transpose() * Q_.uu * k;
    }

   private:
    std::shared_ptr<Problem> problem_;
    StateActionValueQ Q_;
    StateValueV V_;
    ExpectedCostChange expected_cost_change_;
};
