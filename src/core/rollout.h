#pragma once

#include <Eigen/Dense>
#include <vector>

#include "core/constants.h"
#include "core/dynamics.h"
#include "core/policy.h"
#include "core/space.h"
#include "core/trajectory.h"

// Special function to efficiently compute the end state after using zero action for t seconds.
StateVector rolloutZeroAction(const StateVector& start, const double t) {
    const double x = start(0);
    const double y = start(1);
    const double yaw = start(2);
    const double v = start(3);

    const double vx = v * std::cos(yaw);
    const double vy = v * std::sin(yaw);
    const double dx = vx * t;
    const double dy = vy * t;

    return {x + dx, y + dy, yaw, v};
}

template <int N>
inline void rolloutOpenLoop(const ActionSequence<N>& action_sequence, const StateVector& initial_state, Trajectory<N>& traj) {
    // Initialize the first state in the state sequence.
    traj.setStateAt(0, initial_state);

    // Simulate dynamics forward using open-loop action sequence.
    for (size_t stage_idx = 0; stage_idx < traj.length; ++stage_idx) {
        traj.setActionAt(stage_idx, action_sequence.col(stage_idx));
        traj.setStateAt(stage_idx + 1, Dynamics::forward(traj.stateAt(stage_idx), traj.actionAt(stage_idx)));
    }
}

template <int N>
inline void rolloutOpenLoopConstrained(const ActionSequence<N>& action_sequence, const StateVector& initial_state, Trajectory<N>& traj) {
    // Initialize the first state in the state sequence.
    traj.setStateAt(0, initial_state);

    // Simulate dynamics forward using open-loop action sequence.
    for (size_t stage_idx = 0; stage_idx < traj.length; ++stage_idx) {
        // Extract state and action.
        const StateVector& state = traj.stateAt(stage_idx);
        ActionVector action = action_sequence.col(stage_idx);

        // Extract intermediate quantities.
        const double v_sq = square(state(3));
        double lon_accel = action(0);
        double curvature = action(1);

        // Minimum squared speed to prevent division by zero.
        static constexpr double v_sq_min = 1e-6;

        // Maximum curvature limit due to lateral acceleration limit.
        const double a_curvature_max = ACCEL_LAT_MAX / std::max(v_sq, v_sq_min);

        // Dynamic curvature limit, the more restrictive of
        // 1. Static max curvature limit.
        // 2. Lateral acceleration induced max curvature limit.
        const double dyn_curvature_max = std::min(CURVATURE_MAX, a_curvature_max);

        // Clamp action.
        lon_accel = std::clamp(lon_accel, -ACCEL_LON_MAX, ACCEL_LON_MAX);
        curvature = std::clamp(curvature, -dyn_curvature_max, dyn_curvature_max);
        action << lon_accel, curvature;

        // Simulate forward one step.
        traj.setActionAt(stage_idx, action);
        traj.setStateAt(stage_idx + 1, Dynamics::forward(state, action));
    }
}

template <int N>
inline void rolloutClosedLoop(const Policy<N>& policy, const Trajectory<N>& traj_ref, Trajectory<N>& traj) {
    // Copy initial state from reference trajectory to trajectory.
    traj.setStateAt(0, traj_ref.stateAt(0));

    // Simulate dynamics forward using closed-loop feedback policy.
    for (size_t stage_idx = 0; stage_idx < traj.length; ++stage_idx) {
        // Get states.
        const StateVector& state_ref = traj_ref.stateAt(stage_idx);
        const StateVector& state = traj.stateAt(stage_idx);

        // Compute state deviation.
        const StateVector state_dev = state - state_ref;

        // Compute action as feedforward + feedback control.
        const ActionVector action = traj_ref.actionAt(stage_idx) + policy.act(state_dev, stage_idx);
        traj.setActionAt(stage_idx, action);

        // Simulate forward one step.
        traj.setStateAt(stage_idx + 1, Dynamics::forward(state, action));
    }
}
