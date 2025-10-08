#pragma once

#include <Eigen/Dense>
#include <utility>

#include "core/space.h"

struct Jacobian {
    const StateStateMatrix A;
    const StateActionMatrix B;
};

// Dynamics of a kinematic bicycle with forward Euler discretization.
struct Dynamics {
    static StateVector forward(const StateVector& state, const ActionVector& action) {
        // Extract states and actions.
        const double yaw = state(2);
        const double speed = state(3);
        const double accel = action(0);
        const double curvature = action(1);

        // Assemble output.
        return state + DT * StateVector{
                                speed * std::cos(yaw),
                                speed * std::sin(yaw),
                                speed * curvature,
                                accel};
    }

    static Jacobian jacobian(const StateVector& state, const ActionVector& action) {
        // Extract states and actions.
        const double yaw = state(2);
        const double speed = state(3);
        const double accel = action(0);
        const double curvature = action(1);

        // Compute intermediate quantities.
        const double dt_cos_yaw = DT * std::cos(yaw);
        const double dt_sin_yaw = DT * std::sin(yaw);

        // Assemble output.
        StateStateMatrix A;
        A << 1, 0, -speed * dt_sin_yaw, dt_cos_yaw,
            0, 1, speed * dt_cos_yaw, dt_sin_yaw,
            0, 0, 1, 0,
            0, 0, 0, 1;

        StateActionMatrix B;
        B << 0, 0,
            0, 0,
            0, speed * DT,
            DT, 0;

        return {A, B};
    }
};
