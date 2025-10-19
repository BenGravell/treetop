#pragma once

#include <Eigen/Dense>
#include <array>
#include <memory>
#include <vector>

#include "core/constants.h"
#include "core/loss.h"
#include "core/trajectory.h"

// Struct for a trajectory optimization problem
struct Problem {
    Loss loss;
    StateVector initial_state;
    double total_time;
};

inline Problem makeProblem(const StateVector initial_state, const StateVector terminal_state_target, const double total_time) {
    // ---- Define the loss function.

    const double inverse_traj_length = DT / total_time;

    // Soft terms
    static constexpr double soft_scale = 0.1;
    static constexpr double accel_lon_scale = 5.0 * soft_scale;
    static constexpr double accel_lat_scale = 5.0 * soft_scale;
    static constexpr double curvature_scale = 1.0 * soft_scale;
    static constexpr double accel_lon_tol = 0.5;
    static constexpr double accel_lat_tol = 0.5;
    static constexpr double curvature_tol = 0.05;
    static constexpr SoftParams soft_params{accel_lon_scale,
                                            accel_lat_scale,
                                            curvature_scale,
                                            accel_lon_tol,
                                            accel_lat_tol,
                                            curvature_tol};

    // Hard terms
    static constexpr VehicleLimits vehicle_limits{V_MAX,
                                                  V_MIN,
                                                  ACCEL_LON_MAX,
                                                  ACCEL_LAT_MAX,
                                                  CURVATURE_MAX};

    static constexpr double speed_lim_scale = 0.01;
    static constexpr double accel_lon_max_scale = 0.01;
    static constexpr double accel_lat_max_scale = 0.01;
    static constexpr double curvature_max_scale = 0.01;

    static constexpr double speed_free_pos = 0.99 * V_MAX;
    static constexpr double speed_free_neg = 0.99 * V_MIN;
    static constexpr double accel_lon_free = 0.99 * ACCEL_LON_MAX;
    static constexpr double accel_lat_free = 0.99 * ACCEL_LAT_MAX;
    static constexpr double curvature_free = 0.99 * CURVATURE_MAX;

    static constexpr VehicleLimitsParams vehicle_limits_params{speed_lim_scale,
                                                               speed_free_pos,
                                                               speed_free_neg,
                                                               accel_lon_max_scale,
                                                               accel_lon_free,
                                                               accel_lat_max_scale,
                                                               accel_lat_free,
                                                               curvature_max_scale,
                                                               curvature_free};

    // Terminal state terms
    static constexpr double terminal_xy_scale = 1.0;        // 1 m per m
    static constexpr double terminal_xy_tol = 0.01;         // 1 cm
    static constexpr double terminal_yaw_scale = 5.0 / PI;  // 5 m per 180 deg
    static constexpr double terminal_yaw_tol = 0.02;        // ~1 degree
    static constexpr double terminal_speed_scale = 0.5;     // 1 m per 0.5 m/s
    static constexpr double terminal_speed_tol = 0.01;      // 1 cm/s

    static constexpr TerminalStateParams terminal_state_params{terminal_xy_scale,
                                                               terminal_xy_tol,
                                                               terminal_yaw_scale,
                                                               terminal_yaw_tol,
                                                               terminal_speed_scale,
                                                               terminal_speed_tol};

    // Obstacle terms
    static constexpr double obstacle_loss_weight = 10.0;
    static constexpr double clearance_free = 0.05;
    static constexpr ObstacleAvoidanceParams obs_avoid_params{obstacle_loss_weight, clearance_free};

    const Loss loss{soft_params, vehicle_limits, vehicle_limits_params, terminal_state_params, obs_avoid_params, terminal_state_target, inverse_traj_length};

    return Problem{loss, initial_state, total_time};
}
