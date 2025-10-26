#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <tuple>
#include <vector>

#include "core/obstacle.h"
#include "core/search_space.h"
#include "core/smooth_functions.h"
#include "core/space.h"
#include "core/trajectory.h"
#include "core/util.h"

// ---- SCENARIO: Slalom with parking space and border.
inline std::vector<Obstacle> makeScenario() {
    static constexpr double ob_spacing_factor = 1.3;
    static constexpr double ob_r = 1.0;
    static constexpr double gap = 6.0;
    static constexpr double gap_half = 0.5 * gap;
    static constexpr double x_mid = X_MIN + 0.5 * (X_MAX - X_MIN);

    // ---- Slalom
    static constexpr double x_offset = 6.0;
    std::vector<Obstacle> obstacles = {
        {{x_mid - x_offset, 1.0 * ob_spacing_factor}, 2.4 * ob_r},
        {{x_mid + x_offset, -1.0 * ob_spacing_factor}, 2.4 * ob_r}};

    // ---- Parking space.
    std::vector<Obstacle> ps_obstacles = {
        {{x_mid - gap_half - ob_r, 1 * ob_spacing_factor * ob_r}, ob_r},
        {{x_mid - gap_half - ob_r, 0 * ob_spacing_factor * ob_r}, ob_r},
        {{x_mid - gap_half - ob_r, -1 * ob_spacing_factor * ob_r}, ob_r},
        {{x_mid - gap_half - ob_r + ob_spacing_factor * ob_r, 1 * ob_spacing_factor * ob_r}, ob_r},
        {{x_mid - gap_half - ob_r + ob_spacing_factor * ob_r, -1 * ob_spacing_factor * ob_r}, ob_r},
        {{x_mid - gap_half - ob_r + 2 * ob_spacing_factor * ob_r, 1 * ob_spacing_factor * ob_r}, ob_r},
        {{x_mid - gap_half - ob_r + 2 * ob_spacing_factor * ob_r, -1 * ob_spacing_factor * ob_r}, ob_r},
        {{x_mid + gap_half + ob_r, 1 * ob_spacing_factor * ob_r}, ob_r},
        {{x_mid + gap_half + ob_r, 0 * ob_spacing_factor * ob_r}, ob_r},
        {{x_mid + gap_half + ob_r, -1 * ob_spacing_factor * ob_r}, ob_r}};

    obstacles.insert(obstacles.end(), ps_obstacles.begin(), ps_obstacles.end());

    // ---- Border
    std::vector<Obstacle> b_obstacles;
    static constexpr float spacing = 5.0;
    static constexpr double radius = 5.0;

    // Top and bottom edges
    for (float x = X_MIN; x <= X_MAX; x += spacing) {
        b_obstacles.push_back({{x, (float)Y_MAX + (float)radius}, radius});
        b_obstacles.push_back({{x, (float)Y_MIN - (float)radius}, radius});
    }

    // Left and right edges
    for (float y = Y_MIN; y <= Y_MAX; y += spacing) {
        b_obstacles.push_back({{(float)X_MIN - (float)radius, y}, radius});
        b_obstacles.push_back({{(float)X_MAX + (float)radius, y}, radius});
    }

    obstacles.insert(obstacles.end(), b_obstacles.begin(), b_obstacles.end());

    return obstacles;
}


extern std::vector<Obstacle> obstacles;


struct VehicleLimits {
    const double speed_max;
    const double speed_min;
    const double accel_lon_max;
    const double accel_lat_max;
    const double curvature_max;
};

struct SoftParams {
    const double accel_lon_scale;
    const double accel_lat_scale;
    const double curvature_scale;
    const double accel_lon_tol;
    const double accel_lat_tol;
    const double curvature_tol;
};

struct VehicleLimitsParams {
    const double speed_lim_scale;
    const double speed_free_pos;
    const double speed_free_neg;
    const double accel_lon_max_scale;
    const double accel_lon_free;
    const double accel_lat_max_scale;
    const double accel_lat_free;
    const double curvature_max_scale;
    const double curvature_free;
};

struct TerminalStateParams {
    const double xy_scale;
    const double xy_tol;
    const double yaw_scale;
    const double yaw_tol;
    const double speed_scale;
    const double speed_tol;
};

struct ObstacleAvoidanceParams {
    // Weight of obstacle loss
    const double weight;

    // Amount of clearance distance before loss starts kicking in.
    // clearance > clearance_free  ->      loss = 0
    // clearance < clearance_free  ->  0 < loss < 1
    // clearance = 0               ->      loss = 1
    const double clearance_free;
};

// Struct representing a linear-quadratic state value (V) function.
struct StateValueV {
    StateVector x;
    StateStateMatrix xx;
};

// Struct representing a linear-quadratic state-action value (Q) function.
struct StateActionValueQ {
    StateVector x;
    ActionVector u;
    StateStateMatrix xx;
    StateActionMatrix xu;
    ActionActionMatrix uu;
};

// TODO move this somewhere better
inline double clearanceLoss(const double c, const double c_free) {
    const double p = c_free - c;
    return (p <= 0.0) ? 0.0 : quart(p) / quart(c_free);
}

inline double clearanceLossGrad(const double c, const double c_free) {
    const double p = c_free - c;
    return (p <= 0.0) ? 0.0 : -4 * (cube(p) / quart(c_free));
}

inline double clearanceLossHess(const double c, const double c_free) {
    const double p = c_free - c;
    return (p <= 0.0) ? 0.0 : 12 * (square(p) / quart(c_free));
}

inline double obstacleLoss(const Obstacle& obstacle, const StateVector& state, const double clearance_free) {
    return clearanceLoss(obstacle.clearance(state), clearance_free);
}

struct GradAndHess2d {
    const Eigen::Vector2d grad;
    const Eigen::Matrix2d hess;
};

inline GradAndHess2d obstacleLossGradAndHess(const Obstacle& obstacle, const StateVector& state, const double clearance_free) {
    Eigen::Vector2d grad = Eigen::Vector2d::Zero();
    Eigen::Matrix2d hess = Eigen::Matrix2d::Zero();

    const Eigen::Vector2d offset = positionDelta(obstacle.center, state);
    double distance = offset.norm();
    const double clearance = distance - obstacle.radius;
    if (clearance < clearance_free) {
        // Clamp distance to avoid division by zero.
        static constexpr double distance_min_denominator = 1e-3;
        distance = std::max(distance, distance_min_denominator);

        const Eigen::Vector2d offset_normalized = offset / distance;
        const double cg = clearanceLossGrad(clearance, clearance_free);
        const double ch = clearanceLossHess(clearance, clearance_free);
        const double cg_over_d = cg / distance;
        grad = cg * offset_normalized;
        hess = (ch - cg_over_d) * (offset_normalized * offset_normalized.transpose()) + cg_over_d * Eigen::Matrix2d::Identity();
    }
    return {grad, hess};
}

inline double multiObstacleLoss(const std::vector<Obstacle>& obstacles, const StateVector& state, const double clearance_free) {
    double loss = 0.0;
    for (const Obstacle& obstacle : obstacles) {
        loss += obstacleLoss(obstacle, state, clearance_free);
    }
    return loss;
}

inline GradAndHess2d multiObstacleLossGradAndHess(const std::vector<Obstacle>& obstacles, const StateVector& state, const double clearance_free) {
    Eigen::Vector2d grad = Eigen::Vector2d::Zero();
    Eigen::Matrix2d hess = Eigen::Matrix2d::Zero();
    for (const Obstacle& obstacle : obstacles) {
        const GradAndHess2d gradAndHessThisObs = obstacleLossGradAndHess(obstacle, state, clearance_free);
        grad += gradAndHessThisObs.grad;
        hess += gradAndHessThisObs.hess;
    }
    return {grad, hess};
}

struct Loss {
    // ---- Attributes

    SoftParams soft_params;
    VehicleLimits vehicle_limits;
    VehicleLimitsParams vehicle_limits_params;
    TerminalStateParams terminal_state_params;
    ObstacleAvoidanceParams obs_avoid_params;
    StateVector terminal_state_target;
    // Scale all non-terminal loss terms by inverse of trajectory length
    // to make terminal loss terms less sensitive to trajectory length.
    double inverse_traj_length;

    // ---- Methods

    double value(const StateVector& state, const ActionVector& action) const {
        // Extract
        const double speed = state(3);
        const double accel_lon = action(0);
        const double curvature = action(1);
        const double accel_lat = square(speed) * curvature;

        // Compute components
        // Soft terms
        const double soft_accel_lon_loss = soft_params.accel_lon_scale * smoothAbs(accel_lon, soft_params.accel_lon_tol);
        const double soft_accel_lat_loss = soft_params.accel_lat_scale * smoothAbs(accel_lat, soft_params.accel_lat_tol);
        const double soft_curvature_loss = soft_params.curvature_scale * smoothAbs(curvature, soft_params.curvature_tol);
        const double soft_loss = soft_accel_lon_loss + soft_accel_lat_loss + soft_curvature_loss;
        // Hard terms
        const double hard_speed_loss = vehicle_limits_params.speed_lim_scale * smoothBoundedDeadzone(speed, vehicle_limits_params.speed_free_pos, vehicle_limits.speed_max, vehicle_limits_params.speed_free_neg, vehicle_limits.speed_min);
        const double hard_accel_lon_loss = vehicle_limits_params.accel_lon_max_scale * smoothBoundedDeadzoneSymmetric(accel_lon, vehicle_limits_params.accel_lon_free, vehicle_limits.accel_lon_max);
        const double hard_accel_lat_loss = vehicle_limits_params.accel_lat_max_scale * smoothBoundedDeadzoneSymmetric(accel_lat, vehicle_limits_params.accel_lat_free, vehicle_limits.accel_lat_max);
        const double hard_curvature_loss = vehicle_limits_params.curvature_max_scale * smoothBoundedDeadzoneSymmetric(curvature, vehicle_limits_params.curvature_free, vehicle_limits.curvature_max);
        const double hard_loss = hard_speed_loss + hard_accel_lon_loss + hard_accel_lat_loss + hard_curvature_loss;

        // Obstacles
        const double obstacle_loss = obs_avoid_params.weight * multiObstacleLoss(obstacles, state, obs_avoid_params.clearance_free);

        return inverse_traj_length * (soft_loss + hard_loss + obstacle_loss);
    }

    StateActionValueQ gradientAndHessian(const StateVector& state, const ActionVector& action) const {
        // ---- Extract states and actions
        const double speed = state(3);
        const double accel_lon = action(0);
        const double curvature = action(1);

        // ---- Intermediate quantities
        const double speed2 = speed * speed;
        const double speed3 = speed * speed2;
        const double speed4 = speed * speed3;
        const double accel_lat = speed2 * curvature;
        const double curvature2 = curvature * curvature;

        // ---- Obstacle loss
        const GradAndHess2d obstacle_grad_and_hess = multiObstacleLossGradAndHess(obstacles, state, obs_avoid_params.clearance_free);
        const Eigen::Vector2d obstacle_grad = obs_avoid_params.weight * obstacle_grad_and_hess.grad;
        const Eigen::Matrix2d obstacle_hess = obs_avoid_params.weight * obstacle_grad_and_hess.hess;

        // ---- Gradient
        // Compute components
        // Soft terms
        const double soft_accel_lat_grad = soft_params.accel_lat_scale * smoothAbsGrad(accel_lat, soft_params.accel_lat_tol);
        const double soft_speed_grad = soft_accel_lat_grad * (2 * speed * curvature);
        const double soft_accel_lon_grad = soft_params.accel_lon_scale * smoothAbsGrad(accel_lon, soft_params.accel_lon_tol);
        const double soft_curvature_grad_from_curvature = soft_params.curvature_scale * smoothAbsGrad(curvature, soft_params.curvature_tol);
        const double soft_curvature_grad_from_accel_lat = soft_accel_lat_grad * speed2;
        const double soft_curvature_grad = soft_curvature_grad_from_curvature + soft_curvature_grad_from_accel_lat;

        // Hard terms
        const double hard_speed_grad_from_speed_lim = vehicle_limits_params.speed_lim_scale * smoothBoundedDeadzoneGrad(speed, vehicle_limits_params.speed_free_pos, vehicle_limits.speed_max, vehicle_limits_params.speed_free_neg, vehicle_limits.speed_min);
        const double hard_accel_lon_grad = vehicle_limits_params.accel_lon_max_scale * smoothBoundedDeadzoneSymmetricGrad(accel_lon, vehicle_limits_params.accel_lon_free, vehicle_limits.accel_lon_max);
        const double hard_curvature_grad_from_curvature_max = vehicle_limits_params.curvature_max_scale * smoothBoundedDeadzoneSymmetricGrad(curvature, vehicle_limits_params.curvature_free, vehicle_limits.curvature_max);
        // Terms due to lateral acceleration limit.
        const double hard_accel_lat_grad = vehicle_limits_params.accel_lat_max_scale * smoothBoundedDeadzoneSymmetricGrad(accel_lat, vehicle_limits_params.accel_lat_free, vehicle_limits.accel_lat_max);
        const double hard_speed_grad_from_accel_lat = 2.0 * speed * curvature * hard_accel_lat_grad;
        const double hard_curvature_grad_from_accel_lat = speed2 * hard_accel_lat_grad;
        // Speed and curvature get contributions from explicit limits on speed and curvature, as well as lateral acceleration.
        const double hard_speed_grad = hard_speed_grad_from_speed_lim + hard_speed_grad_from_accel_lat;
        const double hard_curvature_grad = hard_curvature_grad_from_curvature_max + hard_curvature_grad_from_accel_lat;

        // Add the soft and hard parts.
        const double speed_grad = soft_speed_grad + hard_speed_grad;
        const double accel_lon_grad = soft_accel_lon_grad + hard_accel_lon_grad;
        const double curvature_grad = soft_curvature_grad + hard_curvature_grad;

        // Assemble
        const StateVector lx{inverse_traj_length * obstacle_grad(0), inverse_traj_length * obstacle_grad(1), 0.0, inverse_traj_length * speed_grad};
        const ActionVector lu{inverse_traj_length * accel_lon_grad, inverse_traj_length * curvature_grad};

        // ---- Hessian
        // Compute components
        // Soft terms
        const double soft_accel_lat_hess = soft_params.accel_lat_scale * smoothAbsHess(accel_lat, soft_params.accel_lat_tol);
        const double soft_vv = soft_accel_lat_hess * (2 * speed2 * curvature2) + soft_accel_lat_grad * (2 * curvature);
        const double soft_vk = soft_accel_lat_hess * (speed2 * (2 * speed * curvature)) + soft_accel_lat_grad * (2 * speed);
        const double soft_aa = soft_params.accel_lon_scale * smoothAbsHess(accel_lon, soft_params.accel_lon_tol);
        const double soft_kk_from_curvature = soft_params.curvature_scale * smoothAbsHess(curvature, soft_params.curvature_tol);
        const double soft_kk_from_accel_lat = soft_params.accel_lat_scale * smoothAbsHess(accel_lat, soft_params.accel_lat_tol) * speed4;
        const double soft_kk = soft_kk_from_curvature + soft_kk_from_accel_lat;

        // Hard terms from simple bounds.
        const double hard_vv_from_speed_lim = vehicle_limits_params.speed_lim_scale * smoothBoundedDeadzoneHess(speed, vehicle_limits_params.speed_free_pos, vehicle_limits.speed_max, vehicle_limits_params.speed_free_neg, vehicle_limits.speed_min);
        const double hard_aa_hess = vehicle_limits_params.accel_lon_max_scale * smoothBoundedDeadzoneSymmetricHess(accel_lon, vehicle_limits_params.accel_lon_free, vehicle_limits.accel_lon_max);
        const double hard_kk_from_curvature_max = vehicle_limits_params.curvature_max_scale * smoothBoundedDeadzoneSymmetricHess(curvature, vehicle_limits_params.curvature_free, vehicle_limits.curvature_max);

        // Hard terms from lat accel.
        const double hard_accel_lat_hess = vehicle_limits_params.accel_lat_max_scale * smoothBoundedDeadzoneSymmetricHess(accel_lat, vehicle_limits_params.accel_lat_free, vehicle_limits.accel_lat_max);

        const double hard_vv_from_lat_accel = 4.0 * speed2 * curvature2 * hard_accel_lat_hess + 2.0 * curvature * hard_accel_lat_grad;
        const double hard_kk_from_lat_accel = speed4 * hard_accel_lat_hess;
        const double hard_vk_from_lat_accel = 2.0 * speed * (hard_accel_lat_grad + accel_lat * hard_accel_lat_hess);

        // Add simple and lat accel parts.
        const double hard_vv = hard_vv_from_speed_lim + hard_vv_from_lat_accel;
        const double hard_vk = hard_vk_from_lat_accel;
        const double hard_aa = hard_aa_hess;
        const double hard_kk = hard_kk_from_curvature_max + hard_kk_from_lat_accel;

        // Add soft and hard terms.
        const double vv = soft_vv + hard_vv;
        const double vk = soft_vk + hard_vk;
        const double aa = soft_aa + hard_aa;
        const double kk = soft_kk + hard_kk;

        // Assemble
        const StateStateMatrix lxx{
            {inverse_traj_length * obstacle_hess(0, 0), inverse_traj_length * obstacle_hess(0, 1), 0.0, 0.0},
            {inverse_traj_length * obstacle_hess(1, 0), inverse_traj_length * obstacle_hess(1, 1), 0.0, 0.0},
            {0.0, 0.0, 0.0, 0.0},
            {0.0, 0.0, 0.0, inverse_traj_length * vv},
        };
        const StateActionMatrix lxu{
            {0.0, 0.0},
            {0.0, 0.0},
            {0.0, 0.0},
            {0.0, inverse_traj_length * vk},
        };
        const ActionActionMatrix luu{
            {inverse_traj_length * aa, 0.0},
            {0.0, inverse_traj_length * kk},
        };

        return {lx, lu, lxx, lxu, luu};
    }

    double terminalValue(const StateVector& state) const {
        // Deltas
        const StateVector delta = state - terminal_state_target;
        const double dx = delta(0);
        const double dy = delta(1);
        const double dyaw = delta(2);
        const double dv = delta(3);

        // Compute components
        const double xy_loss = terminal_state_params.xy_scale * (smoothAbs(dx, terminal_state_params.xy_tol) + smoothAbs(dy, terminal_state_params.xy_tol));
        const double yaw_loss = terminal_state_params.yaw_scale * smoothAbs(dyaw, terminal_state_params.yaw_tol);
        const double v_loss = terminal_state_params.speed_scale * smoothAbs(dv, terminal_state_params.speed_tol);

        // Assemble
        return xy_loss + yaw_loss + v_loss;
    }

    StateValueV terminalGradientAndHessian(const StateVector& state) const {
        // TODO include obstacle loss

        // ---- Deltas
        const StateVector delta = state - terminal_state_target;
        const double dx = delta(0);
        const double dy = delta(1);
        const double dyaw = delta(2);
        const double dv = delta(3);

        // ---- Gradient
        const double x_grad = terminal_state_params.xy_scale * smoothAbsGrad(dx, terminal_state_params.xy_tol);
        const double y_grad = terminal_state_params.xy_scale * smoothAbsGrad(dy, terminal_state_params.xy_tol);
        const double yaw_grad = terminal_state_params.yaw_scale * smoothAbsGrad(dyaw, terminal_state_params.yaw_tol);
        const double v_grad = terminal_state_params.speed_scale * smoothAbsGrad(dv, terminal_state_params.speed_tol);
        const StateVector lx{x_grad, y_grad, yaw_grad, v_grad};

        // ---- Hessian
        const double x_hess = terminal_state_params.xy_scale * smoothAbsHess(dx, terminal_state_params.xy_tol);
        const double y_hess = terminal_state_params.xy_scale * smoothAbsHess(dy, terminal_state_params.xy_tol);
        const double yaw_hess = terminal_state_params.yaw_scale * smoothAbsHess(dyaw, terminal_state_params.yaw_tol);
        const double v_hess = terminal_state_params.speed_scale * smoothAbsHess(dv, terminal_state_params.speed_tol);
        const StateStateMatrix lxx{
            {x_hess, 0.0, 0.0, 0.0},
            {0.0, y_hess, 0.0, 0.0},
            {0.0, 0.0, yaw_hess, 0.0},
            {0.0, 0.0, 0.0, v_hess},
        };

        return {lx, lxx};
    }

    template <int N>
    double totalValue(const Trajectory<N>& traj) const {
        double val = 0.0;
        for (size_t stage_idx = 0; stage_idx < traj.length; ++stage_idx) {
            val += value(traj.stateAt(stage_idx), traj.actionAt(stage_idx));
        }
        val += terminalValue(traj.stateTerminal());
        return val;
    }
};
