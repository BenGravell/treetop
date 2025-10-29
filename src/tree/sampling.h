#pragma once

#include "core/rng.h"
#include "core/space.h"

// Sampling settings
static constexpr double goal_sampling_proba = 0.1;
static constexpr double warm_sampling_proba = 0.2;
static constexpr double cold_sampling_proba = 1.0 - warm_sampling_proba - goal_sampling_proba;

struct SamplingSettings {
    bool use_warm;
    bool use_cold;
    bool use_goal;
};

enum class SampleReason : uint8_t {
    kZeroActionPoint = 0,
    kHot = 1,
    kWarm = 2,
    kCold = 3,
    kGoal = 4
};

struct StateAndReason {
    StateVector state;
    SampleReason reason;
};

inline double urand() {
    static std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(rng);
}

inline double urand(const double x_min, const double x_max) {
    std::uniform_real_distribution<double> dist(x_min, x_max);
    return dist(rng);
}

inline StateVector sampleCold() {
    const double x = urand(X_MIN, X_MAX);
    const double y = urand(Y_MIN, Y_MAX);
    const double yaw = urand(YAW_MIN, YAW_MAX);
    const double v = urand(V_MIN, V_MAX);
    return {x, y, yaw, v};
}

inline StateVector sampleNear(const StateVector& state, const double perturb_factor = 1.0) {
    const double d = perturb_factor * 2.0;
    const double d_yaw = perturb_factor * 0.5 * PI;
    const double dv = perturb_factor * 4.0;

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

inline StateVector sampleWarm(const Trajectory<TRAJ_LENGTH_OPT>& warm_traj, const int time_ix) {
    const int stage_ix = time_ix * TRAJ_LENGTH_STEER;
    const StateVector& state = warm_traj.stateAt(stage_ix);
    return sampleNear(state);
}

inline StateAndReason sample(const StateVector& goal, const std::optional<Trajectory<TRAJ_LENGTH_OPT>>& warm_traj, const int time_ix, const SamplingSettings& settings) {
    // Compute total active probability
    double p_total = 0.0;
    if (settings.use_goal) {
        p_total += goal_sampling_proba;
    }
    if (settings.use_warm && warm_traj) {
        p_total += warm_sampling_proba;
    }
    if (settings.use_cold) {
        p_total += cold_sampling_proba;
    }

    if (p_total < 1e-6) {
        // All disabled: fallback deterministically to goal
        return {goal, SampleReason::kGoal};
    }

    // Draw random value in [0, p_total)
    const double selector = urand(0.0, p_total);

    double acc = 0.0;

    // Sample goal
    if (settings.use_goal) {
        acc += goal_sampling_proba;
        if (selector < acc) {
            return {goal, SampleReason::kGoal};
        }
    }

    // Sample near warm trajectory
    if (settings.use_warm && warm_traj) {
        acc += warm_sampling_proba;
        if (selector < acc) {
            return {sampleWarm(warm_traj.value(), time_ix), SampleReason::kWarm};
        }
    }

    // Sample cold
    if (settings.use_cold) {
        acc += cold_sampling_proba;
        if (selector < acc) {
            return {sampleCold(), SampleReason::kCold};
        }
    }

    // Fallback (should not happen)
    return {goal, SampleReason::kGoal};
}
