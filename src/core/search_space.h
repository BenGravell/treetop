#pragma once

#include "core/constants.h"
#include "core/space.h"

static constexpr double X_MAX = 40.0;
static constexpr double X_MIN = 0.0;
static constexpr double X_SIZE = X_MAX - X_MIN;

static constexpr double Y_MAX = 5.0;
static constexpr double Y_MIN = -Y_MAX;
static constexpr double Y_SIZE = Y_MAX - Y_MIN;

static constexpr double YAW_MAX = 0.5 * M_PI;
static constexpr double YAW_MIN = -YAW_MAX;

inline StateVector clampToSearchSpace(const StateVector& state) {
    const double x = std::clamp(state(0), X_MIN, X_MAX);
    const double y = std::clamp(state(1), Y_MIN, Y_MAX);
    const double yaw = std::clamp(state(2), YAW_MIN, YAW_MAX);
    const double v = std::clamp(state(3), V_MIN, V_MAX);
    return {x, y, yaw, v};
}
