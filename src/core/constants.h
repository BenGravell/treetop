#pragma once

static const double LOG2 = std::log(2.0);

// TODO move to a config struct - VehicleLimits ?
// TODO - move to search_space.h ?
static constexpr double V_MAX = 10.0;
static constexpr double V_MIN = -V_MAX;

// static constexpr double V_MAX = 40.0;   // ~90 mph
// static constexpr double V_MIN = -10.0;  // ~22 mph

static constexpr double V_ABS_MIN_FOR_STEER = 0.01;

static constexpr double ACCEL_LON_MAX = 3.0;  // ~0.3g
static constexpr double ACCEL_LAT_MAX = 6.0;  // ~0.6g
static constexpr double CURVATURE_MAX = 0.2;  // 5 meter turning radius, typical for small sedan