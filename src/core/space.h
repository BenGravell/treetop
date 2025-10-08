#pragma once

#include <Eigen/Dense>

// [x, y, yaw, speed]
static constexpr uint64_t NUM_STATES = 4;

// [acceleration, curvature]
static constexpr uint64_t NUM_ACTIONS = 2;

using StateVector = Eigen::Vector<double, NUM_STATES>;
using ActionVector = Eigen::Vector<double, NUM_ACTIONS>;

using StateStateMatrix = Eigen::Matrix<double, NUM_STATES, NUM_STATES>;
using StateActionMatrix = Eigen::Matrix<double, NUM_STATES, NUM_ACTIONS>;
using ActionActionMatrix = Eigen::Matrix<double, NUM_ACTIONS, NUM_ACTIONS>;
using ActionStateMatrix = Eigen::Matrix<double, NUM_ACTIONS, NUM_STATES>;
