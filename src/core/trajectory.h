#pragma once

#include <Eigen/Dense>
#include <cstddef>
#include <vector>

#include "core/space.h"

// Length of trajectory for steering function, i.e. for one node.
// NOTE: Need to choose a reasonable middle ground.
// 1. Larger TRAJ_LENGTH_STEER -> more reliance on steering function, more nodes per stage.
//    - Typically helps the tree explore state space faster, less coverage.
//    - May miss manuevers involving cusps / direction change.
// 2. Smaller TRAJ_LENGTH_STEER -> less reliance on steering function, less nodes per stage.
//    - Typically makes the tree explore state space slower, less coverage.
//    - Much faster and reliable at finding maneuvers involving cusps / direction change.
static constexpr uint64_t TRAJ_LENGTH_STEER = 5;

// Length of trajectory for trajectory optimization, i.e. the entire trajectory.
static constexpr uint64_t TRAJ_LENGTH_OPT = 100;

static_assert((TRAJ_LENGTH_OPT % TRAJ_LENGTH_STEER) == 0, "TRAJ_LENGTH_OPT must be a multiple of TRAJ_LENGTH_STEER");

// Duration of a single step, seconds
static constexpr double DT = 0.1;

// Duration of a steering function trajectory, in seconds.
static constexpr double TRAJ_DURATION_STEER = DT * TRAJ_LENGTH_STEER;

// Duration of a trajectory optimization trajectory, in seconds.
static constexpr double TRAJ_DURATION_OPT = DT * TRAJ_LENGTH_OPT;

template <int N>
using StateSequence = Eigen::Matrix<double, NUM_STATES, N + 1>;

template <int N>
using ActionSequence = Eigen::Matrix<double, NUM_ACTIONS, N>;

template <int N>
struct Trajectory {
    static constexpr int length = N;

    StateSequence<N> state_sequence;
    ActionSequence<N> action_sequence;

    // Getter for the state at a specific stage index.
    StateVector stateAt(const size_t stage_idx) const {
        return state_sequence.col(stage_idx);
    }

    // Getter for the action at a specific stage index.
    ActionVector actionAt(const size_t stage_idx) const {
        return action_sequence.col(stage_idx);
    }

    // Getter for the terminal state.
    StateVector stateTerminal() const {
        return state_sequence.col(state_sequence.cols() - 1);
    }

    // Setter for the state at a specific stage index.
    void setStateAt(const size_t stage_idx, const StateVector& state) {
        state_sequence.col(stage_idx) = state;
    }

    // Setter for the action at a specific stage index.
    void setActionAt(const size_t stage_idx, const ActionVector& action) {
        action_sequence.col(stage_idx) = action;
    }
};
