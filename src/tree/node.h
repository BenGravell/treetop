#pragma once

#include <optional>

#include "core/space.h"
#include "core/trajectory.h"
#include "tree/sampling.h"

struct Node;

using NodePtr = std::shared_ptr<Node>;

struct Node {
    // State of the node.
    const StateVector state;

    // Pointer to parent node.
    const NodePtr parent;

    // Trajectory leading from parent->state to this->state.
    // Satisfies endpoint conditions:
    // 1. parent->state == traj.stateAt(0)
    // 2.   this->state == traj.stateTerminal()
    const std::optional<Trajectory<TRAJ_LENGTH_STEER>> traj;

    // Cost of traj.
    const double cost;

    // Cost to come to this node from the root of the tree.
    const double cost_to_come;

    // Reason why this node was sampled.
    const SampleReason reason;

    // Whether this node is near the goal.
    const bool near_goal{false};
};
