#pragma once

#include "core/space.h"
#include "core/trajectory.h"
#include "core/util.h"

inline Eigen::Vector2d positionDelta(const Vector2& position, const StateVector& state) {
    return state.head(2) - Eigen::Vector2d(position.x, position.y);
}

inline double distanceSquared(const Vector2& position, const StateVector& state) {
    return positionDelta(position, state).squaredNorm();
}

inline double distance(const Vector2& position, const StateVector& state) {
    return positionDelta(position, state).norm();
}

struct Obstacle {
    const Vector2 center;
    const double radius;

    double clearance(const StateVector& state) const {
        return distance(center, state) - radius;
    }

    bool collidesWith(const StateVector& state) const {
        return distanceSquared(center, state) < square(radius);
    }

    template <int N>
    bool collidesWith(const Trajectory<N>& traj) const {
        for (int stage_ix = N; stage_ix >= 0; --stage_ix) {
            if (collidesWith(traj.stateAt(stage_ix))) {
                return true;
            }
        }
        return false;
    }
};

inline bool obstaclesCollidesWith(std::vector<Obstacle> obstacles, const StateVector& state) {
    for (const Obstacle& obstacle : obstacles) {
        if (obstacle.collidesWith(state)) {
            return true;
        }
    }
    return false;
}

template <int N>
bool obstaclesCollidesWith(std::vector<Obstacle> obstacles, const Trajectory<N>& traj) {
    for (const Obstacle& obstacle : obstacles) {
        if (obstacle.collidesWith(traj)) {
            return true;
        }
    }
    return false;
}
