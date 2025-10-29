#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "core/constants.h"
#include "core/space.h"
#include "core/trajectory.h"
#include "core/util.h"

struct BoundaryCondition {
    const double x;
    const double v;
};

struct BoundaryConditionsStartEnd {
    const BoundaryCondition start;
    const BoundaryCondition end;
};

struct BoundaryConditionsXY {
    const BoundaryConditionsStartEnd x;
    const BoundaryConditionsStartEnd y;
};

// Coefficients defining a cubic polynomial.
// p(x) = a3 * x^3 + a2 * x^2 + a1 * x + a0
struct CubicCoeffs {
    const double a3;
    const double a2;
    const double a1;
    const double a0;
};

// Convert boundary conditions to cubic polynomial coefficients.
// Duration t must be > 0
inline CubicCoeffs bc2coeffs(const BoundaryConditionsStartEnd& bc, const double t) {
    const double t2 = square(t);
    const double t3 = t2 * t;

    const double dx = bc.end.x - bc.start.x;
    const double v_sum = bc.start.v + bc.end.v;

    const double a0 = bc.start.x;
    const double a1 = bc.start.v;
    const double a2 = (3.0 * dx) / t2 - (bc.start.v + v_sum) / t;
    const double a3 = (t * v_sum - 2.0 * dx) / t3;

    return {a3, a2, a1, a0};
}

// Convert full state (x, y, yaw, v) start and goal states to boundary conditions in x and y.
inline BoundaryConditionsXY states2bcs(const StateVector& start, const StateVector& goal) {
    const double x0 = start(0);
    const double y0 = start(1);
    const double yaw_0 = start(2);
    const double v0 = start(3);

    const double x1 = goal(0);
    const double y1 = goal(1);
    const double yaw_1 = goal(2);
    const double v1 = goal(3);

    const double vx0 = v0 * std::cos(yaw_0);
    const double vy0 = v0 * std::sin(yaw_0);

    const double vx1 = v1 * std::cos(yaw_1);
    const double vy1 = v1 * std::sin(yaw_1);

    const BoundaryConditionsStartEnd x_bc{{x0, vx0}, {x1, vx1}};
    const BoundaryConditionsStartEnd y_bc{{y0, vy0}, {y1, vy1}};

    return {x_bc, y_bc};
}

inline double polyval0(const CubicCoeffs coeffs, const double t1) {
    const double t2 = t1 * t1;
    const double t3 = t2 * t1;
    return coeffs.a3 * t3 + coeffs.a2 * t2 + coeffs.a1 * t1 + coeffs.a0;
}

inline double polyval1(const CubicCoeffs coeffs, const double t1) {
    const double t2 = t1 * t1;
    return 3.0 * coeffs.a3 * t2 + 2.0 * coeffs.a2 * t1 + coeffs.a1;
}

inline double polyval2(const CubicCoeffs coeffs, const double t1) {
    return 6.0 * coeffs.a3 * t1 + 2.0 * coeffs.a2;
}

inline int inferTrajectoryDirection(const CubicCoeffs x_coeffs, const CubicCoeffs y_coeffs, const double dt, const double yaw0) {
    // Compute components of a small "secant" stub of the path just after the start,
    // using dt as the secant duration.
    const double t0 = 0.0;
    const double t1 = dt;

    const double x0 = polyval0(x_coeffs, t0);
    const double y0 = polyval0(y_coeffs, t0);
    const double x1 = polyval0(x_coeffs, t1);
    const double y1 = polyval0(y_coeffs, t1);
    const double dx = x1 - x0;
    const double dy = y1 - y0;

    // Compute x- and y-components of the initial yaw.
    const double cos_yaw0 = std::cos(yaw0);
    const double sin_yaw0 = std::sin(yaw0);

    // Dot product between path secant and initial yaw.
    const double dot = dx * cos_yaw0 + dy * sin_yaw0;

    // +1 if direction aligns with yaw (forward), -1 if opposite (reverse)
    return (dot >= 0.0) ? 1 : -1;
}

template <int N>
inline ActionSequence<N> steerCubic(const StateVector& start, const StateVector& goal, const double duration, const double dt) {
    const BoundaryConditionsXY bcs = states2bcs(start, goal);

    const CubicCoeffs x_coeffs = bc2coeffs(bcs.x, duration);
    const CubicCoeffs y_coeffs = bc2coeffs(bcs.y, duration);

    // Infer the trajectory direction.
    const int traj_dir = inferTrajectoryDirection(x_coeffs, y_coeffs, dt, start(2));

    ActionSequence<N> action_sequence;
    for (int i = 0; i < N; ++i) {
        // Sample time at midpoint of the current segment.
        // This yields a bit more accurate sampling compared to sampling times at the start or end of the segment.
        const double t = (i + 0.5) * dt;

        // Get first and second derivatives of x- and y-components of motion with respect to time using analytic polynomial expressions.
        const double dxdt = polyval1(x_coeffs, t);
        const double dydt = polyval1(y_coeffs, t);
        const double d2xdt2 = polyval2(x_coeffs, t);
        const double d2ydt2 = polyval2(y_coeffs, t);

        // Speed and its cube.
        const double v = shypot(dxdt, dydt);
        const double v3 = cube(v);

        // Compute actions.
        double accel = 0;
        double curvature = 0;
        if (v > V_ABS_MIN_FOR_STEER) {
            // Nominal case.
            // Standard equation for acceleration and curvature of a point moving along a curve.
            accel = (dxdt * d2xdt2 + dydt * d2ydt2) / v;
            curvature = (dxdt * d2ydt2 - dydt * d2xdt2) / v3;
        } else {
            // Alternate expression for low speed to prevent division by zero
            // and avoid a singularity.
            accel = shypot(d2xdt2, d2ydt2);
            curvature = 0.0;
        }
        // Assemble and apply trajectory direction.
        const ActionVector action{accel, curvature};
        action_sequence.col(i) = traj_dir * action;
    }

    return action_sequence;
}
