#pragma once

#include <cmath>

#include "core/constants.h"

// Newton's method was used to find this constant, which makes
// smoothBounded(1.0, 1.0) == 1.0
static constexpr double SB_SCALE = 0.2613267250765282656566;
static constexpr double SB_SCALE2 = SB_SCALE * SB_SCALE;

// ---- smoothAbs

// Smooth absolute value function.
// The approximation is close to std::abs()   outside the neighborhood [-a, a],
//                  and close to (x^2) / (2*a) inside the neighborhood [-a, a].
template <typename T>
T smoothAbs(T x, T a) {
    // Use std::log1p() for numerical stability.
    // Same as a * (log(exp(a_inv * x) + exp(-a_inv * x)) - log(2))
    const T a_inv = 1.0 / a;
    const T abs_x_over_a = a_inv * std::abs(x);
    return a * (abs_x_over_a + std::log1p(std::exp(-2.0 * abs_x_over_a)) - LOG2);
}

// Gradient of smoothAbs.
template <typename T>
T smoothAbsGrad(T x, T a) {
    return std::tanh(x / a);
}

// Hessian of smoothAbs.
template <typename T>
T smoothAbsHess(T x, T a) {
    const T a_inv = 1.0 / a;
    const T tanh_val = std::tanh(a_inv * x);
    return a_inv * (1.0 - tanh_val * tanh_val);
}

// ---- smoothBounded

// Smooth bounded function.
template <typename T>
T smoothBounded(T x) {
    return SB_SCALE2 * (std::cosh(x / SB_SCALE) - 1.0) - 0.5 * x * x;
}

// Gradient of smoothBounded.
template <typename T>
T smoothBoundedGrad(T x) {
    return SB_SCALE * std::sinh(x / SB_SCALE) - x;
}

// Hessian of smoothBounded.
template <typename T>
T smoothBoundedHess(T x) {
    return std::cosh(x / SB_SCALE) - 1.0;
}

// ---- smoothBoundedDeadzone

template <typename T>
T smoothBoundedDeadzone(const T x, const T x_free_pos, const T x_limit_pos, const T x_free_neg, const T x_limit_neg) {
    if ((x_free_neg <= x) && (x <= x_free_pos)) {
        return 0.0;
    }
    const bool pos = x > 0;
    const T x_free = pos ? x_free_pos : x_free_neg;
    const T x_limit = pos ? x_limit_pos : x_limit_neg;
    return smoothBounded((x - x_free) / (x_limit - x_free));
}

template <typename T>
T smoothBoundedDeadzoneGrad(const T x, const T x_free_pos, const T x_limit_pos, const T x_free_neg, const T x_limit_neg) {
    if ((x_free_neg <= x) && (x <= x_free_pos)) {
        return 0.0;
    }
    const bool pos = x > 0;
    const T x_free = pos ? x_free_pos : x_free_neg;
    const T x_limit = pos ? x_limit_pos : x_limit_neg;
    const T scale = 1.0 / (x_limit - x_free);
    return scale * smoothBoundedGrad((x - x_free) * scale);
}

template <typename T>
T smoothBoundedDeadzoneHess(const T x, const T x_free_pos, const T x_limit_pos, const T x_free_neg, const T x_limit_neg) {
    if ((x_free_neg <= x) && (x <= x_free_pos)) {
        return 0.0;
    }
    const bool pos = x > 0;
    const T x_free = pos ? x_free_pos : x_free_neg;
    const T x_limit = pos ? x_limit_pos : x_limit_neg;
    const T scale = 1.0 / (x_limit - x_free);
    return (scale * scale) * smoothBoundedHess((x - x_free) * scale);
}

// ---- smoothBoundedDeadzoneSymmetric

template <typename T>
T smoothBoundedDeadzoneSymmetric(const T x, const T x_free, const T x_limit) {
    return smoothBoundedDeadzone(x, x_free, x_limit, -x_free, -x_limit);
}

template <typename T>
T smoothBoundedDeadzoneSymmetricGrad(const T x, const T x_free, const T x_limit) {
    return smoothBoundedDeadzoneGrad(x, x_free, x_limit, -x_free, -x_limit);
}

template <typename T>
T smoothBoundedDeadzoneSymmetricHess(const T x, const T x_free, const T x_limit) {
    return smoothBoundedDeadzoneHess(x, x_free, x_limit, -x_free, -x_limit);
}
