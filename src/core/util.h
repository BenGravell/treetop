#pragma once

#include "core/constants.h"

template <typename T>
T square(const T x) {
    return x * x;
}

template <typename T>
T cube(const T x) {
    return x * square(x);
}

template <typename T>
T quart(const T x) {
    return square(square(x));
}

// Simple hypotenuse. Naive (but faster) version of std::hypot.
template <typename T>
T shypot(const T x, const T y) {
    return std::sqrt(square(x) + square(y));
}

template <typename T>
T rad2deg(const T x) {
    return RAD2DEG * x;
}

template <typename T>
T deg2rad(const T x) {
    return DEG2RAD * x;
}

template <typename T>
T angularDifference(const T angle1, const T angle2) {
    // Compute the difference and shift by PI
    T diff = std::fmod(angle2 - angle1 + PI, 2 * PI);
    // Adjust if fmod returns a negative result
    if (diff < 0) {
        diff += 2 * PI;
    }
    // Shift back by subtracting PI to get a range of [-PI, PI]
    return diff - PI;
}