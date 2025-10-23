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
