#pragma once

#include <random>

inline thread_local std::mt19937 rng(std::random_device{}());
