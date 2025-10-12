#pragma once

#include <random>

std::mt19937 rng(std::random_device{}());
