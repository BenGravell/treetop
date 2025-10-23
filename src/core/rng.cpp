#include "rng.h"

std::mt19937 rng(std::random_device{}());
