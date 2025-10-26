#pragma once

#include <cmath>
#include <vector>

#include "core/obstacle.h"
#include "core/search_space.h"

// ---- SCENARIO: Slalom with parking space and border.
inline std::vector<Obstacle> makeScenario() {
    static constexpr double ob_spacing_factor = 1.3;
    static constexpr double ob_r = 1.0;
    static constexpr double gap = 6.0;
    static constexpr double gap_half = 0.5 * gap;
    static constexpr double x_mid = X_MIN + 0.5 * (X_MAX - X_MIN);

    // ---- Slalom
    static constexpr double x_offset = 6.0;
    std::vector<Obstacle> obstacles = {
        {{x_mid - x_offset, 1.0 * ob_spacing_factor}, 2.4 * ob_r},
        {{x_mid + x_offset, -1.0 * ob_spacing_factor}, 2.4 * ob_r}};

    // ---- Parking space.
    std::vector<Obstacle> ps_obstacles = {
        {{x_mid - gap_half - ob_r, 1 * ob_spacing_factor * ob_r}, ob_r},
        {{x_mid - gap_half - ob_r, 0 * ob_spacing_factor * ob_r}, ob_r},
        {{x_mid - gap_half - ob_r, -1 * ob_spacing_factor * ob_r}, ob_r},
        {{x_mid - gap_half - ob_r + ob_spacing_factor * ob_r, 1 * ob_spacing_factor * ob_r}, ob_r},
        {{x_mid - gap_half - ob_r + ob_spacing_factor * ob_r, -1 * ob_spacing_factor * ob_r}, ob_r},
        {{x_mid - gap_half - ob_r + 2 * ob_spacing_factor * ob_r, 1 * ob_spacing_factor * ob_r}, ob_r},
        {{x_mid - gap_half - ob_r + 2 * ob_spacing_factor * ob_r, -1 * ob_spacing_factor * ob_r}, ob_r},
        {{x_mid + gap_half + ob_r, 1 * ob_spacing_factor * ob_r}, ob_r},
        {{x_mid + gap_half + ob_r, 0 * ob_spacing_factor * ob_r}, ob_r},
        {{x_mid + gap_half + ob_r, -1 * ob_spacing_factor * ob_r}, ob_r}};

    obstacles.insert(obstacles.end(), ps_obstacles.begin(), ps_obstacles.end());

    // ---- Border
    std::vector<Obstacle> b_obstacles;
    static constexpr float spacing = 5.0;
    static constexpr double radius = 5.0;

    // Top and bottom edges
    for (float x = X_MIN; x <= X_MAX; x += spacing) {
        b_obstacles.push_back({{x, (float)Y_MAX + (float)radius}, radius});
        b_obstacles.push_back({{x, (float)Y_MIN - (float)radius}, radius});
    }

    // Left and right edges
    for (float y = Y_MIN; y <= Y_MAX; y += spacing) {
        b_obstacles.push_back({{(float)X_MIN - (float)radius, y}, radius});
        b_obstacles.push_back({{(float)X_MAX + (float)radius, y}, radius});
    }

    obstacles.insert(obstacles.end(), b_obstacles.begin(), b_obstacles.end());

    return obstacles;
}

extern std::vector<Obstacle> obstacles;