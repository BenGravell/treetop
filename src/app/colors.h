#pragma once

#include <raylib.h>

#include <algorithm>

#include "app/guppy_r_hq.h"

inline Color guppyColor(const float x) {
    static constexpr int n = guppy_r_hq_colormap.size();
    const int idx = std::clamp(static_cast<int>(x * float(n - 1) + 0.5f), 0, n - 1);
    const auto& rgb = guppy_r_hq_colormap[idx];
    return Color{rgb[0], rgb[1], rgb[2], 255};
}

inline Color warmColormap(const float x) {
    return guppyColor(0.5f + 0.5f * x);
}

inline Color coolColormap(const float x) {
    return guppyColor(0.5f - 0.5f * x);
}

static constexpr Color COLOR_GRAY_008 = {8, 8, 8, 255};
static constexpr Color COLOR_GRAY_048 = {48, 48, 48, 255};
static constexpr Color COLOR_GRAY_064 = {64, 64, 64, 255};
static constexpr Color COLOR_GRAY_096 = {96, 96, 96, 255};
static constexpr Color COLOR_GRAY_128 = {128, 128, 128, 255};
static constexpr Color COLOR_GRAY_160 = {160, 160, 160, 255};
static constexpr Color COLOR_GRAY_240 = {240, 240, 240, 255};

static constexpr Color COLOR_BACKGROUND = COLOR_GRAY_008;
static constexpr Color COLOR_SEARCH_SPACE_BORDER = COLOR_GRAY_064;

static constexpr Color COLOR_OBSTACLE = COLOR_GRAY_096;

static constexpr Color COLOR_TRAJ_POST_OPT = COLOR_GRAY_240;
static constexpr Color COLOR_TRAJ_PRE_OPT = COLOR_GRAY_128;
static constexpr Color COLOR_NODE_PRE_OPT = COLOR_GRAY_160;

static constexpr Color COLOR_STAT = COLOR_GRAY_240;
static constexpr Color COLOR_STAT_MINOR = COLOR_GRAY_160;

static constexpr Color COLOR_BUTTON_BACKGROUND = COLOR_GRAY_064;
static constexpr Color COLOR_BUTTON_TEXT = COLOR_GRAY_240;