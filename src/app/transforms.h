#pragma once

#include <raylib.h>

#include "core/search_space.h"
#include "core/space.h"

// Gutter sizes, px.
static constexpr int GUTTER_SS_X = 100;
static constexpr int GUTTER_SS_Y = 300;

// Scale factor from state space to screen space.
// 1 meter in state space = SCALE_SS px in screen space.
static constexpr int SCALE_SS = 50;

// Origin in screen space.
static constexpr int ORIGIN_SS_X = GUTTER_SS_X - X_MIN * SCALE_SS;
static constexpr int ORIGIN_SS_Y = GUTTER_SS_Y - Y_MIN * SCALE_SS;
const Vector2 ORIGIN_SS = {ORIGIN_SS_X, ORIGIN_SS_Y};

// Screen dimensions, px.
static constexpr int SCREEN_WIDTH = (2 * GUTTER_SS_X) + (X_SIZE * SCALE_SS);
static constexpr int SCREEN_HEIGHT = (2 * GUTTER_SS_Y) + (Y_SIZE * SCALE_SS);

inline Vector2 state2screen(const Vector2 state) {
    return {ORIGIN_SS.x + SCALE_SS * state.x, ORIGIN_SS.y + SCALE_SS * state.y};
}

inline Vector2 state2screen(const StateVector state) {
    const Vector2 vec{static_cast<float>(state[0]), static_cast<float>(state[1])};
    return state2screen(vec);
}

inline StateVector screen2state(const Vector2 point) {
    return {(point.x - ORIGIN_SS.x) / SCALE_SS, (point.y - ORIGIN_SS.y) / SCALE_SS, 0.0, 0.0};
}
