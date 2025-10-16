#include <raylib.h>
#include <raymath.h>

#include <algorithm>
#include <cmath>
#include <optional>
#include <unordered_map>
#include <vector>

#include "app/colors.h"
#include "app/drawing.h"
#include "app/transforms.h"
#include "core/constants.h"
#include "core/loss.h"
#include "core/obstacle.h"
#include "core/search_space.h"
#include "core/space.h"
#include "core/trajectory.h"
#include "core/util.h"
#include "ilqr/solver.h"
#include "planner/planner.h"
#include "tree/tree.h"

template <int N>
std::vector<double> extractSpeed(const Trajectory<N>& traj) {
    std::vector<double> vals;
    for (const double val : traj.state_sequence.row(3)) {
        vals.push_back(val);
    }
    return vals;
}

template <int N>
std::vector<double> extractLonAccel(const Trajectory<N>& traj) {
    std::vector<double> vals;
    for (const double val : traj.action_sequence.row(0)) {
        vals.push_back(val);
    }
    return vals;
}

template <int N>
std::vector<double> extractLatAccel(const Trajectory<N>& traj) {
    std::vector<double> vals;
    for (int i = 0; i < traj.length; ++i) {
        const double v = traj.stateAt(i)(3);
        const double k = traj.actionAt(i)(1);
        const double a = k * square(v);
        vals.push_back(a);
    }
    return vals;
}

template <int N>
std::vector<double> extractCurvature(const Trajectory<N>& traj) {
    std::vector<double> vals;
    for (const double val : traj.action_sequence.row(1)) {
        vals.push_back(val);
    }
    return vals;
}

template <int N>
std::vector<double> extractYaw(const Trajectory<N>& traj) {
    std::vector<double> vals;
    for (const double val : traj.state_sequence.row(2)) {
        vals.push_back(val);
    }
    return vals;
}

int main() {
    // Initialization
    SetConfigFlags(FLAG_VSYNC_HINT);

    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "TreeTop");

    // Clock times
    int tree_exp_clock_time = -1;
    int traj_opt_clock_time = -1;
    int draw_elm_clock_time = -1;
    int game_upd_clock_time = -1;
    const double tree_exp_clock_momentum = 0.90;
    const double traj_opt_clock_momentum = 0.90;
    const double draw_elm_clock_momentum = 0.90;
    const double game_upd_clock_momentum = 0.90;

    int draw_elm_clock_time_next = 0;

    // Load a monospaced font
    Font mono_font = LoadFont("assets/IBMPlexMono-Bold.ttf");

    static const int button_width = 300;
    static const int button_height = 50;
    static const int button_margin = 10;
    static const int button_x1 = SCREEN_WIDTH - 3 * (button_width + button_margin);
    static const int button_x2 = SCREEN_WIDTH - 2 * (button_width + button_margin);
    static const int button_x3 = SCREEN_WIDTH - 1 * (button_width + button_margin);

    // Column 1
    Rectangle pause_button = {button_x1, button_margin + 1 * (button_height + button_margin), button_width, button_height};
    Rectangle advance_button = {button_x1, button_margin + 2 * (button_height + button_margin), button_width, button_height};

    // Column 2
    Rectangle use_action_jitter_button = {button_x2, button_margin + 0 * (button_height + button_margin), button_width, button_height};
    Rectangle use_warm_start_button = {button_x2, button_margin + 1 * (button_height + button_margin), button_width, button_height};
    Rectangle use_cold_start_button = {button_x2, button_margin + 2 * (button_height + button_margin), button_width, button_height};
    Rectangle use_goal_sampling_button = {button_x2, button_margin + 3 * (button_height + button_margin), button_width, button_height};

    // Column 3
    Rectangle show_tree_button = {button_x3, button_margin + 0 * (button_height + button_margin), button_width, button_height};
    Rectangle show_pre_opt_traj_button = {button_x3, button_margin + 1 * (button_height + button_margin), button_width, button_height};
    Rectangle show_post_opt_traj_button = {button_x3, button_margin + 2 * (button_height + button_margin), button_width, button_height};

    Rectangle search_space_rec = {ORIGIN_SS.x, ORIGIN_SS.y + (float)(Y_MIN * SCALE_SS), X_SIZE * SCALE_SS, Y_SIZE * SCALE_SS};

    // Toggle-able states
    bool paused = false;
    bool use_action_jitter = true;
    bool use_warm = true;
    bool use_cold = true;
    bool use_goal = true;

    SamplingSettings sampling_settings = {use_warm, use_cold, use_goal};

    bool show_tree = true;
    bool show_pre_opt_traj = true;
    bool show_post_opt_traj = true;

    // Define a fixed start point and an initial goal point in state space
    const StateVector start{1.0, 0.0, 0.0, 0.0};
    const StateVector goal{39.0, 0.0, 0.0, 0.0};

    // Convert to screen space
    Vector2 start_point = state2screen(start);
    Vector2 goal_point = state2screen(goal);

    // Initial plan
    PlannerOutputs planner_outputs;
    std::optional<Solution<TRAJ_LENGTH_OPT>> warm = std::nullopt;
    planner_outputs = Planner::plan(start, goal, warm, use_action_jitter, sampling_settings);

    float last_time = GetTime();

    while (!WindowShouldClose()) {
        // Calculate delta time
        const float current_time = GetTime();
        const float delta_time = current_time - last_time;
        last_time = current_time;

        const Vector2 mouse_point = GetMousePosition();

        // check button hitboxes
        const bool mouse_in_pause_button = CheckCollisionPointRec(mouse_point, pause_button);
        const bool mouse_in_advance_button = CheckCollisionPointRec(mouse_point, advance_button);
        const bool mouse_in_use_warm_start_button = CheckCollisionPointRec(mouse_point, use_warm_start_button);
        const bool mouse_in_use_cold_start_button = CheckCollisionPointRec(mouse_point, use_cold_start_button);
        const bool mouse_in_use_goal_sampling_button = CheckCollisionPointRec(mouse_point, use_goal_sampling_button);
        const bool mouse_in_use_action_jitter_button = CheckCollisionPointRec(mouse_point, use_action_jitter_button);

        const bool mouse_in_show_tree_button = CheckCollisionPointRec(mouse_point, show_tree_button);
        const bool mouse_in_show_pre_opt_traj_button = CheckCollisionPointRec(mouse_point, show_pre_opt_traj_button);
        const bool mouse_in_show_post_opt_traj_button = CheckCollisionPointRec(mouse_point, show_post_opt_traj_button);

        // check if mouse is in any button
        const bool mouse_in_button = mouse_in_pause_button || mouse_in_advance_button || mouse_in_use_warm_start_button || mouse_in_use_action_jitter_button || mouse_in_use_cold_start_button || mouse_in_use_goal_sampling_button || mouse_in_show_tree_button || mouse_in_show_pre_opt_traj_button || mouse_in_show_post_opt_traj_button;

        // update toggle states
        if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON) && mouse_in_pause_button) {
            paused = !paused;
        }
        if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON) && mouse_in_use_warm_start_button) {
            use_warm = !use_warm;
        }
        if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON) && mouse_in_use_cold_start_button) {
            use_cold = !use_cold;
        }
        if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON) && mouse_in_use_goal_sampling_button) {
            use_goal = !use_goal;
        }
        if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON) && mouse_in_use_action_jitter_button) {
            use_action_jitter = !use_action_jitter;
        }

        if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON) && mouse_in_show_tree_button) {
            show_tree = !show_tree;
        }
        if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON) && mouse_in_show_pre_opt_traj_button) {
            show_pre_opt_traj = !show_pre_opt_traj;
        }
        if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON) && mouse_in_show_post_opt_traj_button) {
            show_post_opt_traj = !show_post_opt_traj;
        }
        sampling_settings = {use_warm, use_cold, use_goal};

        // Check for explicit advance
        const bool explicit_advance = IsMouseButtonPressed(MOUSE_LEFT_BUTTON) && mouse_in_advance_button;

        // Update goal point from mouse
        if (IsMouseButtonDown(MOUSE_LEFT_BUTTON) && !mouse_in_button) {
            goal_point = mouse_point;
        }

        // Update start point from mouse
        if (IsMouseButtonDown(MOUSE_RIGHT_BUTTON) && !mouse_in_button) {
            // Guard for start point inside obstacle
            if (!obstaclesCollidesWith(obstacles, screen2state(mouse_point))) {
                start_point = mouse_point;
            }
        }

        // Convert from screen space to state space
        StateVector start = screen2state(start_point);
        StateVector goal = screen2state(goal_point);

        // Clamp to search space bounds
        start = clampToSearchSpace(start);
        goal = clampToSearchSpace(goal);

        // Convert back from state space to screen space
        start_point = state2screen(start);
        goal_point = state2screen(goal);

        // Update game state.
        const bool do_update_game = !paused || explicit_advance;
        if (do_update_game) {
            warm = std::make_optional(planner_outputs.solution);
            planner_outputs = Planner::plan(start, goal, warm, use_action_jitter, sampling_settings);
        }

        // Draw everything
        BeginDrawing();
        const float draw_elm_clock_start = GetTime();
        ClearBackground(COLOR_BACKGROUND);

        // Draw the search space
        DrawRectangleLinesEx(search_space_rec, 3, COLOR_SEARCH_SPACE_BORDER);

        // Draw the obstacle
        for (const Obstacle& obstacle : obstacles) {
            // Skip drawing the border obstacles
            if (!((X_MIN <= obstacle.center.x) && (obstacle.center.x <= X_MAX) && (Y_MIN <= obstacle.center.y) && (obstacle.center.y <= Y_MAX))) {
                continue;
            }
            const Vector2 obstacle_center_ss = state2screen(obstacle.center);
            const double obstacle_radius_ss = obstacle.radius * SCALE_SS;
            DrawCircleV(obstacle_center_ss, obstacle_radius_ss, COLOR_OBSTACLE);
        }

        // Draw planner outputs.
        const VisibilitySettings viz_settings{show_tree, show_pre_opt_traj, show_post_opt_traj};

        // Draw tree(s).
        if (viz_settings.show_tree) {
            DrawTree(planner_outputs.tree);
        }

        // Draw pre-opt trajectory (tree solution).
        if (viz_settings.show_pre_opt_traj) {
            static constexpr float line_width = 10;
            static constexpr float node_width = 20;
            // Draw trajectory so that even if DrawPath draws nothing we still see the pre-opt traj.
            DrawTrajectory(planner_outputs.traj_pre_opt, line_width, COLOR_TRAJ_PRE_OPT);
            // Draw path so we see the nodes in the pre-opt traj path, if available.
            DrawPath(planner_outputs.path, line_width, node_width);
        }

        // Draw post-opt trajectory (iLQR solution).
        if (viz_settings.show_post_opt_traj) {
            static constexpr float line_width = 6;
            DrawTrajectory(planner_outputs.solution.traj, line_width, COLOR_TRAJ_POST_OPT);
        }

        // Draw start point and the goal point
        DrawSquare(start_point, 14, WHITE);
        DrawSquare(start_point, 8, BLACK);

        DrawGoalTriangle(goal_point, 20, WHITE);
        DrawGoalTriangle(goal_point, 10, BLACK);

        if (paused) {
            // Show pause overlay
            DrawText("Paused", (SCREEN_WIDTH / 2) - (MeasureText("Paused", 20) / 2), (GUTTER_SS_Y / 2) - (20 / 2), 20, COLOR_STAT);
        }

        // Draw pause button
        DrawRectangleRec(pause_button, COLOR_BUTTON_BACKGROUND);
        DrawText(paused ? "Resume" : "Pause", pause_button.x + 10, pause_button.y + 15, 20, COLOR_BUTTON_TEXT);

        // Draw advance button
        DrawRectangleRec(advance_button, COLOR_BUTTON_BACKGROUND);
        DrawText("Advance", advance_button.x + 10, advance_button.y + 15, 20, COLOR_BUTTON_TEXT);

        // Draw use-warm-start button
        DrawRectangleRec(use_warm_start_button, COLOR_BUTTON_BACKGROUND);
        DrawText(use_warm ? "Disable warm-start sampling" : "Enable warm-start sampling", use_warm_start_button.x + 10, use_warm_start_button.y + 15, 20, COLOR_BUTTON_TEXT);

        // Draw use-cold-start button
        DrawRectangleRec(use_cold_start_button, COLOR_BUTTON_BACKGROUND);
        DrawText(use_cold ? "Disable cold-start sampling" : "Enable cold-start sampling", use_cold_start_button.x + 10, use_cold_start_button.y + 15, 20, COLOR_BUTTON_TEXT);

        // Draw use-goal-sampling button
        DrawRectangleRec(use_goal_sampling_button, COLOR_BUTTON_BACKGROUND);
        DrawText(use_goal ? "Disable goal sampling" : "Enable goal sampling", use_goal_sampling_button.x + 10, use_goal_sampling_button.y + 15, 20, COLOR_BUTTON_TEXT);

        // Draw use-action-jitter button
        DrawRectangleRec(use_action_jitter_button, COLOR_BUTTON_BACKGROUND);
        DrawText(use_action_jitter ? "Disable action jitter" : "Enable action jitter", use_action_jitter_button.x + 10, use_action_jitter_button.y + 15, 20, COLOR_BUTTON_TEXT);

        // Draw show-tree button
        DrawRectangleRec(show_tree_button, COLOR_BUTTON_BACKGROUND);
        DrawText(show_tree ? "Hide tree" : "Show tree", show_tree_button.x + 10, show_tree_button.y + 15, 20, COLOR_BUTTON_TEXT);

        // Draw show-pre-opt-traj button
        DrawRectangleRec(show_pre_opt_traj_button, COLOR_BUTTON_BACKGROUND);
        DrawText(show_pre_opt_traj ? "Hide pre-opt traj" : "Show pre-opt traj", show_pre_opt_traj_button.x + 10, show_pre_opt_traj_button.y + 15, 20, COLOR_BUTTON_TEXT);

        // Draw show-post-opt-traj button
        DrawRectangleRec(show_post_opt_traj_button, COLOR_BUTTON_BACKGROUND);
        DrawText(show_post_opt_traj ? "Hide post-opt traj" : "Show post-opt traj", show_post_opt_traj_button.x + 10, show_post_opt_traj_button.y + 15, 20, COLOR_BUTTON_TEXT);

        // ---- Text stats
        static constexpr int STATS_MARGIN = 10;
        static constexpr int STATS_FONT_SIZE = 20;
        static constexpr int STATS_ROW_HEIGHT = STATS_FONT_SIZE + STATS_MARGIN;
        static constexpr int STATS_WIDTH_1 = 200;

        // Draw the timer info
        if (tree_exp_clock_time < 0) {
            tree_exp_clock_time = planner_outputs.timing_info.tree_exp;
        }
        if (traj_opt_clock_time < 0) {
            traj_opt_clock_time = planner_outputs.timing_info.traj_opt;
        }
        if (draw_elm_clock_time < 0) {
            draw_elm_clock_time = draw_elm_clock_time_next;
        }
        if (game_upd_clock_time < 0) {
            game_upd_clock_time = static_cast<int>(1e6 * delta_time);
        }
        tree_exp_clock_time = static_cast<int>(Lerp(planner_outputs.timing_info.tree_exp, tree_exp_clock_time, paused ? 0.0 : tree_exp_clock_momentum));
        traj_opt_clock_time = static_cast<int>(Lerp(planner_outputs.timing_info.traj_opt, traj_opt_clock_time, paused ? 0.0 : traj_opt_clock_momentum));
        draw_elm_clock_time = static_cast<int>(Lerp(draw_elm_clock_time_next, draw_elm_clock_time, draw_elm_clock_momentum));
        game_upd_clock_time = static_cast<int>(Lerp(static_cast<int>(1e6 * delta_time), game_upd_clock_time, game_upd_clock_momentum));

        // Column 1 - timing info
        DrawTextEx(mono_font, TextFormat("Tree exp: %5.1f ms", 0.001 * static_cast<double>(tree_exp_clock_time)), (Vector2){STATS_MARGIN, STATS_MARGIN + 0 * STATS_ROW_HEIGHT}, STATS_FONT_SIZE, 1, COLOR_STAT);
        DrawTextEx(mono_font, TextFormat("Traj opt: %5.1f ms", 0.001 * static_cast<double>(traj_opt_clock_time)), (Vector2){STATS_MARGIN, STATS_MARGIN + 1 * STATS_ROW_HEIGHT}, STATS_FONT_SIZE, 1, COLOR_STAT);
        DrawTextEx(mono_font, TextFormat("Draw elm: %5.1f ms", 0.001 * static_cast<double>(draw_elm_clock_time)), (Vector2){STATS_MARGIN, STATS_MARGIN + 2 * STATS_ROW_HEIGHT}, STATS_FONT_SIZE, 1, COLOR_STAT_MINOR);
        DrawTextEx(mono_font, TextFormat("Game upd: %5.1f ms", 0.001 * static_cast<double>(game_upd_clock_time)), (Vector2){STATS_MARGIN, STATS_MARGIN + 3 * STATS_ROW_HEIGHT}, STATS_FONT_SIZE, 1, COLOR_STAT_MINOR);

        // Column 2 - planner stats
        const double v_avg = planner_outputs.solution.traj.state_sequence.row(3).cwiseAbs().mean();
        DrawTextEx(mono_font, TextFormat("          Pre-opt cost %5.3f", planner_outputs.cost_pre_opt), (Vector2){STATS_MARGIN + STATS_WIDTH_1, STATS_MARGIN + 0 * STATS_ROW_HEIGHT}, STATS_FONT_SIZE, 1, COLOR_STAT);
        DrawTextEx(mono_font, TextFormat("         Post-opt cost %5.3f", planner_outputs.solution.cost), (Vector2){STATS_MARGIN + STATS_WIDTH_1, STATS_MARGIN + 1 * STATS_ROW_HEIGHT}, STATS_FONT_SIZE, 1, COLOR_STAT);
        DrawTextEx(mono_font, TextFormat("       Traj  avg speed %5.3f m/s", v_avg), (Vector2){STATS_MARGIN + STATS_WIDTH_1, STATS_MARGIN + 2 * STATS_ROW_HEIGHT}, STATS_FONT_SIZE, 1, COLOR_STAT);
        int num_nodes = 0;
        for (const Nodes& nodes : planner_outputs.tree.layers) {
            num_nodes += nodes.size();
        }
        DrawTextEx(mono_font, TextFormat("       Number of nodes %5d", num_nodes), (Vector2){STATS_MARGIN + STATS_WIDTH_1, STATS_MARGIN + 3 * STATS_ROW_HEIGHT}, STATS_FONT_SIZE, 1, COLOR_STAT);
        DrawTextEx(mono_font, TextFormat("       Traj  opt iters %5d", planner_outputs.solution.solve_record.iters), (Vector2){STATS_MARGIN + STATS_WIDTH_1, STATS_MARGIN + 4 * STATS_ROW_HEIGHT}, 20, 1, COLOR_STAT);

        // Time plots.
        {
            // Common data.
            const Trajectory<TRAJ_LENGTH_OPT>& traj_pre_opt = planner_outputs.traj_pre_opt;
            const Trajectory<TRAJ_LENGTH_OPT>& traj_post_opt = planner_outputs.solution.traj;

            // Speed data
            const TimePlotDataValues speed_time_plot_data_vals = {extractSpeed(traj_post_opt), extractSpeed(traj_pre_opt)};

            // Lon accel data
            const TimePlotDataValues lon_accel_time_plot_data_vals = {extractLonAccel(traj_post_opt), extractLonAccel(traj_pre_opt)};

            // Lat accel data
            const TimePlotDataValues lat_accel_time_plot_data_vals = {extractLatAccel(traj_post_opt), extractLatAccel(traj_pre_opt)};

            // Curvature data
            const TimePlotDataValues curvature_time_plot_data_vals = {extractCurvature(traj_post_opt), extractCurvature(traj_pre_opt)};

            // Yaw data
            const TimePlotDataValues yaw_time_plot_data_vals = {extractYaw(traj_post_opt), extractYaw(traj_pre_opt)};

            const double total_time = TRAJ_DURATION_OPT;
            DrawTimePlot(speed_time_plot_data_vals, V_MAX, DT, total_time, viz_settings, 0, "Speed", mono_font);
            DrawTimePlot(lon_accel_time_plot_data_vals, ACCEL_LON_MAX, DT, total_time, viz_settings, 1, "Lon Accel", mono_font);
            DrawTimePlot(lat_accel_time_plot_data_vals, ACCEL_LAT_MAX, DT, total_time, viz_settings, 2, "Lat Accel", mono_font);
            DrawTimePlot(curvature_time_plot_data_vals, CURVATURE_MAX, DT, total_time, viz_settings, 3, "Curvature", mono_font);
            DrawTimePlot(yaw_time_plot_data_vals, YAW_MAX, DT, total_time, viz_settings, 4, "Yaw", mono_font);
        }

        const float draw_elm_clock_stop = GetTime();
        draw_elm_clock_time_next = static_cast<int>(std::ceil(1e6 * (draw_elm_clock_stop - draw_elm_clock_start)));
        EndDrawing();
    }

    // Teardown
    UnloadFont(mono_font);
    CloseWindow();
    return 0;
}
