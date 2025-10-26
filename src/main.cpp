#include <raylib.h>
#include <raymath.h>

#define RAYGUI_IMPLEMENTATION

#include <raygui.h>

#include <algorithm>
#include <cmath>
#include <optional>
#include <unordered_map>
#include <vector>

#include "app/colors.h"
#include "app/config.h"
#include "app/drawing.h"
#include "app/style_dark.h"
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

int roundToNearestPower10Pattern(float e) {
    float exp10 = std::pow(10.0f, std::floor(e));
    float normalized = std::pow(10.0f, e) / exp10;

    // Define the pattern within each decade
    std::vector<float> pattern = {1, 2, 5};

    // Find the closest value in the pattern
    float best = pattern[0];
    for (float p : pattern) {
        if (std::fabs(normalized - p) < std::fabs(normalized - best))
            best = p;
    }

    // Scale back to the proper magnitude
    return static_cast<int>(std::round(best * exp10));
}

int nodeAttemptsRound(const float num_node_attempts_pow10) {
    return roundToNearestPower10Pattern(num_node_attempts_pow10);
}

int main() {
    // Initialization
    SetConfigFlags(FLAG_VSYNC_HINT);

    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "TreeTop");
    GuiLoadStyleDark();

    // Load a monospaced font
    Font font = LoadFontEx("assets/IBMPlexMono-Bold.ttf", BIG_TEXT_HEIGHT, 0, 0);

    SetTextureFilter(font.texture, TEXTURE_FILTER_BILINEAR);
    GuiSetFont(font);

    GuiSetStyle(DEFAULT, TEXT_SIZE, TEXT_HEIGHT);
    GuiSetIconScale(BUTTON_ICON_SCALE);
    GuiSetStyle(DEFAULT, BORDER_WIDTH, BORDER_THICKNESS);

    // GuiSetStyle(DEFAULT, BORDER_COLOR_NORMAL, ColorToInt(COLOR_GRAY_096));
    // GuiSetStyle(DEFAULT, BASE_COLOR_NORMAL, ColorToInt(COLOR_GRAY_064));
    // GuiSetStyle(DEFAULT, TEXT_COLOR_NORMAL, ColorToInt(COLOR_GRAY_160));

    // GuiSetStyle(DEFAULT, BORDER_COLOR_FOCUSED, ColorToInt(COLOR_GRAY_160));
    // GuiSetStyle(DEFAULT, BASE_COLOR_FOCUSED, ColorToInt(COLOR_GRAY_128));
    // GuiSetStyle(DEFAULT, TEXT_COLOR_FOCUSED, ColorToInt(COLOR_GRAY_240));

    // GuiSetStyle(DEFAULT, BORDER_COLOR_PRESSED, ColorToInt(COLOR_GRAY_240));
    // GuiSetStyle(DEFAULT, BASE_COLOR_PRESSED, ColorToInt(COLOR_GRAY_240));
    // GuiSetStyle(DEFAULT, TEXT_COLOR_PRESSED, ColorToInt(COLOR_GRAY_064));

    // GuiSetStyle(DEFAULT, BORDER_COLOR_DISABLED, ColorToInt(COLOR_GRAY_064));
    // GuiSetStyle(DEFAULT, BASE_COLOR_DISABLED, ColorToInt(COLOR_GRAY_048));
    // GuiSetStyle(DEFAULT, TEXT_COLOR_DISABLED, ColorToInt(COLOR_GRAY_096));

    GuiSetStyle(DEFAULT, TEXT_ALIGNMENT, TEXT_ALIGN_CENTER);
    GuiSetStyle(DEFAULT, TEXT_ALIGNMENT_VERTICAL, TEXT_ALIGN_MIDDLE);

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

    static const int button_col_width = 300;
    static const int button_row_height = 50;
    static const int button_margin = 10;
    static const int button_width = button_col_width - button_margin;
    static const int button_height = button_row_height - button_margin;
    static const int button_x1 = SCREEN_WIDTH - 4 * button_col_width;
    static const int button_x2 = SCREEN_WIDTH - 3 * button_col_width;
    static const int button_x3 = SCREEN_WIDTH - 2 * button_col_width;
    static const int button_x4 = SCREEN_WIDTH - 1 * button_col_width;

    // Column 1
    Rectangle advance_button = {button_x1, button_margin + 0 * button_row_height, button_width, button_height};
    Rectangle pause_button = {button_x1, button_margin + 1 * button_row_height, button_width, button_height};

    // Column 2
    Rectangle use_traj_opt_button = {button_x2, button_margin + 0 * button_row_height, button_width, button_height};
    Rectangle use_hot_button = {button_x2, button_margin + 1 * button_row_height, button_width, button_height};
    Rectangle use_action_jitter_button = {button_x2, button_margin + 2 * button_row_height, button_width, button_height};

    // Column 3
    Rectangle use_warm_start_button = {button_x3, button_margin + 0 * button_row_height, button_width, button_height};
    Rectangle use_cold_start_button = {button_x3, button_margin + 1 * button_row_height, button_width, button_height};
    Rectangle use_goal_sampling_button = {button_x3, button_margin + 2 * button_row_height, button_width, button_height};

    // Column 4
    Rectangle show_tree_button = {button_x4, button_margin + 0 * button_row_height, button_width, button_height};
    Rectangle show_pre_opt_traj_button = {button_x4, button_margin + 1 * button_row_height, button_width, button_height};
    Rectangle show_post_opt_traj_button = {button_x4, button_margin + 2 * button_row_height, button_width, button_height};

    Rectangle search_space_rec = {ORIGIN_SS.x, ORIGIN_SS.y + (float)(Y_MIN * SCALE_SS), X_SIZE * SCALE_SS, Y_SIZE * SCALE_SS};

    // Toggle-able states
    bool paused = false;
    bool use_hot = true;
    bool use_traj_opt = true;
    bool use_action_jitter = true;
    bool use_warm = true;
    bool use_cold = true;
    bool use_goal = true;

    bool explicit_advance = false;

    int num_node_attempts = 5000;
    float num_node_attempts_pow10 = std::log10(num_node_attempts);

    int num_path_candidates = 2;
    float num_path_candidates_float = static_cast<float>(num_path_candidates);

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
    planner_outputs = Planner::plan(start, goal, warm, use_hot, use_traj_opt, use_action_jitter, sampling_settings, num_node_attempts, num_path_candidates);

    float last_time = GetTime();

    while (!WindowShouldClose()) {
        // Calculate delta time
        const float current_time = GetTime();
        const float delta_time = current_time - last_time;
        last_time = current_time;

        const Vector2 mouse_point = GetMousePosition();

        const bool mouse_in_env = CheckCollisionPointRec(mouse_point, search_space_rec);

        sampling_settings = {use_warm, use_cold, use_goal};

        // Update goal point from mouse
        if (IsMouseButtonDown(MOUSE_LEFT_BUTTON) && mouse_in_env) {
            goal_point = mouse_point;
        }

        // Update start point from mouse
        if (IsMouseButtonDown(MOUSE_RIGHT_BUTTON) && mouse_in_env) {
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

        num_node_attempts = nodeAttemptsRound(num_node_attempts_pow10);
        num_path_candidates = static_cast<int>(num_path_candidates_float);

        // Update game state.
        const bool do_update_game = !paused || explicit_advance;
        if (do_update_game) {
            warm = std::make_optional(planner_outputs.solution);
            planner_outputs = Planner::plan(start, goal, warm, use_hot, use_traj_opt, use_action_jitter, sampling_settings, num_node_attempts, num_path_candidates);
        }

        // Draw everything
        const float draw_elm_clock_start = GetTime();
        BeginDrawing();
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

        explicit_advance = GuiButton(advance_button, GuiIconText(ICON_PLAYER_NEXT, NULL));
        GuiToggle(pause_button, GuiIconText(ICON_PLAYER_PAUSE, NULL), &paused);
        GuiToggle(use_hot_button, "Hot-start", &use_hot);
        GuiToggle(use_traj_opt_button, "Optimize Trajectory", &use_traj_opt);
        GuiToggle(use_action_jitter_button, "Action Jitter", &use_action_jitter);
        GuiToggle(use_warm_start_button, "Warm-start Sampling", &use_warm);
        GuiToggle(use_cold_start_button, "Cold-start Sampling", &use_cold);
        GuiToggle(use_goal_sampling_button, "Goal Sampling", &use_goal);
        GuiToggle(show_tree_button, "Show Tree", &show_tree);
        GuiToggle(show_pre_opt_traj_button, "Show Pre-Opt Traj", &show_pre_opt_traj);
        GuiToggle(show_post_opt_traj_button, "Show Post-Opt Traj", &show_post_opt_traj);

        const Rectangle num_node_attempts_box = {button_x1, button_margin + 3 * button_row_height, 2 * button_col_width - button_margin, button_height};
        const Rectangle num_path_candidates_box = {button_x1, button_margin + 4 * button_row_height, 2 * button_col_width - button_margin, button_height};

        GuiSliderBar(num_node_attempts_box, "Samples", TextFormat("%i", num_node_attempts), &num_node_attempts_pow10, 1.0f, 5.0f);
        GuiSliderBar(num_path_candidates_box, "Path Candidates", TextFormat("%i", num_path_candidates), &num_path_candidates_float, 1.0f, 5.0f);

        // ---- Text stats
        static constexpr int STATS_MARGIN = 10;
        static constexpr int STATS_ROW_HEIGHT = TEXT_HEIGHT + STATS_MARGIN;
        static constexpr int STATS_COL_1_WIDTH = 300;
        static constexpr int STATS_COL_2_WIDTH = 400;

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

        GuiSetStyle(DEFAULT, TEXT_SIZE, SMALL_TEXT_HEIGHT);
        GuiSetStyle(DEFAULT, TEXT_ALIGNMENT, TEXT_ALIGN_RIGHT);

        // Column 1 - timing info
        GuiLabel(
            (Rectangle){0, 0 * STATS_ROW_HEIGHT, STATS_COL_1_WIDTH, STATS_ROW_HEIGHT},
            TextFormat("Tree exp: %5.1f ms", 0.001 * static_cast<double>(tree_exp_clock_time)));
        GuiLabel(
            (Rectangle){0, 1 * STATS_ROW_HEIGHT, STATS_COL_1_WIDTH, STATS_ROW_HEIGHT},
            TextFormat("Traj opt: %5.1f ms", 0.001 * static_cast<double>(traj_opt_clock_time)));
        GuiLabel(
            (Rectangle){0, 2 * STATS_ROW_HEIGHT, STATS_COL_1_WIDTH, STATS_ROW_HEIGHT},
            TextFormat("Draw elm: %5.1f ms", 0.001 * static_cast<double>(draw_elm_clock_time)));
        GuiLabel(
            (Rectangle){0, 3 * STATS_ROW_HEIGHT, STATS_COL_1_WIDTH, STATS_ROW_HEIGHT},
            TextFormat("Game upd: %5.1f ms", 0.001 * static_cast<double>(game_upd_clock_time)));

        // Column 2 - planner stats
        const double v_avg = planner_outputs.solution.traj.state_sequence.row(3).cwiseAbs().mean();

        int num_nodes = 0;
        for (const Nodes& nodes : planner_outputs.tree.layers) {
            num_nodes += nodes.size();
        }

        GuiLabel(
            (Rectangle){STATS_COL_1_WIDTH, STATS_MARGIN + 0 * STATS_ROW_HEIGHT, STATS_COL_2_WIDTH, STATS_ROW_HEIGHT},
            TextFormat("          Pre-opt cost  %5.3f", planner_outputs.cost_pre_opt));
        GuiLabel(
            (Rectangle){STATS_COL_1_WIDTH, STATS_MARGIN + 1 * STATS_ROW_HEIGHT, STATS_COL_2_WIDTH, STATS_ROW_HEIGHT},
            TextFormat("         Post-opt cost  %5.3f", planner_outputs.solution.cost));
        GuiLabel(
            (Rectangle){STATS_COL_1_WIDTH, STATS_MARGIN + 2 * STATS_ROW_HEIGHT, STATS_COL_2_WIDTH, STATS_ROW_HEIGHT},
            TextFormat("     Traj avg speed [m/s]  %5.3f", v_avg));
        GuiLabel(
            (Rectangle){STATS_COL_1_WIDTH, STATS_MARGIN + 3 * STATS_ROW_HEIGHT, STATS_COL_2_WIDTH, STATS_ROW_HEIGHT},
            TextFormat("        Traj opt iters  %5d", planner_outputs.solution.solve_record.iters));
        GuiLabel(
            (Rectangle){STATS_COL_1_WIDTH, STATS_MARGIN + 4 * STATS_ROW_HEIGHT, STATS_COL_2_WIDTH, STATS_ROW_HEIGHT},
            TextFormat("              # Nodes  %6d", num_nodes));

        GuiSetStyle(DEFAULT, TEXT_SIZE, TEXT_HEIGHT);
        GuiSetStyle(DEFAULT, TEXT_ALIGNMENT, TEXT_ALIGN_CENTER);

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
            DrawTimePlot(speed_time_plot_data_vals, V_MAX, DT, total_time, viz_settings, 0, "Speed");
            DrawTimePlot(lon_accel_time_plot_data_vals, ACCEL_LON_MAX, DT, total_time, viz_settings, 1, "Lon Accel");
            DrawTimePlot(lat_accel_time_plot_data_vals, ACCEL_LAT_MAX, DT, total_time, viz_settings, 2, "Lat Accel");
            DrawTimePlot(curvature_time_plot_data_vals, CURVATURE_MAX, DT, total_time, viz_settings, 3, "Curvature");
            DrawTimePlot(yaw_time_plot_data_vals, YAW_MAX, DT, total_time, viz_settings, 4, "Yaw");
        }

        EndDrawing();
        const float draw_elm_clock_stop = GetTime();
        draw_elm_clock_time_next = static_cast<int>(std::ceil(1e6 * (draw_elm_clock_stop - draw_elm_clock_start)));
    }

    // Teardown
    UnloadFont(font);
    CloseWindow();
    return 0;
}
