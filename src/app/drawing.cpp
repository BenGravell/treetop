#include "app/drawing.h"

#include <raygui.h>


void DrawTimePlot(const TimePlotDataValues& vals, const double val_max, const double dt, const double total_time, const VisibilitySettings& viz_settings, const int ix_plot, const std::string& name) {
    const float plot_x = TIME_PLOT_MARGIN_X + ix_plot * (PLOT_WIDTH + TIME_PLOT_MARGIN_X);
    const float plot_y = SCREEN_HEIGHT - (2 * PLOT_HALF_HEIGHT) - TIME_PLOT_MARGIN_Y;

    // Draw border
    DrawRectangleLines(plot_x, plot_y, PLOT_WIDTH, PLOT_HALF_HEIGHT, COLOR_GRAY_160);
    DrawRectangleLines(plot_x, plot_y + PLOT_HALF_HEIGHT, PLOT_WIDTH, PLOT_HALF_HEIGHT, COLOR_GRAY_160);

    // Draw title
    const std::string title = name + " vs Time";
    GuiSetStyle(DEFAULT, TEXT_SIZE, SMALL_TEXT_HEIGHT);
    GuiLabel(
        (Rectangle){plot_x, plot_y - TIME_PLOT_TITLE_MARGIN_Y, PLOT_WIDTH, TIME_PLOT_TITLE_MARGIN_Y},
        title.c_str());
    GuiSetStyle(DEFAULT, TEXT_SIZE, TEXT_HEIGHT);

    // Post-opt traj
    if (viz_settings.show_post_opt_traj) {
        static constexpr float line_width = 2.0f;
        static constexpr Color color = COLOR_TRAJ_POST_OPT;
        DrawSeries(vals.post_opt_traj, val_max, dt, total_time, plot_x, plot_y, PLOT_WIDTH, PLOT_HALF_HEIGHT, line_width, color);
    }

    // Pre-opt traj
    if (viz_settings.show_pre_opt_traj) {
        static constexpr float line_width = 1.0f;
        static constexpr Color color = COLOR_TRAJ_PRE_OPT;
        DrawSeries(vals.pre_opt_traj, val_max, dt, total_time, plot_x, plot_y, PLOT_WIDTH, PLOT_HALF_HEIGHT, line_width, color);
    }
}