#pragma once

#include <cstddef>
#include <cstdlib>
#include <iostream>

// Macro to check a condition and abort with the condition printed
#define CHECK(condition)                                                   \
    do {                                                                   \
        if (!(condition)) {                                                \
            std::cerr << "Fatal error at " << __FILE__ << ":" << __LINE__  \
                      << " -> Condition failed: " #condition << std::endl; \
            std::abort();                                                  \
        }                                                                  \
    } while (0)

struct SolverSettings {
    const uint64_t max_iters{500};
    const double cost_change_tolerance{1e-6};
    const double cost_change_ratio_min{0.01};
    const double regularization_init{1.0};
    const double regularization_min{1e-8};
    const double regularization_max{1e8};
    const double regularization_factor_surge{10.0};
    const double regularization_factor_increase{2.0};
    const double regularization_factor_decrease{0.5};
    const uint64_t max_feedfrwd_gain_search_attempts{8};
    const double feedfrwd_gain_factor_decrease{0.2};

    void validate() const {
        CHECK(max_iters > 0);
        CHECK(cost_change_tolerance > 0.0);
        CHECK(cost_change_ratio_min > 0.0);
        CHECK(cost_change_ratio_min < 1.0);
        CHECK(regularization_min > 0.0);
        CHECK(regularization_init >= regularization_min);
        CHECK(regularization_init <= regularization_max);
        CHECK(regularization_factor_decrease > 0.0);
        CHECK(regularization_factor_decrease < 1.0);
        CHECK(regularization_factor_increase > 1.0);
        CHECK(regularization_factor_increase <= regularization_factor_surge);
        CHECK(max_feedfrwd_gain_search_attempts > 1);
        CHECK(feedfrwd_gain_factor_decrease > 0.0);
        CHECK(feedfrwd_gain_factor_decrease < 1.0);
    }
};