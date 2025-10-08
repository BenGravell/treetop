#pragma once

#include <Eigen/Dense>
#include <cstddef>
#include <vector>

#include "core/space.h"
#include "core/trajectory.h"

template <int N>
using FeedbackGainSequence = std::array<ActionStateMatrix, N>;

template <int N>
using FeedfrwdGainSequence = std::array<ActionVector, N>;

// Policy class. Affine in the state.
template <int N>
struct Policy {
    FeedbackGainSequence<N> feedback_gain_sequence;
    FeedfrwdGainSequence<N> feedfrwd_gain_sequence;
    double feedfrwd_gain_scale{1.0};

    // Getters.
    const ActionStateMatrix& feedbackGainAt(size_t stage_idx) const {
        return feedback_gain_sequence[stage_idx];
    }
    const ActionVector& feedfrwdGainAt(size_t stage_idx) const {
        return feedfrwd_gain_sequence[stage_idx];
    }

    // Setters.
    void setFeedbackGainAt(size_t stage_idx, const ActionStateMatrix& feedback_gain) {
        feedback_gain_sequence[stage_idx] = feedback_gain;
    }
    void setFeedfrwdGainAt(size_t stage_idx, const ActionVector& feedfrwd_gain) {
        feedfrwd_gain_sequence[stage_idx] = feedfrwd_gain;
    }

    // Act on a state delta.
    ActionVector act(const StateVector& dx, size_t stage_idx) const {
        return feedbackGainAt(stage_idx) * dx + feedfrwd_gain_scale * feedfrwdGainAt(stage_idx);
    }
};
