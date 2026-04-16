// Phase 0 — FeedbackSignal POD. Produced by executors, consumed by FeedbackManager.
#pragma once

#include <string>
#include <nlohmann/json.hpp>

namespace revia::core {

using json = nlohmann::json;

struct FeedbackSignal {
    // How "good" the outcome was, -1.0 (bad) .. 1.0 (good). 0.0 = neutral.
    double score = 0.0;

    // Optional machine-readable tag. Examples:
    //   "user_positive_reply", "user_silence", "interrupted_by_user",
    //   "filter_rewrote_output", "model_timeout", "tts_failure"
    std::string tag;

    // Free-form extra data for analytics (token counts, latencies, etc.).
    json extra = json::object();
};

} // namespace revia::core
