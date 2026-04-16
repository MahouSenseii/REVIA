// Phase 0 — ExecutionResult. Output of ActionDispatcher.Dispatch(...).
#pragma once

#include <string>
#include "models/Enums.h"
#include "models/RuntimeState.h"   // Timestamp
#include "models/FeedbackSignal.h"

namespace revia::core {

struct ExecutionResult {
    bool was_successful = false;

    ActionType  action_type = ActionType::IgnoreEvent;
    Timestamp   completed_at = Timestamp{};

    // Set when was_successful == false.
    std::string failure_reason;

    // Executor-produced feedback to feed into FeedbackManager.
    FeedbackSignal feedback;

    // Factory for failure results so executors share a single construction path.
    static ExecutionResult Fail(ActionType t, std::string reason) {
        ExecutionResult r;
        r.was_successful  = false;
        r.action_type     = t;
        r.completed_at    = now();
        r.failure_reason  = std::move(reason);
        r.feedback.score  = -0.25; // mild negative signal by default
        r.feedback.tag    = "executor_failure";
        return r;
    }

    static ExecutionResult Ok(ActionType t) {
        ExecutionResult r;
        r.was_successful = true;
        r.action_type    = t;
        r.completed_at   = now();
        return r;
    }
};

} // namespace revia::core
