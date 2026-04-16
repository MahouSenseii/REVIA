// Phase 0 — DecisionResult. Output of DecisionEngine; input to PriorityResolver,
// TimingEngine, SafetyGateway, ActionDispatcher.
#pragma once

#include <memory>
#include <string>
#include "models/ToneProfile.h"
#include "models/DelayProfile.h"
#include "models/SafetyProfile.h"

namespace revia::core {

// Forward declaration; IAction is a pure-virtual interface in interfaces/IAction.h
struct IAction;

struct DecisionResult {
    bool should_act = false;

    // Owned. nullptr when should_act == false, or when the orchestrator is
    // producing a "no-op" result.
    std::unique_ptr<IAction> selected_action;

    ToneProfile   tone;
    DelayProfile  delay;
    SafetyProfile safety;

    // Short human-readable summary for logs and Backend UI.
    std::string reason_summary;

    // Non-copyable because of unique_ptr; movable.
    DecisionResult() = default;
    DecisionResult(const DecisionResult&) = delete;
    DecisionResult& operator=(const DecisionResult&) = delete;
    DecisionResult(DecisionResult&&) noexcept = default;
    DecisionResult& operator=(DecisionResult&&) noexcept = default;
};

} // namespace revia::core
