// Phase 0 — DecisionInfluence. Output of a single IDecisionRule; merged by DecisionEngine.
#pragma once

#include <optional>
#include <string>
#include "models/Enums.h"
#include "models/ToneProfile.h"
#include "models/DelayProfile.h"
#include "models/SafetyProfile.h"

namespace revia::core {

struct DecisionInfluence {
    std::string rule_name;

    // Rule's suggestion for whether to act. std::nullopt = no opinion; merge
    // logic in DecisionEngine treats nullopt as a pass-through.
    std::optional<bool> should_act;

    // Preferred action type, if any. nullopt = rule does not care which action.
    std::optional<ActionType> preferred_action;

    // Rule's priority bias. Higher wins in PriorityResolver's precedence fallback.
    int priority_bias = 0;

    std::optional<ToneProfile>   tone;
    std::optional<DelayProfile>  delay;
    std::optional<SafetyProfile> safety;

    // Free-form explanation for logs ("chose WaitSilently because user is mid-typing").
    std::string explanation;
};

} // namespace revia::core
