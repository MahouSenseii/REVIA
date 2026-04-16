// Phase 2 - first-day decision rules implementation.
#include "rules/DefaultDecisionRules.h"

#include <string>

#include "interfaces/IEvent.h"

namespace revia::core {

namespace {

std::string current_text(const ContextPackage& ctx) {
    const auto* conversation = ctx.find("ConversationContextProvider");
    if (!conversation) {
        return {};
    }
    const auto& payload = conversation->payload;
    if (payload.contains("event_text") && payload["event_text"].is_string()) {
        return payload["event_text"].get<std::string>();
    }
    return {};
}

bool platform_requires_strict_filtering(const ContextPackage& ctx) {
    const auto* platform = ctx.find("PlatformContextProvider");
    if (!platform) {
        return false;
    }
    const auto& payload = platform->payload;
    return payload.contains("requires_strict_filtering") &&
           payload["requires_strict_filtering"].is_boolean() &&
           payload["requires_strict_filtering"].get<bool>();
}

} // namespace

DecisionInfluence ResponseEligibilityRule::evaluate(const ContextPackage& ctx,
                                                    const RuntimeState&) {
    DecisionInfluence influence;
    influence.rule_name = name();
    influence.priority_bias = 20;

    const IEvent* event = ctx.current_event;
    if (!event) {
        influence.should_act = false;
        influence.explanation = "no current event";
        return influence;
    }

    const std::string text = current_text(ctx);
    if (event->type == EventType::UserText && !text.empty()) {
        influence.should_act = true;
        influence.preferred_action = ActionType::UpdateInternalStateOnly;
        influence.explanation = "phase2 accepts non-empty UserText for core planning";
        return influence;
    }

    influence.should_act = false;
    influence.preferred_action = ActionType::IgnoreEvent;
    influence.explanation = "event is not eligible for a phase2 action plan";
    return influence;
}

DecisionInfluence IntentConfidenceRule::evaluate(const ContextPackage& ctx,
                                                 const RuntimeState&) {
    DecisionInfluence influence;
    influence.rule_name = name();
    influence.priority_bias = 10;

    const std::string text = current_text(ctx);
    if (text.size() >= 2) {
        influence.explanation = "text intent is present";
    } else {
        influence.should_act = false;
        influence.preferred_action = ActionType::IgnoreEvent;
        influence.explanation = "text intent is too weak";
    }
    return influence;
}

DecisionInfluence InterruptionRule::evaluate(const ContextPackage& ctx,
                                             const RuntimeState& state) {
    DecisionInfluence influence;
    influence.rule_name = name();
    influence.priority_bias = 100;

    const IEvent* event = ctx.current_event;
    if ((event && event->type == EventType::Interruption) || state.is_interrupted) {
        influence.should_act = true;
        influence.preferred_action = ActionType::InterruptCurrentOutput;
        influence.explanation = "interruption outranks normal response planning";
    }
    return influence;
}

DecisionInfluence PlatformConstraintRule::evaluate(const ContextPackage& ctx,
                                                   const RuntimeState&) {
    DecisionInfluence influence;
    influence.rule_name = name();
    influence.priority_bias = 50;

    if (platform_requires_strict_filtering(ctx)) {
        SafetyProfile safety;
        safety.platform_strict = true;
        safety.strictness = 0.85;
        safety.required_layers = {"HardFilter", "PlatformRules", "FailSafe"};

        influence.safety = safety;
        influence.tone = ToneProfile::SafeNeutral();
        influence.explanation = "platform requires stricter filtering";
    }
    return influence;
}

} // namespace revia::core
