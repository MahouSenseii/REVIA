// Phase 3 - default safety layer implementation.
#include "safety/DefaultSafetyLayers.h"

#include <string>
#include <utility>

#include "interfaces/IEvent.h"

namespace revia::core {

namespace {

SafetyLayerOutcome pass(std::string layer, std::string reason = "passed") {
    SafetyLayerOutcome outcome;
    outcome.layer_name = std::move(layer);
    outcome.passed = true;
    outcome.reason = std::move(reason);
    return outcome;
}

SafetyLayerOutcome fail(std::string layer, std::string reason) {
    SafetyLayerOutcome outcome;
    outcome.layer_name = std::move(layer);
    outcome.passed = false;
    outcome.reason = std::move(reason);
    return outcome;
}

std::string event_text(const ContextPackage& context) {
    const auto* conversation = context.find("ConversationContextProvider");
    if (!conversation) {
        return {};
    }
    const auto& payload = conversation->payload;
    if (payload.contains("event_text") && payload["event_text"].is_string()) {
        return payload["event_text"].get<std::string>();
    }
    return {};
}

bool platform_strict(const ContextPackage& context) {
    const auto* platform = context.find("PlatformContextProvider");
    if (!platform) {
        return false;
    }
    const auto& payload = platform->payload;
    return payload.contains("requires_strict_filtering") &&
           payload["requires_strict_filtering"].is_boolean() &&
           payload["requires_strict_filtering"].get<bool>();
}

} // namespace

SafetyLayerOutcome HardFilterLayer::validate(const DecisionResult&,
                                             const ContextPackage& context) {
    const std::string text = event_text(context);
    if (text.find('\0') != std::string::npos) {
        return fail(name(), "input text contains null byte");
    }
    if (text.size() > 16000) {
        return fail(name(), "input text exceeds hard safety size cap");
    }
    return pass(name());
}

SafetyLayerOutcome AiFilterLayer::validate(const DecisionResult&,
                                           const ContextPackage&) {
    return pass(name(), "phase3 placeholder passed");
}

SafetyLayerOutcome PlatformRulesLayer::validate(const DecisionResult& decision,
                                                const ContextPackage& context) {
    if (platform_strict(context) && decision.safety.strictness < 0.75) {
        return fail(name(), "platform strict mode requires safety strictness >= 0.75");
    }
    return pass(name());
}

SafetyLayerOutcome ModeRulesLayer::validate(const DecisionResult&,
                                            const ContextPackage& context) {
    if (context.state_snapshot.current_state == CoreState::Disabled) {
        return fail(name(), "core is disabled");
    }
    return pass(name());
}

SafetyLayerOutcome ProfileOverridesLayer::validate(const DecisionResult&,
                                                   const ContextPackage&) {
    return pass(name(), "no profile override blocked this action");
}

SafetyLayerOutcome FailSafeLayer::validate(const DecisionResult& decision,
                                           const ContextPackage&) {
    if (decision.should_act && !decision.selected_action) {
        return fail(name(), "decision wants action but selected_action is missing");
    }
    return pass(name());
}

} // namespace revia::core
