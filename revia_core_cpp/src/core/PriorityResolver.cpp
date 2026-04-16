// Phase 2 - PriorityResolver implementation.
#include "core/PriorityResolver.h"

#include <algorithm>
#include <memory>
#include <string>

#include "core/StructuredLogger.h"
#include "interfaces/IEvent.h"
#include "models/CoreAction.h"

namespace revia::core {

DecisionResult PriorityResolver::resolve(DecisionResult decision,
                                         const ContextPackage& context,
                                         const RuntimeState& state) {
    const std::string event_id = context.current_event ? context.current_event->id : "";

    if ((context.current_event && context.current_event->type == EventType::Interruption) ||
        state.is_interrupted) {
        decision.should_act = true;
        decision.selected_action = std::make_unique<CoreAction>(
            ActionType::InterruptCurrentOutput,
            100,
            "PriorityResolver forced interruption action");
        if (context.current_event) {
            decision.selected_action->correlation_id = context.current_event->correlation_id;
        }
        decision.reason_summary += "; priority_override=Interruption";
        StructuredLogger::instance().info("priority.override_applied", {
            {"event_id", event_id},
            {"override", "Interruption"},
            {"precedence", "Interruption > Safety > PlatformConstraint > Profile > Emotion > Relationship > Timing"}
        });
        return decision;
    }

    if (platform_requires_strict_filtering(context)) {
        decision.safety.platform_strict = true;
        decision.safety.strictness = std::max(decision.safety.strictness, 0.85);
        decision.tone = ToneProfile::SafeNeutral();
        decision.reason_summary += "; priority_override=PlatformConstraint";
        StructuredLogger::instance().info("priority.override_applied", {
            {"event_id", event_id},
            {"override", "PlatformConstraint"},
            {"safety_strictness", decision.safety.strictness}
        });
    }

    StructuredLogger::instance().info("priority.resolve_completed", {
        {"event_id", event_id},
        {"should_act", decision.should_act},
        {"action", decision.selected_action ? decision.selected_action->describe() : "none"}
    });
    return decision;
}

bool PriorityResolver::platform_requires_strict_filtering(
    const ContextPackage& context) {
    const auto* platform = context.find("PlatformContextProvider");
    if (!platform) {
        return false;
    }
    const auto& payload = platform->payload;
    return payload.contains("requires_strict_filtering") &&
           payload["requires_strict_filtering"].is_boolean() &&
           payload["requires_strict_filtering"].get<bool>();
}

} // namespace revia::core
