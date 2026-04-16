// Phase 3 - TimingEngine implementation.
#include "core/TimingEngine.h"

#include <memory>
#include <string>

#include "core/StructuredLogger.h"
#include "interfaces/IEvent.h"
#include "models/CoreAction.h"

namespace revia::core {

DecisionResult TimingEngine::apply_timing(DecisionResult decision,
                                          const ContextPackage& context,
                                          const RuntimeState& state) {
    const std::string event_id = context.current_event ? context.current_event->id : "";

    if (!decision.should_act) {
        StructuredLogger::instance().debug("timing.skipped", {
            {"event_id", event_id},
            {"reason", "decision_should_not_act"}
        });
        return decision;
    }

    if (context.current_event &&
        context.current_event->type == EventType::Interruption &&
        state.current_state == CoreState::Responding) {
        decision.selected_action = std::make_unique<CoreAction>(
            ActionType::InterruptCurrentOutput,
            100,
            "TimingEngine forced immediate interrupt");
        decision.delay.pre_action_delay = Milliseconds(0);
        decision.delay.cooldown_applied = Milliseconds(0);
        decision.reason_summary += "; timing_override=ImmediateInterrupt";
        StructuredLogger::instance().info("timing.override_applied", {
            {"event_id", event_id},
            {"override", "ImmediateInterrupt"}
        });
        return decision;
    }

    if (state.is_recovery_mode || state.current_state == CoreState::Recovering) {
        decision.delay.pre_action_delay = Milliseconds(500);
        decision.delay.silence_threshold = Milliseconds(2000);
        decision.delay.allow_burst = false;
        decision.delay.cooldown_applied = Milliseconds(500);
        decision.reason_summary += "; timing=recovery_backoff";
    } else if (platform_allows_burst(context)) {
        decision.delay = DelayProfile::AllowBurst();
        decision.reason_summary += "; timing=burst_allowed";
    } else if (decision.delay.pre_action_delay == Milliseconds(0) &&
               decision.delay.silence_threshold == Milliseconds(0)) {
        decision.delay = DelayProfile::NaturalPause();
        decision.reason_summary += "; timing=natural_pause";
    }

    StructuredLogger::instance().info("timing.apply_completed", {
        {"event_id", event_id},
        {"pre_action_delay_ms", decision.delay.pre_action_delay.count()},
        {"silence_threshold_ms", decision.delay.silence_threshold.count()},
        {"allow_burst", decision.delay.allow_burst},
        {"cooldown_applied_ms", decision.delay.cooldown_applied.count()}
    });
    return decision;
}

bool TimingEngine::platform_allows_burst(const ContextPackage& context) {
    const auto* platform = context.find("PlatformContextProvider");
    if (!platform) {
        return false;
    }
    const auto& payload = platform->payload;
    if (!payload.contains("platform") || !payload["platform"].is_string()) {
        return false;
    }
    return payload["platform"].get<std::string>() == "Discord";
}

} // namespace revia::core
