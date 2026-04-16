// Phase 4 - BackendSyncService implementation.
#include "ui_sync/BackendSyncService.h"

#include <string>

#include "core/StructuredLogger.h"

namespace revia::core {

void BackendSyncService::attach(EventBus& event_bus) {
    if (config_subscription_.has_value()) {
        return;
    }

    event_bus_ = &event_bus;
    config_subscription_ = event_bus.subscribe(
        EventType::ConfigChange,
        [this](const IEvent& event) { on_config_change(event); });

    StructuredLogger::instance().info("backend_sync.attached", {
        {"subscriptions", 1}
    });
}

void BackendSyncService::shutdown() {
    if (event_bus_ && config_subscription_.has_value()) {
        event_bus_->unsubscribe(EventType::ConfigChange, *config_subscription_);
        config_subscription_.reset();
    }
    event_bus_ = nullptr;
    StructuredLogger::instance().info("backend_sync.shutdown", {});
}

void BackendSyncService::publish_state(const RuntimeState& state) {
    StructuredLogger::instance().info("backend_sync.state_published", {
        {"current_state", std::string(to_string(state.current_state))},
        {"current_mode", std::string(to_string(state.current_mode))},
        {"version", state.version},
        {"is_speaking", state.is_speaking},
        {"is_listening", state.is_listening},
        {"is_interrupted", state.is_interrupted}
    });
}

void BackendSyncService::publish_decision(const DecisionResult& decision) {
    StructuredLogger::instance().info("backend_sync.decision_published", {
        {"should_act", decision.should_act},
        {"action", decision.selected_action ? decision.selected_action->describe() : "none"},
        {"reason", decision.reason_summary},
        {"tone", decision.tone.label},
        {"safety_strictness", decision.safety.strictness}
    });
}

void BackendSyncService::publish_execution(const ExecutionResult& execution) {
    StructuredLogger::instance().info("backend_sync.execution_published", {
        {"was_successful", execution.was_successful},
        {"action_type", std::string(to_string(execution.action_type))},
        {"failure_reason", execution.failure_reason},
        {"feedback_tag", execution.feedback.tag},
        {"feedback_score", execution.feedback.score}
    });
}

void BackendSyncService::publish_theme_state(const nlohmann::json& theme_state) {
    StructuredLogger::instance().info("backend_sync.theme_published", {
        {"theme", theme_state}
    });
}

void BackendSyncService::publish_config_snapshot(const nlohmann::json& config_snapshot) {
    StructuredLogger::instance().info("backend_sync.config_snapshot_published", {
        {"config", config_snapshot}
    });
}

void BackendSyncService::on_config_change(const IEvent& event) {
    StructuredLogger::instance().info("backend_sync.config_change_received", {
        {"event_id", event.id},
        {"correlation_id", event.correlation_id},
        {"source", std::string(to_string(event.source))},
        {"payload", event.payload}
    });
}

} // namespace revia::core
