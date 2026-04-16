// Phase 4 - BackendSyncService.
//
// Single ownership:
//   * BackendSyncService owns Core -> backend UI publication and validated
//     backend UI -> Core config ingress.
//   * UI code must not mutate deep runtime state directly.
#pragma once

#include <optional>
#include <string>
#include <nlohmann/json.hpp>

#include "core/EventBus.h"
#include "models/DecisionResult.h"
#include "models/ExecutionResult.h"
#include "models/RuntimeState.h"

namespace revia::core {

class BackendSyncService {
public:
    void attach(EventBus& event_bus);
    void shutdown();

    void publish_state(const RuntimeState& state);
    void publish_decision(const DecisionResult& decision);
    void publish_execution(const ExecutionResult& execution);
    void publish_theme_state(const nlohmann::json& theme_state);
    void publish_config_snapshot(const nlohmann::json& config_snapshot);

private:
    void on_config_change(const IEvent& event);

    EventBus* event_bus_ = nullptr;
    std::optional<EventBus::SubscriptionId> config_subscription_;
};

} // namespace revia::core
