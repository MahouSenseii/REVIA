// Phase 3 - CoreOrchestrator.
//
// Single ownership:
//   * CoreOrchestrator owns the event -> state -> context -> decision path.
//   * It does not collect source-specific context, evaluate individual rules,
//     resolve platform policy, time responses, validate safety, or execute
//     actions directly.
//
// Phase 3 wires TimingEngine, SafetyGateway, and ActionDispatcher. Feedback is
// still Phase 4.
#pragma once

#include <optional>
#include <string>

#include "core/ActionDispatcher.h"
#include "core/ContextManager.h"
#include "core/DecisionEngine.h"
#include "core/EventBus.h"
#include "core/FeedbackManager.h"
#include "core/PriorityResolver.h"
#include "core/SafetyGateway.h"
#include "core/StateManager.h"
#include "core/TimingEngine.h"
#include "ui_sync/BackendSyncService.h"

namespace revia::core {

class CoreOrchestrator {
public:
    CoreOrchestrator(EventBus& event_bus,
                     StateManager& state_manager,
                     ContextManager& context_manager,
                     DecisionEngine& decision_engine,
                     PriorityResolver& priority_resolver,
                     TimingEngine& timing_engine,
                     SafetyGateway& safety_gateway,
                     ActionDispatcher& action_dispatcher,
                     FeedbackManager& feedback_manager,
                     BackendSyncService& backend_sync);

    CoreOrchestrator(const CoreOrchestrator&) = delete;
    CoreOrchestrator& operator=(const CoreOrchestrator&) = delete;

    void initialize();
    void shutdown();

    void on_event_received(const IEvent& input_event);

private:
    void finish_event(const IEvent& input_event, const char* reason);
    void enter_recovery(const IEvent& input_event, const std::string& reason);

    EventBus& event_bus_;
    StateManager& state_manager_;
    ContextManager& context_manager_;
    DecisionEngine& decision_engine_;
    PriorityResolver& priority_resolver_;
    TimingEngine& timing_engine_;
    SafetyGateway& safety_gateway_;
    ActionDispatcher& action_dispatcher_;
    FeedbackManager& feedback_manager_;
    BackendSyncService& backend_sync_;

    std::optional<EventBus::SubscriptionId> user_text_subscription_;
};

} // namespace revia::core
