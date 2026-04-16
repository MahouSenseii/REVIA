// Phase 3 - CoreOrchestrator implementation.
#include "core/CoreOrchestrator.h"

#include <exception>
#include <string>
#include <utility>

#include "core/StructuredLogger.h"

namespace revia::core {

CoreOrchestrator::CoreOrchestrator(EventBus& event_bus,
                                   StateManager& state_manager,
                                   ContextManager& context_manager,
                                   DecisionEngine& decision_engine,
                                   PriorityResolver& priority_resolver,
                                   TimingEngine& timing_engine,
                                   SafetyGateway& safety_gateway,
                                   ActionDispatcher& action_dispatcher,
                                   FeedbackManager& feedback_manager,
                                   BackendSyncService& backend_sync)
    : event_bus_(event_bus),
      state_manager_(state_manager),
      context_manager_(context_manager),
      decision_engine_(decision_engine),
      priority_resolver_(priority_resolver),
      timing_engine_(timing_engine),
      safety_gateway_(safety_gateway),
      action_dispatcher_(action_dispatcher),
      feedback_manager_(feedback_manager),
      backend_sync_(backend_sync) {}

void CoreOrchestrator::initialize() {
    if (user_text_subscription_.has_value()) {
        return;
    }

    user_text_subscription_ = event_bus_.subscribe(
        EventType::UserText,
        [this](const IEvent& event) { on_event_received(event); });

    StructuredLogger::instance().info("orchestrator.initialized", {
        {"subscriptions", 1},
        {"phase", "3"}
    });
}

void CoreOrchestrator::shutdown() {
    if (user_text_subscription_.has_value()) {
        event_bus_.unsubscribe(EventType::UserText, *user_text_subscription_);
        user_text_subscription_.reset();
    }
    StructuredLogger::instance().info("orchestrator.shutdown", {{"phase", "3"}});
}

void CoreOrchestrator::on_event_received(const IEvent& input_event) {
    auto& log = StructuredLogger::instance();
    log.info("orchestrator.event_received", {
        {"event_id", input_event.id},
        {"event_type", std::string(to_string(input_event.type))},
        {"source", std::string(to_string(input_event.source))},
        {"correlation_id", input_event.correlation_id}
    });

    try {
        RuntimeState initial_state = state_manager_.get_current_state();
        backend_sync_.publish_state(initial_state);
        state_manager_.mark_interaction("orchestrator_event_received",
                                        input_event.correlation_id);

        if (initial_state.current_state != CoreState::Thinking) {
            state_manager_.try_transition_to(CoreState::Thinking,
                                             "orchestrator_processing",
                                             input_event.correlation_id);
        }

        ContextPackage context = context_manager_.build_context(input_event, initial_state);
        DecisionResult decision = decision_engine_.evaluate(context, initial_state);
        RuntimeState latest_state = state_manager_.get_current_state();
        DecisionResult resolved = priority_resolver_.resolve(
            std::move(decision),
            context,
            latest_state);
        DecisionResult timed = timing_engine_.apply_timing(
            std::move(resolved),
            context,
            latest_state);
        backend_sync_.publish_decision(timed);

        if (!timed.should_act) {
            log.info("orchestrator.decision_noop", {
                {"event_id", input_event.id},
                {"reason", timed.reason_summary}
            });
            finish_event(input_event, "no_action");
            return;
        }

        SafetyResult safety = safety_gateway_.validate(timed, context);
        if (!safety.is_allowed()) {
            log.warn("orchestrator.safety_not_allowed", {
                {"event_id", input_event.id},
                {"verdict", std::string(to_string(safety.verdict))},
                {"summary", safety.summary}
            });

            DecisionResult fallback = decision_engine_.build_fallback_action(
                context,
                safety.summary.empty() ? "safety fallback" : safety.summary);
            if (fallback.selected_action) {
                ExecutionResult fallback_result =
                    action_dispatcher_.dispatch(*fallback.selected_action);
                feedback_manager_.process(fallback_result, context, latest_state);
                backend_sync_.publish_execution(fallback_result);
                if (!fallback_result.was_successful) {
                    enter_recovery(input_event, fallback_result.failure_reason);
                    return;
                }
            }

            finish_event(input_event, "safety_fallback_dispatched");
            return;
        }

        log.info("orchestrator.action_ready", {
            {"event_id", input_event.id},
            {"action", timed.selected_action ? timed.selected_action->describe() : "none"},
            {"reason", timed.reason_summary},
            {"pre_action_delay_ms", timed.delay.pre_action_delay.count()}
        });

        if (!timed.selected_action) {
            enter_recovery(input_event, "timed decision missing selected_action");
            return;
        }

        ExecutionResult result = action_dispatcher_.dispatch(*timed.selected_action);
        feedback_manager_.process(result, context, latest_state);
        backend_sync_.publish_execution(result);
        if (!result.was_successful) {
            enter_recovery(input_event, result.failure_reason);
            return;
        }

        state_manager_.mark_output("phase3_dispatch_completed",
                                   input_event.correlation_id);
        log.info("orchestrator.execution_completed", {
            {"event_id", input_event.id},
            {"action_type", std::string(to_string(result.action_type))},
            {"feedback_tag", result.feedback.tag},
            {"feedback_score", result.feedback.score}
        });

        // FeedbackManager consumes ExecutionResult in Phase 4.
        finish_event(input_event, "phase3_execution_complete");
    } catch (const std::exception& exc) {
        enter_recovery(input_event, exc.what());
    } catch (...) {
        enter_recovery(input_event, "unknown orchestrator exception");
    }
}

void CoreOrchestrator::finish_event(const IEvent& input_event, const char* reason) {
    state_manager_.try_transition_to(CoreState::Idle, reason, input_event.correlation_id);
    backend_sync_.publish_state(state_manager_.get_current_state());
    StructuredLogger::instance().info("orchestrator.event_finished", {
        {"event_id", input_event.id},
        {"reason", reason}
    });
}

void CoreOrchestrator::enter_recovery(const IEvent& input_event,
                                      const std::string& reason) {
    StructuredLogger::instance().error("orchestrator.event_failed", {
        {"event_id", input_event.id},
        {"error", reason}
    });
    state_manager_.try_transition_to(CoreState::Recovering,
                                     "orchestrator_exception",
                                     input_event.correlation_id);
    backend_sync_.publish_state(state_manager_.get_current_state());
}

} // namespace revia::core
