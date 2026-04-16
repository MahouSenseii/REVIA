#include <iostream>
#include <thread>
#include <csignal>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <memory>
#include <vector>
#include "telemetry/telemetry.h"
#include "pipeline/pipeline.h"
#include "neural/EmotionNet.h"
#include "neural/RouterClassifier.h"
#include "plugins/plugin_manager.h"
#include "api/rest_server.h"
#include "api/ws_server.h"
#include "core/ActionDispatcher.h"
#include "core/ContextManager.h"
#include "core/CoreOrchestrator.h"
#include "core/DecisionEngine.h"
#include "core/EventBus.h"
#include "core/FeedbackManager.h"
#include "core/FeatureFlags.h"
#include "core/PriorityResolver.h"
#include "core/SafetyGateway.h"
#include "core/StateManager.h"
#include "core/StructuredLogger.h"
#include "core/TimingEngine.h"
#include "executors/DefaultActionExecutors.h"
#include "feedback/DefaultLearningSinks.h"
#include "providers/DefaultContextProviders.h"
#include "rules/DefaultDecisionRules.h"
#include "safety/DefaultSafetyLayers.h"
#include "ui_sync/BackendSyncService.h"

static std::atomic<bool> g_running{true};

void signal_handler(int) { g_running = false; }

int main() {
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    std::cout << "[REVIA CORE] v" << revia::VERSION << " Initializing...\n";

    std::filesystem::create_directories("logs");
    std::filesystem::create_directories("plugins");
    std::filesystem::create_directories("profiles");

    auto& telemetry = revia::TelemetryEngine::instance();
    auto& core_flags = revia::core::FeatureFlags::instance();

    revia::core::EventBus event_bus;
    revia::core::StateManager state_manager;
    revia::core::ContextManager context_manager;
    revia::core::DecisionEngine decision_engine;
    revia::core::PriorityResolver priority_resolver;
    revia::core::TimingEngine timing_engine;
    revia::core::SafetyGateway safety_gateway;
    revia::core::ActionDispatcher action_dispatcher;
    revia::core::FeedbackManager feedback_manager;
    revia::core::BackendSyncService backend_sync;
    revia::core::CoreOrchestrator orchestrator(
        event_bus,
        state_manager,
        context_manager,
        decision_engine,
        priority_resolver,
        timing_engine,
        safety_gateway,
        action_dispatcher,
        feedback_manager,
        backend_sync);

    if (core_flags.core_v2_enabled()) {
        context_manager.register_provider(std::make_unique<revia::core::ConversationContextProvider>());
        context_manager.register_provider(std::make_unique<revia::core::ProfileContextProvider>());
        context_manager.register_provider(std::make_unique<revia::core::PlatformContextProvider>());
        context_manager.register_provider(std::make_unique<revia::core::EmotionContextProvider>());
        context_manager.register_provider(std::make_unique<revia::core::RelationshipContextProvider>());
        context_manager.register_provider(std::make_unique<revia::core::MemoryContextProvider>());
        context_manager.register_provider(std::make_unique<revia::core::VoiceContextProvider>());
        context_manager.register_provider(std::make_unique<revia::core::VisionContextProvider>());
        context_manager.register_provider(std::make_unique<revia::core::CapabilityContextProvider>());

        decision_engine.register_rule(std::make_unique<revia::core::ResponseEligibilityRule>());
        decision_engine.register_rule(std::make_unique<revia::core::IntentConfidenceRule>());
        decision_engine.register_rule(std::make_unique<revia::core::InterruptionRule>());
        decision_engine.register_rule(std::make_unique<revia::core::PlatformConstraintRule>());

        safety_gateway.register_layer(std::make_unique<revia::core::HardFilterLayer>());
        safety_gateway.register_layer(std::make_unique<revia::core::AiFilterLayer>());
        safety_gateway.register_layer(std::make_unique<revia::core::PlatformRulesLayer>());
        safety_gateway.register_layer(std::make_unique<revia::core::ModeRulesLayer>());
        safety_gateway.register_layer(std::make_unique<revia::core::ProfileOverridesLayer>());
        safety_gateway.register_layer(std::make_unique<revia::core::FailSafeLayer>());

        action_dispatcher.register_executor(std::make_unique<revia::core::NoopActionExecutor>(
            revia::core::ActionType::SpeakResponse,
            "speak_response_stub"));
        action_dispatcher.register_executor(std::make_unique<revia::core::NoopActionExecutor>(
            revia::core::ActionType::SendTextResponse,
            "send_text_response_stub"));
        action_dispatcher.register_executor(std::make_unique<revia::core::NoopActionExecutor>(
            revia::core::ActionType::AskClarifyingQuestion,
            "ask_clarifying_question_stub"));
        action_dispatcher.register_executor(std::make_unique<revia::core::NoopActionExecutor>(
            revia::core::ActionType::WaitSilently,
            "wait_silently"));
        action_dispatcher.register_executor(std::make_unique<revia::core::NoopActionExecutor>(
            revia::core::ActionType::FollowUp,
            "follow_up_stub"));
        action_dispatcher.register_executor(std::make_unique<revia::core::NoopActionExecutor>(
            revia::core::ActionType::IgnoreEvent,
            "ignore_event"));
        action_dispatcher.register_executor(std::make_unique<revia::core::NoopActionExecutor>(
            revia::core::ActionType::TriggerFallback,
            "trigger_fallback_stub"));
        action_dispatcher.register_executor(std::make_unique<revia::core::NoopActionExecutor>(
            revia::core::ActionType::UpdateInternalStateOnly,
            "update_internal_state_only"));
        action_dispatcher.register_executor(std::make_unique<revia::core::NoopActionExecutor>(
            revia::core::ActionType::InterruptCurrentOutput,
            "interrupt_current_output_stub"));
        action_dispatcher.register_executor(std::make_unique<revia::core::NoopActionExecutor>(
            revia::core::ActionType::QueueProactiveMessage,
            "queue_proactive_message_stub"));

        feedback_manager.register_learning_sink(
            std::make_unique<revia::core::StructuredLogLearningSink>());
        backend_sync.attach(event_bus);
        backend_sync.publish_theme_state({{"active_theme", "unknown"}, {"source", "phase4_boot"}});
        backend_sync.publish_config_snapshot({{"source", "phase4_boot"}});

        orchestrator.initialize();
        event_bus.start();
        state_manager.transition_to(revia::core::CoreState::Idle, "core_boot_complete");
    } else {
        revia::core::StructuredLogger::instance().info("orchestrator.phase2.disabled", {
            {"reason", "REVIA_CORE_V2_ENABLED is false"}
        });
    }

    revia::EmotionNet emotion_net;
    revia::RouterClassifier router;
    revia::PluginManager plugins;
    plugins.discover("plugins");

    revia::Pipeline pipeline(telemetry, emotion_net, router, plugins);

    revia::WsServer ws_server(REVIA_WS_PORT, telemetry);
    revia::RestServer rest_server(REVIA_REST_PORT, telemetry, pipeline,
                                  plugins, emotion_net, router, ws_server,
                                  &event_bus, &state_manager);

    // Store threads instead of detaching - managed thread lifetime
    std::vector<std::thread> threads;
    threads.emplace_back([&] { ws_server.run(g_running); });
    threads.emplace_back([&] { rest_server.run(g_running); });
    threads.emplace_back([&] {
        while (g_running) {
            ws_server.broadcast_telemetry();
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });
    // Flush queued broadcast messages every 20ms for low-latency token delivery
    threads.emplace_back([&] {
        while (g_running) {
            ws_server.flush_broadcasts();
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    });

    std::cout << "[REVIA CORE] REST on :" << REVIA_REST_PORT
              << " | WS on :" << REVIA_WS_PORT << "\n";
    std::cout << "[REVIA CORE] Press Ctrl+C to stop.\n";

    while (g_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::cout << "\n[REVIA CORE] Shutting down...\n";
    orchestrator.shutdown();
    backend_sync.shutdown();
    event_bus.stop();
    state_manager.try_transition_to(revia::core::CoreState::Disabled, "core_shutdown");
    rest_server.stop();
    ws_server.stop();
    // Join all stored threads
    for (auto& t : threads) {
        if (t.joinable()) t.join();
    }

    return 0;
}
