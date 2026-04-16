#pragma once
#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include <httplib.h>

namespace revia {

class TelemetryEngine;
class Pipeline;
class PluginManager;
class EmotionNet;
class RouterClassifier;
class WsServer;

namespace core {
class EventBus;
class StateManager;
}

class RestServer {
public:
    RestServer(int port, TelemetryEngine& telemetry, Pipeline& pipeline,
               PluginManager& plugins, EmotionNet& emotion,
               RouterClassifier& router, WsServer& ws,
               core::EventBus* event_bus = nullptr,
               core::StateManager* state_manager = nullptr);
    ~RestServer();
    void run(std::atomic<bool>& running);
    void stop();

private:
    struct ManagedPipelineThread {
        std::thread thread;
        std::shared_ptr<std::atomic<bool>> completed;
    };

    int port_;
    httplib::Server svr_;
    TelemetryEngine& telemetry_;
    Pipeline& pipeline_;
    PluginManager& plugins_;
    EmotionNet& emotion_;
    RouterClassifier& router_;
    WsServer& ws_;
    core::EventBus* event_bus_;
    core::StateManager* state_manager_;
    std::vector<ManagedPipelineThread> pipeline_threads_;  // Store managed threads with completion flags
    mutable std::mutex threads_mtx_;  // Protect thread storage

    void setup_routes();
};

} // namespace revia
