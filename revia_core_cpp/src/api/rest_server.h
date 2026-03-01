#pragma once
#include <atomic>
#include <httplib.h>

namespace revia {

class TelemetryEngine;
class Pipeline;
class PluginManager;
class EmotionNet;
class RouterClassifier;
class WsServer;

class RestServer {
public:
    RestServer(int port, TelemetryEngine& telemetry, Pipeline& pipeline,
               PluginManager& plugins, EmotionNet& emotion,
               RouterClassifier& router, WsServer& ws);
    void run(std::atomic<bool>& running);
    void stop();

private:
    int port_;
    httplib::Server svr_;
    TelemetryEngine& telemetry_;
    Pipeline& pipeline_;
    PluginManager& plugins_;
    EmotionNet& emotion_;
    RouterClassifier& router_;
    WsServer& ws_;

    void setup_routes();
};

} // namespace revia
