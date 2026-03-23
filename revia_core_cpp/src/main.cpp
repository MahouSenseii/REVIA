#include <iostream>
#include <thread>
#include <csignal>
#include <atomic>
#include <filesystem>
#include "telemetry/telemetry.h"
#include "pipeline/pipeline.h"
#include "neural/EmotionNet.h"
#include "neural/RouterClassifier.h"
#include "plugins/plugin_manager.h"
#include "api/rest_server.h"
#include "api/ws_server.h"

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

    revia::EmotionNet emotion_net;
    revia::RouterClassifier router;
    revia::PluginManager plugins;
    plugins.discover("plugins");

    revia::Pipeline pipeline(telemetry, emotion_net, router, plugins);

    revia::WsServer ws_server(REVIA_WS_PORT, telemetry);
    revia::RestServer rest_server(REVIA_REST_PORT, telemetry, pipeline,
                                  plugins, emotion_net, router, ws_server);

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
    rest_server.stop();
    ws_server.stop();
    // Join all stored threads
    for (auto& t : threads) {
        if (t.joinable()) t.join();
    }

    return 0;
}
