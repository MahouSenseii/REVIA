#pragma once
#include <string>
#include <vector>
#include <atomic>
#include <mutex>
#include <ixwebsocket/IXWebSocketServer.h>

namespace revia {

class TelemetryEngine;

class WsServer {
public:
    WsServer(int port, TelemetryEngine& telemetry);
    void run(std::atomic<bool>& running);
    void stop();
    void broadcast(const std::string& msg);
    void broadcast_async(const std::string& msg);
    void flush_broadcasts();
    void broadcast_telemetry();

private:
    int port_;
    TelemetryEngine& telemetry_;
    ix::WebSocketServer server_;
    mutable std::mutex clients_mtx_;  // Protects client list access
    std::mutex broadcast_queue_mtx_;
    std::vector<std::string> broadcast_queue_;
};

} // namespace revia
