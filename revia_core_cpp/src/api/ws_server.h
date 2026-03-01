#pragma once
#include <string>
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
    void broadcast_telemetry();

private:
    int port_;
    TelemetryEngine& telemetry_;
    ix::WebSocketServer server_;
};

} // namespace revia
