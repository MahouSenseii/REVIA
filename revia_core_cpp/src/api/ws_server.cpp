#include "api/ws_server.h"
#include "telemetry/telemetry.h"
#include <iostream>
#include <thread>
#include <chrono>

namespace revia {

WsServer::WsServer(int port, TelemetryEngine& t)
    : port_(port), telemetry_(t), server_(port, "0.0.0.0") {

    server_.setOnClientMessageCallback(
        [](std::shared_ptr<ix::ConnectionState>,
           ix::WebSocket&, const ix::WebSocketMessagePtr& msg) {
            if (msg->type == ix::WebSocketMessageType::Open) {
                std::cout << "[WS] Client connected from " << msg->openInfo.uri << "\n";
            } else if (msg->type == ix::WebSocketMessageType::Close) {
                std::cout << "[WS] Client disconnected\n";
            }
        }
    );
}

void WsServer::run(std::atomic<bool>& running) {
    auto res = server_.listen();
    if (!res.first) {
        std::cerr << "[WS] Failed to listen on port " << port_ << ": " << res.second << "\n";
        return;
    }
    server_.start();
    std::cout << "[WS] Server started on port " << port_ << "\n";
    while (running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void WsServer::stop() {
    server_.stop();
}

void WsServer::broadcast(const std::string& msg) {
    for (auto& client : server_.getClients()) {
        client->send(msg);
    }
}

void WsServer::broadcast_telemetry() {
    auto snapshot = telemetry_.get_snapshot();
    nlohmann::json j;
    j["type"] = "telemetry_update";
    j["data"] = snapshot;
    broadcast(j.dump());
}

} // namespace revia
