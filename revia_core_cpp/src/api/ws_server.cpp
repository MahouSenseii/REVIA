#include "api/ws_server.h"
#include "telemetry/telemetry.h"
#include <cstdlib>
#include <iostream>
#include <thread>
#include <chrono>
#include <nlohmann/json.hpp>

namespace revia {

namespace {

std::string websocket_bind_host() {
    const char* host = std::getenv("REVIA_WS_HOST");
    return (host && *host) ? std::string(host) : std::string("127.0.0.1");
}

} // namespace

WsServer::WsServer(int port, TelemetryEngine& t)
    : port_(port), telemetry_(t), server_(port, websocket_bind_host()) {

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
    // getClients() returns a snapshot copy under IXWebSocket's own internal lock,
    // so iterating the returned container is safe even if clients connect/disconnect
    // concurrently.  Holding clients_mtx_ here would not protect the IXWebSocket-
    // internal list; copying first is the correct fix.
    auto clients = server_.getClients();
    for (auto& client : clients) {
        client->send(msg);
    }
}

void WsServer::broadcast_async(const std::string& msg) {
    // Add message batching - instead of sending immediately, queue for next flush
    std::lock_guard<std::mutex> lk(broadcast_queue_mtx_);
    broadcast_queue_.push_back(msg);
}

void WsServer::flush_broadcasts() {
    std::vector<std::string> messages;
    {
        std::lock_guard<std::mutex> lk(broadcast_queue_mtx_);
        messages.swap(broadcast_queue_);
    }
    if (messages.empty()) return;
    auto clients = server_.getClients();
    for (auto& client : clients) {
        for (auto& msg : messages) {
            client->send(msg);
        }
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
