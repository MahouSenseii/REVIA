#include "api/rest_server.h"
#include "telemetry/telemetry.h"
#include "pipeline/pipeline.h"
#include "plugins/plugin_manager.h"
#include "neural/EmotionNet.h"
#include "neural/RouterClassifier.h"
#include "api/ws_server.h"
#include <thread>
#include <fstream>
#include <filesystem>
#include <nlohmann/json.hpp>

namespace revia {

using json = nlohmann::json;

RestServer::RestServer(int port, TelemetryEngine& t, Pipeline& p,
                       PluginManager& pl, EmotionNet& e,
                       RouterClassifier& r, WsServer& ws)
    : port_(port), telemetry_(t), pipeline_(p), plugins_(pl),
      emotion_(e), router_(r), ws_(ws) {
    setup_routes();
}

RestServer::~RestServer() {
    // Join all pipeline processing threads on destruction
    std::lock_guard<std::mutex> lk(threads_mtx_);
    for (auto& t : pipeline_threads_) {
        if (t.joinable()) t.join();
    }
}

void RestServer::setup_routes() {
    // Helper: restrict CORS to localhost origins only (defence-in-depth against
    // DNS-rebinding or rogue browser tabs calling this local API).
    auto set_cors = [](const httplib::Request& req, httplib::Response& res) {
        // Whitelist of allowed origins
        static const std::vector<std::string> allowed_origins = {
            "http://127.0.0.1",
            "http://localhost",
            "http://127.0.0.1:3000",
            "http://localhost:3000"
        };

        // Check if request Origin header is in whitelist
        auto origin = req.get_header_value("Origin");
        bool origin_allowed = origin.empty();  // Allow no Origin header
        if (!origin.empty()) {
            for (const auto& allowed : allowed_origins) {
                if (origin == allowed) {
                    origin_allowed = true;
                    break;
                }
            }
        }

        if (origin_allowed && !origin.empty()) {
            res.set_header("Access-Control-Allow-Origin", origin);
        } else if (origin.empty()) {
            res.set_header("Access-Control-Allow-Origin", "http://127.0.0.1");
        }
        res.set_header("Vary", "Origin");
    };

    svr_.Get("/api/status", [this, set_cors](const httplib::Request& req, httplib::Response& res) {
        set_cors(req, res);
        json j;
        j["state"] = telemetry_.get_state();
        j["health"] = "Online";
        j["version"] = revia::VERSION;
        res.set_content(j.dump(), "application/json");
    });

    svr_.Get("/api/telemetry", [this, set_cors](const httplib::Request& req, httplib::Response& res) {
        set_cors(req, res);
        res.set_content(telemetry_.get_snapshot().dump(), "application/json");
    });

    svr_.Get("/api/plugins", [this, set_cors](const httplib::Request& req, httplib::Response& res) {
        set_cors(req, res);
        res.set_content(plugins_.to_json().dump(), "application/json");
    });

    svr_.Post(R"(/api/plugins/([a-zA-Z0-9_\-]{1,32})/enable)", [this, set_cors](const httplib::Request& req, httplib::Response& res) {
        set_cors(req, res);
        auto name = req.matches[1].str();
        bool ok = plugins_.enable(name);
        res.set_content(json({{"ok", ok}}).dump(), "application/json");
    });

    svr_.Post(R"(/api/plugins/([a-zA-Z0-9_\-]{1,32})/disable)", [this, set_cors](const httplib::Request& req, httplib::Response& res) {
        set_cors(req, res);
        auto name = req.matches[1].str();
        bool ok = plugins_.disable(name);
        res.set_content(json({{"ok", ok}}).dump(), "application/json");
    });

    svr_.Get("/api/neural", [this, set_cors](const httplib::Request& req, httplib::Response& res) {
        set_cors(req, res);
        json j;
        j["emotion_net"] = {
            {"enabled", emotion_.enabled},
            {"last_inference_ms", emotion_.last_inference_ms},
            {"last_output", telemetry_.get_emotion().label}
        };
        j["router_classifier"] = {
            {"enabled", router_.enabled},
            {"last_inference_ms", router_.last_inference_ms},
            {"last_output", telemetry_.get_router().mode}
        };
        res.set_content(j.dump(), "application/json");
    });

    svr_.Post(R"(/api/neural/([\w_]+)/enable)", [this, set_cors](const httplib::Request& req, httplib::Response& res) {
        set_cors(req, res);
        auto name = req.matches[1].str();
        if (name == "emotion_net") { emotion_.enabled = true; pipeline_.emotion_enabled = true; }
        else if (name == "router_classifier") { router_.enabled = true; pipeline_.router_enabled = true; }
        res.set_content(json({{"ok", true}}).dump(), "application/json");
    });

    svr_.Post(R"(/api/neural/([\w_]+)/disable)", [this, set_cors](const httplib::Request& req, httplib::Response& res) {
        set_cors(req, res);
        auto name = req.matches[1].str();
        if (name == "emotion_net") { emotion_.enabled = false; pipeline_.emotion_enabled = false; }
        else if (name == "router_classifier") { router_.enabled = false; pipeline_.router_enabled = false; }
        res.set_content(json({{"ok", true}}).dump(), "application/json");
    });

    svr_.Post("/api/chat", [this, set_cors](const httplib::Request& req, httplib::Response& res) {
        set_cors(req, res);
        try {
            auto body = json::parse(req.body);
            std::string text = body.value("text", "");
            if (text.empty()) {
                res.set_content(json({{"error", "empty text"}}).dump(), "application/json");
                return;
            }

            PipelineRequest pr;
            pr.user_text = text;
            pr.on_token = [this](const std::string& token) {
                ws_.broadcast_async(json({{"type", "chat_token"}, {"token", token}}).dump());
            };
            pr.on_complete = [this](const std::string& full) {
                ws_.broadcast(json({{"type", "chat_complete"}, {"text", full}}).dump());
            };
            pr.on_status = [this](const std::string& status) {
                ws_.broadcast(json({{"type", "status_update"}, {"state", status}}).dump());
            };

            // Store thread instead of detaching, use move semantics
            // Add concurrency limit to prevent resource exhaustion
            constexpr int MAX_CONCURRENT_REQUESTS = 8;
            {
                std::lock_guard<std::mutex> lk(threads_mtx_);
                // Clean up finished threads
                auto it = std::remove_if(pipeline_threads_.begin(), pipeline_threads_.end(),
                    [](std::thread& t) {
                        if (!t.joinable()) return true;  // Already joined, remove
                        return false;  // Still running, keep
                    });
                pipeline_threads_.erase(it, pipeline_threads_.end());
                if (pipeline_threads_.size() >= MAX_CONCURRENT_REQUESTS) {
                    res.status = 429;
                    res.set_content(R"({"error":"Too many concurrent requests"})", "application/json");
                    return;
                }
                pipeline_threads_.emplace_back([this, pr = std::move(pr)]() { pipeline_.process(pr); });
            }

            res.set_content(json({{"ok", true}, {"message", "Processing started"}}).dump(),
                            "application/json");
        } catch (const std::exception& e) {
            res.set_content(json({{"error", e.what()}}).dump(), "application/json");
        }
    });

    svr_.Get("/api/profile", [set_cors](const httplib::Request& req, httplib::Response& res) {
        set_cors(req, res);
        std::ifstream f("profiles/default_profile.json");
        if (f.is_open()) {
            std::string content((std::istreambuf_iterator<char>(f)),
                                 std::istreambuf_iterator<char>());
            res.set_content(content, "application/json");
        } else {
            res.set_content("{}", "application/json");
        }
    });

    // F-18: validate that the body is a well-formed JSON object before writing
    // to disk; also enforce a 64 KB size cap to prevent unbounded file growth.
    svr_.Post("/api/profile", [this, set_cors](const httplib::Request& req, httplib::Response& res) {
        set_cors(req, res);
        static constexpr std::size_t kMaxProfileBytes = 64 * 1024;
        if (req.body.size() > kMaxProfileBytes) {
            res.status = 413;
            res.set_content(json({{"error", "payload too large (max 64 KB)"}}).dump(), "application/json");
            return;
        }
        try {
            auto profile = json::parse(req.body);
            if (!profile.is_object()) {
                res.status = 400;
                res.set_content(json({{"error", "profile must be a JSON object"}}).dump(), "application/json");
                return;
            }
            std::filesystem::create_directories("profiles");
            std::ofstream f("profiles/default_profile.json");
            f << profile.dump(2);
            // If the profile carries a system_prompt field, propagate it into
            // the pipeline so the LLM personality is updated without restart.
            if (profile.contains("system_prompt") && profile["system_prompt"].is_string()) {
                pipeline_.set_system_prompt(profile["system_prompt"].get<std::string>());
            }
            res.set_content(json({{"ok", true}}).dump(), "application/json");
        } catch (const json::parse_error& e) {
            res.status = 400;
            res.set_content(json({{"error", std::string("invalid JSON: ") + e.what()}}).dump(), "application/json");
        }
    });

    svr_.Post("/api/model/config", [this, set_cors](const httplib::Request& req, httplib::Response& res) {
        set_cors(req, res);
        try {
            auto body = json::parse(req.body);
            std::string server_url = body.value("local_server_url", "");
            std::string model = body.value("api_model", body.value("local_path", ""));
            pipeline_.set_llm_config(server_url, model);
            res.set_content(json({{"ok", true}}).dump(), "application/json");
        } catch (const std::exception& e) {
            res.set_content(json({{"error", e.what()}}).dump(), "application/json");
        }
    });

    svr_.Get("/api/training-data", [set_cors](const httplib::Request& req, httplib::Response& res) {
        set_cors(req, res);
        json j;
        j["routing_events"] = 0;
        j["emotion_events"] = 0;
        res.set_content(j.dump(), "application/json");
    });

    svr_.Options(R"(/(.*)))", [set_cors](const httplib::Request& req, httplib::Response& res) {
        set_cors(req, res);
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type");
    });
}

void RestServer::run(std::atomic<bool>& running) {
    // Set server to periodically check running flag instead of blocking indefinitely
    svr_.set_post_routing_handler([&running](const httplib::Request&, httplib::Response&) {
        // Check running flag to allow graceful shutdown
        if (!running) {
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });
    svr_.listen("0.0.0.0", port_);
}

void RestServer::stop() {
    svr_.stop();
}

} // namespace revia
