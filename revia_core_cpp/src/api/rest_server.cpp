#include "api/rest_server.h"
#include "telemetry/telemetry.h"
#include "pipeline/pipeline.h"
#include "plugins/plugin_manager.h"
#include "neural/EmotionNet.h"
#include "neural/RouterClassifier.h"
#include "api/ws_server.h"
#include "core/EventBus.h"
#include "core/FeatureFlags.h"
#include "core/StateManager.h"
#include "core/StructuredLogger.h"
#include <algorithm>
#include <cstdlib>
#include <thread>
#include <fstream>
#include <filesystem>
#include <nlohmann/json.hpp>

namespace revia {

using json = nlohmann::json;

namespace {

revia::core::EventType parse_event_type(const std::string& raw) {
    if (raw == "UserText" || raw == "user_text" || raw == "userText") {
        return revia::core::EventType::UserText;
    }
    if (raw == "UserSpeech" || raw == "user_speech" || raw == "userSpeech") {
        return revia::core::EventType::UserSpeech;
    }
    if (raw == "ConfigChange" || raw == "config_change" || raw == "configChange") {
        return revia::core::EventType::ConfigChange;
    }
    return revia::core::EventType::Unknown;
}

revia::core::EventSource parse_event_source(const std::string& raw) {
    if (raw == "Discord" || raw == "discord") {
        return revia::core::EventSource::Discord;
    }
    if (raw == "Twitch" || raw == "twitch") {
        return revia::core::EventSource::Twitch;
    }
    if (raw == "LocalSTT" || raw == "local_stt" || raw == "localSTT") {
        return revia::core::EventSource::LocalSTT;
    }
    if (raw == "ControllerUI" || raw == "controller_ui" || raw == "controllerUI") {
        return revia::core::EventSource::ControllerUI;
    }
    if (raw == "InternalTimer" || raw == "internal_timer" || raw == "internalTimer") {
        return revia::core::EventSource::InternalTimer;
    }
    if (raw == "InternalModel" || raw == "internal_model" || raw == "internalModel") {
        return revia::core::EventSource::InternalModel;
    }
    if (raw == "Plugin" || raw == "plugin") {
        return revia::core::EventSource::Plugin;
    }
    return revia::core::EventSource::Unknown;
}

} // namespace

RestServer::RestServer(int port, TelemetryEngine& t, Pipeline& p,
                       PluginManager& pl, EmotionNet& e,
                       RouterClassifier& r, WsServer& ws,
                       core::EventBus* event_bus,
                       core::StateManager* state_manager)
    : port_(port), telemetry_(t), pipeline_(p), plugins_(pl),
      emotion_(e), router_(r), ws_(ws),
      event_bus_(event_bus), state_manager_(state_manager) {
    setup_routes();
}

RestServer::~RestServer() {
    // Join all pipeline processing threads on destruction
    std::lock_guard<std::mutex> lk(threads_mtx_);
    for (auto& managed_thread : pipeline_threads_) {
        if (managed_thread.thread.joinable()) managed_thread.thread.join();
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
                // Clean up finished threads — join any that are done, then remove
                // Use a shared future-based approach: threads that completed set a flag
                // For simplicity, try joining with zero timeout via native handle
                auto it = pipeline_threads_.begin();
                while (it != pipeline_threads_.end()) {
                    if (it->completed && it->completed->load(std::memory_order_acquire)) {
                        if (it->thread.joinable()) {
                            it->thread.join();
                        }
                        it = pipeline_threads_.erase(it);
                    } else if (!it->thread.joinable()) {
                        it = pipeline_threads_.erase(it);
                    } else {
                        ++it;
                    }
                }
                if (pipeline_threads_.size() >= MAX_CONCURRENT_REQUESTS) {
                    res.status = 429;
                    res.set_content(R"({"error":"Too many concurrent requests"})", "application/json");
                    return;
                }
                auto completed = std::make_shared<std::atomic<bool>>(false);
                pipeline_threads_.push_back(ManagedPipelineThread{
                    std::thread([this, pr = std::move(pr), completed]() mutable {
                        pipeline_.process(pr);
                        completed->store(true, std::memory_order_release);
                    }),
                    completed,
                });
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
            // If the profile carries a system_prompt field, sanitize and propagate
            // it into the pipeline so the LLM personality is updated without restart.
            if (profile.contains("system_prompt") && profile["system_prompt"].is_string()) {
                std::string sp = profile["system_prompt"].get<std::string>();
                // Cap length to prevent excessively large prompts from being injected.
                static constexpr size_t kMaxSystemPromptLen = 10000;
                if (sp.size() > kMaxSystemPromptLen) {
                    sp.resize(kMaxSystemPromptLen);
                }
                // Strip null bytes — they can be used to truncate C-string parsers.
                sp.erase(std::remove(sp.begin(), sp.end(), '\0'), sp.end());
                pipeline_.set_system_prompt(sp);
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

    svr_.Get("/api/core/state", [this, set_cors](const httplib::Request& req, httplib::Response& res) {
        set_cors(req, res);
        if (!state_manager_) {
            res.status = 503;
            res.set_content(json({{"ok", false}, {"error", "core state manager unavailable"}}).dump(),
                            "application/json");
            return;
        }
        res.set_content(json({{"ok", true}, {"state", state_manager_->snapshot_json()}}).dump(),
                        "application/json");
    });

    svr_.Post("/api/core/events", [this, set_cors](const httplib::Request& req, httplib::Response& res) {
        set_cors(req, res);

        if (!core::FeatureFlags::instance().core_v2_enabled()) {
            res.status = 202;
            res.set_content(json({{"ok", false}, {"status", "core_v2_disabled"}}).dump(),
                            "application/json");
            return;
        }
        if (!event_bus_) {
            res.status = 503;
            res.set_content(json({{"ok", false}, {"error", "core event bus unavailable"}}).dump(),
                            "application/json");
            return;
        }

        static constexpr std::size_t kMaxEventBytes = 64 * 1024;
        if (req.body.size() > kMaxEventBytes) {
            res.status = 413;
            res.set_content(json({{"ok", false}, {"error", "event payload too large (max 64 KB)"}}).dump(),
                            "application/json");
            return;
        }

        try {
            auto body = json::parse(req.body);
            if (!body.is_object()) {
                res.status = 400;
                res.set_content(json({{"ok", false}, {"error", "event body must be a JSON object"}}).dump(),
                                "application/json");
                return;
            }

            const std::string raw_type = body.value("type", std::string{"Unknown"});
            const auto event_type = parse_event_type(raw_type);
            if (event_type != core::EventType::UserText &&
                event_type != core::EventType::ConfigChange) {
                res.status = 400;
                res.set_content(json({
                    {"ok", false},
                    {"error", "Core event endpoint accepts UserText or ConfigChange"}
                }).dump(), "application/json");
                return;
            }

            json payload = body.contains("payload") ? body["payload"] : json::object();
            if (!payload.is_object()) {
                res.status = 400;
                res.set_content(json({{"ok", false}, {"error", "payload must be a JSON object"}}).dump(),
                                "application/json");
                return;
            }
            if (event_type == core::EventType::UserText &&
                (!payload.contains("text") || !payload["text"].is_string() ||
                 payload["text"].get<std::string>().empty())) {
                res.status = 400;
                res.set_content(json({{"ok", false}, {"error", "UserText payload requires non-empty text"}}).dump(),
                                "application/json");
                return;
            }

            core::IEvent event;
            event.id = body.value("id", std::string{});
            event.type = event_type;
            event.source = parse_event_source(body.value("source", std::string{"Unknown"}));
            event.payload = std::move(payload);
            event.correlation_id = body.value("correlation_id", std::string{});

            const bool published = event_bus_->publish(std::move(event));
            if (!published) {
                res.status = 503;
                res.set_content(json({{"ok", false}, {"error", "core event queue full"}}).dump(),
                                "application/json");
                return;
            }

            core::StructuredLogger::instance().info("api.core_event.accepted", {
                {"event_type", raw_type},
                {"source", body.value("source", std::string{"Unknown"})}
            });
            res.status = 202;
            res.set_content(json({{"ok", true}, {"status", "queued"}}).dump(),
                            "application/json");
        } catch (const json::parse_error& exc) {
            res.status = 400;
            res.set_content(json({{"ok", false}, {"error", std::string("invalid JSON: ") + exc.what()}}).dump(),
                            "application/json");
        } catch (const std::exception& exc) {
            res.status = 500;
            res.set_content(json({{"ok", false}, {"error", exc.what()}}).dump(),
                            "application/json");
        }
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
    const char* host_env = std::getenv("REVIA_REST_HOST");
    const std::string host = (host_env && *host_env) ? host_env : "127.0.0.1";
    svr_.listen(host, port_);
}

void RestServer::stop() {
    svr_.stop();
}

} // namespace revia
