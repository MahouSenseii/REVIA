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

void RestServer::setup_routes() {
    svr_.Get("/api/status", [this](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        json j;
        j["state"] = telemetry_.get_state();
        j["health"] = "Online";
        j["version"] = revia::VERSION;
        res.set_content(j.dump(), "application/json");
    });

    svr_.Get("/api/telemetry", [this](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_content(telemetry_.get_snapshot().dump(), "application/json");
    });

    svr_.Get("/api/plugins", [this](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_content(plugins_.to_json().dump(), "application/json");
    });

    svr_.Post(R"(/api/plugins/([\w\-]+)/enable)", [this](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        auto name = req.matches[1].str();
        bool ok = plugins_.enable(name);
        res.set_content(json({{"ok", ok}}).dump(), "application/json");
    });

    svr_.Post(R"(/api/plugins/([\w\-]+)/disable)", [this](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        auto name = req.matches[1].str();
        bool ok = plugins_.disable(name);
        res.set_content(json({{"ok", ok}}).dump(), "application/json");
    });

    svr_.Get("/api/neural", [this](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
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

    svr_.Post(R"(/api/neural/([\w_]+)/enable)", [this](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        auto name = req.matches[1].str();
        if (name == "emotion_net") { emotion_.enabled = true; pipeline_.emotion_enabled = true; }
        else if (name == "router_classifier") { router_.enabled = true; pipeline_.router_enabled = true; }
        res.set_content(json({{"ok", true}}).dump(), "application/json");
    });

    svr_.Post(R"(/api/neural/([\w_]+)/disable)", [this](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        auto name = req.matches[1].str();
        if (name == "emotion_net") { emotion_.enabled = false; pipeline_.emotion_enabled = false; }
        else if (name == "router_classifier") { router_.enabled = false; pipeline_.router_enabled = false; }
        res.set_content(json({{"ok", true}}).dump(), "application/json");
    });

    svr_.Post("/api/chat", [this](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
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
                ws_.broadcast(json({{"type", "chat_token"}, {"token", token}}).dump());
            };
            pr.on_complete = [this](const std::string& full) {
                ws_.broadcast(json({{"type", "chat_complete"}, {"text", full}}).dump());
            };
            pr.on_status = [this](const std::string& status) {
                ws_.broadcast(json({{"type", "status_update"}, {"state", status}}).dump());
            };

            std::thread([this, pr]() { pipeline_.process(pr); }).detach();

            res.set_content(json({{"ok", true}, {"message", "Processing started"}}).dump(),
                            "application/json");
        } catch (const std::exception& e) {
            res.set_content(json({{"error", e.what()}}).dump(), "application/json");
        }
    });

    svr_.Get("/api/profile", [](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        std::ifstream f("profiles/default_profile.json");
        if (f.is_open()) {
            std::string content((std::istreambuf_iterator<char>(f)),
                                 std::istreambuf_iterator<char>());
            res.set_content(content, "application/json");
        } else {
            res.set_content("{}", "application/json");
        }
    });

    svr_.Post("/api/profile", [](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        std::filesystem::create_directories("profiles");
        std::ofstream f("profiles/default_profile.json");
        f << req.body;
        res.set_content(json({{"ok", true}}).dump(), "application/json");
    });

    svr_.Post("/api/model/config", [this](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
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

    svr_.Get("/api/training-data", [](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        json j;
        j["routing_events"] = 0;
        j["emotion_events"] = 0;
        res.set_content(j.dump(), "application/json");
    });

    svr_.Options(R"(/(.*)))", [](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type");
    });
}

void RestServer::run(std::atomic<bool>&) {
    svr_.listen("0.0.0.0", port_);
}

void RestServer::stop() {
    svr_.stop();
}

} // namespace revia
