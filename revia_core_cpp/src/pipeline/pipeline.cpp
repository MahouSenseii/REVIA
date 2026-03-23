#include "pipeline/pipeline.h"
#include "neural/EmotionNet.h"
#include "neural/RouterClassifier.h"
#include "plugins/plugin_manager.h"
#include <thread>
#include <chrono>
#include <cstdlib>
#include <sstream>
#include <iostream>
#include <future>
#include <httplib.h>
#include <nlohmann/json.hpp>

namespace revia {

Pipeline::Pipeline(TelemetryEngine& t, EmotionNet& e, RouterClassifier& r, PluginManager& p)
    : telemetry_(t), emotion_(e), router_(r), plugins_(p) {}

void Pipeline::set_llm_config(const std::string& server_url, const std::string& model) {
    std::lock_guard<std::mutex> lk(llm_cfg_mtx_);
    if (!server_url.empty()) llm_url_ = server_url;
    if (!model.empty()) llm_model_ = model;
}

void Pipeline::set_system_prompt(const std::string& prompt) {
    std::lock_guard<std::mutex> lk(llm_cfg_mtx_);
    if (!prompt.empty()) system_prompt_ = prompt;
}

void Pipeline::process(const PipelineRequest& req) {
    if (req.on_status) req.on_status("Processing");
    telemetry_.set_state("Processing");

    stage_input_capture(req);
    stage_stt_batch(req);

    // Parallelize emotion and router classification for better performance
    std::future<void> router_fut;
    std::future<void> emotion_fut;
    if (router_enabled) {
        router_fut = std::async(std::launch::async, [this, text = req.user_text]() {
            stage_router_classify(text);
        });
    }
    if (emotion_enabled) {
        emotion_fut = std::async(std::launch::async, [this, text = req.user_text]() {
            stage_emotion_infer(text);
        });
    }
    // Wait for both to complete, with error recovery
    if (router_fut.valid()) {
        try { router_fut.get(); }
        catch (const std::exception& e) {
            std::cerr << "[Pipeline] Router classification failed: " << e.what() << "\n";
            auto span = telemetry_.begin_span("router_error");
            span.extra["error"] = e.what();
            telemetry_.end_span(span);
        }
    }
    if (emotion_fut.valid()) {
        try { emotion_fut.get(); }
        catch (const std::exception& e) {
            std::cerr << "[Pipeline] Emotion inference failed: " << e.what() << "\n";
            auto span = telemetry_.begin_span("emotion_error");
            span.extra["error"] = e.what();
            telemetry_.end_span(span);
        }
    }

    auto rout = telemetry_.get_router();
    if (rout.rag_enable) stage_rag_retrieve(req.user_text);

    stage_prompt_assemble(req.user_text);

    std::string prompt = "User: " + req.user_text + "\nAssistant:";
    stage_llm_generate(req, prompt);

    stage_memory_write(req.user_text, "");

    telemetry_.set_state("Idle");
    if (req.on_status) req.on_status("Idle");
}

void Pipeline::stage_input_capture(const PipelineRequest&) {
    auto span = telemetry_.begin_span("input_capture");
    // REMOVED: artificial latency
    telemetry_.end_span(span);
}

void Pipeline::stage_stt_batch(const PipelineRequest&) {
    auto span = telemetry_.begin_span("stt_batch_collect");
    span.extra["batch_window_ms"] = 200;
    span.extra["partial_transcript_len"] = 0;
    // REMOVED: artificial latency
    telemetry_.end_span(span);

    auto span2 = telemetry_.begin_span("stt_decode_partial");
    // REMOVED: artificial latency
    telemetry_.end_span(span2);
}

void Pipeline::stage_router_classify(const std::string& text) {
    auto span = telemetry_.begin_span("router_classify");
    auto result = router_.classify(text, "", telemetry_.get_emotion());
    telemetry_.set_router(result);
    telemetry_.end_span(span);
}

void Pipeline::stage_emotion_infer(const std::string& text) {
    auto span = telemetry_.begin_span("emotion_infer");
    auto result = emotion_.infer(text);
    telemetry_.set_emotion(result);
    telemetry_.end_span(span);
}

void Pipeline::stage_rag_retrieve(const std::string&) {
    auto span = telemetry_.begin_span("rag_retrieve");
    // REMOVED: artificial latency
    telemetry_.end_span(span);
}

void Pipeline::stage_prompt_assemble(const std::string&) {
    auto span = telemetry_.begin_span("prompt_assemble");
    // REMOVED: artificial latency
    telemetry_.end_span(span);
}

void Pipeline::stage_llm_generate(const PipelineRequest& req, const std::string&) {
    auto span = telemetry_.begin_span("llm_generate_total");
    telemetry_.set_state("Generating");
    if (req.on_status) req.on_status("Generating");

    std::string url, model, sys_prompt;
    std::string host = "127.0.0.1";
    int port = 8080;
    {
        // Single lock scope for config + URL cache (both guarded by llm_cfg_mtx_)
        std::lock_guard<std::mutex> lk(llm_cfg_mtx_);
        url = llm_url_;
        model = llm_model_;
        sys_prompt = system_prompt_;

        // Cache parsed URL components to avoid re-parsing on every request
        if (cached_host_.empty() || cached_url_ != url) {
            std::string rest = url;
            auto scheme_end = rest.find("://");
            if (scheme_end != std::string::npos) rest = rest.substr(scheme_end + 3);

            // Remove path if present (everything after /)
            auto path_end = rest.find('/');
            if (path_end != std::string::npos) rest = rest.substr(0, path_end);

            auto colon = rest.rfind(':');
            if (colon != std::string::npos) {
                host = rest.substr(0, colon);
                try {
                    std::string port_str = rest.substr(colon + 1);
                    int parsed_port = std::stoi(port_str);
                    if (parsed_port > 0 && parsed_port <= 65535) {
                        port = parsed_port;
                    } else {
                        std::cerr << "[Pipeline] Invalid port number: " << parsed_port << ", using default 8080\n";
                    }
                } catch (const std::invalid_argument& e) {
                    std::cerr << "[Pipeline] Failed to parse port: " << e.what() << ", using default 8080\n";
                } catch (const std::out_of_range& e) {
                    std::cerr << "[Pipeline] Port number out of range: " << e.what() << ", using default 8080\n";
                }
            } else {
                host = rest;
            }
            cached_host_ = host;
            cached_port_ = port;
            cached_url_ = url;
        } else {
            host = cached_host_;
            port = cached_port_;
        }
    }

    nlohmann::json body = {
        {"model", model},
        {"messages", nlohmann::json::array({
            {{"role", "system"}, {"content", sys_prompt}},
            {{"role", "user"}, {"content", req.user_text}}
        })},
        {"temperature", 0.7},
        {"stream", true}
    };

    std::string full_response;
    LLMMetrics m;
    m.context_length = 2048;
    auto gen_start = std::chrono::high_resolution_clock::now();
    int token_count = 0;

    httplib::Client cli(host, port);
    cli.set_connection_timeout(5, 0);
    cli.set_read_timeout(120, 0);

    std::string sse_buffer;
    auto result = cli.Post(
        "/v1/chat/completions",
        httplib::Headers{},
        body.dump(),
        "application/json",
        [&](const char* data, size_t len) -> bool {
            sse_buffer.append(data, len);
            size_t pos;
            while ((pos = sse_buffer.find('\n')) != std::string::npos) {
                std::string line = sse_buffer.substr(0, pos);
                sse_buffer.erase(0, pos + 1);
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (line.rfind("data: ", 0) != 0) continue;
                std::string payload = line.substr(6);
                if (payload == "[DONE]") return true;
                try {
                    auto chunk = nlohmann::json::parse(payload);
                    std::string tok = chunk["choices"][0]["delta"].value("content", "");
                    if (!tok.empty()) {
                        full_response += tok;
                        ++token_count;
                        if (req.on_token) req.on_token(tok);
                        auto elapsed = std::chrono::duration<double>(
                            std::chrono::high_resolution_clock::now() - gen_start).count();
                        m.tokens_generated = token_count;
                        m.tokens_per_second = elapsed > 0 ? token_count / elapsed : 0;
                        telemetry_.set_llm_metrics(m);
                    }
                } catch (const std::exception& e) {
                    std::cerr << "[Pipeline] Exception parsing LLM response: " << e.what() << "\n";
                }
            }
            return true;
        }
    );

    if (!result || result->status != 200) {
        std::string err = "[LLM Error] Cannot reach " + host + ":" + std::to_string(port)
                        + ". Make sure your LLM server is running with a model loaded.";
        if (req.on_token) req.on_token(err);
        full_response = err;
    }

    if (req.on_complete) req.on_complete(full_response);
    stage_tts_synthesize(full_response);
    telemetry_.end_span(span);
}

void Pipeline::stage_tts_synthesize(const std::string&) {
    auto span = telemetry_.begin_span("tts_synthesize");
    telemetry_.set_state("Answering (TTS)");
    // REMOVED: artificial latency
    telemetry_.end_span(span);
}

void Pipeline::stage_memory_write(const std::string&, const std::string&) {
    auto span = telemetry_.begin_span("memory_write");
    // REMOVED: artificial latency
    telemetry_.end_span(span);
}

} // namespace revia
