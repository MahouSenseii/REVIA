#include "pipeline/pipeline.h"
#include "neural/EmotionNet.h"
#include "neural/RouterClassifier.h"
#include "plugins/plugin_manager.h"
#include <thread>
#include <chrono>
#include <cstdlib>

namespace revia {

Pipeline::Pipeline(TelemetryEngine& t, EmotionNet& e, RouterClassifier& r, PluginManager& p)
    : telemetry_(t), emotion_(e), router_(r), plugins_(p) {}

void Pipeline::process(const PipelineRequest& req) {
    if (req.on_status) req.on_status("Processing");
    telemetry_.set_state("Processing");

    stage_input_capture(req);
    stage_stt_batch(req);

    if (router_enabled) stage_router_classify(req.user_text);
    if (emotion_enabled) stage_emotion_infer(req.user_text);

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
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    telemetry_.end_span(span);
}

void Pipeline::stage_stt_batch(const PipelineRequest&) {
    auto span = telemetry_.begin_span("stt_batch_collect");
    span.extra["batch_window_ms"] = 200;
    span.extra["partial_transcript_len"] = 0;
    std::this_thread::sleep_for(std::chrono::milliseconds(3));
    telemetry_.end_span(span);

    auto span2 = telemetry_.begin_span("stt_decode_partial");
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
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
    std::this_thread::sleep_for(std::chrono::milliseconds(15));
    telemetry_.end_span(span);
}

void Pipeline::stage_prompt_assemble(const std::string&) {
    auto span = telemetry_.begin_span("prompt_assemble");
    std::this_thread::sleep_for(std::chrono::milliseconds(3));
    telemetry_.end_span(span);
}

void Pipeline::stage_llm_generate(const PipelineRequest& req, const std::string&) {
    auto span = telemetry_.begin_span("llm_generate_total");
    telemetry_.set_state("Generating");
    if (req.on_status) req.on_status("Generating");

    std::vector<std::string> tokens = {
        "I", " understand", " your", " request", ".", " Let", " me",
        " help", " you", " with", " that", ".", " As", " your",
        " neural", " assistant", ",", " I'm", " here", " to",
        " provide", " thoughtful", " and", " helpful", " responses", "."
    };

    LLMMetrics m;
    m.context_length = 2048;
    std::string full_response;
    auto gen_start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < tokens.size(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(40 + std::rand() % 30));
        full_response += tokens[i];
        if (req.on_token) req.on_token(tokens[i]);

        auto elapsed = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - gen_start).count();
        m.tokens_generated = static_cast<int>(i + 1);
        m.tokens_per_second = elapsed > 0 ? m.tokens_generated / elapsed : 0;
        telemetry_.set_llm_metrics(m);
    }

    if (req.on_complete) req.on_complete(full_response);

    stage_tts_synthesize(full_response);
    telemetry_.end_span(span);
}

void Pipeline::stage_tts_synthesize(const std::string&) {
    auto span = telemetry_.begin_span("tts_synthesize");
    telemetry_.set_state("Answering (TTS)");
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    telemetry_.end_span(span);
}

void Pipeline::stage_memory_write(const std::string&, const std::string&) {
    auto span = telemetry_.begin_span("memory_write");
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    telemetry_.end_span(span);
}

} // namespace revia
