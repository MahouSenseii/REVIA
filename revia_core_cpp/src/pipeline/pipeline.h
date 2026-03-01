#pragma once
#include <string>
#include <functional>
#include <atomic>
#include "telemetry/telemetry.h"

namespace revia {

class EmotionNet;
class RouterClassifier;
class PluginManager;

struct PipelineRequest {
    std::string user_text;
    bool vision_active = false;
    std::function<void(const std::string&)> on_token;
    std::function<void(const std::string&)> on_complete;
    std::function<void(const std::string&)> on_status;
};

class Pipeline {
public:
    Pipeline(TelemetryEngine& telemetry, EmotionNet& emotion,
             RouterClassifier& router, PluginManager& plugins);

    void process(const PipelineRequest& req);

    std::atomic<bool> emotion_enabled{true};
    std::atomic<bool> router_enabled{true};

private:
    TelemetryEngine& telemetry_;
    EmotionNet& emotion_;
    RouterClassifier& router_;
    PluginManager& plugins_;

    void stage_input_capture(const PipelineRequest& req);
    void stage_stt_batch(const PipelineRequest& req);
    void stage_router_classify(const std::string& text);
    void stage_emotion_infer(const std::string& text);
    void stage_rag_retrieve(const std::string& text);
    void stage_prompt_assemble(const std::string& text);
    void stage_llm_generate(const PipelineRequest& req, const std::string& prompt);
    void stage_tts_synthesize(const std::string& text);
    void stage_memory_write(const std::string& user, const std::string& assistant);
};

} // namespace revia
