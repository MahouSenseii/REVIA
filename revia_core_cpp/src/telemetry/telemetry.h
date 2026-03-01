#pragma once
#include <string>
#include <vector>
#include <mutex>
#include <chrono>
#include <fstream>
#include <nlohmann/json.hpp>

namespace revia {

using json = nlohmann::json;
using Clock = std::chrono::high_resolution_clock;

struct TelemetrySpan {
    std::string stage;
    std::string thread_name;
    std::string device = "CPU";
    double start_ms = 0;
    double end_ms = 0;
    double duration_ms = 0;
    int queue_depth = 0;
    std::string error;
    json extra = json::object();

    json to_json() const {
        return {
            {"stage", stage}, {"thread", thread_name}, {"device", device},
            {"start_ms", start_ms}, {"end_ms", end_ms}, {"duration_ms", duration_ms},
            {"queue_depth", queue_depth}, {"error", error}, {"extra", extra}
        };
    }
};

struct LLMMetrics {
    int tokens_generated = 0;
    double tokens_per_second = 0.0;
    int context_length = 0;
};

struct EmotionOutput {
    float valence = 0.0f;
    float arousal = 0.0f;
    float dominance = 0.0f;
    std::string label = "Neutral";
    float confidence = 0.0f;
    double inference_ms = 0.0;
};

struct RouterOutput {
    std::string mode = "chat";
    float confidence = 0.0f;
    std::string suggested_tool;
    bool rag_enable = false;
    float rag_confidence = 0.0f;
    double inference_ms = 0.0;
};

struct SystemMetrics {
    double cpu_percent = 12.0;
    double gpu_percent = 0.0;
    double ram_mb = 480.0;
    double vram_mb = 0.0;
    std::string health = "Online";
    std::string model_name = "Stub-7B";
    std::string backend = "CPU";
    std::string device = "CPU";
};

class TelemetryEngine {
public:
    static TelemetryEngine& instance();

    TelemetrySpan begin_span(const std::string& stage, const std::string& device = "CPU");
    void end_span(TelemetrySpan& span);

    void set_llm_metrics(const LLMMetrics& m);
    LLMMetrics get_llm_metrics() const;

    void set_emotion(const EmotionOutput& e);
    EmotionOutput get_emotion() const;

    void set_router(const RouterOutput& r);
    RouterOutput get_router() const;

    void set_system_metrics(const SystemMetrics& m);
    SystemMetrics get_system_metrics() const;

    void set_state(const std::string& s);
    std::string get_state() const;

    std::vector<TelemetrySpan> get_recent_spans(int n = 30) const;
    json get_snapshot() const;

private:
    TelemetryEngine();
    mutable std::mutex mtx_;
    std::vector<TelemetrySpan> spans_;
    LLMMetrics llm_;
    EmotionOutput emotion_;
    RouterOutput router_;
    SystemMetrics sys_;
    std::string state_ = "Idle";
    Clock::time_point epoch_;
    std::ofstream log_file_;
};

} // namespace revia
