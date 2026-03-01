#include "telemetry/telemetry.h"
#include <thread>
#include <sstream>
#include <ctime>
#include <iomanip>
#include <filesystem>
#include <algorithm>

namespace revia {

TelemetryEngine& TelemetryEngine::instance() {
    static TelemetryEngine inst;
    return inst;
}

TelemetryEngine::TelemetryEngine() : epoch_(Clock::now()) {
    namespace fs = std::filesystem;
    fs::create_directories("logs");
    auto t = std::time(nullptr);
    std::ostringstream fname;
#ifdef _WIN32
    std::tm tm_buf;
    localtime_s(&tm_buf, &t);
    fname << "logs/telemetry_" << std::put_time(&tm_buf, "%Y%m%d") << ".jsonl";
#else
    fname << "logs/telemetry_" << std::put_time(std::localtime(&t), "%Y%m%d") << ".jsonl";
#endif
    log_file_.open(fname.str(), std::ios::app);
}

TelemetrySpan TelemetryEngine::begin_span(const std::string& stage, const std::string& device) {
    TelemetrySpan s;
    s.stage = stage;
    s.device = device;
    auto now = Clock::now();
    s.start_ms = std::chrono::duration<double, std::milli>(now - epoch_).count();
    std::ostringstream ss;
    ss << std::this_thread::get_id();
    s.thread_name = ss.str();
    return s;
}

void TelemetryEngine::end_span(TelemetrySpan& s) {
    auto now = Clock::now();
    s.end_ms = std::chrono::duration<double, std::milli>(now - epoch_).count();
    s.duration_ms = s.end_ms - s.start_ms;

    std::lock_guard<std::mutex> lk(mtx_);
    spans_.push_back(s);
    if (spans_.size() > 500)
        spans_.erase(spans_.begin(), spans_.begin() + 100);
    if (log_file_.is_open()) {
        log_file_ << s.to_json().dump() << "\n";
        log_file_.flush();
    }
}

void TelemetryEngine::set_llm_metrics(const LLMMetrics& m) { std::lock_guard lk(mtx_); llm_ = m; }
LLMMetrics TelemetryEngine::get_llm_metrics() const { std::lock_guard lk(mtx_); return llm_; }

void TelemetryEngine::set_emotion(const EmotionOutput& e) { std::lock_guard lk(mtx_); emotion_ = e; }
EmotionOutput TelemetryEngine::get_emotion() const { std::lock_guard lk(mtx_); return emotion_; }

void TelemetryEngine::set_router(const RouterOutput& r) { std::lock_guard lk(mtx_); router_ = r; }
RouterOutput TelemetryEngine::get_router() const { std::lock_guard lk(mtx_); return router_; }

void TelemetryEngine::set_system_metrics(const SystemMetrics& m) { std::lock_guard lk(mtx_); sys_ = m; }
SystemMetrics TelemetryEngine::get_system_metrics() const { std::lock_guard lk(mtx_); return sys_; }

void TelemetryEngine::set_state(const std::string& s) { std::lock_guard lk(mtx_); state_ = s; }
std::string TelemetryEngine::get_state() const { std::lock_guard lk(mtx_); return state_; }

std::vector<TelemetrySpan> TelemetryEngine::get_recent_spans(int n) const {
    std::lock_guard lk(mtx_);
    int start = std::max(0, static_cast<int>(spans_.size()) - n);
    return {spans_.begin() + start, spans_.end()};
}

json TelemetryEngine::get_snapshot() const {
    std::lock_guard lk(mtx_);
    json j;
    j["state"] = state_;
    j["llm"] = {
        {"tokens_generated", llm_.tokens_generated},
        {"tokens_per_second", llm_.tokens_per_second},
        {"context_length", llm_.context_length}
    };
    j["emotion"] = {
        {"valence", emotion_.valence}, {"arousal", emotion_.arousal},
        {"dominance", emotion_.dominance}, {"label", emotion_.label},
        {"confidence", emotion_.confidence}, {"inference_ms", emotion_.inference_ms}
    };
    j["router"] = {
        {"mode", router_.mode}, {"confidence", router_.confidence},
        {"suggested_tool", router_.suggested_tool},
        {"rag_enable", router_.rag_enable}, {"inference_ms", router_.inference_ms}
    };
    j["system"] = {
        {"cpu_percent", sys_.cpu_percent}, {"gpu_percent", sys_.gpu_percent},
        {"ram_mb", sys_.ram_mb}, {"vram_mb", sys_.vram_mb},
        {"health", sys_.health}, {"model", sys_.model_name},
        {"backend", sys_.backend}, {"device", sys_.device}
    };
    json spans_j = json::array();
    int start = std::max(0, static_cast<int>(spans_.size()) - 20);
    for (int i = start; i < static_cast<int>(spans_.size()); ++i)
        spans_j.push_back(spans_[i].to_json());
    j["recent_spans"] = spans_j;
    return j;
}

} // namespace revia
