// Phase 0 — StructuredLogger.
//
// Goals:
//   * Every Core module emits structured events through this one sink.
//   * One JSON object per line in logs/revia_core.jsonl.
//   * Human-readable line on stderr for local dev.
//   * Thread-safe. Batched flush (every 50 entries) matching TelemetryEngine.
//   * Zero coupling to any other Core module — StructuredLogger is a leaf.
//
// Usage:
//   auto& log = revia::core::StructuredLogger::instance();
//   log.event("orchestrator.event_received", {
//       {"event_id", e.id},
//       {"event_type", to_string(e.type)}
//   });
//
// The free helper REVIA_CORE_LOG(stage, fields) is provided for brevity.
#pragma once

#include <atomic>
#include <chrono>
#include <fstream>
#include <mutex>
#include <string>
#include <string_view>
#include <nlohmann/json.hpp>

namespace revia::core {

using json = nlohmann::json;

enum class LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error
};

constexpr std::string_view to_string(LogLevel l) {
    switch (l) {
        case LogLevel::Trace: return "trace";
        case LogLevel::Debug: return "debug";
        case LogLevel::Info:  return "info";
        case LogLevel::Warn:  return "warn";
        case LogLevel::Error: return "error";
    }
    return "info";
}

class StructuredLogger {
public:
    // Singleton. Note: Phase 6 cleanup will DI-inject this everywhere.
    // For Phase 0 a singleton matches the existing TelemetryEngine pattern
    // and keeps the scaffolding simple.
    static StructuredLogger& instance();

    // Log a single structured event.
    //   stage  — dotted identifier like "orchestrator.decision_merged"
    //   fields — JSON object of structured fields; merged into the output line
    //   level  — default Info
    void event(std::string_view stage,
               json fields = json::object(),
               LogLevel level = LogLevel::Info);

    // Convenience wrappers.
    void trace(std::string_view stage, json fields = json::object()) {
        event(stage, std::move(fields), LogLevel::Trace);
    }
    void debug(std::string_view stage, json fields = json::object()) {
        event(stage, std::move(fields), LogLevel::Debug);
    }
    void info(std::string_view stage, json fields = json::object()) {
        event(stage, std::move(fields), LogLevel::Info);
    }
    void warn(std::string_view stage, json fields = json::object()) {
        event(stage, std::move(fields), LogLevel::Warn);
    }
    void error(std::string_view stage, json fields = json::object()) {
        event(stage, std::move(fields), LogLevel::Error);
    }

    // Configure sinks. Default: JSONL file at logs/revia_core.jsonl AND stderr.
    void set_file_sink_enabled(bool enabled);
    void set_stderr_sink_enabled(bool enabled);

    // Override the JSONL file path (call before first event to avoid race).
    void set_file_path(std::string path);

    // Control the stderr verbosity independently of file verbosity.
    void set_stderr_min_level(LogLevel level);
    void set_file_min_level(LogLevel level);

    // Force-flush the JSONL file buffer.
    void flush();

    // Entries written since boot. Useful for tests.
    [[nodiscard]] std::uint64_t total_events() const { return total_events_.load(); }

    StructuredLogger(const StructuredLogger&) = delete;
    StructuredLogger& operator=(const StructuredLogger&) = delete;

private:
    StructuredLogger();
    ~StructuredLogger();

    void open_file_locked();
    void write_locked(const json& line, LogLevel level);

    mutable std::mutex mtx_;
    std::ofstream file_;
    std::string   file_path_ = "logs/revia_core.jsonl";
    bool          file_open_ = false;

    bool          file_sink_enabled_   = true;
    bool          stderr_sink_enabled_ = true;

    LogLevel      file_min_level_   = LogLevel::Trace;
    LogLevel      stderr_min_level_ = LogLevel::Info;

    int           unflushed_writes_ = 0;
    static constexpr int kFlushEvery = 50;

    std::atomic<std::uint64_t> total_events_{0};
};

// -------- Free helper: REVIA_CORE_LOG("stage", { {"k","v"} }) --------

inline void core_log(std::string_view stage,
                     json fields = json::object(),
                     LogLevel level = LogLevel::Info) {
    StructuredLogger::instance().event(stage, std::move(fields), level);
}

#define REVIA_CORE_LOG(stage, fields) \
    ::revia::core::StructuredLogger::instance().event((stage), (fields))

} // namespace revia::core
