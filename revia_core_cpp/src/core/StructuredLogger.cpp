// Phase 0 — StructuredLogger implementation.
#include "core/StructuredLogger.h"

#include <filesystem>
#include <iostream>
#include <thread>
#include <sstream>

namespace revia::core {

namespace {

// Format a timestamp as ISO-8601 with millisecond precision.
std::string iso8601_now() {
    using namespace std::chrono;
    const auto tp = system_clock::now();
    const auto t  = system_clock::to_time_t(tp);
    const auto ms = duration_cast<milliseconds>(tp.time_since_epoch()).count() % 1000;

    std::tm tm_utc{};
#if defined(_WIN32)
    gmtime_s(&tm_utc, &t);
#else
    gmtime_r(&t, &tm_utc);
#endif

    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &tm_utc);

    std::ostringstream oss;
    oss << buf << '.';
    oss.width(3); oss.fill('0'); oss << ms << 'Z';
    return oss.str();
}

std::string this_thread_id() {
    std::ostringstream oss;
    oss << std::this_thread::get_id();
    return oss.str();
}

bool level_enabled(LogLevel actual, LogLevel threshold) {
    return static_cast<int>(actual) >= static_cast<int>(threshold);
}

// Human-readable stderr line:
//   2026-04-16T10:20:31.042Z  info  orchestrator.event_received  {"event_id":"…"}
std::string format_stderr_line(const json& entry) {
    std::ostringstream oss;
    oss << entry.value("ts", "")      << "  "
        << entry.value("level", "info") << "  "
        << entry.value("stage", "?");

    if (entry.contains("fields") && !entry["fields"].empty()) {
        oss << "  " << entry["fields"].dump();
    }
    return oss.str();
}

} // namespace

StructuredLogger& StructuredLogger::instance() {
    static StructuredLogger inst;
    return inst;
}

StructuredLogger::StructuredLogger() = default;

StructuredLogger::~StructuredLogger() {
    std::lock_guard<std::mutex> lock(mtx_);
    if (file_open_ && file_.is_open()) {
        file_.flush();
        file_.close();
    }
}

void StructuredLogger::open_file_locked() {
    if (file_open_) return;

    try {
        const auto path = std::filesystem::path(file_path_);
        if (path.has_parent_path()) {
            std::filesystem::create_directories(path.parent_path());
        }
        file_.open(file_path_, std::ios::app);
        file_open_ = file_.is_open();
        if (!file_open_) {
            // Fall back to stderr-only; don't throw from a logger.
            std::cerr << "[StructuredLogger] could not open " << file_path_
                      << " — file sink disabled.\n";
            file_sink_enabled_ = false;
        }
    } catch (const std::exception& e) {
        std::cerr << "[StructuredLogger] filesystem error: " << e.what()
                  << " — file sink disabled.\n";
        file_sink_enabled_ = false;
        file_open_ = false;
    }
}

void StructuredLogger::set_file_sink_enabled(bool enabled) {
    std::lock_guard<std::mutex> lock(mtx_);
    file_sink_enabled_ = enabled;
}

void StructuredLogger::set_stderr_sink_enabled(bool enabled) {
    std::lock_guard<std::mutex> lock(mtx_);
    stderr_sink_enabled_ = enabled;
}

void StructuredLogger::set_file_path(std::string path) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (file_open_ && file_.is_open()) {
        file_.flush();
        file_.close();
    }
    file_open_ = false;
    file_path_ = std::move(path);
}

void StructuredLogger::set_stderr_min_level(LogLevel level) {
    std::lock_guard<std::mutex> lock(mtx_);
    stderr_min_level_ = level;
}

void StructuredLogger::set_file_min_level(LogLevel level) {
    std::lock_guard<std::mutex> lock(mtx_);
    file_min_level_ = level;
}

void StructuredLogger::flush() {
    std::lock_guard<std::mutex> lock(mtx_);
    if (file_open_ && file_.is_open()) {
        file_.flush();
    }
    unflushed_writes_ = 0;
}

void StructuredLogger::event(std::string_view stage,
                             json fields,
                             LogLevel level) {
    // Build the entry outside the lock to keep the critical section small.
    json entry = {
        {"ts",     iso8601_now()},
        {"level",  to_string(level)},
        {"stage",  std::string(stage)},
        {"thread", this_thread_id()},
        {"fields", std::move(fields)}
    };

    std::lock_guard<std::mutex> lock(mtx_);
    write_locked(entry, level);
}

void StructuredLogger::write_locked(const json& entry, LogLevel level) {
    total_events_.fetch_add(1, std::memory_order_relaxed);

    if (file_sink_enabled_ && level_enabled(level, file_min_level_)) {
        if (!file_open_) open_file_locked();
        if (file_open_ && file_.is_open()) {
            file_ << entry.dump() << '\n';
            ++unflushed_writes_;
            if (unflushed_writes_ >= kFlushEvery) {
                file_.flush();
                unflushed_writes_ = 0;
            }
        }
    }

    if (stderr_sink_enabled_ && level_enabled(level, stderr_min_level_)) {
        std::cerr << format_stderr_line(entry) << '\n';
    }
}

} // namespace revia::core
