// Phase 0 — FeatureFlags implementation.
#include "core/FeatureFlags.h"
#include "core/StructuredLogger.h"

#include <algorithm>
#include <cstdlib>
#include <cctype>
#include <string>

namespace revia::core {

namespace {

std::string to_lower(std::string_view s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    return out;
}

const char* getenv_safe(const char* name) {
#if defined(_WIN32)
    // _dupenv_s would be the MSVC-safe path, but we accept the simple form
    // and let the CRT warn; this matches the rest of the codebase style.
    #pragma warning(push)
    #pragma warning(disable : 4996)
    return std::getenv(name);
    #pragma warning(pop)
#else
    return std::getenv(name);
#endif
}

} // namespace

bool FeatureFlags::parse_bool(std::string_view s, bool default_value) {
    if (s.empty()) return default_value;

    const auto lower = to_lower(s);
    if (lower == "1" || lower == "true" || lower == "on" ||
        lower == "yes" || lower == "y" || lower == "t") {
        return true;
    }
    if (lower == "0" || lower == "false" || lower == "off" ||
        lower == "no" || lower == "n" || lower == "f") {
        return false;
    }
    return default_value;
}

FeatureFlags& FeatureFlags::instance() {
    static FeatureFlags inst;
    return inst;
}

FeatureFlags::FeatureFlags() {
    reload_from_env();
}

void FeatureFlags::reload_from_env() {
    const char* v2 = getenv_safe("REVIA_CORE_V2_ENABLED");
    const bool v2_on = parse_bool(v2 ? v2 : "", /*default=*/false);
    core_v2_enabled_.store(v2_on, std::memory_order_release);

    const char* stderr_flag = getenv_safe("REVIA_CORE_V2_LOG_STDERR");
    const bool stderr_on = parse_bool(stderr_flag ? stderr_flag : "", /*default=*/true);
    log_stderr_enabled_.store(stderr_on, std::memory_order_release);

    // Propagate stderr preference to the logger immediately.
    StructuredLogger::instance().set_stderr_sink_enabled(stderr_on);

    StructuredLogger::instance().event("feature_flags.loaded", {
        {"core_v2_enabled",     v2_on},
        {"log_stderr_enabled",  stderr_on},
        {"source",              "env"}
    });
}

} // namespace revia::core
