// Phase 0 — FeatureFlags.
//
// Single source of truth for runtime toggles that gate the strangler-fig
// migration. Read once at boot, cached. Writable via set_* for tests.
//
// Environment variables
// ---------------------
//   REVIA_CORE_V2_ENABLED      "1"/"true"/"on"/"yes"  -> on   (default off)
//   REVIA_CORE_V2_LOG_STDERR   "0"/"false"/"off"/"no" -> off  (default on)
//
// All parsing is case-insensitive for on-values: 1, true, on, yes, y, t.
#pragma once

#include <atomic>
#include <string>
#include <string_view>

namespace revia::core {

class FeatureFlags {
public:
    static FeatureFlags& instance();

    // Master switch for the Phase 1+ orchestrator path. When false, legacy
    // code paths run unchanged. When true, events route through the new
    // CoreOrchestrator.
    [[nodiscard]] bool core_v2_enabled() const {
        return core_v2_enabled_.load(std::memory_order_acquire);
    }

    // Whether StructuredLogger's stderr sink is enabled. Separate from the
    // master switch so ops can silence console noise in prod.
    [[nodiscard]] bool log_stderr_enabled() const {
        return log_stderr_enabled_.load(std::memory_order_acquire);
    }

    // Test / runtime overrides. Prefer environment variables in production.
    void set_core_v2_enabled(bool v) {
        core_v2_enabled_.store(v, std::memory_order_release);
    }
    void set_log_stderr_enabled(bool v) {
        log_stderr_enabled_.store(v, std::memory_order_release);
    }

    // Force a re-read from the environment. Useful in tests.
    void reload_from_env();

    FeatureFlags(const FeatureFlags&) = delete;
    FeatureFlags& operator=(const FeatureFlags&) = delete;

    // Exposed for the Python bridge / admin tooling to reuse the parser.
    [[nodiscard]] static bool parse_bool(std::string_view s, bool default_value);

private:
    FeatureFlags();

    std::atomic<bool> core_v2_enabled_{false};
    std::atomic<bool> log_stderr_enabled_{true};
};

} // namespace revia::core
