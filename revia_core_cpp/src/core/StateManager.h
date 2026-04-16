// Phase 1 - StateManager.
//
// Single ownership:
//   * StateManager owns RuntimeState and all CoreState transitions.
//   * Other modules receive snapshots and request transitions; they do not
//     mutate RuntimeState directly.
//
// Invalid transitions throw from transition_to(...). Call try_transition_to(...)
// only at external boundaries where a failed transition should be logged and
// degraded instead of crashing the caller.
#pragma once

#include <cstdint>
#include <mutex>
#include <string>

#include <nlohmann/json.hpp>

#include "models/RuntimeState.h"

namespace revia::core {

class StateManager {
public:
    StateManager();

    StateManager(const StateManager&) = delete;
    StateManager& operator=(const StateManager&) = delete;

    [[nodiscard]] RuntimeState get_current_state() const;
    [[nodiscard]] CoreState current_state() const;
    [[nodiscard]] RuntimeMode current_mode() const;

    [[nodiscard]] bool can_transition(CoreState from, CoreState to) const;
    [[nodiscard]] bool can_transition_to(CoreState to) const;

    // Throws std::invalid_argument when the transition is not in the explicit
    // Phase 1 transition table.
    RuntimeState transition_to(CoreState next,
                               std::string reason = {},
                               std::string correlation_id = {});

    // Safe boundary helper. Returns false on invalid transition and logs it.
    [[nodiscard]] bool try_transition_to(CoreState next,
                                         std::string reason = {},
                                         std::string correlation_id = {});

    RuntimeState set_mode(RuntimeMode mode,
                          std::string reason = {},
                          std::string correlation_id = {});

    void mark_interaction(std::string reason = {}, std::string correlation_id = {});
    void mark_output(std::string reason = {}, std::string correlation_id = {});

    [[nodiscard]] nlohmann::json snapshot_json() const;

private:
    void apply_state_flags_locked(CoreState state);
    void log_transition_locked(CoreState previous,
                               CoreState next,
                               const std::string& reason,
                               const std::string& correlation_id) const;
    [[nodiscard]] static std::int64_t timestamp_ms(Timestamp ts);

    mutable std::mutex mtx_;
    RuntimeState state_;
};

} // namespace revia::core
