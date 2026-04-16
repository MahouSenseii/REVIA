// Phase 0 — RuntimeState POD. Authoritative runtime state owned by StateManager (Phase 1).
#pragma once

#include <chrono>
#include <string>
#include "models/Enums.h"

namespace revia::core {

using Timestamp = std::chrono::system_clock::time_point;

inline Timestamp now() { return std::chrono::system_clock::now(); }

struct RuntimeState {
    RuntimeMode current_mode   = RuntimeMode::Normal;
    CoreState   current_state  = CoreState::Booting;

    bool is_speaking      = false;
    bool is_listening     = false;
    bool is_interrupted   = false;
    bool is_recovery_mode = false;

    Timestamp last_interaction_time = Timestamp{};
    Timestamp last_output_time      = Timestamp{};

    // A monotonically-increasing version for optimistic concurrency / sync.
    // Incremented by StateManager on any mutation. UI/BackendSync reads it.
    std::uint64_t version = 0;
};

} // namespace revia::core
