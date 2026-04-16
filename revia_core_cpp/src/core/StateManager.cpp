// Phase 1 - StateManager implementation.
#include "core/StateManager.h"

#include <chrono>
#include <stdexcept>
#include <utility>

#include "core/StructuredLogger.h"

namespace revia::core {

StateManager::StateManager() {
    apply_state_flags_locked(state_.current_state);
    StructuredLogger::instance().info("state_manager.initialized", {
        {"state", std::string(to_string(state_.current_state))},
        {"mode", std::string(to_string(state_.current_mode))},
        {"version", state_.version}
    });
}

RuntimeState StateManager::get_current_state() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return state_;
}

CoreState StateManager::current_state() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return state_.current_state;
}

RuntimeMode StateManager::current_mode() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return state_.current_mode;
}

bool StateManager::can_transition(CoreState from, CoreState to) const {
    if (from == to) {
        return true;
    }

    switch (from) {
        case CoreState::Booting:
            return to == CoreState::Idle ||
                   to == CoreState::Recovering ||
                   to == CoreState::Disabled;

        case CoreState::Idle:
            return to == CoreState::Listening ||
                   to == CoreState::Thinking ||
                   to == CoreState::Waiting ||
                   to == CoreState::Proactive ||
                   to == CoreState::Recovering ||
                   to == CoreState::Disabled;

        case CoreState::Listening:
            return to == CoreState::Thinking ||
                   to == CoreState::Idle ||
                   to == CoreState::Interrupted ||
                   to == CoreState::Recovering ||
                   to == CoreState::Disabled;

        case CoreState::Thinking:
            return to == CoreState::Responding ||
                   to == CoreState::Waiting ||
                   to == CoreState::Idle ||
                   to == CoreState::Interrupted ||
                   to == CoreState::Recovering ||
                   to == CoreState::Disabled;

        case CoreState::Responding:
            return to == CoreState::Idle ||
                   to == CoreState::Waiting ||
                   to == CoreState::Interrupted ||
                   to == CoreState::Recovering ||
                   to == CoreState::Disabled;

        case CoreState::Interrupted:
            return to == CoreState::Listening ||
                   to == CoreState::Thinking ||
                   to == CoreState::Idle ||
                   to == CoreState::Recovering ||
                   to == CoreState::Disabled;

        case CoreState::Waiting:
            return to == CoreState::Idle ||
                   to == CoreState::Listening ||
                   to == CoreState::Proactive ||
                   to == CoreState::Recovering ||
                   to == CoreState::Disabled;

        case CoreState::Proactive:
            return to == CoreState::Thinking ||
                   to == CoreState::Responding ||
                   to == CoreState::Idle ||
                   to == CoreState::Recovering ||
                   to == CoreState::Disabled;

        case CoreState::Recovering:
            return to == CoreState::Idle ||
                   to == CoreState::Disabled;

        case CoreState::Disabled:
            return to == CoreState::Booting ||
                   to == CoreState::Idle;
    }

    return false;
}

bool StateManager::can_transition_to(CoreState to) const {
    std::lock_guard<std::mutex> lock(mtx_);
    return can_transition(state_.current_state, to);
}

RuntimeState StateManager::transition_to(CoreState next,
                                         std::string reason,
                                         std::string correlation_id) {
    std::lock_guard<std::mutex> lock(mtx_);
    const CoreState previous = state_.current_state;

    if (!can_transition(previous, next)) {
        StructuredLogger::instance().warn("state.transition.rejected", {
            {"from", std::string(to_string(previous))},
            {"to", std::string(to_string(next))},
            {"reason", reason},
            {"correlation_id", correlation_id}
        });
        throw std::invalid_argument(
            "invalid CoreState transition from " + std::string(to_string(previous)) +
            " to " + std::string(to_string(next)));
    }

    if (previous == next) {
        StructuredLogger::instance().debug("state.transition.noop", {
            {"state", std::string(to_string(next))},
            {"reason", reason},
            {"correlation_id", correlation_id},
            {"version", state_.version}
        });
        return state_;
    }

    state_.current_state = next;
    apply_state_flags_locked(next);
    ++state_.version;

    log_transition_locked(previous, next, reason, correlation_id);
    return state_;
}

bool StateManager::try_transition_to(CoreState next,
                                     std::string reason,
                                     std::string correlation_id) {
    try {
        transition_to(next, std::move(reason), std::move(correlation_id));
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

RuntimeState StateManager::set_mode(RuntimeMode mode,
                                    std::string reason,
                                    std::string correlation_id) {
    std::lock_guard<std::mutex> lock(mtx_);
    const RuntimeMode previous = state_.current_mode;
    if (previous == mode) {
        return state_;
    }

    state_.current_mode = mode;
    state_.is_recovery_mode = (mode == RuntimeMode::Recovery) ||
                              (state_.current_state == CoreState::Recovering);
    ++state_.version;

    StructuredLogger::instance().info("state.mode_changed", {
        {"from", std::string(to_string(previous))},
        {"to", std::string(to_string(mode))},
        {"reason", reason},
        {"correlation_id", correlation_id},
        {"version", state_.version}
    });
    return state_;
}

void StateManager::mark_interaction(std::string reason, std::string correlation_id) {
    std::lock_guard<std::mutex> lock(mtx_);
    state_.last_interaction_time = now();
    ++state_.version;
    StructuredLogger::instance().debug("state.interaction_marked", {
        {"reason", reason},
        {"correlation_id", correlation_id},
        {"version", state_.version}
    });
}

void StateManager::mark_output(std::string reason, std::string correlation_id) {
    std::lock_guard<std::mutex> lock(mtx_);
    state_.last_output_time = now();
    ++state_.version;
    StructuredLogger::instance().debug("state.output_marked", {
        {"reason", reason},
        {"correlation_id", correlation_id},
        {"version", state_.version}
    });
}

nlohmann::json StateManager::snapshot_json() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return {
        {"current_mode", std::string(to_string(state_.current_mode))},
        {"current_state", std::string(to_string(state_.current_state))},
        {"is_speaking", state_.is_speaking},
        {"is_listening", state_.is_listening},
        {"is_interrupted", state_.is_interrupted},
        {"is_recovery_mode", state_.is_recovery_mode},
        {"last_interaction_time_ms", timestamp_ms(state_.last_interaction_time)},
        {"last_output_time_ms", timestamp_ms(state_.last_output_time)},
        {"version", state_.version}
    };
}

void StateManager::apply_state_flags_locked(CoreState state) {
    state_.is_speaking = state == CoreState::Responding;
    state_.is_listening = state == CoreState::Listening;
    state_.is_interrupted = state == CoreState::Interrupted;
    state_.is_recovery_mode = state == CoreState::Recovering ||
                              state_.current_mode == RuntimeMode::Recovery;

    if (state == CoreState::Thinking || state == CoreState::Listening) {
        state_.last_interaction_time = now();
    }
    if (state == CoreState::Responding) {
        state_.last_output_time = now();
    }
}

void StateManager::log_transition_locked(CoreState previous,
                                         CoreState next,
                                         const std::string& reason,
                                         const std::string& correlation_id) const {
    StructuredLogger::instance().info("state.transition.accepted", {
        {"from", std::string(to_string(previous))},
        {"to", std::string(to_string(next))},
        {"reason", reason},
        {"correlation_id", correlation_id},
        {"version", state_.version}
    });
}

std::int64_t StateManager::timestamp_ms(Timestamp ts) {
    if (ts == Timestamp{}) {
        return 0;
    }
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        ts.time_since_epoch()).count();
}

} // namespace revia::core
