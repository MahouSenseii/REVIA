// Phase 3 - ActionDispatcher implementation.
#include "core/ActionDispatcher.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "core/StructuredLogger.h"

namespace revia::core {

namespace {

std::string make_action_id() {
    static std::atomic<std::uint64_t> counter{1};
    const auto value = counter.fetch_add(1, std::memory_order_relaxed);
    const auto ticks = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();

    std::ostringstream oss;
    oss << "act-" << std::hex << ticks << '-' << value;
    return oss.str();
}

} // namespace

void ActionDispatcher::register_executor(std::unique_ptr<IActionExecutor> executor) {
    if (!executor) {
        throw std::invalid_argument("ActionDispatcher::register_executor received null executor");
    }

    const ActionType action_type = executor->handles();
    StructuredLogger::instance().info("action.executor_registered", {
        {"action_type", std::string(to_string(action_type))}
    });
    executors_[action_type] = std::move(executor);
}

ExecutionResult ActionDispatcher::dispatch(IAction& action) {
    if (action.id.empty()) {
        action.id = make_action_id();
        StructuredLogger::instance().debug("action.id_filled", {
            {"action_type", std::string(to_string(action.type))},
            {"action_id", action.id}
        });
    }

    StructuredLogger::instance().info("action.dispatch_started", {
        {"action_id", action.id},
        {"action_type", std::string(to_string(action.type))},
        {"priority", action.priority},
        {"correlation_id", action.correlation_id},
        {"description", action.describe()}
    });

    auto it = executors_.find(action.type);
    if (it == executors_.end()) {
        auto result = ExecutionResult::Fail(action.type, "No executor registered");
        StructuredLogger::instance().error("action.dispatch_failed", {
            {"action_type", std::string(to_string(action.type))},
            {"reason", result.failure_reason}
        });
        return result;
    }

    try {
        ExecutionResult result = it->second->execute(action);
        StructuredLogger::instance().info("action.dispatch_completed", {
            {"action_id", action.id},
            {"action_type", std::string(to_string(action.type))},
            {"was_successful", result.was_successful},
            {"failure_reason", result.failure_reason},
            {"feedback_tag", result.feedback.tag},
            {"feedback_score", result.feedback.score}
        });
        return result;
    } catch (const std::exception& exc) {
        auto result = ExecutionResult::Fail(action.type, exc.what());
        StructuredLogger::instance().error("action.executor_exception", {
            {"action_type", std::string(to_string(action.type))},
            {"error", exc.what()}
        });
        return result;
    } catch (...) {
        auto result = ExecutionResult::Fail(action.type, "unknown executor exception");
        StructuredLogger::instance().error("action.executor_exception", {
            {"action_type", std::string(to_string(action.type))},
            {"error", "unknown executor exception"}
        });
        return result;
    }
}

} // namespace revia::core
