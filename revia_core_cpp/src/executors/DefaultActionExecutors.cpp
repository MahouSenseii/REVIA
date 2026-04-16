// Phase 3 - default action executors implementation.
#include "executors/DefaultActionExecutors.h"

#include <string>
#include <utility>

#include "core/StructuredLogger.h"

namespace revia::core {

NoopActionExecutor::NoopActionExecutor(ActionType action_type,
                                       std::string feedback_tag)
    : action_type_(action_type),
      feedback_tag_(std::move(feedback_tag)) {}

ExecutionResult NoopActionExecutor::execute(const IAction& action) {
    StructuredLogger::instance().info("executor.noop_executed", {
        {"action_type", std::string(to_string(action.type))},
        {"correlation_id", action.correlation_id},
        {"feedback_tag", feedback_tag_}
    });

    ExecutionResult result = ExecutionResult::Ok(action.type);
    result.feedback.tag = feedback_tag_;
    result.feedback.score = 0.05;
    return result;
}

} // namespace revia::core
