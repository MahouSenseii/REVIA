// Phase 4 - default learning sinks implementation.
#include "feedback/DefaultLearningSinks.h"

#include <string>

#include "core/StructuredLogger.h"

namespace revia::core {

void StructuredLogLearningSink::consume(const ExecutionResult& result,
                                        const ContextPackage& context,
                                        const RuntimeState& state) {
    StructuredLogger::instance().info("learning.signal_recorded", {
        {"event_id", context.current_event ? context.current_event->id : ""},
        {"action_type", std::string(to_string(result.action_type))},
        {"was_successful", result.was_successful},
        {"feedback_tag", result.feedback.tag},
        {"feedback_score", result.feedback.score},
        {"state", std::string(to_string(state.current_state))},
        {"sink", name()}
    });
}

} // namespace revia::core
