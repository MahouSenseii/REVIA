// Phase 4 - FeedbackManager implementation.
#include "core/FeedbackManager.h"

#include <exception>
#include <stdexcept>
#include <string>
#include <utility>

#include "core/StructuredLogger.h"

namespace revia::core {

void FeedbackManager::register_learning_sink(std::unique_ptr<ILearningSink> sink) {
    if (!sink) {
        throw std::invalid_argument("FeedbackManager::register_learning_sink received null sink");
    }

    StructuredLogger::instance().info("feedback.sink_registered", {
        {"sink", sink->name()}
    });
    sinks_.push_back(std::move(sink));
}

void FeedbackManager::process(const ExecutionResult& result,
                              const ContextPackage& context,
                              const RuntimeState& state) {
    const std::string event_id = context.current_event ? context.current_event->id : "";
    StructuredLogger::instance().info("feedback.process_started", {
        {"event_id", event_id},
        {"was_successful", result.was_successful},
        {"action_type", std::string(to_string(result.action_type))},
        {"feedback_tag", result.feedback.tag},
        {"feedback_score", result.feedback.score},
        {"sink_count", sinks_.size()}
    });

    for (const auto& sink : sinks_) {
        try {
            sink->consume(result, context, state);
            StructuredLogger::instance().debug("feedback.sink_consumed", {
                {"event_id", event_id},
                {"sink", sink->name()}
            });
        } catch (const std::exception& exc) {
            StructuredLogger::instance().warn("feedback.sink_failed", {
                {"event_id", event_id},
                {"sink", sink->name()},
                {"error", exc.what()}
            });
        } catch (...) {
            StructuredLogger::instance().warn("feedback.sink_failed", {
                {"event_id", event_id},
                {"sink", sink->name()},
                {"error", "unknown exception"}
            });
        }
    }

    StructuredLogger::instance().info("feedback.process_completed", {
        {"event_id", event_id},
        {"sink_count", sinks_.size()}
    });
}

} // namespace revia::core
