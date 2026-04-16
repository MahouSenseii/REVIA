// Phase 3 - SafetyGateway implementation.
#include "core/SafetyGateway.h"

#include <stdexcept>
#include <utility>

#include "core/StructuredLogger.h"

namespace revia::core {

void SafetyGateway::register_layer(std::unique_ptr<ISafetyLayer> layer) {
    if (!layer) {
        throw std::invalid_argument("SafetyGateway::register_layer received null layer");
    }

    StructuredLogger::instance().info("safety.layer_registered", {
        {"layer", layer->name()}
    });
    layers_.push_back(std::move(layer));
}

SafetyResult SafetyGateway::validate(const DecisionResult& decision,
                                     const ContextPackage& context) {
    SafetyResult result;
    const std::string event_id = context.current_event ? context.current_event->id : "";

    StructuredLogger::instance().debug("safety.validate_started", {
        {"event_id", event_id},
        {"layer_count", layers_.size()},
        {"action", decision.selected_action ? decision.selected_action->describe() : "none"}
    });

    for (const auto& layer : layers_) {
        SafetyLayerOutcome outcome;
        try {
            outcome = layer->validate(decision, context);
            if (outcome.layer_name.empty()) {
                outcome.layer_name = layer->name();
            }
        } catch (const std::exception& exc) {
            outcome.layer_name = layer->name();
            outcome.passed = false;
            outcome.reason = std::string("layer exception: ") + exc.what();
        } catch (...) {
            outcome.layer_name = layer->name();
            outcome.passed = false;
            outcome.reason = "layer exception: unknown";
        }

        result.layers.push_back(outcome);
        StructuredLogger::instance().debug("safety.layer_evaluated", {
            {"event_id", event_id},
            {"layer", outcome.layer_name},
            {"passed", outcome.passed},
            {"reason", outcome.reason}
        });

        if (!outcome.passed) {
            result.verdict = decision.safety.allow_fallback_on_block
                ? SafetyVerdict::RequiresFallback
                : SafetyVerdict::Blocked;
            result.summary = outcome.layer_name + ": " + outcome.reason;
            StructuredLogger::instance().warn("safety.validate_blocked", {
                {"event_id", event_id},
                {"verdict", std::string(to_string(result.verdict))},
                {"summary", result.summary}
            });
            return result;
        }
    }

    result.verdict = SafetyVerdict::Allowed;
    result.summary = "all safety layers passed";
    StructuredLogger::instance().info("safety.validate_completed", {
        {"event_id", event_id},
        {"verdict", std::string(to_string(result.verdict))},
        {"summary", result.summary}
    });
    return result;
}

} // namespace revia::core
