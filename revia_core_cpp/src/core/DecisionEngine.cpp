// Phase 2 - DecisionEngine implementation.
#include "core/DecisionEngine.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "core/StructuredLogger.h"
#include "models/CoreAction.h"

namespace revia::core {

void DecisionEngine::register_rule(std::unique_ptr<IDecisionRule> rule) {
    if (!rule) {
        throw std::invalid_argument("DecisionEngine::register_rule received null rule");
    }

    StructuredLogger::instance().info("decision.rule_registered", {
        {"rule", rule->name()},
        {"category", rule->category()},
        {"parallel_safe", rule->is_parallel_safe()}
    });
    rules_.push_back(std::move(rule));
}

DecisionResult DecisionEngine::evaluate(const ContextPackage& context,
                                        const RuntimeState& state) {
    std::vector<DecisionInfluence> influences;
    influences.reserve(rules_.size());

    const std::string event_id = context.current_event ? context.current_event->id : "";
    StructuredLogger::instance().debug("decision.evaluate_started", {
        {"event_id", event_id},
        {"rule_count", rules_.size()}
    });

    for (const auto& rule : rules_) {
        try {
            DecisionInfluence influence = rule->evaluate(context, state);
            if (influence.rule_name.empty()) {
                influence.rule_name = rule->name();
            }
            StructuredLogger::instance().debug("decision.rule_evaluated", {
                {"event_id", event_id},
                {"rule", rule->name()},
                {"category", rule->category()},
                {"priority_bias", influence.priority_bias},
                {"has_should_act", influence.should_act.has_value()},
                {"has_action", influence.preferred_action.has_value()}
            });
            influences.push_back(std::move(influence));
        } catch (const std::exception& exc) {
            StructuredLogger::instance().warn("decision.rule_failed", {
                {"event_id", event_id},
                {"rule", rule->name()},
                {"error", exc.what()}
            });
        }
    }

    DecisionInfluence merged = merge_influences(influences);

    DecisionResult result;
    result.should_act = merged.should_act.value_or(false);
    result.tone = merged.tone.value_or(ToneProfile{});
    result.delay = merged.delay.value_or(DelayProfile{});
    result.safety = merged.safety.value_or(SafetyProfile{});
    result.reason_summary = summarize(influences, merged);

    if (result.should_act) {
        const ActionType action_type =
            merged.preferred_action.value_or(ActionType::UpdateInternalStateOnly);
        result.selected_action = std::make_unique<CoreAction>(
            action_type,
            merged.priority_bias,
            result.reason_summary);
        if (context.current_event) {
            result.selected_action->correlation_id = context.current_event->correlation_id;
        }
    }

    StructuredLogger::instance().info("decision.evaluate_completed", {
        {"event_id", event_id},
        {"should_act", result.should_act},
        {"action", result.selected_action ? result.selected_action->describe() : "none"},
        {"reason", result.reason_summary}
    });
    return result;
}

DecisionResult DecisionEngine::build_fallback_action(const ContextPackage& context,
                                                     std::string reason) {
    DecisionResult result;
    result.should_act = true;
    result.tone = ToneProfile::SafeNeutral();
    result.safety.strictness = 1.0;
    result.reason_summary = std::move(reason);
    result.selected_action = std::make_unique<CoreAction>(
        ActionType::TriggerFallback,
        100,
        result.reason_summary);
    if (context.current_event) {
        result.selected_action->correlation_id = context.current_event->correlation_id;
    }
    return result;
}

DecisionInfluence DecisionEngine::merge_influences(
    const std::vector<DecisionInfluence>& influences) const {
    DecisionInfluence merged;
    merged.rule_name = "DecisionEngine.Merge";
    merged.should_act = false;
    merged.priority_bias = 0;

    for (const auto& influence : influences) {
        if (influence.should_act.has_value()) {
            const bool should_replace =
                !merged.should_act.has_value() ||
                influence.priority_bias >= merged.priority_bias;
            if (should_replace) {
                merged.should_act = influence.should_act;
            }
        }

        if (influence.preferred_action.has_value() &&
            influence.priority_bias >= merged.priority_bias) {
            merged.preferred_action = influence.preferred_action;
        }

        if (influence.tone.has_value()) {
            merged.tone = influence.tone;
        }
        if (influence.delay.has_value()) {
            merged.delay = influence.delay;
        }
        if (influence.safety.has_value()) {
            merged.safety = influence.safety;
        }

        merged.priority_bias = std::max(merged.priority_bias, influence.priority_bias);
    }

    return merged;
}

std::string DecisionEngine::summarize(
    const std::vector<DecisionInfluence>& influences,
    const DecisionInfluence& merged) {
    std::ostringstream oss;
    oss << "merged " << influences.size() << " influence(s)";
    if (merged.preferred_action.has_value()) {
        oss << "; action=" << std::string(to_string(*merged.preferred_action));
    }

    bool first_explanation = true;
    for (const auto& influence : influences) {
        if (influence.explanation.empty()) {
            continue;
        }
        oss << (first_explanation ? "; reasons=" : " | ");
        first_explanation = false;
        oss << influence.rule_name << ": " << influence.explanation;
    }
    return oss.str();
}

} // namespace revia::core
