// Phase 2 - concrete action plan.
//
// This is not an executor. It is the smallest concrete IAction needed for the
// DecisionEngine to describe what should happen next. Phase 3's ActionDispatcher
// will consume IAction values and route them to executors.
#pragma once

#include <string>
#include <utility>
#include <nlohmann/json.hpp>

#include "interfaces/IAction.h"

namespace revia::core {

class CoreAction final : public IAction {
public:
    nlohmann::json payload = nlohmann::json::object();

    CoreAction(ActionType action_type = ActionType::IgnoreEvent,
               int action_priority = 0,
               std::string reason = {}) {
        type = action_type;
        priority = action_priority;
        if (!reason.empty()) {
            payload["reason"] = std::move(reason);
        }
    }

    [[nodiscard]] std::string describe() const override {
        std::string out = std::string(to_string(type));
        if (payload.contains("reason") && payload["reason"].is_string()) {
            out += "(" + payload["reason"].get<std::string>() + ")";
        }
        return out;
    }
};

} // namespace revia::core
