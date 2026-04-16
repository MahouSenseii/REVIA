// Phase 0 — IAction contract. Matches REVIA_CORE.md §Example Interfaces.
//
// IAction is a polymorphic base (virtual destructor) because different action
// types carry different payloads, and ActionDispatcher dispatches on the
// concrete subtype to the correct IActionExecutor.
#pragma once

#include <memory>
#include <string>
#include "models/Enums.h"

namespace revia::core {

class IAction {
public:
    std::string id;               // UUID-v4; ActionDispatcher fills if empty
    ActionType  type = ActionType::IgnoreEvent;
    int         priority = 0;

    // Correlation id of the originating event, so logs can be joined.
    std::string correlation_id;

    virtual ~IAction() = default;

    // Short human-readable description for logs. Default = enum name.
    // Subtypes override when they want to add detail (e.g. "SpeakResponse(text_len=142)").
    [[nodiscard]] virtual std::string describe() const {
        return std::string(to_string(type));
    }
};

// Convenience alias for executors/dispatchers.
using ActionPtr = std::unique_ptr<IAction>;

} // namespace revia::core
