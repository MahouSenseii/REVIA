// Phase 0 — IActionExecutor contract.
//
// Executors live under src/executors/ (Phase 3+). ActionDispatcher holds a
// registry Map<ActionType, IActionExecutor*> and routes actions by type.
//
// Executors MUST NOT decide. They only execute. If an executor finds itself
// branching on context to choose *what* to do, that logic belongs in a rule.
#pragma once

#include "interfaces/IAction.h"
#include "models/Enums.h"
#include "models/ExecutionResult.h"

namespace revia::core {

class IActionExecutor {
public:
    virtual ~IActionExecutor() = default;

    // Which ActionType this executor handles. The dispatcher uses this to
    // populate its registry.
    [[nodiscard]] virtual ActionType handles() const = 0;

    // Execute the action. MUST return within a bounded time (suggested: 5s
    // for synchronous work; longer async work should post follow-up events
    // onto the EventBus rather than blocking).
    [[nodiscard]] virtual ExecutionResult execute(const IAction& action) = 0;
};

} // namespace revia::core
