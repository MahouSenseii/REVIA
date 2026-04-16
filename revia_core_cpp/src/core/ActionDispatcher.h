// Phase 3 - ActionDispatcher.
//
// Single ownership:
//   * ActionDispatcher owns action executor lookup and dispatch.
//   * Executors execute only; they do not decide behavior.
#pragma once

#include <cstddef>
#include <memory>
#include <unordered_map>

#include "interfaces/IActionExecutor.h"

namespace revia::core {

class ActionDispatcher {
public:
    void register_executor(std::unique_ptr<IActionExecutor> executor);

    [[nodiscard]] ExecutionResult dispatch(IAction& action);

    [[nodiscard]] std::size_t executor_count() const {
        return executors_.size();
    }

private:
    struct ActionTypeHash {
        std::size_t operator()(ActionType type) const {
            return static_cast<std::size_t>(type);
        }
    };

    std::unordered_map<ActionType, std::unique_ptr<IActionExecutor>, ActionTypeHash> executors_;
};

} // namespace revia::core
