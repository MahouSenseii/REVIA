// Phase 3 - default action executors.
//
// These executors intentionally do not perform visible output yet. They make
// dispatch real while preserving legacy response behavior until adapters are
// introduced.
#pragma once

#include <string>

#include "interfaces/IActionExecutor.h"

namespace revia::core {

class NoopActionExecutor final : public IActionExecutor {
public:
    NoopActionExecutor(ActionType action_type, std::string feedback_tag);

    [[nodiscard]] ActionType handles() const override { return action_type_; }
    [[nodiscard]] ExecutionResult execute(const IAction& action) override;

private:
    ActionType action_type_;
    std::string feedback_tag_;
};

} // namespace revia::core
