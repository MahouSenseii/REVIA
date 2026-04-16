// Phase 2 - DecisionEngine.
//
// Single ownership:
//   * DecisionEngine owns action selection from rule influences.
//   * Rules can suggest; they do not mutate state or execute actions.
//   * PriorityResolver can still override the merged decision later.
#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "interfaces/IDecisionRule.h"
#include "models/ContextPackage.h"
#include "models/DecisionInfluence.h"
#include "models/DecisionResult.h"
#include "models/RuntimeState.h"

namespace revia::core {

class DecisionEngine {
public:
    void register_rule(std::unique_ptr<IDecisionRule> rule);

    [[nodiscard]] DecisionResult evaluate(const ContextPackage& context,
                                          const RuntimeState& state);

    [[nodiscard]] DecisionResult build_fallback_action(const ContextPackage& context,
                                                       std::string reason);

    [[nodiscard]] std::size_t rule_count() const {
        return rules_.size();
    }

private:
    [[nodiscard]] DecisionInfluence merge_influences(
        const std::vector<DecisionInfluence>& influences) const;

    [[nodiscard]] static std::string summarize(
        const std::vector<DecisionInfluence>& influences,
        const DecisionInfluence& merged);

    std::vector<std::unique_ptr<IDecisionRule>> rules_;
};

} // namespace revia::core
