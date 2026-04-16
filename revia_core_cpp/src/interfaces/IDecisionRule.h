// Phase 0 — IDecisionRule contract.
//
// Rules live under src/rules/ (Phase 2+). Each rule returns a DecisionInfluence.
// Rules MUST be:
//   - stateless (no mutation of runtime state from within Evaluate)
//   - cheap (<1 ms typical); long-running work belongs in a context provider
//   - side-effect free (write only to logs via StructuredLogger)
//
// Python decision plugins (Phase 4+) conform to the same contract over an
// IPC bridge loaded by PluginManager. The cpp-side shim forwards Evaluate()
// to a subprocess.
#pragma once

#include <string>
#include "models/ContextPackage.h"
#include "models/DecisionInfluence.h"
#include "models/RuntimeState.h"

namespace revia::core {

class IDecisionRule {
public:
    virtual ~IDecisionRule() = default;

    [[nodiscard]] virtual std::string name() const = 0;

    // Category is used by PriorityResolver for grouping; see
    // REVIA_CORE.md §Recommended Decision Rule Categories.
    // Examples: "ResponseEligibility", "Interruption", "EmotionInfluence".
    [[nodiscard]] virtual std::string category() const = 0;

    [[nodiscard]] virtual DecisionInfluence evaluate(const ContextPackage& ctx,
                                                     const RuntimeState& state) = 0;

    // Optional: declare whether this rule is safe to run in parallel with
    // others of the same category. Defaults to true; override to false only
    // when the rule must observe a specific ordering.
    [[nodiscard]] virtual bool is_parallel_safe() const { return true; }
};

} // namespace revia::core
