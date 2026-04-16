// Phase 2 - PriorityResolver.
//
// Single ownership:
//   * PriorityResolver owns cross-system conflict arbitration.
//   * DecisionEngine chooses an initial plan; this module applies explicit
//     precedence rules from REVIA_CORE.md.
#pragma once

#include "models/ContextPackage.h"
#include "models/DecisionResult.h"
#include "models/RuntimeState.h"

namespace revia::core {

class PriorityResolver {
public:
    [[nodiscard]] DecisionResult resolve(DecisionResult decision,
                                         const ContextPackage& context,
                                         const RuntimeState& state);

private:
    [[nodiscard]] static bool platform_requires_strict_filtering(
        const ContextPackage& context);
};

} // namespace revia::core
