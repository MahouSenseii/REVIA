// Phase 2 - ContextRanker.
//
// Single ownership:
//   * ContextRanker owns context importance scoring.
//   * Providers may offer hints, but they do not decide final ranking.
#pragma once

#include <vector>

#include "models/ContextPackage.h"
#include "models/RuntimeState.h"

namespace revia::core {

class ContextRanker {
public:
    [[nodiscard]] std::vector<ContextSignal> score(const ContextPackage& package,
                                                   const RuntimeState& state) const;

private:
    [[nodiscard]] double score_fragment(const ContextFragment& fragment,
                                        const RuntimeState& state) const;
};

} // namespace revia::core
