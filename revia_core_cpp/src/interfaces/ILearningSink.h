// Phase 0 — ILearningSink contract.
//
// FeedbackManager holds zero or more ILearningSink implementations. This is
// the seam across which py-side analytics (reinforcement_learner, metrics
// exporters) consume core feedback without the cpp core knowing anything
// about their internals.
//
// See REVIA_CORE.md §Feedback Loop.
#pragma once

#include <string>
#include "models/ContextPackage.h"
#include "models/ExecutionResult.h"
#include "models/RuntimeState.h"

namespace revia::core {

class ILearningSink {
public:
    virtual ~ILearningSink() = default;

    [[nodiscard]] virtual std::string name() const = 0;

    // Called by FeedbackManager after every execution (success OR failure).
    // Implementations MUST NOT throw. Exceptions escaping here will be
    // swallowed + logged; persistent failures should degrade the sink to
    // no-op rather than propagate.
    virtual void consume(const ExecutionResult& result,
                         const ContextPackage&  ctx,
                         const RuntimeState&    state) = 0;
};

} // namespace revia::core
