// Phase 0 — IContextProvider contract.
//
// Implementations live under src/providers/ (Phase 2+). Each provider returns
// a single ContextFragment per event. Providers MUST be:
//   - stateless OR own their own cache (no shared global mutation)
//   - deterministic given the same (event, state) input
//   - fast-returning OR marked async for ContextManager thread-pool handling
#pragma once

#include <string>
#include "models/ContextFragment.h"
#include "models/RuntimeState.h"

namespace revia::core {

struct IEvent;

class IContextProvider {
public:
    virtual ~IContextProvider() = default;

    // Provider's canonical name (used in logs + fragment.provider_name).
    [[nodiscard]] virtual std::string name() const = 0;

    // Collect a fragment for this event/state. May return an "empty" fragment
    // (payload = {}) if the provider has nothing to say; ContextManager will
    // drop those.
    [[nodiscard]] virtual ContextFragment collect(const IEvent& event,
                                                  const RuntimeState& state) = 0;

    // Optional budget hint (milliseconds). ContextManager enforces a hard
    // timeout; this is an advisory the ranker can use to deprioritize
    // expensive providers. 0 = unknown/cheap.
    [[nodiscard]] virtual int budget_ms_hint() const { return 0; }
};

} // namespace revia::core
