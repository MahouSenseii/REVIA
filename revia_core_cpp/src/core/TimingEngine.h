// Phase 3 - TimingEngine.
//
// Single ownership:
//   * TimingEngine owns pacing, cooldown, delay, burst, and recovery timing.
//   * It does not sleep or execute actions. ActionDispatcher handles execution.
//   * Other modules may display timers, but behavior-affecting timing belongs here.
#pragma once

#include "models/ContextPackage.h"
#include "models/DecisionResult.h"
#include "models/RuntimeState.h"

namespace revia::core {

class TimingEngine {
public:
    [[nodiscard]] DecisionResult apply_timing(DecisionResult decision,
                                              const ContextPackage& context,
                                              const RuntimeState& state);

private:
    [[nodiscard]] static bool platform_allows_burst(const ContextPackage& context);
};

} // namespace revia::core
