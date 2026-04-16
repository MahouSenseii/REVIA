// Phase 0 — ContextPackage. Assembled by ContextManager; consumed by DecisionEngine.
#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <nlohmann/json.hpp>
#include "models/ContextFragment.h"
#include "models/RuntimeState.h"

namespace revia::core {

// Forward declaration; IEvent lives in interfaces/ and has a full definition there.
struct IEvent;

// A ranked signal is a pointer into the package's fragment list plus a score.
// We use an index + score so the package remains copyable without shared_ptr
// overhead; see also ContextRanker (Phase 2).
struct ContextSignal {
    std::size_t fragment_index = 0;
    double      score          = 0.0;
    std::string reason;   // optional: why this signal scored as it did
};

struct ContextPackage {
    // The triggering event. Borrowed pointer; lifetime is owned by the bus/
    // orchestrator call-frame, not by the package.
    const IEvent* current_event = nullptr;

    // Raw fragments as produced by providers.
    std::vector<ContextFragment> fragments;

    // Ranked view. Order = descending score. Computed by ContextRanker.
    std::vector<ContextSignal> ranked_signals;

    // Read-only snapshot of the runtime state at build time. DecisionEngine
    // should prefer this over live state to keep a decision frame consistent.
    RuntimeState state_snapshot;

    // Convenience: lookup a fragment by provider name (returns nullptr if missing).
    const ContextFragment* find(const std::string& provider_name) const {
        for (const auto& f : fragments) {
            if (f.provider_name == provider_name) return &f;
        }
        return nullptr;
    }

    void add(ContextFragment fragment) {
        fragments.push_back(std::move(fragment));
    }
};

} // namespace revia::core
