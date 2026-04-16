// Phase 2 - ContextManager.
//
// Single ownership:
//   * ContextManager owns context collection.
//   * Providers own their own source-specific collection logic.
//   * ContextRanker owns scoring; ContextManager only coordinates it.
#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "core/ContextRanker.h"
#include "interfaces/IContextProvider.h"
#include "interfaces/IEvent.h"
#include "models/ContextPackage.h"
#include "models/RuntimeState.h"

namespace revia::core {

class ContextManager {
public:
    explicit ContextManager(ContextRanker ranker = ContextRanker{});

    ContextManager(const ContextManager&) = delete;
    ContextManager& operator=(const ContextManager&) = delete;

    void register_provider(std::unique_ptr<IContextProvider> provider);

    [[nodiscard]] ContextPackage build_context(const IEvent& input_event,
                                               const RuntimeState& runtime_state);

    [[nodiscard]] std::size_t provider_count() const {
        return providers_.size();
    }

private:
    std::vector<std::unique_ptr<IContextProvider>> providers_;
    ContextRanker ranker_;
};

} // namespace revia::core
