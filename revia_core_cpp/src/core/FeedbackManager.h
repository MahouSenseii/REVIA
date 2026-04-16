// Phase 4 - FeedbackManager.
//
// Single ownership:
//   * FeedbackManager owns execution outcome processing and learning sink fanout.
//   * Sinks consume feedback; they do not mutate Core runtime state directly.
#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "interfaces/ILearningSink.h"
#include "models/ContextPackage.h"
#include "models/ExecutionResult.h"
#include "models/RuntimeState.h"

namespace revia::core {

class FeedbackManager {
public:
    void register_learning_sink(std::unique_ptr<ILearningSink> sink);

    void process(const ExecutionResult& result,
                 const ContextPackage& context,
                 const RuntimeState& state);

    [[nodiscard]] std::size_t sink_count() const {
        return sinks_.size();
    }

private:
    std::vector<std::unique_ptr<ILearningSink>> sinks_;
};

} // namespace revia::core
