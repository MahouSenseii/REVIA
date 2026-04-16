// Phase 4 - default learning sinks.
#pragma once

#include <string>

#include "interfaces/ILearningSink.h"

namespace revia::core {

class StructuredLogLearningSink final : public ILearningSink {
public:
    [[nodiscard]] std::string name() const override {
        return "StructuredLogLearningSink";
    }

    void consume(const ExecutionResult& result,
                 const ContextPackage& context,
                 const RuntimeState& state) override;
};

} // namespace revia::core
