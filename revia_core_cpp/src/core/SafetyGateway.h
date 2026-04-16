// Phase 3 - SafetyGateway.
//
// Single ownership:
//   * SafetyGateway owns output/action safety validation.
//   * Safety layers are small and ordered. They do not execute actions.
#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "models/ContextPackage.h"
#include "models/DecisionResult.h"
#include "models/SafetyResult.h"

namespace revia::core {

class ISafetyLayer {
public:
    virtual ~ISafetyLayer() = default;

    [[nodiscard]] virtual std::string name() const = 0;

    [[nodiscard]] virtual SafetyLayerOutcome validate(const DecisionResult& decision,
                                                      const ContextPackage& context) = 0;
};

class SafetyGateway {
public:
    void register_layer(std::unique_ptr<ISafetyLayer> layer);

    [[nodiscard]] SafetyResult validate(const DecisionResult& decision,
                                        const ContextPackage& context);

    [[nodiscard]] std::size_t layer_count() const {
        return layers_.size();
    }

private:
    std::vector<std::unique_ptr<ISafetyLayer>> layers_;
};

} // namespace revia::core
