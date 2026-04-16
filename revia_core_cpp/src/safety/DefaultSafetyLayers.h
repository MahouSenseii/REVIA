// Phase 3 - default safety layer chain.
#pragma once

#include <string>

#include "core/SafetyGateway.h"

namespace revia::core {

class HardFilterLayer final : public ISafetyLayer {
public:
    [[nodiscard]] std::string name() const override { return "HardFilter"; }
    [[nodiscard]] SafetyLayerOutcome validate(const DecisionResult& decision,
                                              const ContextPackage& context) override;
};

class AiFilterLayer final : public ISafetyLayer {
public:
    [[nodiscard]] std::string name() const override { return "AiFilter"; }
    [[nodiscard]] SafetyLayerOutcome validate(const DecisionResult& decision,
                                              const ContextPackage& context) override;
};

class PlatformRulesLayer final : public ISafetyLayer {
public:
    [[nodiscard]] std::string name() const override { return "PlatformRules"; }
    [[nodiscard]] SafetyLayerOutcome validate(const DecisionResult& decision,
                                              const ContextPackage& context) override;
};

class ModeRulesLayer final : public ISafetyLayer {
public:
    [[nodiscard]] std::string name() const override { return "ModeRules"; }
    [[nodiscard]] SafetyLayerOutcome validate(const DecisionResult& decision,
                                              const ContextPackage& context) override;
};

class ProfileOverridesLayer final : public ISafetyLayer {
public:
    [[nodiscard]] std::string name() const override { return "ProfileOverrides"; }
    [[nodiscard]] SafetyLayerOutcome validate(const DecisionResult& decision,
                                              const ContextPackage& context) override;
};

class FailSafeLayer final : public ISafetyLayer {
public:
    [[nodiscard]] std::string name() const override { return "FailSafe"; }
    [[nodiscard]] SafetyLayerOutcome validate(const DecisionResult& decision,
                                              const ContextPackage& context) override;
};

} // namespace revia::core
