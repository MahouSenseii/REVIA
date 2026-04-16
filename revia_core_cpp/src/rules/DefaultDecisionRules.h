// Phase 2 - first-day decision rules.
//
// Rules are deliberately small and side-effect free. They produce influences;
// DecisionEngine and PriorityResolver own the merge and conflict resolution.
#pragma once

#include <string>

#include "interfaces/IDecisionRule.h"

namespace revia::core {

class ResponseEligibilityRule final : public IDecisionRule {
public:
    [[nodiscard]] std::string name() const override {
        return "ResponseEligibilityRule";
    }

    [[nodiscard]] std::string category() const override {
        return "ResponseEligibility";
    }

    [[nodiscard]] DecisionInfluence evaluate(const ContextPackage& ctx,
                                             const RuntimeState& state) override;
};

class IntentConfidenceRule final : public IDecisionRule {
public:
    [[nodiscard]] std::string name() const override {
        return "IntentConfidenceRule";
    }

    [[nodiscard]] std::string category() const override {
        return "IntentConfidence";
    }

    [[nodiscard]] DecisionInfluence evaluate(const ContextPackage& ctx,
                                             const RuntimeState& state) override;
};

class InterruptionRule final : public IDecisionRule {
public:
    [[nodiscard]] std::string name() const override {
        return "InterruptionRule";
    }

    [[nodiscard]] std::string category() const override {
        return "Interruption";
    }

    [[nodiscard]] DecisionInfluence evaluate(const ContextPackage& ctx,
                                             const RuntimeState& state) override;
};

class PlatformConstraintRule final : public IDecisionRule {
public:
    [[nodiscard]] std::string name() const override {
        return "PlatformConstraintRule";
    }

    [[nodiscard]] std::string category() const override {
        return "PlatformConstraint";
    }

    [[nodiscard]] DecisionInfluence evaluate(const ContextPackage& ctx,
                                             const RuntimeState& state) override;
};

} // namespace revia::core
