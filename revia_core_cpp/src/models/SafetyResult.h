// Phase 0 — SafetyResult. Output of SafetyGateway.Validate(...).
#pragma once

#include <string>
#include <vector>
#include "models/Enums.h"

namespace revia::core {

struct SafetyLayerOutcome {
    std::string layer_name;   // e.g. "HardFilter", "AiFilter"
    bool passed = true;
    std::string reason;       // filled when passed == false
};

struct SafetyResult {
    SafetyVerdict verdict = SafetyVerdict::Allowed;

    // Per-layer trace. Always populated, even when allowed, for explainability.
    std::vector<SafetyLayerOutcome> layers;

    // If the gateway rewrote the output (e.g. softened tone, redacted PII),
    // the modified text lives here. Empty string = no modification.
    std::string modified_text;

    // Short human-readable summary for logs and UI.
    std::string summary;

    [[nodiscard]] bool is_allowed() const {
        return verdict == SafetyVerdict::Allowed
            || verdict == SafetyVerdict::AllowedWithModification;
    }
};

} // namespace revia::core
