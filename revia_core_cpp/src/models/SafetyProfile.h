// Phase 0 — SafetyProfile POD. Carries the declared safety posture for a decision.
// Consumed by SafetyGateway (Phase 3). Do not confuse with SafetyResult (the *output*
// of the gateway run).
#pragma once

#include <string>
#include <vector>

namespace revia::core {

struct SafetyProfile {
    // Which filter layers to run. Empty = use platform default.
    std::vector<std::string> required_layers;

    // Strictness knob 0.0 (permissive) .. 1.0 (max strict).
    double strictness = 0.5;

    // Platform flag — set when the platform mandates strict filtering
    // (e.g. Twitch content rules). Takes precedence over profile overrides.
    bool platform_strict = false;

    // If true, a failed safety run triggers fallback action rather than
    // dropping the output entirely.
    bool allow_fallback_on_block = true;
};

} // namespace revia::core
