// Phase 0 — ToneProfile POD. Consumed by executors to shape output voice/tone.
#pragma once

#include <string>

namespace revia::core {

struct ToneProfile {
    // High-level tone label. Free-form string so new tones can be added
    // without changing every consumer.
    std::string label = "neutral";

    // 0.0 .. 1.0 scalars. Default = middle ground.
    double warmth    = 0.5;
    double formality = 0.5;
    double energy    = 0.5;
    double humor     = 0.5;

    // Set to true when SafetyGateway's ProfileOverrides says to stay soft.
    bool soften = false;

    static ToneProfile SafeNeutral() {
        ToneProfile t;
        t.label = "safe_neutral";
        t.warmth = 0.5; t.formality = 0.6; t.energy = 0.3; t.humor = 0.2;
        t.soften = true;
        return t;
    }
};

} // namespace revia::core
