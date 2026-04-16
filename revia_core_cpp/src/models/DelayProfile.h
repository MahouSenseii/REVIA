// Phase 0 — DelayProfile POD. Owned/produced by TimingEngine (Phase 3).
#pragma once

#include <chrono>

namespace revia::core {

using Milliseconds = std::chrono::milliseconds;

struct DelayProfile {
    // Delay before the action begins executing.
    Milliseconds pre_action_delay { 0 };

    // Minimum silence to wait before a follow-up proactive action.
    Milliseconds silence_threshold { 0 };

    // If true, TimingEngine has determined a burst of fast responses is OK.
    bool allow_burst = false;

    // Anti-loop cooldown already applied to this decision (for logging).
    Milliseconds cooldown_applied { 0 };

    static DelayProfile NaturalPause() {
        DelayProfile d;
        d.pre_action_delay   = Milliseconds(250);
        d.silence_threshold  = Milliseconds(1500);
        d.allow_burst        = false;
        return d;
    }

    static DelayProfile ShortFollowUp() {
        DelayProfile d;
        d.pre_action_delay   = Milliseconds(80);
        d.silence_threshold  = Milliseconds(400);
        d.allow_burst        = true;
        return d;
    }

    static DelayProfile AllowBurst() {
        DelayProfile d;
        d.pre_action_delay   = Milliseconds(40);
        d.silence_threshold  = Milliseconds(200);
        d.allow_burst        = true;
        return d;
    }
};

} // namespace revia::core
