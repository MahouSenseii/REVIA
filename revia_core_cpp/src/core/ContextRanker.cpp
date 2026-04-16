// Phase 2 - ContextRanker implementation.
#include "core/ContextRanker.h"

#include <algorithm>
#include <string>

namespace revia::core {

std::vector<ContextSignal> ContextRanker::score(const ContextPackage& package,
                                                const RuntimeState& state) const {
    std::vector<ContextSignal> signals;
    signals.reserve(package.fragments.size());

    for (std::size_t i = 0; i < package.fragments.size(); ++i) {
        const auto& fragment = package.fragments[i];
        const double value = score_fragment(fragment, state);
        if (value <= 0.0) {
            continue;
        }

        ContextSignal signal;
        signal.fragment_index = i;
        signal.score = value;
        signal.reason = fragment.importance_hint >= 0.0
            ? "provider_hint"
            : "ranker_default";
        signals.push_back(std::move(signal));
    }

    std::sort(signals.begin(), signals.end(), [](const auto& a, const auto& b) {
        return a.score > b.score;
    });
    return signals;
}

double ContextRanker::score_fragment(const ContextFragment& fragment,
                                     const RuntimeState& state) const {
    if (fragment.importance_hint >= 0.0) {
        return std::clamp(fragment.importance_hint, 0.0, 1.0);
    }

    const auto& payload = fragment.payload;
    if (payload.empty()) {
        return 0.0;
    }

    if (payload.contains("event_text") || payload.contains("transcript")) {
        return 0.95;
    }
    if (payload.contains("requires_strict_filtering") ||
        payload.contains("platform")) {
        return 0.75;
    }
    if (state.is_recovery_mode || state.current_state == CoreState::Recovering) {
        return 0.70;
    }
    if (payload.contains("profile_id")) {
        return 0.50;
    }
    if (payload.contains("emotion_label")) {
        return 0.35;
    }
    return 0.25;
}

} // namespace revia::core
