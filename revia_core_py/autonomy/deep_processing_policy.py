from __future__ import annotations


def should_use_deep_processing(candidate, scored_candidate) -> bool:
    if getattr(candidate, "requires_deep_processing", False):
        return True
    if getattr(scored_candidate, "confidence", 1.0) < 0.60:
        return True
    if getattr(scored_candidate, "interruption_risk", 0.0) >= 0.45:
        return True
    return candidate.type in {
        "memory_reflection",
        "emotional_checkin",
        "suggest_action",
    }
