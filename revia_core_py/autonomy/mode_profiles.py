from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModeProfile:
    name: str
    min_check_seconds: int
    max_check_seconds: int
    speak_threshold: float
    base_silence_score: float
    min_idle_seconds: float
    global_cooldown_seconds: float
    same_topic_cooldown_seconds: float
    same_phrase_cooldown_seconds: float
    new_topic_cooldown_seconds: float
    quiet_request_cooldown_seconds: float
    interruption_cooldown_seconds: float


MODE_PROFILES: dict[str, ModeProfile] = {
    "quiet": ModeProfile(
        name="quiet",
        min_check_seconds=90,
        max_check_seconds=240,
        speak_threshold=0.92,
        base_silence_score=0.90,
        min_idle_seconds=90,
        global_cooldown_seconds=180,
        same_topic_cooldown_seconds=900,
        same_phrase_cooldown_seconds=1800,
        new_topic_cooldown_seconds=900,
        quiet_request_cooldown_seconds=1800,
        interruption_cooldown_seconds=180,
    ),
    "companion": ModeProfile(
        name="companion",
        min_check_seconds=30,
        max_check_seconds=120,
        speak_threshold=0.75,
        base_silence_score=0.70,
        min_idle_seconds=40,
        global_cooldown_seconds=120,
        same_topic_cooldown_seconds=420,
        same_phrase_cooldown_seconds=900,
        new_topic_cooldown_seconds=420,
        quiet_request_cooldown_seconds=900,
        interruption_cooldown_seconds=90,
    ),
    "stream": ModeProfile(
        name="stream",
        min_check_seconds=10,
        max_check_seconds=60,
        speak_threshold=0.65,
        base_silence_score=0.55,
        min_idle_seconds=20,
        global_cooldown_seconds=45,
        same_topic_cooldown_seconds=180,
        same_phrase_cooldown_seconds=420,
        new_topic_cooldown_seconds=240,
        quiet_request_cooldown_seconds=600,
        interruption_cooldown_seconds=60,
    ),
    "work": ModeProfile(
        name="work",
        min_check_seconds=120,
        max_check_seconds=300,
        speak_threshold=0.88,
        base_silence_score=0.85,
        min_idle_seconds=120,
        global_cooldown_seconds=240,
        same_topic_cooldown_seconds=1200,
        same_phrase_cooldown_seconds=1800,
        new_topic_cooldown_seconds=1200,
        quiet_request_cooldown_seconds=1800,
        interruption_cooldown_seconds=180,
    ),
    "emotional_support": ModeProfile(
        name="emotional_support",
        min_check_seconds=30,
        max_check_seconds=90,
        speak_threshold=0.70,
        base_silence_score=0.62,
        min_idle_seconds=35,
        global_cooldown_seconds=90,
        same_topic_cooldown_seconds=360,
        same_phrase_cooldown_seconds=900,
        new_topic_cooldown_seconds=600,
        quiet_request_cooldown_seconds=900,
        interruption_cooldown_seconds=90,
    ),
}


def get_mode_profile(name: str | None) -> ModeProfile:
    key = str(name or "companion").strip().lower()
    return MODE_PROFILES.get(key, MODE_PROFILES["companion"])
