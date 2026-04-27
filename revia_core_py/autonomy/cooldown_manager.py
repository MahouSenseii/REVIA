from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from .candidate_generator import SelfInitiationCandidate


_SIG_RE = re.compile(r"[^a-z0-9]+")


@dataclass
class CooldownSnapshot:
    active: dict[str, float] = field(default_factory=dict)
    repetition_risk: float = 0.0


class CooldownManager:
    def __init__(self):
        self._last_spoken_at = 0.0
        self._last_new_topic_at = 0.0
        self._topic_last: dict[str, float] = {}
        self._phrase_last: dict[str, float] = {}
        self._quiet_until = 0.0
        self._interruption_until = 0.0

    def note_quiet_request(self, duration_s: float) -> None:
        self._quiet_until = max(self._quiet_until, time.monotonic() + max(0.0, duration_s))

    def note_interruption(self, duration_s: float) -> None:
        self._interruption_until = max(self._interruption_until, time.monotonic() + max(0.0, duration_s))

    def register_spoken_candidate(self, candidate: SelfInitiationCandidate, state) -> None:
        now = time.monotonic()
        self._last_spoken_at = now
        topic = str(candidate.topic or "").strip()
        if topic:
            previous_topic = state.active_topic
            self._topic_last[topic] = now
            if previous_topic and topic != previous_topic:
                self._last_new_topic_at = now
        sig = self._phrase_signature(candidate.text or candidate.reason or candidate.type)
        if sig:
            self._phrase_last[sig] = now

    def snapshot_for(self, candidate: SelfInitiationCandidate, state) -> CooldownSnapshot:
        now = time.monotonic()
        active: dict[str, float] = {}
        if self._last_spoken_at:
            rem = state.profile.global_cooldown_seconds - (now - self._last_spoken_at)
            if rem > 0:
                active["global"] = rem
        topic = str(candidate.topic or "").strip()
        if topic and topic in self._topic_last:
            rem = state.profile.same_topic_cooldown_seconds - (now - self._topic_last[topic])
            if rem > 0:
                active["same_topic"] = rem
        if candidate.type == "start_new_topic" and self._last_new_topic_at:
            rem = state.profile.new_topic_cooldown_seconds - (now - self._last_new_topic_at)
            if rem > 0:
                active["new_topic"] = rem
        sig = self._phrase_signature(candidate.text or candidate.reason or candidate.type)
        if sig and sig in self._phrase_last:
            rem = state.profile.same_phrase_cooldown_seconds - (now - self._phrase_last[sig])
            if rem > 0:
                active["same_phrase"] = rem
        if self._quiet_until > now:
            active["quiet_request"] = self._quiet_until - now
        if self._interruption_until > now:
            active["interruption"] = self._interruption_until - now

        repetition_risk = 0.0
        if "same_topic" in active:
            repetition_risk += 0.45
        if "same_phrase" in active:
            repetition_risk += 0.45
        if "global" in active:
            repetition_risk += 0.30
        return CooldownSnapshot(
            active={k: round(v, 1) for k, v in active.items()},
            repetition_risk=max(0.0, min(1.0, repetition_risk)),
        )

    def active_cooldowns(self) -> dict[str, float]:
        now = time.monotonic()
        active: dict[str, float] = {}
        if self._quiet_until > now:
            active["quiet_request"] = round(self._quiet_until - now, 1)
        if self._interruption_until > now:
            active["interruption"] = round(self._interruption_until - now, 1)
        return active

    @staticmethod
    def _phrase_signature(text: str) -> str:
        words = _SIG_RE.sub(" ", str(text or "").lower()).split()[:8]
        return " ".join(words)
