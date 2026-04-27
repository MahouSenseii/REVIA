from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from .mode_profiles import ModeProfile, get_mode_profile


_QUIET_RE = re.compile(
    r"\b(stop|wait|hold on|quiet|be quiet|shush|shut up|not now|leave me alone)\b",
    re.IGNORECASE,
)

_FOCUS_RE = re.compile(
    r"\b(focus|working|debugging|coding|reading|busy|one sec|give me a minute)\b",
    re.IGNORECASE,
)

_NEGATIVE_MOODS = {
    "angry",
    "frustrated",
    "sad",
    "fear",
    "nervous",
    "stressed",
    "upset",
}


@dataclass
class AutonomyState:
    now: float
    current_mode: str
    profile: ModeProfile
    user_is_speaking: bool = False
    user_is_typing: bool = False
    revia_is_speaking: bool = False
    response_in_progress: bool = False
    seconds_since_last_user_message: float = 9999.0
    seconds_since_last_revia_message: float = 9999.0
    seconds_since_last_user_activity: float = 9999.0
    runtime_state: str = "Idle"
    active_request_id: str = ""
    active_topic: str = ""
    user_mood: str = "Neutral"
    conversation_energy: float = 0.0
    interruption_risk: float = 0.0
    quiet_requested: bool = False
    user_appears_focused: bool = False
    recent_user_text: str = ""
    recent_revia_text: str = ""
    profile_name: str = "Revia"
    metadata: dict[str, Any] = field(default_factory=dict)

    def do_not_talk_reasons(self) -> list[str]:
        reasons: list[str] = []
        if self.user_is_speaking:
            reasons.append("user is speaking")
        if self.user_is_typing:
            reasons.append("user is typing")
        if self.revia_is_speaking:
            reasons.append("Revia is already speaking")
        if self.response_in_progress:
            reasons.append("a response is already being generated")
        if self.quiet_requested:
            reasons.append("user recently asked for quiet")
        if self.seconds_since_last_revia_message < self.profile.global_cooldown_seconds:
            reasons.append("global autonomy cooldown active")
        if self.interruption_risk >= 0.70:
            reasons.append("interruption risk is high")
        if self.user_appears_focused and self.current_mode in {"quiet", "work"}:
            reasons.append("user appears focused")
        return reasons


class StateTracker:
    def __init__(
        self,
        *,
        conversation_manager,
        turn_manager,
        memory_store,
        profile_provider: Callable[[], dict[str, Any]],
        emotion_provider: Callable[[], dict[str, Any]],
        user_activity_seconds_provider: Callable[[], float],
    ):
        self._conversation_manager = conversation_manager
        self._turn_manager = turn_manager
        self._memory_store = memory_store
        self._profile_provider = profile_provider
        self._emotion_provider = emotion_provider
        self._user_activity_seconds_provider = user_activity_seconds_provider

    def get_state(
        self,
        *,
        metadata: dict[str, Any] | None = None,
        active_topic: str = "",
    ) -> AutonomyState:
        metadata = dict(metadata or {})
        now = time.monotonic()
        profile = self._profile_provider() or {}
        behavior = profile.get("behavior", {}) or {}
        mode_name = str(
            metadata.get("autonomy_mode")
            or behavior.get("autonomy_mode")
            or profile.get("autonomy_mode")
            or "companion"
        ).strip().lower()
        mode_profile = get_mode_profile(mode_name)
        recent = self._memory_store.get_short_term(limit=20)
        last_user, last_revia = self._last_messages(recent)
        mood_payload = self._emotion_provider() or {}
        mood = str(mood_payload.get("label") or "Neutral")
        runtime_state = str(self._conversation_manager.current_state or "Idle")
        turn_snapshot = self._turn_manager.snapshot() or {}
        active_request_id = str(turn_snapshot.get("active_request_id") or "")
        active_turn = turn_snapshot.get("active_turn") or {}
        response_in_progress = (
            bool(active_turn)
            or runtime_state in {"Thinking", "Listening"}
            or bool(metadata.get("response_in_progress", False))
        )
        revia_is_speaking = runtime_state == "Speaking" or bool(metadata.get("revia_is_speaking", False))
        user_is_speaking = bool(metadata.get("user_is_speaking", False))
        user_is_typing = bool(metadata.get("user_is_typing", False))

        recent_user_text = str(last_user.get("content") or "")
        recent_revia_text = str(last_revia.get("content") or "")
        quiet_requested = self._recent_quiet_request(recent_user_text, last_user)
        focused = bool(_FOCUS_RE.search(recent_user_text or ""))
        energy = self._conversation_energy(recent)
        interruption_risk = self._interruption_risk(
            user_is_speaking=user_is_speaking,
            user_is_typing=user_is_typing,
            revia_is_speaking=revia_is_speaking,
            response_in_progress=response_in_progress,
            seconds_since_last_user=self._age_seconds(last_user, default=9999.0),
            seconds_since_activity=float(self._user_activity_seconds_provider()),
        )
        if mood.lower() in _NEGATIVE_MOODS:
            interruption_risk = max(0.15, interruption_risk)

        return AutonomyState(
            now=now,
            current_mode=mode_profile.name,
            profile=mode_profile,
            user_is_speaking=user_is_speaking,
            user_is_typing=user_is_typing,
            revia_is_speaking=revia_is_speaking,
            response_in_progress=response_in_progress,
            seconds_since_last_user_message=self._age_seconds(last_user, default=9999.0),
            seconds_since_last_revia_message=self._age_seconds(last_revia, default=9999.0),
            seconds_since_last_user_activity=float(self._user_activity_seconds_provider()),
            runtime_state=runtime_state,
            active_request_id=active_request_id,
            active_topic=active_topic,
            user_mood=mood,
            conversation_energy=energy,
            interruption_risk=interruption_risk,
            quiet_requested=quiet_requested,
            user_appears_focused=focused,
            recent_user_text=recent_user_text,
            recent_revia_text=recent_revia_text,
            profile_name=str(profile.get("character_name") or "Revia"),
            metadata=metadata,
        )

    @staticmethod
    def _last_messages(messages: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
        last_user: dict[str, Any] = {}
        last_revia: dict[str, Any] = {}
        for msg in reversed(messages or []):
            role = str(msg.get("role") or "").lower()
            if not last_user and role == "user":
                last_user = dict(msg)
            elif not last_revia and role == "assistant":
                last_revia = dict(msg)
            if last_user and last_revia:
                break
        return last_user, last_revia

    @staticmethod
    def _age_seconds(entry: dict[str, Any], default: float) -> float:
        ts = str(entry.get("timestamp") or "").strip()
        if not ts:
            return default
        try:
            dt = datetime.fromisoformat(ts)
            return max(0.0, (datetime.now(dt.tzinfo) - dt).total_seconds())
        except Exception:
            return default

    def _recent_quiet_request(self, text: str, entry: dict[str, Any]) -> bool:
        if not text or not _QUIET_RE.search(text):
            return False
        return self._age_seconds(entry, default=9999.0) <= 900.0

    @staticmethod
    def _conversation_energy(messages: list[dict[str, Any]]) -> float:
        if not messages:
            return 0.0
        recent = messages[-8:]
        words = sum(len(str(m.get("content") or "").split()) for m in recent)
        return max(0.0, min(1.0, words / 260.0))

    @staticmethod
    def _interruption_risk(
        *,
        user_is_speaking: bool,
        user_is_typing: bool,
        revia_is_speaking: bool,
        response_in_progress: bool,
        seconds_since_last_user: float,
        seconds_since_activity: float,
    ) -> float:
        risk = 0.0
        if user_is_speaking:
            risk += 0.60
        if user_is_typing:
            risk += 0.45
        if revia_is_speaking:
            risk += 0.35
        if response_in_progress:
            risk += 0.45
        if seconds_since_last_user < 20.0:
            risk += 0.30
        elif seconds_since_last_user < 45.0:
            risk += 0.15
        if seconds_since_activity < 15.0:
            risk += 0.25
        return max(0.0, min(1.0, risk))
