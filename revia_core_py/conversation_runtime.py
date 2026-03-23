from __future__ import annotations

import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum


class LLMConnectionState(str, Enum):
    DISCONNECTED = "Disconnected"
    CONNECTING = "Connecting"
    READY = "Ready"
    ERROR = "Error"


class ReviaState(str, Enum):
    BOOTING = "Booting"
    INITIALIZING = "Initializing"
    IDLE = "Idle"
    LISTENING = "Listening"
    THINKING = "Thinking"
    SPEAKING = "Speaking"
    COOLDOWN = "Cooldown"
    ERROR = "Error"
    # PRD §8 — IHS states
    INTERRUPTED = "Interrupted"   # user barged in while Revia was speaking
    RECOVERING  = "Recovering"    # pipeline is rebuilding after interruption


class TriggerKind(str, Enum):
    RESPONSE = "response"
    AUTONOMOUS = "autonomous"
    INTERRUPTION = "interruption"


class TriggerSource(str, Enum):
    USER_MESSAGE = "UserMessage"
    VOICE_INPUT = "VoiceInput"
    STARTUP = "Startup"
    IDLE_TIMER = "IdleTimer"
    EMOTION = "Emotion"
    FOCUS = "Focus"
    WAKE_WORD = "WakeWord"
    RANDOM_BANTER = "RandomBanter"
    MANUAL_UI = "ManualUI"
    UI_EVENT = "UIEvent"
    INTERRUPTION = "Interruption"


@dataclass
class TriggerRequest:
    source: str
    kind: str
    reason: str
    text: str = ""
    force: bool = False
    require_voice_input: bool = False
    require_speech_output: bool = False
    require_emotion: bool = True
    metadata: dict = field(default_factory=dict)
    received_at: float = field(default_factory=time.monotonic)


@dataclass
class SubsystemStatus:
    required: bool
    ready: bool
    state: str
    detail: str = ""

    def to_dict(self):
        return {
            "required": self.required,
            "ready": self.ready,
            "state": self.state,
            "detail": self.detail,
        }


@dataclass
class ReadinessSnapshot:
    startup_phase: str
    startup_complete: bool
    checks: dict[str, SubsystemStatus]
    blocking_reasons: list[str]
    ready: bool
    can_start_conversation: bool
    can_auto_initiate: bool

    def to_dict(self):
        return {
            "startup_phase": self.startup_phase,
            "startup_complete": self.startup_complete,
            "checks": {
                name: status.to_dict() for name, status in self.checks.items()
            },
            "blocking_reasons": list(self.blocking_reasons),
            "ready": self.ready,
            "can_start_conversation": self.can_start_conversation,
            "can_auto_initiate": self.can_auto_initiate,
        }


@dataclass
class BehaviorDecision:
    allowed: bool
    reason: str
    trigger_kind: str
    trigger_source: str
    cooldown_name: str = ""
    cooldown_remaining_s: float = 0.0

    def to_dict(self):
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "trigger_kind": self.trigger_kind,
            "trigger_source": self.trigger_source,
            "cooldown_name": self.cooldown_name,
            "cooldown_remaining_s": round(self.cooldown_remaining_s, 3),
        }


@dataclass
class ResponseFilterResult:
    accepted: bool
    text: str
    reason: str
    speakable: bool = True


class ConversationStateMachine:
    _ALLOWED_TRANSITIONS = {
        ReviaState.BOOTING: {ReviaState.INITIALIZING, ReviaState.ERROR},
        ReviaState.INITIALIZING: {ReviaState.IDLE, ReviaState.ERROR},
        ReviaState.IDLE: {
            ReviaState.LISTENING,
            ReviaState.THINKING,
            ReviaState.COOLDOWN,
            ReviaState.ERROR,
        },
        ReviaState.LISTENING: {
            ReviaState.IDLE,
            ReviaState.THINKING,
            ReviaState.ERROR,
        },
        ReviaState.THINKING: {
            ReviaState.SPEAKING,
            ReviaState.COOLDOWN,
            ReviaState.IDLE,
            ReviaState.ERROR,
        },
        ReviaState.SPEAKING: {
            ReviaState.COOLDOWN,
            ReviaState.IDLE,
            ReviaState.ERROR,
            ReviaState.INTERRUPTED,   # PRD §8 — barge-in while speaking
        },
        ReviaState.COOLDOWN: {
            ReviaState.IDLE,
            ReviaState.ERROR,
        },
        ReviaState.ERROR: {
            ReviaState.INITIALIZING,
            ReviaState.IDLE,
        },
        # PRD §8 — IHS recovery transitions
        ReviaState.INTERRUPTED: {
            ReviaState.RECOVERING,
            ReviaState.LISTENING,
            ReviaState.IDLE,
            ReviaState.ERROR,
        },
        ReviaState.RECOVERING: {
            ReviaState.THINKING,
            ReviaState.IDLE,
            ReviaState.ERROR,
        },
    }

    def __init__(self, log_fn):
        self._log = log_fn
        self._lock = threading.Lock()
        self._state = ReviaState.BOOTING
        self._updated_at = time.monotonic()

        # ── PRD §8 — IHS extended state fields ───────────────────────────
        self.interruption_type: str   = ""     # last InterruptionType value
        self.partial_response_spoken: str = "" # text Revia had delivered before barge-in
        self.topic_shift_confidence: float = 0.0

        # ── PRD §10/§12 — AVS / ALE extended state fields ────────────────
        self.answer_confidence_last: float = 0.0  # last AVS composite score
        self.regen_count: int    = 0   # regens on the current turn
        self.loop_risk_score: float = 0.0  # last ALE loop_risk_score

    @property
    def state(self) -> str:
        with self._lock:
            return self._state.value

    # ── PRD §8 IHS helpers ────────────────────────────────────────────────

    def record_interruption(
        self,
        interruption_type: str,
        partial_spoken: str = "",
        topic_shift_confidence: float = 0.0,
    ) -> None:
        """Persist IHS classification results into the FSM for downstream use."""
        with self._lock:
            self.interruption_type        = interruption_type
            self.partial_response_spoken  = partial_spoken
            self.topic_shift_confidence   = topic_shift_confidence

    def record_avs_result(self, composite: float, regen_count: int = 0) -> None:
        """Persist the latest AVS composite score and regen counter."""
        with self._lock:
            self.answer_confidence_last = composite
            self.regen_count            = regen_count

    def record_ale_result(self, loop_risk_score: float) -> None:
        """Persist the latest ALE loop_risk_score."""
        with self._lock:
            self.loop_risk_score = loop_risk_score

    def get_extended_state(self) -> dict:
        """Return a snapshot of all PRD extended state fields."""
        with self._lock:
            return {
                "interruption_type":       self.interruption_type,
                "partial_response_spoken": self.partial_response_spoken[:80],
                "topic_shift_confidence":  round(self.topic_shift_confidence, 4),
                "answer_confidence_last":  round(self.answer_confidence_last, 4),
                "regen_count":             self.regen_count,
                "loop_risk_score":         round(self.loop_risk_score, 4),
            }

    def transition(self, new_state: ReviaState | str, reason: str = "", force: bool = False) -> bool:
        if not isinstance(new_state, ReviaState):
            new_state = ReviaState(str(new_state))
        with self._lock:
            current = self._state
            if current == new_state:
                return True
            if not force and new_state not in self._ALLOWED_TRANSITIONS.get(current, set()):
                self._log(
                    f"Blocked invalid state transition: {current.value} -> "
                    f"{new_state.value} ({reason or 'no reason'})"
                )
                return False
            self._state = new_state
            self._updated_at = time.monotonic()
        self._log(
            f"State change: {current.value} -> {new_state.value}"
            + (f" ({reason})" if reason else "")
        )
        return True


class BehaviorController:
    def __init__(
        self,
        log_fn,
        startup_grace_s: float = 4.0,
        autonomous_warmup_s: float = 8.0,
        response_cooldown_s: float = 0.75,
        autonomous_cooldown_s: float = 120.0,
        interruption_cooldown_s: float = 5.0,
    ):
        self._log = log_fn
        self._lock = threading.Lock()
        self._startup_phase = ReviaState.BOOTING.value
        self._startup_complete_at = 0.0
        self._startup_grace_s = float(startup_grace_s)
        self._autonomous_warmup_s = float(autonomous_warmup_s)
        self._cooldown_defaults = {
            "response": float(response_cooldown_s),
            "autonomous": float(autonomous_cooldown_s),
            "interruption": float(interruption_cooldown_s),
        }
        self._cooldowns: dict[str, float] = {}

    def mark_startup_phase(self, phase: str):
        with self._lock:
            self._startup_phase = str(phase or ReviaState.INITIALIZING.value)

    def mark_startup_complete(self):
        with self._lock:
            now = time.monotonic()
            if self._startup_complete_at <= 0.0:
                self._startup_complete_at = now
            self._startup_phase = "Ready"
        self._log("Startup complete: readiness gate unlocked")

    def startup_complete(self) -> bool:
        with self._lock:
            return self._startup_complete_at > 0.0

    def startup_phase(self) -> str:
        with self._lock:
            if self._startup_complete_at > 0.0:
                return "Ready"
            return self._startup_phase

    def can_auto_initiate(self) -> bool:
        with self._lock:
            if self._startup_complete_at <= 0.0:
                return False
            return time.monotonic() >= (
                self._startup_complete_at
                + self._startup_grace_s
                + self._autonomous_warmup_s
            )

    def active_cooldowns(self) -> dict[str, float]:
        now = time.monotonic()
        with self._lock:
            expired = [
                name for name, expires_at in self._cooldowns.items()
                if expires_at <= now
            ]
            for name in expired:
                self._cooldowns.pop(name, None)
            return {
                name: round(max(0.0, expires_at - now), 3)
                for name, expires_at in self._cooldowns.items()
                if expires_at > now
            }

    def start_cooldown(self, name: str, duration_s: float | None = None):
        duration = self._cooldown_defaults.get(name, 0.0)
        if duration_s is not None:
            duration = float(duration_s)
        if duration <= 0.0:
            return
        expires_at = time.monotonic() + duration
        with self._lock:
            self._cooldowns[name] = expires_at
        self._log(f"Cooldown started: {name} ({duration:.1f}s)")

    def remaining_cooldown(self, name: str) -> float:
        """Get remaining cooldown time without re-acquiring lock. Call within lock context if available."""
        now = time.monotonic()
        with self._lock:
            expires_at = self._cooldowns.get(name, 0.0)
            if expires_at <= now:
                return 0.0
            return round(max(0.0, expires_at - now), 3)

    def evaluate(
        self,
        trigger: TriggerRequest,
        readiness: ReadinessSnapshot,
        current_state: str,
        *,
        user_speaking: bool = False,
        assistant_speaking: bool = False,
        auto_initiation_allowed: bool = True,
    ) -> BehaviorDecision:
        kind = str(trigger.kind or TriggerKind.RESPONSE.value)
        source = str(trigger.source or TriggerSource.MANUAL_UI.value)
        reason = str(trigger.reason or "").strip()
        current_state = str(current_state or ReviaState.BOOTING.value)
        if not reason:
            return self._blocked(trigger, "invalid trigger reason")

        if not readiness.can_start_conversation:
            detail = readiness.blocking_reasons[0] if readiness.blocking_reasons else "system not ready"
            return self._blocked(trigger, f"system not ready ({detail})")

        if current_state in (
            ReviaState.BOOTING.value,
            ReviaState.INITIALIZING.value,
            ReviaState.ERROR.value,
        ):
            return self._blocked(trigger, f"state={current_state}")

        if assistant_speaking and kind != TriggerKind.INTERRUPTION.value:
            return self._blocked(trigger, "assistant already speaking")

        if kind == TriggerKind.RESPONSE.value:
            if current_state in (ReviaState.THINKING.value, ReviaState.SPEAKING.value, ReviaState.COOLDOWN.value):
                return self._blocked(trigger, f"state={current_state}")
            remaining = self.remaining_cooldown("response")
            if remaining > 0 and not trigger.force:
                return self._blocked(trigger, "response cooldown active", "response", remaining)

        elif kind == TriggerKind.AUTONOMOUS.value:
            if not auto_initiation_allowed:
                return self._blocked(trigger, "auto-initiation disabled")
            if not readiness.can_auto_initiate:
                return self._blocked(trigger, "startup warmup active")
            if current_state != ReviaState.IDLE.value:
                return self._blocked(trigger, f"state={current_state}")
            if user_speaking:
                return self._blocked(trigger, "user is speaking")
            recent_user_activity_s = float(trigger.metadata.get("recent_user_activity_s", 9999.0) or 9999.0)
            if recent_user_activity_s < 25.0 and not trigger.force:
                return self._blocked(trigger, "recent user activity")
            remaining = self.remaining_cooldown("autonomous")
            if remaining > 0 and not trigger.force:
                return self._blocked(trigger, "autonomous cooldown active", "autonomous", remaining)

        elif kind == TriggerKind.INTERRUPTION.value:
            if not assistant_speaking:
                return self._blocked(trigger, "nothing to interrupt")
            remaining = self.remaining_cooldown("interruption")
            if remaining > 0 and not trigger.force:
                return self._blocked(trigger, "interruption cooldown active", "interruption", remaining)

        self._log(f"Allowed: {source} ({reason})")
        return BehaviorDecision(
            allowed=True,
            reason=reason,
            trigger_kind=kind,
            trigger_source=source,
        )

    def _blocked(
        self,
        trigger: TriggerRequest,
        reason: str,
        cooldown_name: str = "",
        cooldown_remaining_s: float = 0.0,
    ) -> BehaviorDecision:
        self._log(
            f"Blocked: {reason} | source={trigger.source} | kind={trigger.kind}"
        )
        return BehaviorDecision(
            allowed=False,
            reason=reason,
            trigger_kind=str(trigger.kind or ""),
            trigger_source=str(trigger.source or ""),
            cooldown_name=cooldown_name,
            cooldown_remaining_s=cooldown_remaining_s,
        )


class ResponseFilter:
    _WHITESPACE_RE = re.compile(r"[ \t]+")
    _SIGNATURE_RE = re.compile(r"[^a-z0-9]+")

    def __init__(self, log_fn):
        self._log = log_fn
        self._lock = threading.Lock()
        self._recent_autonomous = deque(maxlen=6)

    def apply(self, text: str, trigger: TriggerRequest, emotion_label: str = "Neutral") -> ResponseFilterResult:
        cleaned = self._normalize(text)
        if not cleaned:
            fallback = ""
            if str(trigger.kind) == TriggerKind.RESPONSE.value:
                fallback = "I need another second to finish lining that up."
            self._log("Response filter rejected: empty output")
            return ResponseFilterResult(False, fallback, "empty output", speakable=False)

        if str(trigger.kind) == TriggerKind.AUTONOMOUS.value:
            cleaned = self._trim_sentences(cleaned, max_sentences=2)
            if cleaned.startswith("["):
                self._log("Response filter rejected: autonomous error output")
                return ResponseFilterResult(False, "", "autonomous error output", speakable=False)
            sig = self._signature(cleaned)
            with self._lock:
                if sig in self._recent_autonomous:
                    self._log("Response filter rejected: repetitive autonomous output")
                    return ResponseFilterResult(False, "", "repetitive autonomous output", speakable=False)
                self._recent_autonomous.append(sig)

        speakable = not cleaned.startswith("[")
        self._log(
            f"Response filter accepted: {trigger.source} | emotion={emotion_label} | speakable={speakable}"
        )
        return ResponseFilterResult(True, cleaned, "accepted", speakable=speakable)

    def _normalize(self, text: str) -> str:
        normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
        lines = [self._WHITESPACE_RE.sub(" ", line).strip() for line in normalized.split("\n")]
        return "\n".join(line for line in lines if line).strip()

    def _trim_sentences(self, text: str, max_sentences: int = 2) -> str:
        chunks = re.split(r"(?<=[.!?])\s+", text.strip())
        kept = [chunk.strip() for chunk in chunks if chunk.strip()][:max_sentences]
        return " ".join(kept).strip() or text.strip()

    def _signature(self, text: str) -> str:
        return self._SIGNATURE_RE.sub("", text.lower()).strip()


class ConversationManager:
    def __init__(self, log_fn):
        self._log = log_fn
        self.state_machine = ConversationStateMachine(log_fn)
        self.behavior = BehaviorController(log_fn)
        self.response_filter = ResponseFilter(log_fn)

    @property
    def current_state(self) -> str:
        return self.state_machine.state

    def mark_booting(self, reason: str = "Booting core"):
        self.behavior.mark_startup_phase(ReviaState.BOOTING.value)
        self.state_machine.transition(ReviaState.BOOTING, reason, force=True)

    def mark_initializing(self, reason: str = "Initializing subsystems"):
        self.behavior.mark_startup_phase(ReviaState.INITIALIZING.value)
        self.state_machine.transition(ReviaState.INITIALIZING, reason, force=True)

    def mark_startup_complete(self, reason: str = "Startup complete"):
        self.behavior.mark_startup_complete()
        self.state_machine.transition(ReviaState.IDLE, reason, force=True)

    def transition_state(self, new_state: ReviaState | str, reason: str = "", force: bool = False):
        changed = self.state_machine.transition(new_state, reason, force=force)
        if isinstance(new_state, ReviaState) and changed and new_state == ReviaState.COOLDOWN:
            self.behavior.active_cooldowns()
        return changed

    def maybe_leave_cooldown(self):
        active = self.behavior.active_cooldowns()
        if not active and self.current_state == ReviaState.COOLDOWN.value:
            self.state_machine.transition(ReviaState.IDLE, "Cooldown finished")
        return active

    def build_readiness_snapshot(self, checks: dict[str, SubsystemStatus]) -> ReadinessSnapshot:
        startup_phase = self.behavior.startup_phase()
        startup_complete = self.behavior.startup_complete()
        blocking_reasons = []
        for name, status in checks.items():
            if status.required and not status.ready:
                detail = status.detail or status.state or "not ready"
                blocking_reasons.append(f"{name}: {detail}")
        ready = not blocking_reasons
        return ReadinessSnapshot(
            startup_phase=startup_phase,
            startup_complete=startup_complete,
            checks=checks,
            blocking_reasons=blocking_reasons,
            ready=ready,
            can_start_conversation=ready,
            can_auto_initiate=ready and self.behavior.can_auto_initiate(),
        )

    def behavior_snapshot(self) -> dict:
        cooldowns = self.behavior.active_cooldowns()
        return {
            "cooldowns": cooldowns,
            "startup_phase": self.behavior.startup_phase(),
            "startup_complete": self.behavior.startup_complete(),
            "can_auto_initiate": self.behavior.can_auto_initiate(),
        }
