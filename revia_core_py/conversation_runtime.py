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
    # PRD section 8 - IHS states
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
            ReviaState.THINKING,  # User-driven responses can interrupt cooldown
            ReviaState.ERROR,
        },
        ReviaState.ERROR: {
            ReviaState.INITIALIZING,
            ReviaState.IDLE,
        },
        # PRD section 8 - IHS recovery transitions
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
        # Monotonic stamp of when the *current* state was entered.  Used by
        # the FSM watchdog (Issue #1) to detect a stuck THINKING state when
        # the pipeline crashes between THINKING -> SPEAKING.
        self._state_entered_at: float = self._updated_at

        # PRD section 8 - IHS extended state fields
        self.interruption_type: str   = ""     # last InterruptionType value
        self.partial_response_spoken: str = "" # text Revia had delivered before barge-in
        self.topic_shift_confidence: float = 0.0

        # PRD section 10/section 12 - AVS / ALE extended state fields
        self.answer_confidence_last: float = 0.0  # last AVS composite score
        self.regen_count: int    = 0   # regens on the current turn
        self.loop_risk_score: float = 0.0  # last ALE loop_risk_score

    @property
    def state(self) -> str:
        with self._lock:
            return self._state.value

    # PRD section 8 IHS helpers

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
            self._state_entered_at = self._updated_at
        self._log(
            f"State change: {current.value} -> {new_state.value}"
            + (f" ({reason})" if reason else "")
        )
        return True

    # ------------------------------------------------------------------
    # Issue #1 — FSM-level watchdog
    # ------------------------------------------------------------------

    def time_in_state(self) -> float:
        """Seconds elapsed since the FSM last transitioned."""
        with self._lock:
            return max(0.0, time.monotonic() - self._state_entered_at)

    def force_recover_if_stuck(
        self,
        *,
        thinking_timeout_s: float = 60.0,
        speaking_timeout_s: float = 90.0,
    ) -> bool:
        """Force-recover the FSM if it has been stuck in a transient state.

        Guards against the failure mode described in REVIA_DEEP_DIVE Issue
        #1: if the pipeline raises between ``THINKING`` and ``SPEAKING``
        (e.g. LLM backend crashes mid-generation), the FSM remains pinned
        in THINKING and ``BehaviorController.evaluate()`` rejects every
        new RESPONSE-kind trigger as ``state=Thinking`` until the server
        is restarted.

        Returns ``True`` when a forced transition occurred, ``False``
        when the FSM is in a healthy state.

        The transition path is THINKING/SPEAKING -> ERROR -> IDLE (forced
        through validation), mirroring the behaviour of
        :class:`TurnWatchdog` so observers see the same recovery story.
        """
        with self._lock:
            current = self._state
            elapsed = time.monotonic() - self._state_entered_at
            if current == ReviaState.THINKING:
                threshold = float(thinking_timeout_s)
                if elapsed < threshold:
                    return False
                stuck_state = current
            elif current == ReviaState.SPEAKING:
                threshold = float(speaking_timeout_s)
                if elapsed < threshold:
                    return False
                stuck_state = current
            else:
                return False

        self._log(
            f"FSM watchdog: forcing recovery from stuck {stuck_state.value} "
            f"after {elapsed:.1f}s (threshold={threshold:.0f}s)"
        )
        # ERROR is an explicit allowed target from THINKING/SPEAKING; IDLE
        # is reachable from ERROR.  Both are forced so the recovery cannot
        # itself be rejected by the validation rules.
        self.transition(ReviaState.ERROR, reason="fsm_watchdog_recovery", force=True)
        self.transition(ReviaState.IDLE, reason="fsm_watchdog_recovery", force=True)
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
        thinking_timeout_s: float = 60.0,
        speaking_timeout_s: float = 90.0,
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
        # Issue #1 — watchdog timeouts for stuck FSM recovery.  Wired by
        # ConversationManager via :meth:`bind_state_machine` so evaluate()
        # can force-recover a permanently-pinned THINKING/SPEAKING state
        # before rejecting an inbound user trigger.
        self._thinking_timeout_s = float(thinking_timeout_s)
        self._speaking_timeout_s = float(speaking_timeout_s)
        self._state_machine: ConversationStateMachine | None = None

    def bind_state_machine(self, state_machine: "ConversationStateMachine") -> None:
        """Wire the FSM so :meth:`evaluate` can run the stuck-state watchdog."""
        self._state_machine = state_machine

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

        # Issue #1 — FSM watchdog: if the state machine has been pinned in
        # THINKING (or, defensively, SPEAKING) past the configured timeout,
        # force a recovery to IDLE before evaluating the trigger.  Without
        # this, a pipeline crash between THINKING and SPEAKING leaves every
        # subsequent user message rejected as "state=Thinking" until the
        # server is restarted.
        if self._state_machine is not None:
            try:
                recovered = self._state_machine.force_recover_if_stuck(
                    thinking_timeout_s=self._thinking_timeout_s,
                    speaking_timeout_s=self._speaking_timeout_s,
                )
                if recovered:
                    current_state = self._state_machine.state
            except Exception as _exc:
                # Watchdog must never raise into the trigger path.
                self._log(
                    f"FSM watchdog raised: {type(_exc).__name__}: {_exc}"
                )

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
            # User-driven replies should remain conversational: let the
            # controller interrupt any existing speech first, and do not gate
            # the human behind pacing cooldowns. Only an actively thinking turn
            # remains a hard block.
            if current_state == ReviaState.THINKING.value:
                return self._blocked(trigger, f"state={current_state}")

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
        # Build a compact trigger-context header reused by all log lines below.
        # Knowing which trigger produced a rejected output (and a peek at the
        # text) makes post-mortems possible without diffing the full pipeline.
        raw_preview = str(text or "").replace("\n", " ").strip()
        if len(raw_preview) > 80:
            raw_preview = raw_preview[:77] + "..."
        trig_ctx = (
            f"source={trigger.source} | kind={trigger.kind} "
            f"| emotion={emotion_label} | reason={trigger.reason}"
        )

        cleaned = self._normalize(text)
        if not cleaned:
            fallback = ""
            if str(trigger.kind) == TriggerKind.RESPONSE.value:
                fallback = "I need another second to finish lining that up."
            self._log(
                f"Response filter rejected: empty output | {trig_ctx} "
                f"| input_len={len(str(text or ''))}"
            )
            return ResponseFilterResult(False, fallback, "empty output", speakable=False)

        # Startup greetings are exempt from the autonomous repetition filter
        # — they naturally repeat across sessions and that's expected behavior.
        is_startup_greeting = (
            str(trigger.source) == TriggerSource.STARTUP.value
            or str(trigger.reason or "").startswith("startup")
        )

        if str(trigger.kind) == TriggerKind.AUTONOMOUS.value and not is_startup_greeting:
            cleaned = self._trim_sentences(cleaned, max_sentences=2)
            if cleaned.startswith("["):
                self._log(
                    f"Response filter rejected: autonomous error output | "
                    f"{trig_ctx} | preview={raw_preview!r}"
                )
                return ResponseFilterResult(False, "", "autonomous error output", speakable=False)
            sig = self._signature(cleaned)
            with self._lock:
                if sig in self._recent_autonomous:
                    self._log(
                        f"Response filter rejected: repetitive autonomous output "
                        f"| {trig_ctx} | sig={sig[:8]} | preview={raw_preview!r}"
                    )
                    return ResponseFilterResult(False, "", "repetitive autonomous output", speakable=False)
                self._recent_autonomous.append(sig)

        speakable = not cleaned.startswith("[")
        self._log(
            f"Response filter accepted: {trig_ctx} | speakable={speakable} "
            f"| chars={len(cleaned)}"
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
        self.behavior.bind_state_machine(self.state_machine)
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

    def transition_state(self, new_state, reason: str = "", force: bool = False):
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
