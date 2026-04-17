from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from app.revia_states import (
    STATE_IDLE,
    STATE_THINKING,
    INTERRUPTIBLE_STATES,
)


@dataclass
class BehaviorDecision:
    """Controller-side behaviour gate result.

    Note: the core server has a richer ``BehaviorDecision`` in
    ``revia_core_py.conversation_runtime`` that additionally carries
    ``trigger_kind``, ``trigger_source``, ``cooldown_name``, and
    ``cooldown_remaining_s``.  Keep the two definitions in sync when adding
    fields that are relevant to both layers.
    """

    allowed: bool
    reason: str


class ConversationBehaviorController:
    def __init__(
        self,
        status_provider,
        log_fn,
        *,
        is_user_speaking=None,
        is_assistant_speaking=None,
        is_tts_enabled=None,
        is_tts_ready=None,
        is_stt_ready=None,
    ):
        self._status_provider = status_provider
        self._log = log_fn
        self._is_user_speaking = is_user_speaking or (lambda: False)
        self._is_assistant_speaking = is_assistant_speaking or (lambda: False)
        self._is_tts_enabled = is_tts_enabled or (lambda: False)
        self._is_tts_ready = is_tts_ready or (lambda: True)
        self._is_stt_ready = is_stt_ready or (lambda: True)
        # Throttle state for _blocked: prevents "Blocked: Startup (state=Thinking)"
        # from spamming the log tab every telemetry tick when the FSM is pinned.
        self._block_log_lock = threading.Lock()
        self._block_log_last_ts: dict[tuple[str, str], float] = {}
        self._block_log_suppressed: dict[tuple[str, str], int] = {}
        self._block_log_cooldown_s = 30.0

    def should_respond(
        self,
        *,
        source: str,
        reason: str,
        require_voice_input: bool = False,
        require_speech_output: bool = False,
    ) -> BehaviorDecision:
        return self._evaluate(
            source=source,
            reason=reason,
            autonomous=False,
            require_voice_input=require_voice_input,
            require_speech_output=require_speech_output,
        )

    def should_initiate_conversation(
        self,
        *,
        source: str,
        reason: str,
        force: bool = False,
        require_speech_output: bool = False,
    ) -> BehaviorDecision:
        return self._evaluate(
            source=source,
            reason=reason,
            autonomous=True,
            require_voice_input=False,
            require_speech_output=require_speech_output,
            force=force,
        )

    def should_interrupt(self, *, source: str, reason: str) -> BehaviorDecision:
        status = self._status_provider() or {}
        if not self._is_assistant_speaking():
            return self._blocked(source, "nothing is speaking")
        if status.get("state") not in INTERRUPTIBLE_STATES:
            return self._blocked(source, "assistant is not interruptible right now")
        return self._allowed(source, reason)

    def can_speak_now(self, *, autonomous: bool = False, require_speech_output: bool = False) -> BehaviorDecision:
        return self._evaluate(
            source="BehaviorCheck",
            reason="speak now check",
            autonomous=autonomous,
            require_voice_input=False,
            require_speech_output=require_speech_output,
        )

    def _evaluate(
        self,
        *,
        source: str,
        reason: str,
        autonomous: bool,
        require_voice_input: bool,
        require_speech_output: bool,
        force: bool = False,
    ) -> BehaviorDecision:
        status = self._status_provider() or {}
        if not status:
            return self._blocked(source, "core status unavailable")

        readiness = status.get("conversation_readiness", {}) or {}
        llm_state = ((status.get("llm_connection", {}) or {}).get("state") or "Disconnected").strip()
        runtime_state = (status.get("state") or "Booting").strip()
        behavior = status.get("behavior", {}) or {}
        cooldowns = behavior.get("cooldowns", {}) or {}
        blocking_reasons = list(readiness.get("blocking_reasons", []) or [])

        if llm_state != "Ready":
            blocking_reasons.insert(0, f"llm: {llm_state}")

        if require_voice_input and not self._is_stt_ready():
            blocking_reasons.insert(0, "stt: local speech input is not ready")

        if require_speech_output and self._is_tts_enabled() and not self._is_tts_ready():
            blocking_reasons.insert(0, "tts: local speech output is not ready")

        if not readiness.get("can_start_conversation", False):
            return self._blocked(source, blocking_reasons[0] if blocking_reasons else "system not ready", status)

        if autonomous and not readiness.get("can_auto_initiate", False) and not force:
            return self._blocked(source, "startup warmup or autonomous gate is active", status)

        if autonomous:
            if self._is_user_speaking():
                return self._blocked(source, "user is speaking", status)
            if self._is_assistant_speaking():
                return self._blocked(source, "assistant is already speaking", status)
            if runtime_state != STATE_IDLE:
                return self._blocked(source, f"state={runtime_state}", status)
            remaining = float(cooldowns.get("autonomous", 0.0) or 0.0)
            if remaining > 0.0 and not force:
                return self._blocked(source, f"autonomous cooldown active ({remaining:.1f}s)", status)
        else:
            # User-initiated messages should feel like a real conversation:
            # the user can always send, can interrupt Revia mid-sentence,
            # and is never gated by pacing cooldowns (those exist to pace
            # Revia's autonomous speech, not the human on the other end).
            #
            # The only legitimate block is THINKING - a request is already
            # in-flight and we'd race with it. SPEAKING is handled by the
            # chat panel (it interrupts before sending). COOLDOWN is
            # explicitly non-blocking here.
            if runtime_state == STATE_THINKING:
                return self._blocked(source, f"state={runtime_state}", status)

        return self._allowed(source, reason)

    def _allowed(self, source: str, reason: str) -> BehaviorDecision:
        self._log(f"[Revia] Allowed: {source} ({reason})")
        return BehaviorDecision(True, reason)

    def _blocked(self, source: str, reason: str, status: dict | None = None) -> BehaviorDecision:
        # Pull extra context so the first log line tells you *why* the block
        # fired, not just "state=Thinking". Re-use the status snapshot passed
        # from _evaluate() when available so the log reflects the exact state
        # that triggered the block — not a potentially stale re-fetch.
        extras = []
        try:
            status = (status or self._status_provider()) or {}
            readiness = status.get("conversation_readiness", {}) or {}
            llm = status.get("llm_connection", {}) or {}
            behavior = status.get("behavior", {}) or {}
            cooldowns = behavior.get("cooldowns", {}) or {}
            blocking = list(readiness.get("blocking_reasons", []) or [])
            runtime_state = status.get("state") or "?"
            extras.append(f"runtime_state={runtime_state}")
            if llm.get("state") and llm.get("state") != "Ready":
                extras.append(f"llm={llm.get('state')}")
            if cooldowns:
                cd = ",".join(f"{k}={v:.1f}s" for k, v in cooldowns.items())
                extras.append(f"cooldowns=[{cd}]")
            if blocking:
                extras.append(f"blocking={blocking[0]}")
            if self._is_user_speaking():
                extras.append("user_speaking=true")
            if self._is_assistant_speaking():
                extras.append("assistant_speaking=true")
        except Exception as e:
            extras.append(f"(context gather failed: {type(e).__name__})")

        enriched = (
            f"[Revia] Blocked: {source} ({reason})"
            + (f" | {' | '.join(extras)}" if extras else "")
        )

        # Throttle repeats of the same (source, reason) tuple. The FSM can
        # rearm the startup autonomous line on every telemetry update, which
        # otherwise produces one "Blocked: Startup" per tick.
        key = (str(source), str(reason))
        now = time.monotonic()
        should_log = True
        suppressed = 0
        with self._block_log_lock:
            last = self._block_log_last_ts.get(key, 0.0)
            if last > 0.0 and (now - last) < self._block_log_cooldown_s:
                self._block_log_suppressed[key] = (
                    self._block_log_suppressed.get(key, 0) + 1
                )
                should_log = False
            else:
                suppressed = self._block_log_suppressed.pop(key, 0)
                self._block_log_last_ts[key] = now

        if should_log:
            if suppressed > 0:
                self._log(
                    f"{enriched} (+{suppressed} repeats suppressed in last "
                    f"{int(self._block_log_cooldown_s)}s)"
                )
            else:
                self._log(enriched)
        return BehaviorDecision(False, reason)
