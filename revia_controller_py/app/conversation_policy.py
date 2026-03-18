from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BehaviorDecision:
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
        if status.get("state") not in ("Speaking", "Thinking"):
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
            return self._blocked(source, blocking_reasons[0] if blocking_reasons else "system not ready")

        if autonomous and not readiness.get("can_auto_initiate", False) and not force:
            return self._blocked(source, "startup warmup or autonomous gate is active")

        if autonomous:
            if self._is_user_speaking():
                return self._blocked(source, "user is speaking")
            if self._is_assistant_speaking():
                return self._blocked(source, "assistant is already speaking")
            if runtime_state != "Idle":
                return self._blocked(source, f"state={runtime_state}")
            remaining = float(cooldowns.get("autonomous", 0.0) or 0.0)
            if remaining > 0.0 and not force:
                return self._blocked(source, f"autonomous cooldown active ({remaining:.1f}s)")
        else:
            if self._is_assistant_speaking():
                return self._blocked(source, "assistant is already speaking")
            if runtime_state in ("Thinking", "Speaking", "Cooldown"):
                return self._blocked(source, f"state={runtime_state}")
            remaining = float(cooldowns.get("response", 0.0) or 0.0)
            if remaining > 0.0 and not force:
                return self._blocked(source, f"response cooldown active ({remaining:.1f}s)")

        return self._allowed(source, reason)

    def _allowed(self, source: str, reason: str) -> BehaviorDecision:
        self._log(f"[Revia] Allowed: {source} ({reason})")
        return BehaviorDecision(True, reason)

    def _blocked(self, source: str, reason: str) -> BehaviorDecision:
        self._log(f"[Revia] Blocked: {source} ({reason})")
        return BehaviorDecision(False, reason)
