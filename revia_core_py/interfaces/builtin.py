"""Built-in interfaces: text chat, voice TTS, vision avatar, audit log, notification.

Each one is a thin :class:`Interface` subclass that calls a host-supplied
callback (so we don't need a hard dependency on the GUI / TTS / Avatar
modules from inside the agent layer).
"""
from __future__ import annotations

from typing import Any, Callable

from .base import Interface, InterfaceContext


# ---------------------------------------------------------------------------
# Text chat — always-on canonical channel.
# ---------------------------------------------------------------------------

class TextChatInterface(Interface):
    """Sends the canonical text to the chat panel via a broadcast callback.

    The callback signature mirrors the existing ``broadcast_fn`` used by
    ``LLMBackend.generate_response``::

        broadcast_fn(channel: str, payload: dict) -> None
    """

    name = "text_chat"
    kind = "text"

    def __init__(
        self,
        broadcast_fn: Callable[..., None] | None = None,
        enabled: bool = True,
    ):
        super().__init__(enabled=enabled)
        self._broadcast = broadcast_fn or (lambda *a, **k: None)

    def deliver(self, ctx: InterfaceContext) -> dict[str, Any]:
        text = ctx.text.strip()
        payload = {
            "text": text,
            "emotion": ctx.emotion_label,
            "voice_style": ctx.voice_style,
        }
        try:
            self._broadcast("chat:text", payload)
        except TypeError:
            # Some legacy callbacks expect (channel, **kwargs).
            self._broadcast("chat:text", **payload)
        return {"emitted_chars": len(text)}


# ---------------------------------------------------------------------------
# Voice TTS — opt-in.
# ---------------------------------------------------------------------------

class VoiceInterface(Interface):
    """Sends the answer to a TTS engine.

    The TTS callback receives ``(text, rate, pitch_shift_st, emotion)``
    and returns a truthy value on success.  If no callback is wired we
    skip with reason ``no_tts_callback``.
    """

    name = "voice"
    kind = "voice"

    # Intent labels that should NEVER trigger TTS.
    SKIP_INTENTS: frozenset[str] = frozenset(
        {"system_log", "audit", "internal"}
    )

    def __init__(
        self,
        speak_fn: Callable[..., Any] | None = None,
        enabled: bool = False,
        skip_intents: frozenset[str] | None = None,
    ):
        super().__init__(enabled=enabled)
        self._speak = speak_fn
        self._skip_intents = frozenset(skip_intents) if skip_intents else self.SKIP_INTENTS

    def accept(self, ctx: InterfaceContext) -> tuple[bool, str]:
        ok, reason = super().accept(ctx)
        if not ok:
            return ok, reason
        intent_label = self._intent_label(ctx).lower()
        if intent_label in self._skip_intents:
            return False, f"intent_{intent_label}_skips_voice"
        if self._speak is None:
            return False, "no_tts_callback"
        return True, ""

    def deliver(self, ctx: InterfaceContext) -> dict[str, Any]:
        prosody = ctx.prosody
        rate = float(prosody.get("speech_rate", 1.0))
        pitch = float(prosody.get("pitch_shift_st", 0.0))
        ok = self._speak(
            ctx.text,
            rate=rate,
            pitch_shift_st=pitch,
            emotion=ctx.emotion_label,
        )
        return {
            "ok": bool(ok),
            "rate": round(rate, 3),
            "pitch_shift_st": round(pitch, 2),
            "emotion": ctx.emotion_label,
        }

    @staticmethod
    def _intent_label(ctx: InterfaceContext) -> str:
        intent = ctx.intent or {}
        if isinstance(intent, dict):
            return str(intent.get("label") or "")
        return ""


# ---------------------------------------------------------------------------
# Vision / Avatar — opt-in.
# ---------------------------------------------------------------------------

class VisionInterface(Interface):
    """Drives an avatar / visual cue layer based on emotion + prosody."""

    name = "vision"
    kind = "vision"

    def __init__(
        self,
        avatar_fn: Callable[..., Any] | None = None,
        enabled: bool = False,
    ):
        super().__init__(enabled=enabled)
        self._avatar = avatar_fn

    def accept(self, ctx: InterfaceContext) -> tuple[bool, str]:
        if not self._enabled:
            return False, "disabled"
        if self._avatar is None:
            return False, "no_avatar_callback"
        return True, ""

    def deliver(self, ctx: InterfaceContext) -> dict[str, Any]:
        prosody = ctx.prosody
        tone = str(prosody.get("tone") or ctx.voice_style.get("tone") or "warm")
        emotion = ctx.emotion_label
        self._avatar(emotion=emotion, tone=tone, text=ctx.text)
        return {"emotion": emotion, "tone": tone}


# ---------------------------------------------------------------------------
# Audit log — always on.
# ---------------------------------------------------------------------------

class LogInterface(Interface):
    """Append a one-line audit entry per turn.

    Always accepts (even on empty text) so we record blocked turns too.
    """

    name = "audit_log"
    kind = "system"

    def __init__(
        self,
        log_fn: Callable[[str], None] | None = None,
        enabled: bool = True,
    ):
        super().__init__(enabled=enabled)
        self._log = log_fn or (lambda *_a, **_k: None)

    def accept(self, ctx: InterfaceContext) -> tuple[bool, str]:
        # Audit channel runs even when the orchestrator yielded no text.
        return (self._enabled, "" if self._enabled else "disabled")

    def deliver(self, ctx: InterfaceContext) -> dict[str, Any]:
        intent_label = ""
        if isinstance(ctx.intent, dict):
            intent_label = str(ctx.intent.get("label") or "")
        text = (ctx.text or "")[:160].replace("\n", " ")
        line = (
            f"[Interface] turn={ctx.metadata.get('turn_id', '')} "
            f"intent={intent_label} emotion={ctx.emotion_label} "
            f"text={text!r}"
        )
        self._log(line)
        return {"line_chars": len(line)}


# ---------------------------------------------------------------------------
# Notification — opt-in, intent-gated.
# ---------------------------------------------------------------------------

class NotificationInterface(Interface):
    """Sends a system notification on selected intents only.

    Default trigger set targets *priority* user states (complaint /
    emotional share) so REVIA can flag attention even while the user is
    not in the chat tab.
    """

    name = "notification"
    kind = "system"

    DEFAULT_INTENTS: frozenset[str] = frozenset(
        {"complaint", "emotional_share"}
    )

    def __init__(
        self,
        notify_fn: Callable[..., Any] | None = None,
        enabled: bool = False,
        intents: frozenset[str] | None = None,
    ):
        super().__init__(enabled=enabled)
        self._notify = notify_fn
        self._intents = frozenset(intents) if intents else self.DEFAULT_INTENTS

    def accept(self, ctx: InterfaceContext) -> tuple[bool, str]:
        ok, reason = super().accept(ctx)
        if not ok:
            return ok, reason
        if self._notify is None:
            return False, "no_notify_callback"
        intent = ctx.intent if isinstance(ctx.intent, dict) else {}
        label = str(intent.get("label") or "").lower()
        if label not in self._intents:
            return False, f"intent_{label or 'unknown'}_not_in_trigger_set"
        return True, ""

    def deliver(self, ctx: InterfaceContext) -> dict[str, Any]:
        title = f"REVIA · {ctx.emotion_label}"
        ok = self._notify(title=title, body=ctx.text[:280])
        return {"ok": bool(ok), "title": title}
