"""Interface ABC + dataclasses for the V2.4 InterfaceRouter.

An *interface* is an output channel — the way a single canonical Revia
answer reaches the user.  Each interface subclasses :class:`Interface`,
exposes ``deliver(ctx)``, and is registered with an
:class:`InterfaceRouter`.

Per-spec rule: the orchestrator builds ONE final response.  The router
then fans it out to every enabled+accepting channel (text, voice, vision,
notification, audit log).  Channels never mutate the answer; they only
*present* it.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class InterfaceContext:
    """Read-only context handed to every interface during dispatch.

    ``final`` is whatever the FinalResponseBuilder produced — a duck-typed
    object with at least ``text``, ``emotion_label``, ``prosody`` and
    ``voice_style`` attributes.  Tests can substitute a plain dict.
    """

    final: Any
    user_text: str = ""
    intent: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    cancel_token: Any = None

    @property
    def text(self) -> str:
        if hasattr(self.final, "text"):
            return str(getattr(self.final, "text") or "")
        if isinstance(self.final, dict):
            return str(self.final.get("text") or "")
        return ""

    @property
    def emotion_label(self) -> str:
        if hasattr(self.final, "emotion_label"):
            return str(getattr(self.final, "emotion_label") or "neutral")
        if isinstance(self.final, dict):
            return str(self.final.get("emotion_label") or "neutral")
        return "neutral"

    @property
    def prosody(self) -> dict[str, Any]:
        if hasattr(self.final, "prosody"):
            return dict(getattr(self.final, "prosody") or {})
        if isinstance(self.final, dict):
            return dict(self.final.get("prosody") or {})
        return {}

    @property
    def voice_style(self) -> dict[str, Any]:
        if hasattr(self.final, "voice_style"):
            return dict(getattr(self.final, "voice_style") or {})
        if isinstance(self.final, dict):
            return dict(self.final.get("voice_style") or {})
        return {}


@dataclass
class InterfaceDecision:
    """Result of one interface's ``execute`` call."""

    interface: str
    kind: str = "text"
    delivered: bool = False
    elapsed_ms: float = 0.0
    skipped_reason: str = ""
    error: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface": self.interface,
            "kind": self.kind,
            "delivered": bool(self.delivered),
            "elapsed_ms": round(float(self.elapsed_ms), 2),
            "skipped_reason": self.skipped_reason,
            "error": self.error,
            "payload": dict(self.payload),
        }


# ---------------------------------------------------------------------------
# Interface ABC
# ---------------------------------------------------------------------------

class Interface(ABC):
    """Base class for every output channel.

    Subclasses implement :meth:`deliver`.  The base class wraps it in
    :meth:`execute` to enforce uniform timing, accept-gating, and
    error capture so the router never has to special-case any channel.
    """

    name: str = "Interface"
    kind: str = "text"   # "text" | "voice" | "vision" | "system"

    def __init__(self, enabled: bool = True):
        self._enabled = bool(enabled)

    # --- enable/disable -----------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, on: bool) -> None:
        self._enabled = bool(on)

    # --- accept gate (override-able) ----------------------------------

    def accept(self, ctx: InterfaceContext) -> tuple[bool, str]:
        """Return ``(accept, skip_reason)``.  Default: accept iff enabled."""
        if not self._enabled:
            return False, "disabled"
        if not (ctx.text or "").strip():
            return False, "empty_text"
        return True, ""

    @abstractmethod
    def deliver(self, ctx: InterfaceContext) -> dict[str, Any]:
        """Send / render / play the canonical response.

        Implementations must NOT mutate ``ctx``.  Return a small payload
        dict for audit (e.g. ``{"emitted": 42}``).  Raising an exception
        marks the decision as failed; the router moves on.
        """
        raise NotImplementedError

    # --- top-level wrapper --------------------------------------------

    def execute(self, ctx: InterfaceContext) -> InterfaceDecision:
        t0 = time.monotonic()
        ok, reason = self.accept(ctx)
        if not ok:
            return InterfaceDecision(
                interface=self.name,
                kind=self.kind,
                delivered=False,
                elapsed_ms=(time.monotonic() - t0) * 1000.0,
                skipped_reason=reason,
            )
        try:
            payload = self.deliver(ctx) or {}
            return InterfaceDecision(
                interface=self.name,
                kind=self.kind,
                delivered=True,
                elapsed_ms=(time.monotonic() - t0) * 1000.0,
                payload=payload if isinstance(payload, dict) else {"value": payload},
            )
        except Exception as exc:
            return InterfaceDecision(
                interface=self.name,
                kind=self.kind,
                delivered=False,
                elapsed_ms=(time.monotonic() - t0) * 1000.0,
                error=f"{type(exc).__name__}: {exc}",
            )
