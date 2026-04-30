"""V4.1 Skill ABC + dataclasses for the SkillRegistry.

A *skill* is an externally-callable capability the ReasoningAgent can
hand off to: a calculator, a clock, a memory recall, a websearch, a
calendar lookup, a code execution sandbox.  Skills are intentionally
*deterministic and offline-friendly* by default — V4 ships only
zero-dep built-ins so REVIA stays usable on a fresh machine.

Each skill exposes:

    * ``name``           - registry key (e.g. ``"calculator"``)
    * ``description``    - one-line capability blurb the ToolUseAgent uses
    * ``triggers``       - regex / keyword patterns that hint when to run
    * ``handle(req)``    - returns a :class:`SkillResponse`

Skills MUST be safe to call without user confirmation when their
``cost_class == "free"``.  Anything that costs money or touches the
network must declare ``cost_class != "free"`` so the orchestrator can
gate it behind a confirmation prompt.
"""
from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SkillRequest:
    """Structured input passed to :meth:`Skill.handle`."""

    user_text: str
    intent: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    arguments: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_text": self.user_text[:200],
            "intent": dict(self.intent),
            "metadata": dict(self.metadata),
            "arguments": dict(self.arguments),
        }


@dataclass
class SkillResponse:
    """Structured output returned by :meth:`Skill.handle`."""

    success: bool
    text: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": bool(self.success),
            "text": self.text,
            "data": dict(self.data),
            "elapsed_ms": round(float(self.elapsed_ms), 2),
            "error": self.error,
        }


@dataclass
class SkillMatch:
    """How well a candidate skill fits a user request."""

    skill_name: str
    score: float
    reasons: list[str] = field(default_factory=list)


class Skill(ABC):
    """Base class for any skill plugged into the SkillRegistry."""

    name: str = "Skill"
    description: str = ""
    cost_class: str = "free"      # "free" | "metered" | "paid"
    triggers: tuple[str, ...] = ()  # regex strings the registry uses to score

    def __init__(self, enabled: bool = True):
        self._enabled = bool(enabled)

    # --- enable/disable -----------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, on: bool) -> None:
        self._enabled = bool(on)

    # --- match -------------------------------------------------------

    def match(self, req: SkillRequest) -> SkillMatch:
        """Score how strongly this skill should answer ``req``.

        Default uses regex triggers; subclasses can override with
        smarter matching (LLM classifier, embedding similarity, etc.).
        """
        if not self._enabled or not req.user_text:
            return SkillMatch(self.name, 0.0)
        text = req.user_text.lower()
        hits = 0
        reasons: list[str] = []
        for pattern in self.triggers:
            try:
                if re.search(pattern, text, re.IGNORECASE):
                    hits += 1
                    reasons.append(f"trigger:{pattern}")
            except re.error:
                continue
        score = min(1.0, hits / max(1, len(self.triggers))) if self.triggers else 0.0
        return SkillMatch(self.name, score=score, reasons=reasons)

    # --- handle (top-level wrapper around _execute) -------------------

    def handle(self, req: SkillRequest) -> SkillResponse:
        if not self._enabled:
            return SkillResponse(success=False, error="skill_disabled")
        t0 = time.monotonic()
        try:
            resp = self._execute(req)
            if resp is None:
                resp = SkillResponse(success=False, error="empty_skill_response")
            resp.elapsed_ms = max(resp.elapsed_ms, (time.monotonic() - t0) * 1000.0)
            return resp
        except Exception as exc:
            return SkillResponse(
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                elapsed_ms=(time.monotonic() - t0) * 1000.0,
            )

    @abstractmethod
    def _execute(self, req: SkillRequest) -> SkillResponse:
        """Subclasses implement the actual capability here."""
        raise NotImplementedError

    # --- introspection ------------------------------------------------

    def info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "cost_class": self.cost_class,
            "triggers": list(self.triggers),
            "enabled": bool(self._enabled),
        }
