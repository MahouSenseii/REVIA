"""V4.2 ToolUseAgent — decides if a turn needs a skill, dispatches to the registry.

Runs in parallel with the other pre-agents.  When the
:class:`SkillRegistry` finds a high-confidence match the agent emits a
``SkillResponse`` payload that the :class:`ReasoningAgent` can fold into
its prompt (or surface verbatim if the skill is sufficient on its own).

Output schema::

    {
        "used_skill": str | "",
        "score": 0..1,
        "result_text": str,         # the skill's text output
        "result_data": dict,        # the skill's structured data
        "candidates": [{name, score, reasons}, ...],
        "elapsed_ms": float,
    }
"""
from __future__ import annotations

import time
from typing import Any

from .agent_base import Agent, AgentContext

try:  # pragma: no cover - both package + direct contexts
    from ..skills import SkillRegistry, SkillRequest
except ImportError:  # pragma: no cover
    from skills import SkillRegistry, SkillRequest  # type: ignore[no-redef]


class ToolUseAgent(Agent):
    """Lightweight gateway between the agent layer and the SkillRegistry.

    Returns a successful but empty payload when no skill matches — the
    orchestrator can simply ignore it and proceed with reasoning.
    """

    name = "ToolUseAgent"
    default_timeout_ms = 600

    def __init__(
        self,
        registry: SkillRegistry | None = None,
        explicit_skill_metadata_key: str = "skill",
    ):
        self._registry = registry
        self._explicit_key = explicit_skill_metadata_key

    def run(self, context: AgentContext) -> dict[str, Any]:
        context.cancel_token.raise_if_cancelled()

        if self._registry is None:
            return {
                "_confidence": 0.0,
                "used_skill": "",
                "score": 0.0,
                "result_text": "",
                "result_data": {},
                "candidates": [],
                "elapsed_ms": 0.0,
                "note": "no_registry",
            }

        intent = context.metadata.get("intent") or {}
        explicit = str(context.metadata.get(self._explicit_key) or "") or None

        req = SkillRequest(
            user_text=context.user_text or "",
            intent=intent if isinstance(intent, dict) else {},
            metadata=dict(context.metadata or {}),
            arguments=dict(context.metadata.get("skill_arguments") or {}),
        )

        t0 = time.monotonic()
        result = self._registry.dispatch(req, explicit_skill=explicit)
        elapsed = (time.monotonic() - t0) * 1000.0

        candidates_payload = [
            {"name": c.skill_name, "score": round(c.score, 4),
             "reasons": list(c.reasons)}
            for c in result.candidates
        ]

        if not result.chosen:
            return {
                "_confidence": 0.0,
                "used_skill": "",
                "score": 0.0,
                "result_text": "",
                "result_data": {},
                "candidates": candidates_payload,
                "elapsed_ms": round(elapsed, 2),
                "note": "no_skill_above_threshold",
            }

        resp = result.response
        success = bool(resp and resp.success)
        confidence = float(result.score) if success else 0.0
        return {
            "_confidence": confidence,
            "used_skill": result.chosen,
            "score": round(float(result.score), 4),
            "result_text": (resp.text if resp else ""),
            "result_data": (dict(resp.data) if resp else {}),
            "candidates": candidates_payload,
            "success": success,
            "error": (resp.error if resp else ""),
            "elapsed_ms": round(elapsed, 2),
        }
