"""ReasoningAgent — Option A "delegate-mode".

V1 delegates the heavy lifting to the existing :class:`ReplyPlanner`,
which already does intent parsing + emotional routing + AVS-graded
candidate generation + HFL post-processing.  This means:

* The LLM call uses whatever ``LLMBackend`` is currently configured.
* AVS regen-patience and ALE diversity rules keep working unchanged.
* HFL prosody is computed per the active ProfileEngine.

The agent's ``AgentResult.result`` payload contains both the final reply
text and the AVS metadata so the QualityGate / FinalResponseBuilder can
use them without re-running scoring.

If no planner / llm_fn is supplied (e.g. unit tests), the agent falls
back to a stub answer — never crashes the orchestrator.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from .agent_base import Agent, AgentContext


class ReasoningAgent(Agent):
    name = "ReasoningAgent"
    default_timeout_ms = 8000

    def __init__(
        self,
        reply_planner=None,
        model_router=None,
        system_prompt_provider: Optional[Callable[[AgentContext], str]] = None,
        recent_replies_provider: Optional[Callable[[AgentContext], list[str]]] = None,
    ):
        self._planner = reply_planner
        self._router = model_router
        self._system_prompt_provider = system_prompt_provider
        self._recent_replies_provider = recent_replies_provider

    def run(self, context: AgentContext) -> dict[str, Any]:
        context.cancel_token.raise_if_cancelled()

        emotion_label = str(context.metadata.get("emotion_label", "neutral"))
        system_prompt = ""
        if self._system_prompt_provider is not None:
            try:
                system_prompt = str(self._system_prompt_provider(context) or "")
            except Exception:
                system_prompt = ""

        recent_replies: list[str] = []
        if self._recent_replies_provider is not None:
            try:
                recent_replies = list(self._recent_replies_provider(context) or [])
            except Exception:
                recent_replies = []

        # Path A: delegate to the existing ReplyPlanner if injected.
        if self._planner is not None:
            plan = self._planner.plan(
                user_utterance=context.user_text,
                system_prompt=system_prompt,
                emotion_label=emotion_label,
                recent_replies=recent_replies,
            )
            avs_passed = bool(getattr(plan, "avs_results", None)) and \
                any(r.passed for r in plan.avs_results)
            best_score = 0.0
            for r in getattr(plan, "avs_results", []) or []:
                if r.scores.composite > best_score:
                    best_score = float(r.scores.composite)
            return {
                "_confidence": best_score if best_score > 0.0 else (0.6 if plan.final_reply else 0.0),
                "text": plan.final_reply,
                "candidates": list(plan.candidates),
                "avs_passed": avs_passed,
                "avs_best_score": round(best_score, 4),
                "fallback_accept": bool(plan.fallback_accept),
                "emotion_label": plan.emotion_label,
                "intent": plan.intent.to_dict(),
                "elapsed_ms": round(plan.total_elapsed_ms, 2),
                "notes": list(plan.notes),
            }

        # Path B: direct LLM call via the model router.
        if self._router is not None and self._router.has("reason_chat"):
            try:
                text = self._router.call(
                    "reason_chat",
                    context.user_text,
                    None,  # broadcast_fn
                )
            except TypeError:
                # Some handlers do not accept broadcast_fn; retry without it.
                text = self._router.call("reason_chat", context.user_text)
            text = str(text or "")
            return {
                "_confidence": 0.5 if text else 0.0,
                "text": text,
                "candidates": [text] if text else [],
                "avs_passed": False,
                "avs_best_score": 0.0,
                "fallback_accept": True,
                "emotion_label": emotion_label,
                "intent": {},
                "notes": ["direct_router_call"],
            }

        # Path C: deterministic stub (used by tests when nothing is wired).
        stub = f"[stub-reasoning] You said: {context.user_text[:120]}"
        return {
            "_confidence": 0.1,
            "text": stub,
            "candidates": [stub],
            "avs_passed": False,
            "avs_best_score": 0.0,
            "fallback_accept": True,
            "emotion_label": emotion_label,
            "intent": {},
            "notes": ["stub_no_planner_no_router"],
        }
