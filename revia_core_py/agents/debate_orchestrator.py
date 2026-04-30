"""V4.4 DebateOrchestrator — multi-variant reasoning with critic-picked winner.

For high-stakes turns (commands, factual questions, complaints) the
caller can run several :class:`ReasoningAgent` variants in parallel,
each with a different system-prompt slant, and let the
:class:`CriticAgent` pick the strongest candidate.  Falls back to the
single-variant result when only one variant is provided.

The class is *opt-in*: the standard :class:`AgentOrchestrator` does not
spawn debate by default; callers wrap a turn with
:meth:`DebateOrchestrator.run_debate` to use it.

Each variant dict has the shape::

    {"name": "concise", "system_prompt": "Be terse and concrete.", "rank": 10}

The orchestrator yields a structured result with every variant + the
winner pick + the score gap.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from typing import Any, Iterable

from .agent_base import AgentContext

_log = logging.getLogger(__name__)


@dataclass
class DebateVariantResult:
    name: str
    system_prompt: str = ""
    text: str = ""
    success: bool = False
    confidence: float = 0.0
    critic_score: float = 0.0
    critic_recommendation: str = ""
    elapsed_ms: float = 0.0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "system_prompt": self.system_prompt[:160],
            "text": self.text,
            "success": bool(self.success),
            "confidence": round(float(self.confidence), 4),
            "critic_score": round(float(self.critic_score), 4),
            "critic_recommendation": self.critic_recommendation,
            "elapsed_ms": round(float(self.elapsed_ms), 2),
            "error": self.error,
        }


@dataclass
class DebateOutput:
    variants: list[DebateVariantResult] = field(default_factory=list)
    winner: str = ""
    winner_text: str = ""
    score_gap: float = 0.0
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "variants": [v.to_dict() for v in self.variants],
            "winner": self.winner,
            "winner_text": self.winner_text,
            "score_gap": round(float(self.score_gap), 4),
            "elapsed_ms": round(float(self.elapsed_ms), 2),
        }


class DebateOrchestrator:
    """Run N reasoning variants in parallel and pick the critic's best.

    The class does NOT own a reasoning agent or critic — it accepts them
    via constructor injection so it stays testable and composable.
    """

    def __init__(
        self,
        reasoning_agent,
        critic_agent,
        executor: ThreadPoolExecutor | None = None,
        max_workers: int = 3,
        per_variant_timeout_s: float = 12.0,
        log_fn=None,
    ):
        self._reasoning = reasoning_agent
        self._critic = critic_agent
        self._owns_executor = executor is None
        self._executor = executor or ThreadPoolExecutor(
            max_workers=max(2, int(max_workers)),
            thread_name_prefix="revia-debate",
        )
        self._timeout_s = float(per_variant_timeout_s)
        self._log = log_fn or _log.info

    def run_debate(
        self,
        context: AgentContext,
        variants: Iterable[dict[str, Any]],
    ) -> DebateOutput:
        variant_list = list(variants or [])
        t0 = time.monotonic()
        if not variant_list:
            return DebateOutput(elapsed_ms=0.0)

        # Phase 1: run reasoning per variant (parallel).
        futures = {
            v.get("name", f"v{idx}"): self._executor.submit(
                self._run_variant, context, v,
            )
            for idx, v in enumerate(variant_list)
        }
        variant_results: list[DebateVariantResult] = []
        for name, fut in futures.items():
            try:
                variant_results.append(fut.result(timeout=self._timeout_s))
            except FuturesTimeoutError:
                fut.cancel()
                variant_results.append(DebateVariantResult(
                    name=name, error=f"timeout after {int(self._timeout_s * 1000)}ms",
                ))
            except Exception as exc:
                variant_results.append(DebateVariantResult(
                    name=name, error=f"{type(exc).__name__}: {exc}",
                ))

        # Phase 2: critic on each successful candidate.
        for v in variant_results:
            if not v.success or not v.text:
                continue
            critic_score, recommendation = self._score_with_critic(context, v.text)
            v.critic_score = critic_score
            v.critic_recommendation = recommendation

        # Pick winner.
        winner = self._pick_winner(variant_results)
        winner_text = winner.text if winner else ""
        gap = self._score_gap(variant_results)

        return DebateOutput(
            variants=variant_results,
            winner=winner.name if winner else "",
            winner_text=winner_text,
            score_gap=gap,
            elapsed_ms=(time.monotonic() - t0) * 1000.0,
        )

    def shutdown(self) -> None:
        if self._owns_executor:
            try:
                self._executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:  # pragma: no cover
                self._executor.shutdown(wait=False)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run_variant(
        self,
        context: AgentContext,
        variant: dict[str, Any],
    ) -> DebateVariantResult:
        name = str(variant.get("name") or "variant")
        sys_prompt = str(variant.get("system_prompt") or "")
        t0 = time.monotonic()

        # Inject the variant's system prompt into a copy of metadata so
        # the ReasoningAgent's system_prompt_provider (or planner) can use it.
        variant_meta = dict(context.metadata or {})
        variant_meta["debate_variant"] = name
        variant_meta["debate_system_prompt"] = sys_prompt
        variant_ctx = AgentContext(
            user_text=context.user_text,
            turn_id=f"{context.turn_id}#{name}",
            conversation_id=context.conversation_id,
            user_profile=context.user_profile,
            response_threshold=context.response_threshold,
            cancel_token=context.cancel_token,
            metadata=variant_meta,
        )

        result = self._reasoning.execute(variant_ctx)
        text = ""
        success = False
        confidence = 0.0
        if result.success:
            payload = result.result or {}
            text = str(payload.get("text") or "").strip()
            success = bool(text)
            confidence = float(result.confidence)

        return DebateVariantResult(
            name=name,
            system_prompt=sys_prompt,
            text=text,
            success=success,
            confidence=confidence,
            elapsed_ms=(time.monotonic() - t0) * 1000.0,
            error=result.error or "",
        )

    def _score_with_critic(
        self,
        context: AgentContext,
        candidate_text: str,
    ) -> tuple[float, str]:
        if self._critic is None:
            return 0.0, ""
        critic_meta = dict(context.metadata or {})
        critic_meta["candidate_text"] = candidate_text
        critic_ctx = AgentContext(
            user_text=context.user_text,
            turn_id=context.turn_id + "#critic",
            cancel_token=context.cancel_token,
            metadata=critic_meta,
        )
        result = self._critic.execute(critic_ctx)
        if not result.success:
            return 0.0, ""
        payload = result.result or {}
        return (
            float(payload.get("score", 0.0)),
            str(payload.get("recommendation", "")),
        )

    @staticmethod
    def _pick_winner(
        variants: list[DebateVariantResult],
    ) -> DebateVariantResult | None:
        usable = [v for v in variants if v.success]
        if not usable:
            return None
        # Composite: critic_score weighted above raw confidence.
        usable.sort(
            key=lambda v: (
                # Reject candidates whose critic_recommendation == "regen".
                0 if v.critic_recommendation == "regen" else 1,
                v.critic_score,
                v.confidence,
            ),
            reverse=True,
        )
        return usable[0]

    @staticmethod
    def _score_gap(variants: list[DebateVariantResult]) -> float:
        scores = sorted(
            (v.critic_score for v in variants if v.success), reverse=True,
        )
        if len(scores) < 2:
            return 0.0
        return float(scores[0] - scores[1])
