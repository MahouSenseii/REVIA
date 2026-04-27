"""AgentOrchestrator — fan-out, collect, post-agents, regen-on-rejection.

Two-phase per turn::

    Phase 1 (parallel, pre-agents)
        AgentContext
          |--> [MemoryAgent, EmotionAgent, IntentAgent, ReasoningAgent,
                VoiceStyleAgent, HardwareAgent, ...]
          |    each with its own timeout + cancel token

    Phase 2 (parallel, post-agents)
        candidate_text + intent + emotion in metadata
          |--> [CriticAgent, ...]   (typically just 1)
          |    runs only when there's a non-empty candidate

    FinalResponseBuilder            -- one canonical answer
    QualityGate                     -- approve / reject

    Phase 3 (regen-on-rejection)
        If quality.rejected OR critic.recommendation in {regen, clarify}
        and regen_attempts < max_regen:
          - rebuild context with feedback in metadata['regen_hint']
          - re-run ONLY the ReasoningAgent
          - re-run post-agents
          - re-build, re-gate

The orchestrator returns an :class:`OrchestratorOutput` with a full audit
trail (every agent run, including those produced during regen attempts).
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from typing import Any, Iterable

from .agent_base import (
    Agent,
    AgentContext,
    AgentResult,
    CancellationToken,
    CancelledError,
)
from .final_response import FinalResponse, FinalResponseBuilder
from .quality_gate import QualityGate, QualityVerdict

_log = logging.getLogger(__name__)


_POST_AGENT_NAMES = ("CriticAgent",)


@dataclass
class OrchestratorOutput:
    """Everything produced by one orchestrator turn."""

    final: FinalResponse
    quality: QualityVerdict
    agent_results: list[AgentResult] = field(default_factory=list)
    elapsed_ms: float = 0.0
    cancelled: bool = False
    regen_attempts: int = 0
    critic: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "final": self.final.to_dict(),
            "quality": self.quality.to_dict(),
            "agent_results": [r.to_dict() for r in self.agent_results],
            "elapsed_ms": round(float(self.elapsed_ms), 2),
            "cancelled": bool(self.cancelled),
            "regen_attempts": int(self.regen_attempts),
            "critic": dict(self.critic),
        }


class AgentOrchestrator:
    """Coordinates one parallel-agents turn end-to-end.

    Backwards compatible with V1: pass only ``agents`` and the orchestrator
    behaves as it did before.  V2 unlocks two extra features:

    * ``post_agents`` — agents that run *after* Phase 1, with the
      candidate reply available via ``context.metadata['candidate_text']``.
    * ``max_regen`` — regenerate the reasoning step (only) when the
      quality gate or critic asks for it.
    """

    def __init__(
        self,
        agents: Iterable[Agent],
        final_builder: FinalResponseBuilder,
        quality_gate: QualityGate,
        executor: ThreadPoolExecutor | None = None,
        max_workers: int = 4,
        agent_timeouts_ms: dict[str, int] | None = None,
        post_agents: Iterable[Agent] | None = None,
        max_regen: int = 1,
        log_fn=None,
    ):
        self._agents: dict[str, Agent] = {a.name: a for a in agents}
        if not self._agents:
            raise ValueError("AgentOrchestrator requires at least one agent")
        self._post_agents: dict[str, Agent] = {
            a.name: a for a in (post_agents or [])
        }
        self._final = final_builder
        self._quality = quality_gate
        self._owns_executor = executor is None
        self._executor = executor or ThreadPoolExecutor(
            max_workers=max(2, int(max_workers)),
            thread_name_prefix="revia-agent",
        )
        self._timeouts_ms = dict(agent_timeouts_ms or {})
        self._max_regen = max(0, int(max_regen))
        self._log = log_fn or _log.info

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def agent_names(self) -> list[str]:
        return list(self._agents.keys()) + list(self._post_agents.keys())

    def run_turn(self, context: AgentContext) -> OrchestratorOutput:
        t0 = time.monotonic()

        # ------------- Phase 1: parallel pre-agents -------------
        pre_results = self._run_parallel(self._agents, context)

        all_cancelled = bool(pre_results) and all(
            (not r.success) and (r.error or "").startswith("cancelled")
            for r in pre_results
        )

        reasoning_payload = self._extract_payload(pre_results, "ReasoningAgent")
        emotion_label = self._extract_field(pre_results, "EmotionAgent",
                                             "label", "neutral").lower()
        intent_payload = self._extract_payload(pre_results, "IntentAgent") or {}
        voice_style = self._extract_payload(pre_results, "VoiceStyleAgent") or {}

        candidate_text = (
            str(reasoning_payload.get("text") or "").strip()
            if reasoning_payload else ""
        )

        # ------------- Phase 2: parallel post-agents -------------
        post_results: list[AgentResult] = []
        critic_payload: dict[str, Any] = {}
        if self._post_agents and candidate_text:
            post_ctx = self._with_post_metadata(
                context, candidate_text, intent_payload, emotion_label,
            )
            post_results = self._run_parallel(self._post_agents, post_ctx)
            critic_payload = self._extract_payload(post_results, "CriticAgent") or {}

        # ------------- Build + gate -------------
        final = self._final.build(
            agent_results=pre_results + post_results,
            reasoning_result_payload=reasoning_payload,
            emotion_label_default=emotion_label,
            voice_style=voice_style,
        )

        recent_replies = list(context.metadata.get("recent_replies") or [])
        upstream_score = None
        if reasoning_payload is not None:
            upstream_score = float(reasoning_payload.get("avs_best_score") or 0.0) or None

        verdict = self._quality.check(
            reply=final.text,
            user_utterance=context.user_text,
            emotion_label=final.emotion_label,
            recent_replies=recent_replies,
            upstream_score=upstream_score,
            threshold=context.response_threshold,
        )

        # ------------- Phase 3: regen-on-rejection -------------
        all_results: list[AgentResult] = list(pre_results) + list(post_results)
        regen_attempts = 0
        while (
            regen_attempts < self._max_regen
            and self._needs_regen(verdict, critic_payload)
            and not all_cancelled
            and not context.cancel_token.cancelled
        ):
            regen_attempts += 1
            regen_result = self._regen_once(
                context, verdict, critic_payload, candidate_text,
                intent_payload, emotion_label, voice_style,
            )
            if regen_result is None:
                break

            new_pre, new_post, new_payload, new_critic, new_text = regen_result
            all_results.extend(new_pre)
            all_results.extend(new_post)
            reasoning_payload = new_payload or reasoning_payload
            candidate_text = new_text or candidate_text
            critic_payload = new_critic or critic_payload

            final = self._final.build(
                agent_results=all_results,
                reasoning_result_payload=reasoning_payload,
                emotion_label_default=emotion_label,
                voice_style=voice_style,
            )
            upstream_score = None
            if reasoning_payload is not None:
                upstream_score = (
                    float(reasoning_payload.get("avs_best_score") or 0.0) or None
                )
            verdict = self._quality.check(
                reply=final.text,
                user_utterance=context.user_text,
                emotion_label=final.emotion_label,
                recent_replies=recent_replies,
                upstream_score=upstream_score,
                threshold=context.response_threshold,
            )

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        return OrchestratorOutput(
            final=final,
            quality=verdict,
            agent_results=all_results,
            elapsed_ms=elapsed_ms,
            cancelled=all_cancelled,
            regen_attempts=regen_attempts,
            critic=critic_payload,
        )

    def shutdown(self) -> None:
        if self._owns_executor:
            try:
                self._executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:  # py < 3.9
                self._executor.shutdown(wait=False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_parallel(
        self,
        agent_map: dict[str, Agent],
        context: AgentContext,
    ) -> list[AgentResult]:
        futures: dict[str, tuple[Future, float]] = {}
        for name, agent in agent_map.items():
            timeout_ms = float(
                self._timeouts_ms.get(name, agent.default_timeout_ms)
            )
            fut = self._executor.submit(_safe_execute, agent, context)
            futures[name] = (fut, timeout_ms)

        results: list[AgentResult] = []
        for name, (fut, timeout_ms) in futures.items():
            try:
                result = fut.result(timeout=max(0.001, timeout_ms / 1000.0))
                results.append(result)
            except FuturesTimeoutError:
                # Trip the cancel token so cooperative agents can stop.
                context.cancel_token.cancel()
                fut.cancel()
                results.append(AgentResult(
                    agent=name,
                    success=False,
                    confidence=0.0,
                    elapsed_ms=float(timeout_ms),
                    error=f"timeout after {timeout_ms:.0f}ms",
                ))
                self._log(f"[Orchestrator] {name} timed out after {timeout_ms:.0f}ms")
            except CancelledError as exc:
                results.append(AgentResult(
                    agent=name, success=False, confidence=0.0,
                    elapsed_ms=0.0, error=f"cancelled: {exc}",
                ))
            except Exception as exc:  # pragma: no cover - defensive
                results.append(AgentResult(
                    agent=name, success=False, confidence=0.0,
                    elapsed_ms=0.0, error=f"{type(exc).__name__}: {exc}",
                ))
        return results

    def _run_single(self, agent: Agent, context: AgentContext) -> AgentResult:
        return _safe_execute(agent, context)

    @staticmethod
    def _extract_payload(
        results: list[AgentResult],
        agent_name: str,
    ) -> dict[str, Any] | None:
        for r in results:
            if r.agent == agent_name and r.success:
                return r.result
        return None

    @staticmethod
    def _extract_field(
        results: list[AgentResult],
        agent_name: str,
        field_name: str,
        default: Any,
    ) -> Any:
        for r in results:
            if r.agent == agent_name and r.success:
                return r.result.get(field_name, default)
        return default

    @staticmethod
    def _with_post_metadata(
        context: AgentContext,
        candidate_text: str,
        intent: dict[str, Any],
        emotion_label: str,
    ) -> AgentContext:
        meta = dict(context.metadata or {})
        meta["candidate_text"] = candidate_text
        meta["intent"] = dict(intent or {})
        meta["emotion_label"] = emotion_label
        return AgentContext(
            user_text=context.user_text,
            turn_id=context.turn_id,
            conversation_id=context.conversation_id,
            user_profile=context.user_profile,
            response_threshold=context.response_threshold,
            cancel_token=context.cancel_token,
            metadata=meta,
        )

    @staticmethod
    def _needs_regen(
        verdict: QualityVerdict,
        critic_payload: dict[str, Any],
    ) -> bool:
        if not verdict.approved:
            return True
        if isinstance(critic_payload, dict):
            rec = str(critic_payload.get("recommendation") or "").lower()
            if rec in ("regen", "clarify"):
                return True
        return False

    def _regen_once(
        self,
        context: AgentContext,
        verdict: QualityVerdict,
        critic_payload: dict[str, Any],
        prior_text: str,
        intent_payload: dict[str, Any],
        emotion_label: str,
        voice_style: dict[str, Any],
    ):
        reasoning = self._agents.get("ReasoningAgent")
        if reasoning is None:
            return None

        hint_lines: list[str] = []
        if verdict.reasons:
            hint_lines.append("Quality gate notes: " + "; ".join(verdict.reasons))
        if isinstance(critic_payload, dict):
            issues = critic_payload.get("issues") or {}
            tripped = [k for k, v in issues.items() if v]
            if tripped:
                hint_lines.append(
                    "Critic flagged: " + ", ".join(tripped)
                    + ". Avoid those issues this time."
                )
            if critic_payload.get("reasons"):
                hint_lines.append(
                    "Critic reasons: " + "; ".join(critic_payload["reasons"])
                )
        if not hint_lines:
            hint_lines.append("Improve clarity and intent alignment.")

        regen_hint = "\n".join(hint_lines)
        regen_meta = dict(context.metadata or {})
        regen_meta["regen_hint"] = regen_hint
        regen_meta["prior_candidate"] = prior_text
        regen_meta["intent"] = dict(intent_payload or {})
        regen_meta["emotion_label"] = emotion_label
        regen_meta["voice_style"] = dict(voice_style or {})
        regen_ctx = AgentContext(
            user_text=context.user_text,
            turn_id=f"{context.turn_id}#regen{int(time.time() * 1000)}",
            conversation_id=context.conversation_id,
            user_profile=context.user_profile,
            response_threshold=context.response_threshold,
            cancel_token=context.cancel_token,
            metadata=regen_meta,
        )

        new_reasoning = self._run_single(reasoning, regen_ctx)
        if not new_reasoning.success:
            self._log(
                f"[Orchestrator] regen reasoning failed: "
                f"{new_reasoning.error or 'unknown'}"
            )
            return [new_reasoning], [], None, critic_payload, prior_text

        new_payload = new_reasoning.result
        new_text = str(new_payload.get("text") or "").strip() or prior_text

        # Re-run post-agents on the new candidate (Critic).
        new_post: list[AgentResult] = []
        new_critic = critic_payload
        if self._post_agents and new_text:
            post_ctx = self._with_post_metadata(
                context, new_text, intent_payload, emotion_label,
            )
            new_post = self._run_parallel(self._post_agents, post_ctx)
            new_critic = (
                self._extract_payload(new_post, "CriticAgent") or critic_payload
            )

        return [new_reasoning], new_post, new_payload, new_critic, new_text


def _safe_execute(agent: Agent, context: AgentContext) -> AgentResult:
    """Top-level executor target.  Module-level so ThreadPool can pickle it."""
    try:
        return agent.execute(context)
    except Exception as exc:  # pragma: no cover - last-line defence
        return AgentResult(
            agent=getattr(agent, "name", agent.__class__.__name__),
            success=False,
            confidence=0.0,
            elapsed_ms=0.0,
            error=f"{type(exc).__name__}: {exc}",
        )
