"""Unit tests for the V1 parallel-agents spine.

Run from inside ``revia_core_py``::

    python -m unittest test_agents -v
"""
from __future__ import annotations

import time
import unittest
from typing import Any

from agents.agent_base import (
    Agent,
    AgentContext,
    AgentResult,
    CancellationToken,
)
from agents.emotion_agent import EmotionAgent
from agents.final_response import FinalResponseBuilder
from agents.memory_agent import MemoryAgent
from agents.model_router import ModelRouter
from agents.orchestrator import AgentOrchestrator
from agents.quality_gate import QualityGate
from agents.reasoning_agent import ReasoningAgent


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

class _FakeAgent(Agent):
    name = "FakeAgent"
    default_timeout_ms = 1000

    def __init__(self, name: str, sleep_s: float = 0.0,
                 confidence: float = 0.7, payload: dict | None = None,
                 raise_exc: Exception | None = None):
        self.name = name
        self._sleep_s = sleep_s
        self._confidence = confidence
        self._payload = payload or {}
        self._raise = raise_exc

    def run(self, context: AgentContext) -> dict[str, Any]:
        if self._sleep_s > 0:
            # Sleep in small slices so cancellation can fire mid-flight.
            steps = max(1, int(self._sleep_s * 50))
            for _ in range(steps):
                context.cancel_token.raise_if_cancelled()
                time.sleep(self._sleep_s / steps)
        if self._raise is not None:
            raise self._raise
        return {"_confidence": self._confidence, **self._payload}


class _StubMemory:
    def __init__(self, hits: list[str]):
        self._hits = hits
        self.short_term = [{"role": "user", "content": "prior turn"}]

    def get_short_term(self, limit=50):
        return list(self.short_term[-limit:])

    def search(self, query, max_results=5):
        return [{"content": h} for h in self._hits[:max_results]]


class _StubEmotionNet:
    enabled = True
    def infer(self, text, recent_messages=None, prev_emotion=None,
             profile_name=None, profile_state=None):
        return {
            "label": "Focused",
            "secondary_label": "Neutral",
            "confidence": 0.81,
            "uncertainty": 0.19,
            "valence": 0.2,
            "arousal": 0.4,
            "dominance": 0.3,
            "emotion_probs": {"Focused": 0.81, "Neutral": 0.19},
            "top_emotions": [{"label": "Focused", "prob": 0.81}],
            "signals": {},
            "temporal": {},
            "model": "stub",
            "inference_ms": 1.0,
        }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOrchestratorBasics(unittest.TestCase):

    def _orch(self, agents, threshold: float = 0.0) -> AgentOrchestrator:
        return AgentOrchestrator(
            agents=agents,
            final_builder=FinalResponseBuilder(hfl=None),
            quality_gate=QualityGate(profile_engine=None),
        )

    def _ctx(self, text: str = "hello", threshold: float = 0.0) -> AgentContext:
        return AgentContext(
            user_text=text,
            turn_id="t-1",
            conversation_id="c-1",
            user_profile="test",
            response_threshold=threshold,
            cancel_token=CancellationToken(turn_id="t-1"),
        )

    def test_orchestrator_runs_three_agents_in_parallel(self):
        """Sum of per-agent sleeps > total elapsed -> they ran concurrently."""
        per_sleep_s = 0.30
        agents = [
            _FakeAgent("MemoryAgent", sleep_s=per_sleep_s, confidence=0.9,
                       payload={"relevant_facts": ["x"]}),
            _FakeAgent("EmotionAgent", sleep_s=per_sleep_s, confidence=0.8,
                       payload={"label": "Focused"}),
            _FakeAgent("ReasoningAgent", sleep_s=per_sleep_s, confidence=0.7,
                       payload={"text": "Hi.", "avs_best_score": 0.9}),
        ]
        orch = self._orch(agents)
        try:
            t0 = time.monotonic()
            out = orch.run_turn(self._ctx("hello world", 0.5))
            elapsed = time.monotonic() - t0

            self.assertEqual(len(out.agent_results), 3)
            for r in out.agent_results:
                self.assertTrue(r.success, msg=f"{r.agent}: {r.error}")
            # Sequential would be ~0.9s; parallel should be ~0.30-0.45s.
            self.assertLess(elapsed, per_sleep_s * 3 - 0.05,
                            f"agents look serial (elapsed={elapsed:.2f}s)")
            self.assertEqual(out.final.text, "Hi.")
            self.assertTrue(out.quality.approved)
        finally:
            orch.shutdown()

    def test_orchestrator_continues_when_memory_times_out(self):
        slow_memory = _FakeAgent("MemoryAgent", sleep_s=2.0,
                                 confidence=0.5, payload={"relevant_facts": []})
        emotion = _FakeAgent("EmotionAgent", sleep_s=0.0,
                             confidence=0.8, payload={"label": "Focused"})
        reasoning = _FakeAgent("ReasoningAgent", sleep_s=0.0,
                               confidence=0.7,
                               payload={"text": "ok", "avs_best_score": 0.85})
        orch = AgentOrchestrator(
            agents=[slow_memory, emotion, reasoning],
            final_builder=FinalResponseBuilder(hfl=None),
            quality_gate=QualityGate(profile_engine=None),
            agent_timeouts_ms={"MemoryAgent": 100,
                               "EmotionAgent": 1000,
                               "ReasoningAgent": 1000},
        )
        try:
            out = orch.run_turn(self._ctx("hi", 0.5))
            mem = next(r for r in out.agent_results if r.agent == "MemoryAgent")
            self.assertFalse(mem.success)
            self.assertIn("timeout", (mem.error or "").lower())

            other = [r for r in out.agent_results if r.agent != "MemoryAgent"]
            self.assertTrue(all(r.success or "cancelled" in (r.error or "")
                                for r in other))
            # Final still produced from reasoning if it wasn't cancelled.
            if any(r.agent == "ReasoningAgent" and r.success
                   for r in out.agent_results):
                self.assertEqual(out.final.text, "ok")
        finally:
            orch.shutdown()

    def test_quality_gate_rejects_below_threshold(self):
        reasoning = _FakeAgent("ReasoningAgent", sleep_s=0.0,
                               confidence=0.30,
                               payload={"text": "ok", "avs_best_score": 0.30})
        memory = _FakeAgent("MemoryAgent", sleep_s=0.0, confidence=0.5,
                            payload={"relevant_facts": []})
        emotion = _FakeAgent("EmotionAgent", sleep_s=0.0, confidence=0.5,
                             payload={"label": "Focused"})
        orch = AgentOrchestrator(
            agents=[memory, emotion, reasoning],
            final_builder=FinalResponseBuilder(hfl=None),
            quality_gate=QualityGate(profile_engine=None),
        )
        try:
            out = orch.run_turn(self._ctx("hi", threshold=0.70))
            self.assertEqual(out.final.text, "ok")
            self.assertFalse(out.quality.approved)
            self.assertGreaterEqual(out.quality.threshold, 0.69)
        finally:
            orch.shutdown()

    def test_cancellation_token_stops_inflight_agents(self):
        slow1 = _FakeAgent("MemoryAgent", sleep_s=1.0, confidence=0.5)
        slow2 = _FakeAgent("ReasoningAgent", sleep_s=1.0, confidence=0.5)
        ctx = self._ctx("cancel me", 0.5)
        orch = AgentOrchestrator(
            agents=[slow1, slow2],
            final_builder=FinalResponseBuilder(hfl=None),
            quality_gate=QualityGate(profile_engine=None),
            agent_timeouts_ms={"MemoryAgent": 5000, "ReasoningAgent": 5000},
        )
        try:
            # Cancel BEFORE running; agents should bail immediately.
            ctx.cancel_token.cancel()
            out = orch.run_turn(ctx)
            for r in out.agent_results:
                self.assertFalse(r.success)
                self.assertIn("cancelled", (r.error or "").lower())
            self.assertTrue(out.cancelled)
        finally:
            orch.shutdown()

    def test_agent_exception_does_not_break_orchestrator(self):
        bad = _FakeAgent("MemoryAgent", sleep_s=0.0,
                         raise_exc=RuntimeError("boom"))
        good = _FakeAgent("ReasoningAgent", sleep_s=0.0, confidence=0.6,
                          payload={"text": "still here", "avs_best_score": 0.8})
        orch = AgentOrchestrator(
            agents=[bad, good],
            final_builder=FinalResponseBuilder(hfl=None),
            quality_gate=QualityGate(profile_engine=None),
        )
        try:
            out = orch.run_turn(self._ctx("hi", 0.5))
            mem = next(r for r in out.agent_results if r.agent == "MemoryAgent")
            self.assertFalse(mem.success)
            self.assertIn("RuntimeError", mem.error or "")
            self.assertEqual(out.final.text, "still here")
            self.assertTrue(out.quality.approved)
        finally:
            orch.shutdown()


class TestModelRouter(unittest.TestCase):

    def test_router_routes_by_task_type(self):
        called = []
        router = ModelRouter()
        router.register("emotion_classify", "stub_emo",
                        lambda *a, **k: called.append(("emo", a, k)) or {"label": "Focused"})
        router.register("reason_chat", "stub_llm",
                        lambda *a, **k: called.append(("llm", a, k)) or "answer")

        self.assertTrue(router.has("emotion_classify"))
        self.assertTrue(router.has("reason_chat"))
        self.assertFalse(router.has("vision"))

        out_emo = router.call("emotion_classify", "hello")
        out_llm = router.call("reason_chat", "hello")
        self.assertEqual(out_emo, {"label": "Focused"})
        self.assertEqual(out_llm, "answer")
        self.assertEqual([c[0] for c in called], ["emo", "llm"])

    def test_router_unknown_task_raises(self):
        router = ModelRouter()
        with self.assertRaises(KeyError):
            router.get("nonexistent_task")


class TestRealAgentsWithStubs(unittest.TestCase):
    """End-to-end with real agent classes + stub backends."""

    def test_memory_agent_returns_relevant_facts(self):
        mem = MemoryAgent(memory_store=_StubMemory(["fact A", "fact B"]))
        ctx = AgentContext(user_text="parallel agents",
                           cancel_token=CancellationToken("t"))
        result = mem.execute(ctx)
        self.assertTrue(result.success)
        self.assertGreater(result.confidence, 0.0)
        self.assertGreaterEqual(len(result.result["relevant_facts"]), 1)

    def test_emotion_agent_uses_emotion_net(self):
        emo = EmotionAgent(emotion_net=_StubEmotionNet())
        ctx = AgentContext(user_text="I love this",
                           cancel_token=CancellationToken("t"))
        result = emo.execute(ctx)
        self.assertTrue(result.success)
        self.assertEqual(result.result["label"], "Focused")
        self.assertGreater(result.confidence, 0.5)

    def test_reasoning_agent_stub_path(self):
        agent = ReasoningAgent()  # no planner, no router
        ctx = AgentContext(user_text="hello?", cancel_token=CancellationToken("t"))
        result = agent.execute(ctx)
        self.assertTrue(result.success)
        self.assertIn("hello?", result.result["text"])
        self.assertEqual(result.result["notes"], ["stub_no_planner_no_router"])


if __name__ == "__main__":
    unittest.main()
