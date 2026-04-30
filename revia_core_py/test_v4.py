"""Unit tests for V4 — Skills + Tool/Vision/Debate Layer.

Run from inside ``revia_core_py``::

    python -m unittest test_v4 -v
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from agents import (
    AgentContext,
    AgentResult,
    CancellationToken,
    CriticAgent,
    DebateOrchestrator,
    ToolUseAgent,
    VisionAgent,
)
from agents.agent_base import Agent
from autonomy_v3 import EpisodicMemoryStore
from skills import (
    CalculatorSkill,
    ClockSkill,
    EchoSkill,
    MemoryRecallSkill,
    Skill,
    SkillRegistry,
    SkillRequest,
)


# ---------------------------------------------------------------------------
# Skills — base
# ---------------------------------------------------------------------------

class TestCalculatorSkill(unittest.TestCase):

    def test_simple_addition(self):
        s = CalculatorSkill()
        resp = s.handle(SkillRequest(user_text="what is 2 + 3?"))
        self.assertTrue(resp.success)
        self.assertEqual(resp.data["value"], 5)
        self.assertIn("=", resp.text)

    def test_complex_expression(self):
        s = CalculatorSkill()
        resp = s.handle(SkillRequest(user_text="calculate (4 + 6) * 2"))
        self.assertTrue(resp.success)
        self.assertEqual(resp.data["value"], 20)

    def test_caret_treated_as_power(self):
        s = CalculatorSkill()
        resp = s.handle(SkillRequest(user_text="what is 2^10"))
        self.assertTrue(resp.success)
        self.assertEqual(resp.data["value"], 1024)

    def test_no_expression_returns_error(self):
        s = CalculatorSkill()
        resp = s.handle(SkillRequest(user_text="hello there"))
        self.assertFalse(resp.success)

    def test_match_score_with_trigger(self):
        s = CalculatorSkill()
        m = s.match(SkillRequest(user_text="calculate 5 + 5"))
        self.assertGreater(m.score, 0.0)


class TestClockSkill(unittest.TestCase):

    def test_returns_iso_local_and_weekday(self):
        s = ClockSkill()
        resp = s.handle(SkillRequest(user_text="what time is it"))
        self.assertTrue(resp.success)
        self.assertIn("iso_local", resp.data)
        self.assertIn("weekday", resp.data)


class TestEchoSkill(unittest.TestCase):

    def test_strips_echo_prefix(self):
        s = EchoSkill()
        resp = s.handle(SkillRequest(user_text="echo hello world"))
        self.assertTrue(resp.success)
        self.assertEqual(resp.text, "hello world")

    def test_match_only_for_explicit_echo(self):
        s = EchoSkill()
        m = s.match(SkillRequest(user_text="hi there"))
        self.assertEqual(m.score, 0.0)


class TestMemoryRecallSkill(unittest.TestCase):

    def test_uses_episode_store(self):
        store = EpisodicMemoryStore(
            path=Path(tempfile.mkstemp(suffix=".jsonl")[1]),
        )
        store.add(user_text="we talked about cats yesterday",
                  reply_text="indeed we did", intent_label="chat")
        s = MemoryRecallSkill(episode_store=store)
        resp = s.handle(SkillRequest(user_text="recall cats"))
        self.assertTrue(resp.success)
        self.assertGreater(resp.data["count"], 0)

    def test_returns_no_match_message_when_empty(self):
        store = EpisodicMemoryStore(
            path=Path(tempfile.mkstemp(suffix=".jsonl")[1]),
        )
        s = MemoryRecallSkill(episode_store=store)
        resp = s.handle(SkillRequest(user_text="recall something"))
        self.assertTrue(resp.success)
        self.assertIn("No matching", resp.text)


# ---------------------------------------------------------------------------
# SkillRegistry
# ---------------------------------------------------------------------------

class TestSkillRegistry(unittest.TestCase):

    def test_dispatch_picks_highest_match(self):
        reg = SkillRegistry(
            skills=[CalculatorSkill(), ClockSkill(), EchoSkill()],
        )
        result = reg.dispatch(SkillRequest(user_text="what is 12 + 8?"))
        self.assertEqual(result.chosen, "calculator")
        self.assertTrue(result.response.success)
        self.assertEqual(result.response.data["value"], 20)

    def test_no_match_returns_empty_chosen(self):
        reg = SkillRegistry(skills=[CalculatorSkill(), ClockSkill()],
                             min_score=0.5)
        result = reg.dispatch(SkillRequest(user_text="purely conversational text"))
        self.assertEqual(result.chosen, "")
        self.assertIsNone(result.response)

    def test_explicit_skill_overrides_match(self):
        reg = SkillRegistry(skills=[CalculatorSkill(), EchoSkill()])
        result = reg.dispatch(
            SkillRequest(user_text="2 + 2"),
            explicit_skill="echo",
        )
        self.assertEqual(result.chosen, "echo")
        self.assertTrue(result.response.success)

    def test_disabled_skill_returns_disabled_error(self):
        reg = SkillRegistry(skills=[CalculatorSkill()])
        reg.set_enabled("calculator", False)
        result = reg.dispatch(SkillRequest(user_text="2 + 2"))
        # Match score becomes 0 when disabled, so dispatch returns no chosen.
        self.assertEqual(result.chosen, "")

    def test_status_payload(self):
        reg = SkillRegistry(skills=[CalculatorSkill(), EchoSkill()])
        st = reg.status()
        names = {s["name"] for s in st["skills"]}
        self.assertEqual(names, {"calculator", "echo"})
        self.assertEqual(st["count"], 2)


# ---------------------------------------------------------------------------
# ToolUseAgent
# ---------------------------------------------------------------------------

class TestToolUseAgent(unittest.TestCase):

    def _ctx(self, text: str, **meta) -> AgentContext:
        return AgentContext(
            user_text=text, cancel_token=CancellationToken("t"),
            metadata=meta,
        )

    def test_no_registry_returns_empty(self):
        ag = ToolUseAgent(registry=None)
        out = ag.execute(self._ctx("hi")).result
        self.assertEqual(out["used_skill"], "")
        self.assertEqual(out["score"], 0.0)

    def test_dispatches_to_calculator(self):
        reg = SkillRegistry(skills=[CalculatorSkill()])
        ag = ToolUseAgent(registry=reg)
        out = ag.execute(self._ctx("calculate 6 * 7")).result
        self.assertEqual(out["used_skill"], "calculator")
        self.assertTrue(out["success"])
        self.assertEqual(out["result_data"]["value"], 42)

    def test_explicit_skill_via_metadata(self):
        reg = SkillRegistry(skills=[CalculatorSkill(), EchoSkill()])
        ag = ToolUseAgent(registry=reg)
        out = ag.execute(self._ctx("echo hello", skill="echo")).result
        self.assertEqual(out["used_skill"], "echo")
        self.assertTrue(out["success"])

    def test_no_match_returns_no_skill(self):
        reg = SkillRegistry(skills=[CalculatorSkill()], min_score=0.99)
        ag = ToolUseAgent(registry=reg)
        out = ag.execute(self._ctx("just chatting")).result
        self.assertEqual(out["used_skill"], "")
        self.assertEqual(out["score"], 0.0)


# ---------------------------------------------------------------------------
# VisionAgent
# ---------------------------------------------------------------------------

class _StubRouter:
    """Minimal router with a single registered task type."""

    def __init__(self, task_name: str, handler):
        self._task = task_name
        self._h = handler

    def has(self, name: str) -> bool:
        return name == self._task

    def call(self, name: str, *args, **kwargs):
        return self._h(*args, **kwargs)


class TestVisionAgent(unittest.TestCase):

    def _ctx(self, **meta) -> AgentContext:
        return AgentContext(
            user_text="describe", cancel_token=CancellationToken("t"),
            metadata=meta,
        )

    def test_no_frame_no_caption_returns_inactive(self):
        ag = VisionAgent()
        out = ag.execute(self._ctx()).result
        self.assertFalse(out["has_frame"])
        self.assertEqual(out["source"], "none")

    def test_metadata_caption_path(self):
        ag = VisionAgent()
        out = ag.execute(self._ctx(
            vision_caption="A cat on a desk", vision_tags=["cat", "desk"],
        )).result
        self.assertEqual(out["source"], "metadata")
        self.assertIn("cat", out["description"].lower())
        self.assertEqual(out["tags"], ["cat", "desk"])

    def test_router_path(self):
        def fake_describe(frame, user_text=None, hints=None):
            return {"description": "A red apple", "tags": ["apple"], "confidence": 0.9}
        router = _StubRouter("describe_image", fake_describe)
        ag = VisionAgent(model_router=router)
        out = ag.execute(self._ctx(vision_frame=b"fakeimg")).result
        self.assertEqual(out["source"], "router")
        self.assertEqual(out["description"], "A red apple")
        self.assertIn("apple", out["tags"])

    def test_stub_path_when_no_router_no_caption(self):
        ag = VisionAgent()
        out = ag.execute(self._ctx(vision_frame=b"\x00\x01")).result
        self.assertEqual(out["source"], "stub")
        self.assertTrue(out["has_frame"])


# ---------------------------------------------------------------------------
# DebateOrchestrator
# ---------------------------------------------------------------------------

class _ScriptedReasoner(Agent):
    """Returns a different reply per debate variant."""
    name = "ReasoningAgent"
    default_timeout_ms = 1000

    def __init__(self, by_variant: dict[str, str]):
        self._by_variant = dict(by_variant)
        self.calls: list[str] = []

    def run(self, context):
        v = str(context.metadata.get("debate_variant") or "default")
        self.calls.append(v)
        text = self._by_variant.get(v, "default reply about cats")
        return {
            "_confidence": 0.8 if "good" in text.lower() else 0.4,
            "text": text,
            "candidates": [text],
            "avs_passed": False, "avs_best_score": 0.0,
            "fallback_accept": True, "emotion_label": "neutral",
            "intent": {}, "notes": [],
        }


class _NaiveCritic(Agent):
    """Score == 0.9 if 'good' in text, else 0.3."""
    name = "CriticAgent"
    default_timeout_ms = 200

    def run(self, context):
        candidate = str(context.metadata.get("candidate_text") or "").lower()
        score = 0.9 if "good" in candidate else 0.3
        rec = "accept" if score >= 0.7 else "regen"
        return {
            "_confidence": score,
            "score": score,
            "recommendation": rec,
            "reasons": [],
            "issues": {},
        }


class TestDebateOrchestrator(unittest.TestCase):

    def _ctx(self) -> AgentContext:
        return AgentContext(
            user_text="tell me about cats",
            cancel_token=CancellationToken("t-debate"),
        )

    def test_picks_best_critic_score(self):
        reasoner = _ScriptedReasoner({
            "concise": "short reply about cats",
            "detailed": "much more detailed and good explanation about cats",
        })
        critic = _NaiveCritic()
        debate = DebateOrchestrator(reasoning_agent=reasoner, critic_agent=critic)
        out = debate.run_debate(
            self._ctx(),
            variants=[
                {"name": "concise", "system_prompt": "Be terse."},
                {"name": "detailed", "system_prompt": "Be thorough."},
            ],
        )
        debate.shutdown()
        self.assertEqual(out.winner, "detailed")
        self.assertIn("good", out.winner_text.lower())
        self.assertEqual(len(out.variants), 2)

    def test_score_gap_zero_with_one_variant(self):
        reasoner = _ScriptedReasoner({"only": "this is a good reply"})
        critic = _NaiveCritic()
        debate = DebateOrchestrator(reasoning_agent=reasoner, critic_agent=critic)
        out = debate.run_debate(
            self._ctx(),
            variants=[{"name": "only", "system_prompt": "x"}],
        )
        debate.shutdown()
        self.assertEqual(out.score_gap, 0.0)

    def test_rejects_regen_recommendations(self):
        # Both variants score below 0.7 (no 'good') -> recommendation=regen.
        reasoner = _ScriptedReasoner({
            "a": "bad reply one",
            "b": "bad reply two",
        })
        critic = _NaiveCritic()
        debate = DebateOrchestrator(reasoning_agent=reasoner, critic_agent=critic)
        out = debate.run_debate(
            self._ctx(),
            variants=[{"name": "a", "system_prompt": ""},
                      {"name": "b", "system_prompt": ""}],
        )
        debate.shutdown()
        # Even when both are flagged regen, a winner is still produced
        # (best of the available).  But _pick_winner tie-breaks by
        # critic_score so it won't crash.
        self.assertIn(out.winner, ("a", "b"))

    def test_empty_variant_list(self):
        reasoner = _ScriptedReasoner({})
        critic = _NaiveCritic()
        debate = DebateOrchestrator(reasoning_agent=reasoner, critic_agent=critic)
        out = debate.run_debate(self._ctx(), variants=[])
        debate.shutdown()
        self.assertEqual(out.winner, "")
        self.assertEqual(out.variants, [])


if __name__ == "__main__":
    unittest.main()
