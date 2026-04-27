"""Unit tests for the V2.2 agents (Intent / VoiceStyle / Critic + regen loop).

Run from inside ``revia_core_py``::

    python -m unittest test_v22 -v
"""
from __future__ import annotations

import unittest
from typing import Any

from agents import (
    AgentContext,
    AgentOrchestrator,
    CancellationToken,
    CriticAgent,
    EmotionAgent,
    FinalResponseBuilder,
    IntentAgent,
    MemoryAgent,
    QualityGate,
    QualityVerdict,
    ReasoningAgent,
    VoiceStyleAgent,
)
from agents.agent_base import Agent


# ---------------------------------------------------------------------------
# Test stubs
# ---------------------------------------------------------------------------

class _StubMemory:
    short_term: list[dict[str, str]] = []
    long_term: list[dict[str, str]] = []

    def get_short_term(self, limit: int = 50):
        return list(self.short_term[-limit:])

    def search(self, query: str, max_results: int = 5):
        return []


class _StubEmotionNet:
    enabled = True

    def infer(self, text, recent_messages=None, prev_emotion=None,
              profile_name=None, profile_state=None):
        return {
            "label": "Neutral",
            "secondary_label": "Neutral",
            "confidence": 0.6,
            "uncertainty": 0.4,
            "valence": 0.0,
            "arousal": 0.0,
            "dominance": 0.0,
            "emotion_probs": {"Neutral": 1.0},
            "top_emotions": [{"label": "Neutral", "prob": 1.0}],
        }


class _ScriptedReasoner(Agent):
    """Returns a different reply on each call so we can prove regen ran."""
    name = "ReasoningAgent"
    default_timeout_ms = 1000

    def __init__(self, replies: list[str]):
        self._replies = list(replies)
        self.calls = 0

    def run(self, context):
        idx = min(self.calls, len(self._replies) - 1)
        self.calls += 1
        text = self._replies[idx]
        return {
            "_confidence": 0.8 if "good" in text else 0.4,
            "text": text,
            "candidates": [text],
            "avs_passed": False,
            "avs_best_score": 0.0,
            "fallback_accept": True,
            "emotion_label": "neutral",
            "intent": {},
            "notes": ["scripted"],
            "regen_hint_seen": str(context.metadata.get("regen_hint") or ""),
        }


class _StubProfile:
    minimum_answer_threshold = 0.50
    regen_patience = 2
    verbosity = 0.5
    current = {"behavior": {"formality": 0.4}}


# ---------------------------------------------------------------------------
# IntentAgent
# ---------------------------------------------------------------------------

class TestIntentAgent(unittest.TestCase):

    def _run(self, text: str) -> dict[str, Any]:
        agent = IntentAgent()
        ctx = AgentContext(user_text=text, cancel_token=CancellationToken())
        result = agent.execute(ctx)
        self.assertTrue(result.success, msg=result.error)
        return result.result

    def test_question_marker(self):
        out = self._run("Why is the sky blue?")
        self.assertEqual(out["label"], "question")
        self.assertTrue(out["is_question"])
        self.assertTrue(out["expects_facts"])

    def test_command_imperative(self):
        out = self._run("write a haiku about mountains")
        self.assertEqual(out["label"], "command")
        self.assertTrue(out["is_imperative"])

    def test_greeting(self):
        out = self._run("hey there")
        self.assertEqual(out["label"], "greeting")
        self.assertEqual(out["polarity"], "positive")

    def test_emotional_share(self):
        out = self._run("I feel sad today")
        self.assertEqual(out["label"], "emotional_share")
        self.assertEqual(out["polarity"], "negative")

    def test_compliment(self):
        out = self._run("Thanks, that was awesome")
        self.assertEqual(out["label"], "compliment")
        self.assertEqual(out["polarity"], "positive")

    def test_complaint(self):
        out = self._run("this sucks, doesn't work")
        self.assertEqual(out["label"], "complaint")
        self.assertEqual(out["polarity"], "negative")

    def test_clarification_short_question(self):
        out = self._run("why?")
        self.assertEqual(out["label"], "clarification")

    def test_empty_input_returns_chat(self):
        out = self._run("")
        self.assertEqual(out["label"], "chat")
        self.assertEqual(out["confidence"], 0.0)


# ---------------------------------------------------------------------------
# VoiceStyleAgent
# ---------------------------------------------------------------------------

class TestVoiceStyleAgent(unittest.TestCase):

    def _run(self, intent: dict, emotion: str = "neutral") -> dict[str, Any]:
        agent = VoiceStyleAgent(profile_engine=_StubProfile())
        ctx = AgentContext(
            user_text="ignored",
            cancel_token=CancellationToken(),
            metadata={"intent": intent, "emotion_label": emotion},
        )
        result = agent.execute(ctx)
        self.assertTrue(result.success, msg=result.error)
        return result.result

    def test_reassuring_for_sad_user(self):
        out = self._run(
            intent={"label": "emotional_share", "polarity": "negative"},
            emotion="Sadness",
        )
        self.assertEqual(out["tone"], "reassuring")
        self.assertLess(out["speech_rate"], 1.0)

    def test_playful_for_compliment(self):
        out = self._run(
            intent={"label": "compliment", "polarity": "positive"},
            emotion="Joy",
        )
        self.assertEqual(out["tone"], "playful")
        self.assertGreater(out["speech_rate"], 1.0)

    def test_focused_for_command(self):
        out = self._run(intent={"label": "command", "polarity": "neutral"})
        self.assertEqual(out["tone"], "focused")

    def test_max_sentences_short_for_small_talk(self):
        out = self._run(intent={"label": "small_talk", "polarity": "neutral"})
        self.assertLessEqual(out["max_sentences"], 2)


# ---------------------------------------------------------------------------
# CriticAgent
# ---------------------------------------------------------------------------

class TestCriticAgent(unittest.TestCase):

    def _run(
        self,
        candidate: str,
        user_text: str = "tell me about cats",
        intent: dict | None = None,
        recent: list[str] | None = None,
        banned: list[str] | None = None,
    ) -> dict[str, Any]:
        agent = CriticAgent()
        ctx = AgentContext(
            user_text=user_text,
            cancel_token=CancellationToken(),
            metadata={
                "candidate_text": candidate,
                "intent": intent or {"label": "question", "is_question": True,
                                     "expects_facts": True},
                "recent_replies": recent or [],
                "banned_phrases": banned or [],
            },
        )
        result = agent.execute(ctx)
        self.assertTrue(result.success, msg=result.error)
        return result.result

    def test_empty_candidate_recommends_regen(self):
        out = self._run("")
        self.assertEqual(out["recommendation"], "regen")
        self.assertTrue(out["issues"]["empty"])

    def test_refusal_spam_recommends_regen(self):
        out = self._run("As an AI, I cannot help with that request, sorry.")
        self.assertEqual(out["recommendation"], "regen")
        self.assertTrue(out["issues"]["refusal_spam"])

    def test_repetition_recommends_regen(self):
        prior = "Cats are small mammals that purr and sleep a lot every day."
        out = self._run(prior, recent=[prior])
        self.assertEqual(out["recommendation"], "regen")
        self.assertTrue(out["issues"]["repetition"])

    def test_too_short_for_question(self):
        out = self._run("yes.", intent={"label": "question", "is_question": True})
        self.assertTrue(out["issues"]["too_short"])
        self.assertEqual(out["recommendation"], "regen")

    def test_uncertain_facts_clarify(self):
        out = self._run(
            "I think it's probably because of some kind of reflection maybe.",
            intent={"label": "question", "is_question": True,
                    "expects_facts": True},
        )
        self.assertTrue(out["issues"]["uncertain_facts"])
        self.assertEqual(out["recommendation"], "clarify")

    def test_banned_phrase_violation(self):
        out = self._run("Sure thing, here's the answer about cats.",
                        banned=["sure thing"])
        self.assertTrue(out["issues"]["profile_violation"])
        self.assertEqual(out["recommendation"], "regen")

    def test_accepts_clean_answer(self):
        out = self._run(
            "Cats are domesticated felines known for independence and grooming.",
            intent={"label": "question", "is_question": True,
                    "expects_facts": True},
        )
        self.assertEqual(out["recommendation"], "accept")
        self.assertGreaterEqual(out["score"], 0.85)


# ---------------------------------------------------------------------------
# FinalResponseBuilder voice-style hooks
# ---------------------------------------------------------------------------

class TestFinalResponseVoiceStyle(unittest.TestCase):

    def test_max_sentence_cap_trims_long_reply(self):
        builder = FinalResponseBuilder(hfl=None)
        # 5 sentences, cap=2, so it should trim once exceeded by 2+.
        text = "A. B. C. D. E."
        final = builder.build(
            agent_results=[],
            reasoning_result_payload={"text": text, "emotion_label": "neutral"},
            voice_style={"max_sentences": 2, "speech_rate": 1.0},
        )
        # We expect the cap to trigger at most 2 sentences.
        self.assertLessEqual(final.text.count("."), 2 + 1)
        self.assertIn("voice_style_sentence_cap_2", final.notes)

    def test_signature_phrase_added_for_greeting(self):
        builder = FinalResponseBuilder(hfl=None)
        final = builder.build(
            agent_results=[],
            reasoning_result_payload={"text": "Welcome to Revia",
                                      "emotion_label": "neutral"},
            voice_style={
                "signature_phrase": "hey there",
                "tone": "warm",
                "intent_label": "greeting",
                "max_sentences": 2,
            },
        )
        self.assertIn("hey there", final.text.lower())
        self.assertIn("voice_style_signature_lead", final.notes)
        self.assertEqual(final.prosody.get("tone"), "warm")

    def test_signature_skipped_when_already_present(self):
        builder = FinalResponseBuilder(hfl=None)
        final = builder.build(
            agent_results=[],
            reasoning_result_payload={"text": "hey there friend",
                                      "emotion_label": "neutral"},
            voice_style={"signature_phrase": "hey there",
                         "tone": "warm", "intent_label": "greeting",
                         "max_sentences": 3},
        )
        # Should appear exactly once, not twice.
        self.assertEqual(final.text.lower().count("hey there"), 1)


# ---------------------------------------------------------------------------
# Orchestrator regen-on-rejection
# ---------------------------------------------------------------------------

class _FailingQualityGate:
    """Approves only when the candidate contains the word 'good'."""

    def check(self, reply, user_utterance, emotion_label="neutral",
              recent_replies=None, upstream_score=None, threshold=None):
        approved = "good" in (reply or "").lower()
        return QualityVerdict(
            score=0.9 if approved else 0.2,
            threshold=threshold or 0.7,
            approved=approved,
            reasons=[] if approved else ["fail_pattern"],
            source="stub",
        )


class TestRegenLoop(unittest.TestCase):

    def _build(self, replies, max_regen=2):
        memory = _StubMemory()
        emotion = _StubEmotionNet()
        reasoner = _ScriptedReasoner(replies)
        agents = [
            MemoryAgent(memory_store=memory),
            EmotionAgent(emotion_net=emotion),
            IntentAgent(),
            VoiceStyleAgent(profile_engine=_StubProfile()),
            reasoner,
        ]
        orch = AgentOrchestrator(
            agents=agents,
            final_builder=FinalResponseBuilder(hfl=None),
            quality_gate=_FailingQualityGate(),
            post_agents=[CriticAgent()],
            max_regen=max_regen,
            agent_timeouts_ms={
                "MemoryAgent": 500, "EmotionAgent": 500,
                "IntentAgent": 250, "VoiceStyleAgent": 250,
                "ReasoningAgent": 1000, "CriticAgent": 400,
            },
        )
        return orch, reasoner

    def test_regen_loop_runs_until_approved(self):
        orch, reasoner = self._build(
            replies=[
                "this is a fine reply about cats",       # rejected (no 'good')
                "this is a much better, longer good reply about cats and how they sleep",   # approved
            ],
            max_regen=2,
        )
        ctx = AgentContext(
            user_text="tell me about cats",
            cancel_token=CancellationToken("t1"),
            response_threshold=0.7,
        )
        out = orch.run_turn(ctx)
        orch.shutdown()
        self.assertEqual(reasoner.calls, 2)
        self.assertEqual(out.regen_attempts, 1)
        self.assertTrue(out.quality.approved)
        self.assertIn("good", out.final.text.lower())

    def test_regen_stops_at_budget_when_still_failing(self):
        orch, reasoner = self._build(
            replies=[
                "bad reply one about cats and stuff that goes on a while",
                "bad reply two about cats and stuff that goes on a while",
                "bad reply three about cats and stuff that goes on a while",
            ],
            max_regen=1,
        )
        ctx = AgentContext(
            user_text="tell me about cats",
            cancel_token=CancellationToken("t2"),
            response_threshold=0.7,
        )
        out = orch.run_turn(ctx)
        orch.shutdown()
        self.assertEqual(reasoner.calls, 2)   # initial + 1 regen
        self.assertEqual(out.regen_attempts, 1)
        self.assertFalse(out.quality.approved)

    def test_regen_hint_passed_to_reasoner(self):
        orch, reasoner = self._build(
            replies=[
                "first attempt about cats and life",
                "second attempt is good now",
            ],
            max_regen=2,
        )
        ctx = AgentContext(
            user_text="tell me about cats",
            cancel_token=CancellationToken("t3"),
            response_threshold=0.7,
        )
        out = orch.run_turn(ctx)
        orch.shutdown()
        # The regen run should have populated regen_hint metadata.
        regen_runs = [
            r for r in out.agent_results
            if r.agent == "ReasoningAgent" and r.success
        ]
        self.assertEqual(len(regen_runs), 2)
        self.assertEqual(regen_runs[0].result.get("regen_hint_seen"), "")
        self.assertNotEqual(regen_runs[1].result.get("regen_hint_seen"), "")


# ---------------------------------------------------------------------------
# Backwards compat: V1-style orchestrator (no post_agents) still works.
# ---------------------------------------------------------------------------

class TestOrchestratorBackcompat(unittest.TestCase):

    def test_v1_style_no_post_agents_no_regen(self):
        memory = _StubMemory()
        emotion = _StubEmotionNet()
        reasoner = _ScriptedReasoner(["a good reply about cats and life"])
        orch = AgentOrchestrator(
            agents=[
                MemoryAgent(memory_store=memory),
                EmotionAgent(emotion_net=emotion),
                reasoner,
            ],
            final_builder=FinalResponseBuilder(hfl=None),
            quality_gate=_FailingQualityGate(),
            # No post_agents, no max_regen kwarg => default 1 but no rejection
            # since the reply already contains 'good'.
        )
        ctx = AgentContext(
            user_text="tell me about cats",
            cancel_token=CancellationToken("t-bc"),
            response_threshold=0.7,
        )
        out = orch.run_turn(ctx)
        orch.shutdown()
        self.assertTrue(out.quality.approved)
        self.assertEqual(out.regen_attempts, 0)
        self.assertEqual(reasoner.calls, 1)


if __name__ == "__main__":
    unittest.main()
