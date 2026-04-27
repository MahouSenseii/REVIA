import random
import unittest
from datetime import datetime, timedelta

from autonomy.autonomy_loop import ReviaAutonomyLoop
from autonomy.candidate_generator import CandidateGenerator
from autonomy.cooldown_manager import CooldownManager
from autonomy.memory_retriever import MemoryRetriever
from autonomy.self_initiation_scorer import SelfInitiationScorer
from autonomy.state_tracker import StateTracker
from autonomy.topic_manager import TopicManager


class _Conversation:
    current_state = "Idle"


class _Turns:
    def snapshot(self):
        return {"active_request_id": "", "active_turn": None}


class _Memory:
    def __init__(self, messages=None):
        self._messages = messages or []

    def get_short_term(self, limit=50):
        return list(self._messages[-limit:])

    def get_long_term(self, limit=50, category=None):
        return []

    def search(self, query, max_results=5):
        return []


def _ago(seconds):
    return (datetime.now() - timedelta(seconds=seconds)).isoformat()


def _build_loop(messages, *, mode="stream", idle_s=400):
    memory = _Memory(messages)
    state = StateTracker(
        conversation_manager=_Conversation(),
        turn_manager=_Turns(),
        memory_store=memory,
        profile_provider=lambda: {
            "character_name": "Revia",
            "behavior": {"autonomy_mode": mode},
        },
        emotion_provider=lambda: {"label": "Neutral"},
        user_activity_seconds_provider=lambda: idle_s,
    )
    topics = TopicManager(memory_store=memory)
    return ReviaAutonomyLoop(
        state_tracker=state,
        topic_manager=topics,
        memory_retriever=MemoryRetriever(memory_store=memory),
        candidate_generator=CandidateGenerator(),
        scorer=SelfInitiationScorer(),
        cooldown_manager=CooldownManager(),
        rng=random.Random(1),
    )


class TestAutonomyLoop(unittest.TestCase):
    def test_quiet_request_blocks_before_generation(self):
        loop = _build_loop([
            {"role": "user", "content": "wait, quiet", "timestamp": _ago(30)}
        ])

        decision = loop.evaluate_once()

        self.assertFalse(decision.should_speak)
        self.assertIn("user recently asked for quiet", decision.do_not_talk_reasons)

    def test_topic_candidate_can_beat_silence_in_stream_mode(self):
        loop = _build_loop([
            {
                "role": "user",
                "content": "we should continue the REVIA autonomy system next",
                "timestamp": _ago(360),
            }
        ])

        decision = loop.evaluate_once()

        self.assertTrue(decision.should_speak)
        self.assertIsNotNone(decision.selected)
        self.assertNotEqual(decision.selected.candidate.type, "silence")
        self.assertIn("Candidate type:", decision.prompt)


if __name__ == "__main__":
    unittest.main()
