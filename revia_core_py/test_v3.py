"""Unit tests for V3 — Autonomy & Persistence Layer.

Covers:
    * EpisodicMemoryStore add / search / persistence / time-decay scoring
    * ReflectionAgent label classification + lesson attachment
    * GoalTracker detection + lifecycle + persistence
    * AutonomyScheduler tick + pressure gating + task lifecycle

Run from inside ``revia_core_py``::

    python -m unittest test_v3 -v
"""
from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from typing import Any

from agents.agent_base import AgentContext, CancellationToken

from autonomy_v3 import (
    AutonomyScheduler,
    Episode,
    EpisodicMemoryStore,
    GoalTracker,
    ReflectionAgent,
    register_default_tasks,
)


# ---------------------------------------------------------------------------
# EpisodicMemoryStore
# ---------------------------------------------------------------------------

class TestEpisodicMemoryStore(unittest.TestCase):

    def _store(self, **kw) -> EpisodicMemoryStore:
        return EpisodicMemoryStore(
            path=Path(tempfile.mkstemp(suffix=".jsonl")[1]),
            max_episodes=50,
            **kw,
        )

    def test_add_assigns_id_keywords_timestamp(self):
        s = self._store()
        ep = s.add(user_text="The quick fox jumped",
                   reply_text="That's a classic pangram",
                   intent_label="chat", emotion_label="neutral")
        self.assertTrue(ep.id)
        self.assertGreater(ep.timestamp, 0)
        self.assertIn("quick", ep.keywords)
        self.assertIn("classic", ep.keywords)
        self.assertNotIn("the", ep.keywords)  # filler removed

    def test_search_by_keyword(self):
        s = self._store()
        s.add(user_text="cats are cool", reply_text="indeed")
        s.add(user_text="dogs bark", reply_text="they do")
        s.add(user_text="weather today", reply_text="sunny")
        out = s.search("cats", limit=5)
        self.assertGreaterEqual(len(out), 1)
        self.assertIn("cats", out[0].episode.user_text)

    def test_search_with_intent_filter(self):
        s = self._store()
        s.add(user_text="what is python", reply_text="lang", intent_label="question")
        s.add(user_text="hello there", reply_text="hi", intent_label="greeting")
        out = s.search("python", limit=5, intent_label="question")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].episode.intent_label, "question")

    def test_persistence_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ep.jsonl"
            s1 = EpisodicMemoryStore(path=path, max_episodes=10)
            s1.add(user_text="persist me", reply_text="ok")
            s1.add(user_text="and me too", reply_text="ok2")
            s1.save()
            self.assertTrue(path.exists())
            s2 = EpisodicMemoryStore(path=path, max_episodes=10)
            n = s2.load()
            self.assertEqual(n, 2)
            self.assertEqual(len(s2.all()), 2)

    def test_attach_lesson(self):
        s = self._store()
        ep = s.add(user_text="hi", reply_text="hello")
        ok = s.attach_lesson(ep.id, "happy_path")
        self.assertTrue(ok)
        self.assertEqual(s.recent(1)[0].lesson, "happy_path")
        self.assertFalse(s.attach_lesson("nonexistent", "x"))

    def test_max_episodes_drops_oldest(self):
        s = EpisodicMemoryStore(
            path=Path(tempfile.mkstemp(suffix=".jsonl")[1]),
            max_episodes=3,
        )
        for i in range(5):
            s.add(user_text=f"msg {i}", reply_text=f"reply {i}")
        all_eps = s.all()
        self.assertEqual(len(all_eps), 3)
        self.assertEqual(all_eps[0].user_text, "msg 2")
        self.assertEqual(all_eps[-1].user_text, "msg 4")

    def test_stats_payload_shape(self):
        s = self._store()
        s.add(user_text="a", reply_text="x", intent_label="chat",
              emotion_label="neutral", quality_score=0.8)
        s.add(user_text="b", reply_text="y", intent_label="question",
              emotion_label="curious", quality_score=0.6)
        st = s.stats()
        self.assertEqual(st["count"], 2)
        self.assertIn("avg_quality", st)
        self.assertEqual(st["intents"]["chat"], 1)
        self.assertEqual(st["intents"]["question"], 1)


# ---------------------------------------------------------------------------
# ReflectionAgent
# ---------------------------------------------------------------------------

class TestReflectionAgent(unittest.TestCase):

    def _ctx(self, **meta):
        return AgentContext(
            user_text=meta.pop("user_text", "ignored"),
            cancel_token=CancellationToken("t"),
            metadata=meta,
        )

    def test_user_complaint_label(self):
        ag = ReflectionAgent()
        ctx = self._ctx(
            intent={"label": "complaint", "polarity": "negative"},
            quality={"score": 0.7, "approved": True},
            critic={"recommendation": "accept", "issues": {}},
        )
        out = ag.execute(ctx).result
        self.assertEqual(out["label"], "user_complaint")
        self.assertIn("formality", out["rl_hint"])

    def test_critic_repetition_label(self):
        ag = ReflectionAgent()
        ctx = self._ctx(
            intent={"label": "chat"},
            quality={"score": 0.8, "approved": True},
            critic={"recommendation": "regen",
                    "issues": {"repetition": True}},
        )
        out = ag.execute(ctx).result
        self.assertEqual(out["label"], "critic_repetition")

    def test_low_quality_after_regen(self):
        ag = ReflectionAgent()
        ctx = self._ctx(
            intent={"label": "question"},
            quality={"score": 0.3, "approved": False, "threshold": 0.7},
            critic={"recommendation": "regen", "issues": {}},
            regen_attempts=1,
            threshold=0.7,
        )
        out = ag.execute(ctx).result
        self.assertEqual(out["label"], "regen_overhead")

    def test_attaches_lesson_to_store(self):
        store = EpisodicMemoryStore(
            path=Path(tempfile.mkstemp(suffix=".jsonl")[1]),
            max_episodes=10,
        )
        ep = store.add(user_text="hi", reply_text="hello")
        ag = ReflectionAgent(episode_store=store)
        ctx = self._ctx(
            episode_id=ep.id,
            intent={"label": "chat"},
            quality={"score": 0.85, "approved": True},
            critic={"recommendation": "accept", "issues": {}},
        )
        out = ag.execute(ctx).result
        self.assertEqual(out["label"], "stable_chat")
        # Lesson should be attached.
        self.assertEqual(store.recent(1)[0].lesson, out["lesson"])

    def test_stable_chat_default(self):
        ag = ReflectionAgent()
        ctx = self._ctx(
            intent={"label": "chat", "polarity": "neutral"},
            quality={"score": 0.9, "approved": True},
            critic={"recommendation": "accept", "issues": {}},
        )
        out = ag.execute(ctx).result
        self.assertEqual(out["label"], "stable_chat")


# ---------------------------------------------------------------------------
# GoalTracker
# ---------------------------------------------------------------------------

class TestGoalTracker(unittest.TestCase):

    def test_add_and_open_goals(self):
        t = GoalTracker(path=Path(tempfile.mkstemp(suffix=".json")[1]))
        g = t.add(kind="reminder", title="ping me later")
        self.assertEqual(t.open_goals()[0].id, g.id)
        self.assertEqual(t.stats()["total"], 1)

    def test_status_transitions(self):
        t = GoalTracker(path=Path(tempfile.mkstemp(suffix=".json")[1]))
        g = t.add(kind="follow_up_command", title="do thing")
        self.assertTrue(t.update_status(g.id, "in_progress"))
        self.assertTrue(t.update_status(g.id, "resolved"))
        self.assertFalse(t.update_status(g.id, "bogus"))
        # Resolved goals don't appear in open_goals().
        self.assertEqual(t.open_goals(), [])

    def test_detect_reminder_pattern(self):
        t = GoalTracker(path=Path(tempfile.mkstemp(suffix=".json")[1]))
        added = t.detect_from_turn(
            user_text="please remind me to call mom tomorrow",
            reply_text="I will",
            intent={"label": "command"},
        )
        kinds = {g.kind for g in added}
        self.assertIn("reminder", kinds)

    def test_detect_short_command_followup(self):
        t = GoalTracker(path=Path(tempfile.mkstemp(suffix=".json")[1]))
        added = t.detect_from_turn(
            user_text="write the code please",
            reply_text="ok done",   # very short
            intent={"label": "command"},
        )
        self.assertTrue(any(g.kind == "follow_up_command" for g in added))

    def test_detect_uncertain_question(self):
        t = GoalTracker(path=Path(tempfile.mkstemp(suffix=".json")[1]))
        added = t.detect_from_turn(
            user_text="why is the sky blue?",
            reply_text="It's probably because of light",
            intent={"label": "question"},
            critic={"issues": {"uncertain_facts": True}},
        )
        self.assertTrue(any(g.kind == "answer_pending" for g in added))

    def test_detect_well_being_check(self):
        t = GoalTracker(path=Path(tempfile.mkstemp(suffix=".json")[1]))
        added = t.detect_from_turn(
            user_text="I feel really sad today",
            reply_text="I hear you",
            intent={"label": "emotional_share", "polarity": "negative"},
        )
        wb = [g for g in added if g.kind == "well_being_check_in"]
        self.assertEqual(len(wb), 1)
        self.assertGreater(wb[0].deadline_at, time.time())

    def test_persistence_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "g.json"
            t1 = GoalTracker(path=path)
            t1.add(kind="reminder", title="A")
            t1.add(kind="follow_up_command", title="B")
            t1.save()
            t2 = GoalTracker(path=path)
            n = t2.load()
            self.assertEqual(n, 2)

    def test_expire_overdue(self):
        t = GoalTracker(
            path=Path(tempfile.mkstemp(suffix=".json")[1]),
            auto_expire_s=0.05,
        )
        t.add(kind="reminder", title="x")
        time.sleep(0.06)
        n = t.expire_overdue()
        self.assertEqual(n, 1)
        # Now nothing remains "open".
        self.assertEqual(t.open_goals(), [])


# ---------------------------------------------------------------------------
# AutonomyScheduler
# ---------------------------------------------------------------------------

class TestAutonomyScheduler(unittest.TestCase):

    def test_register_run_once(self):
        sched = AutonomyScheduler(tick_s=0.05)
        calls: list[int] = []
        sched.register("count", fn=lambda: calls.append(1) or len(calls), interval_s=1.0)
        results = sched.run_all()
        self.assertEqual(results.get("count", "").startswith("ok"), True)
        self.assertEqual(len(calls), 1)

    def test_pressure_skip_critical(self):
        # Snapshot reports pressure="critical" -> tick should skip everything.
        class FakeSnap:
            pressure = "critical"
        sched = AutonomyScheduler(snapshot_provider=lambda: FakeSnap())
        sched.register("noop", fn=lambda: 1, interval_s=1.0)
        results = sched.run_once()
        self.assertEqual(results.get("_skipped"), "hardware_pressure_critical")

    def test_failure_increments_failures(self):
        sched = AutonomyScheduler()
        def boom():
            raise ValueError("oops")
        task = sched.register("bad", fn=boom, interval_s=1.0)
        sched.run_all()
        self.assertEqual(task.runs, 0)
        self.assertEqual(task.failures, 1)
        self.assertIn("ValueError", task.last_status)

    def test_set_enabled_skips_task(self):
        sched = AutonomyScheduler()
        calls: list[int] = []
        sched.register("a", fn=lambda: calls.append(1), interval_s=1.0)
        sched.set_enabled("a", False)
        sched.run_all()
        self.assertEqual(calls, [])

    def test_register_default_tasks(self):
        ep_store = EpisodicMemoryStore(
            path=Path(tempfile.mkstemp(suffix=".jsonl")[1]),
        )
        goal_tracker = GoalTracker(
            path=Path(tempfile.mkstemp(suffix=".json")[1]),
        )
        sched = AutonomyScheduler()
        register_default_tasks(
            sched, episode_store=ep_store, goal_tracker=goal_tracker,
        )
        names = [t["name"] for t in sched.status()["tasks"]]
        self.assertIn("persist_memory", names)
        self.assertIn("expire_goals", names)
        self.assertIn("persist_goals", names)

    def test_start_stop_thread(self):
        sched = AutonomyScheduler(tick_s=0.05)
        calls: list[int] = []
        sched.register("a", fn=lambda: calls.append(1), interval_s=0.05)
        sched.start()
        time.sleep(0.20)
        sched.stop()
        self.assertGreaterEqual(len(calls), 1)
        self.assertFalse(sched.is_running())


if __name__ == "__main__":
    unittest.main()
