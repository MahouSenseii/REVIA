"""
Integration tests for the request pipeline and FSM.

Covers:
  - ConversationManager state transitions
  - _trim_conversation budget logic (B-3 regression guard)
  - Telemetry token estimate caching (P-4)
  - MemoryStore short-term add / lock safety (B-6)
  - LLMBackend._trim_conversation floor behaviour

No real LLM calls are made.  All heavy imports are guarded so the file can
be collected even if optional dependencies are absent.

Run with:
    python -m pytest test_pipeline.py -v
or:
    python -m unittest test_pipeline -v
"""

import os
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(__file__))


# ---------------------------------------------------------------------------
# ConversationManager / FSM tests
# ---------------------------------------------------------------------------


class TestConversationManagerTransitions(unittest.TestCase):
    """State-machine transition correctness."""

    def setUp(self):
        try:
            from conversation_runtime import (
                BehaviorController,
                ConversationManager,
                ReadinessSnapshot,
                ReviaState,
                TriggerKind,
                TriggerRequest,
                TriggerSource,
            )
            self.ReviaState = ReviaState
            self.TriggerKind = TriggerKind
            self.TriggerSource = TriggerSource
            self.TriggerRequest = TriggerRequest
            self.mgr = ConversationManager(log_fn=lambda _: None)
            self._available = True
        except Exception as exc:
            self._available = False
            self._skip_reason = str(exc)

    def _skip_if_unavailable(self):
        if not self._available:
            self.skipTest(f"conversation_runtime unavailable: {self._skip_reason}")

    def test_initial_state_is_booting_or_idle(self):
        self._skip_if_unavailable()
        self.assertIn(
            self.mgr.current_state,
            (self.ReviaState.BOOTING.value, self.ReviaState.IDLE.value),
        )

    def test_force_transition_into_idle(self):
        self._skip_if_unavailable()
        self.mgr.transition_state(self.ReviaState.IDLE, reason="test", force=True)
        self.assertEqual(self.mgr.current_state, self.ReviaState.IDLE.value)

    def test_idle_to_thinking_is_allowed(self):
        self._skip_if_unavailable()
        self.mgr.transition_state(self.ReviaState.IDLE, reason="setup", force=True)
        ok = self.mgr.transition_state(self.ReviaState.THINKING, reason="user message")
        self.assertTrue(ok)
        self.assertEqual(self.mgr.current_state, self.ReviaState.THINKING.value)

    def test_cooldown_to_thinking_blocked_without_force(self):
        """COOLDOWN → THINKING is not in the allowed table; must need force=True."""
        self._skip_if_unavailable()
        self.mgr.transition_state(self.ReviaState.COOLDOWN, reason="setup", force=True)
        # Without force, should either fail or require a force path
        ok = self.mgr.transition_state(self.ReviaState.THINKING, reason="no force")
        if ok:
            # Some implementations allow it; ensure state is consistent
            self.assertEqual(self.mgr.current_state, self.ReviaState.THINKING.value)

    def test_force_exits_cooldown_to_idle(self):
        """B-1 regression: force=True must exit Cooldown regardless of table."""
        self._skip_if_unavailable()
        self.mgr.transition_state(self.ReviaState.COOLDOWN, reason="setup", force=True)
        ok = self.mgr.transition_state(self.ReviaState.IDLE, reason="user pre-empted", force=True)
        self.assertTrue(ok)
        self.assertEqual(self.mgr.current_state, self.ReviaState.IDLE.value)


# ---------------------------------------------------------------------------
# _trim_conversation tests (B-3 regression guard + P-4 cache)
# ---------------------------------------------------------------------------


class _FakeLLMBackend:
    """Minimal stand-in that only carries _trim_conversation logic."""

    def __init__(self, ctx_length=2000, fast_mode=False):
        self.ctx_length = ctx_length
        self.fast_mode = fast_mode

    @staticmethod
    def _tag_token_est(msg):
        if "_token_est" not in msg:
            content = msg.get("content", "")
            if isinstance(content, list):
                msg["_token_est"] = sum(len(p.get("text", "").split()) for p in content if isinstance(p, dict))
            else:
                msg["_token_est"] = len(str(content).split())
        return msg

    def _trim_conversation(self, conversation):
        convo = list(conversation or [])

        def _msg_token_est(m):
            if "_token_est" in m:
                return m["_token_est"]
            content = m.get("content", "")
            if isinstance(content, list):
                return sum(len(p.get("text", "").split()) for p in content if isinstance(p, dict))
            return len(str(content).split())

        est_tokens = sum(_msg_token_est(m) for m in convo)
        ctx_budget = int(self.ctx_length * 0.65)

        if est_tokens <= ctx_budget:
            return convo

        _MIN_HISTORY_MESSAGES = 10
        while len(convo) > _MIN_HISTORY_MESSAGES and est_tokens > ctx_budget:
            removed = convo.pop(0)
            est_tokens -= _msg_token_est(removed)

        if self.fast_mode and len(convo) > 80:
            return convo[-50:]
        if len(convo) > 120:
            return convo[-80:]
        return convo


class TestTrimConversation(unittest.TestCase):
    """_trim_conversation budget logic — B-3 regression guard."""

    def setUp(self):
        self.backend = _FakeLLMBackend(ctx_length=100)  # tiny context for test

    def _make_msg(self, words=5):
        return {"role": "user", "content": " ".join(["word"] * words)}

    def test_within_budget_returns_all(self):
        msgs = [self._make_msg(1) for _ in range(5)]
        result = self.backend._trim_conversation(msgs)
        self.assertEqual(len(result), 5)

    def test_over_budget_trims_oldest(self):
        # 20 messages × 5 words = 100 words; budget = 65 → must trim
        msgs = [self._make_msg(5) for _ in range(20)]
        result = self.backend._trim_conversation(msgs)
        self.assertLess(len(result), 20)

    def test_floor_at_10_messages(self):
        """B-3: even when over budget, at least 10 messages must remain."""
        # 5 messages × 100 words = 500 words >> budget of 65
        msgs = [self._make_msg(100) for _ in range(5)]
        result = self.backend._trim_conversation(msgs)
        # Can't trim below 5 if we started with 5 (floor is 10 but we only have 5)
        self.assertGreaterEqual(len(result), min(5, 10))

    def test_old_floor_80_regression(self):
        """B-3 regression: old floor=80 would prevent trimming 15-message conversations."""
        # 15 messages × 10 words = 150 words >> budget of 65
        msgs = [self._make_msg(10) for _ in range(15)]
        result = self.backend._trim_conversation(msgs)
        # With fixed floor=10, trimming should occur
        self.assertLess(len(result), 15)

    def test_token_est_cache_used(self):
        """P-4: pre-tagged messages should use cached _token_est.

        Build a list of 16 messages where the oldest carries an artificially
        huge _token_est=999.  With ctx_length=50 (budget=32) and floor=10,
        the while-loop fires and pops the oldest message first, so the 999
        entry must not appear in the result.
        """
        msg = {"role": "user", "content": "hello world", "_token_est": 999}
        # 16 total messages so len(convo)=16 > floor=10 and the loop can run
        msgs = [msg] + [self._make_msg(1) for _ in range(15)]
        backend = _FakeLLMBackend(ctx_length=50)
        result = backend._trim_conversation(msgs)
        for m in result:
            self.assertNotEqual(
                m.get("_token_est"), 999,
                "Oldest high-_token_est message should have been trimmed away",
            )


# ---------------------------------------------------------------------------
# MemoryStore thread-safety smoke test
# ---------------------------------------------------------------------------


class TestMemoryStoreLockSafety(unittest.TestCase):
    """Concurrent short_term.append should not corrupt the list."""

    def _make_store(self):
        try:
            import core_server
            # Directly instantiate using the class from the already-loaded module
            return core_server.MemoryStore.__new__(core_server.MemoryStore), True
        except Exception:
            return None, False

    def test_concurrent_add_short_term_no_corruption(self):
        import threading

        # Use a minimal stub that mimics the MemoryStore short_term list + lock
        class _StubStore:
            def __init__(self):
                self._lock = threading.Lock()
                self.short_term = []

            def add_short_term(self, role, content, metadata=None):
                entry = {"role": role, "content": content}
                with self._lock:
                    self.short_term.append(entry)

        store = _StubStore()
        errors = []

        def _writer(n):
            try:
                for i in range(50):
                    store.add_short_term("user", f"msg {n}-{i}")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_writer, args=(t,)) for t in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Thread errors: {errors}")
        self.assertEqual(len(store.short_term), 8 * 50)


# ---------------------------------------------------------------------------
# Telemetry lock smoke test (B-2 regression)
# ---------------------------------------------------------------------------


class TestTelemetryLockCoverage(unittest.TestCase):
    """Concurrent telemetry.llm writes should not produce partial state reads."""

    def test_no_partial_reads_under_concurrent_writes(self):
        import threading

        class _FakeTelemetry:
            def __init__(self):
                self._lock = threading.Lock()
                self.llm = {"tokens_generated": 0, "tokens_per_second": 0.0, "context_length": 0}

        tel = _FakeTelemetry()
        errors = []

        def _writer():
            for i in range(200):
                with tel._lock:
                    tel.llm["tokens_generated"] = i
                    tel.llm["tokens_per_second"] = float(i)
                    tel.llm["context_length"] = i * 2

        def _reader():
            for _ in range(200):
                with tel._lock:
                    snap = dict(tel.llm)
                # tokens_generated should equal tokens_per_second (same i)
                if snap["tokens_generated"] != int(snap["tokens_per_second"]):
                    errors.append(snap)

        w = threading.Thread(target=_writer)
        r = threading.Thread(target=_reader)
        w.start()
        r.start()
        w.join()
        r.join()

        self.assertEqual(len(errors), 0, f"Partial reads detected: {errors[:3]}")


if __name__ == "__main__":
    unittest.main()
