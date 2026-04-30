"""Tests for the High-severity fixes from REVIA_DEEP_DIVE.md.

Covers:
  * Issue #1  — ConversationStateMachine FSM watchdog & BehaviorController wiring
  * Issue #5  — Phase 6 adapter stubs raise loudly on instantiation
  * Issue #10 — IntentAgent uses the registered ``intent_classify`` route
  * Issue #16 — WebSocketBackend / CompositeBackend / ReviaErrorHandler wiring
"""
from __future__ import annotations

import os
import sys
import time
import unittest

# Make the package directory importable when running this file directly.
_HERE = os.path.dirname(__file__)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


# ---------------------------------------------------------------------------
# Issue #1 — FSM watchdog
# ---------------------------------------------------------------------------

class TestFSMWatchdog(unittest.TestCase):
    def setUp(self):
        from conversation_runtime import (
            BehaviorController,
            ConversationStateMachine,
            ReviaState,
            TriggerKind,
            TriggerRequest,
            TriggerSource,
            ReadinessSnapshot,
            SubsystemStatus,
        )
        self.RS = ReviaState
        self.TR = TriggerRequest
        self.TK = TriggerKind
        self.TS = TriggerSource
        self.logs: list[str] = []
        self.sm = ConversationStateMachine(self.logs.append)
        self.bc = BehaviorController(
            self.logs.append,
            thinking_timeout_s=0.05,
            speaking_timeout_s=0.05,
        )
        self.bc.bind_state_machine(self.sm)
        self.readiness = ReadinessSnapshot(
            startup_phase="Ready",
            startup_complete=True,
            checks={"core": SubsystemStatus(True, True, "Ready", "")},
            blocking_reasons=[],
            ready=True,
            can_start_conversation=True,
            can_auto_initiate=True,
        )

    def _drive_to_thinking(self):
        self.sm.transition(self.RS.INITIALIZING, force=True)
        self.sm.transition(self.RS.IDLE, force=True)
        self.sm.transition(self.RS.THINKING)
        self.assertEqual(self.sm.state, self.RS.THINKING.value)

    def test_force_recover_does_nothing_when_not_stuck(self):
        self._drive_to_thinking()
        recovered = self.sm.force_recover_if_stuck(thinking_timeout_s=10.0)
        self.assertFalse(recovered)
        self.assertEqual(self.sm.state, self.RS.THINKING.value)

    def test_force_recover_unsticks_thinking_after_timeout(self):
        self._drive_to_thinking()
        time.sleep(0.06)
        recovered = self.sm.force_recover_if_stuck(thinking_timeout_s=0.05)
        self.assertTrue(recovered)
        self.assertEqual(self.sm.state, self.RS.IDLE.value)

    def test_evaluate_recovers_stuck_thinking_and_allows_user_reply(self):
        self._drive_to_thinking()
        time.sleep(0.06)  # exceed timeout
        trigger = self.TR(
            source=self.TS.USER_MESSAGE.value,
            kind=self.TK.RESPONSE.value,
            reason="user typed something",
            text="hello revia",
        )
        decision = self.bc.evaluate(
            trigger,
            self.readiness,
            self.sm.state,
        )
        # The watchdog should have unstuck the FSM, so the user reply is allowed.
        self.assertTrue(
            decision.allowed,
            f"expected allowed=True after watchdog recovery, got {decision}",
        )
        self.assertEqual(self.sm.state, self.RS.IDLE.value)

    def test_evaluate_blocks_when_thinking_is_recent(self):
        self._drive_to_thinking()
        # Do NOT sleep past the timeout — the FSM is healthily THINKING.
        trigger = self.TR(
            source=self.TS.USER_MESSAGE.value,
            kind=self.TK.RESPONSE.value,
            reason="user typed something",
            text="hello revia",
        )
        decision = self.bc.evaluate(
            trigger,
            self.readiness,
            self.sm.state,
        )
        self.assertFalse(decision.allowed)
        self.assertIn("Thinking", decision.reason)


# ---------------------------------------------------------------------------
# Issue #5 — Phase 6 adapter stubs raise loudly on construction
# ---------------------------------------------------------------------------

class TestPhase6AdapterGuards(unittest.TestCase):
    def test_memory_adapter_raises_on_construction(self):
        from adapters.memory_adapter import MemoryAdapter
        with self.assertRaises(NotImplementedError) as ctx:
            MemoryAdapter()
        self.assertIn("MemoryAdapter", str(ctx.exception))

    def test_llm_adapter_raises_on_construction(self):
        from adapters.llm_adapter import LlmAdapter
        with self.assertRaises(NotImplementedError) as ctx:
            LlmAdapter()
        self.assertIn("LlmAdapter", str(ctx.exception))

    def test_web_search_adapter_raises_on_construction(self):
        from adapters.web_search_adapter import WebSearchAdapter
        with self.assertRaises(NotImplementedError) as ctx:
            WebSearchAdapter()
        self.assertIn("WebSearchAdapter", str(ctx.exception))

    def test_subclasses_can_still_be_constructed(self):
        """Concrete subclasses must remain instantiable so the abstract
        boundary stays useful for the eventual migration."""
        from adapters.memory_adapter import MemoryAdapter

        class _Concrete(MemoryAdapter):
            def __init__(self):
                # Deliberately do not call super().__init__()
                pass

            def search(self, query, limit=5):
                return []

            def write_turn(self, user_text, assistant_text):
                return None

        instance = _Concrete()
        self.assertEqual(instance.search("anything"), [])


# ---------------------------------------------------------------------------
# Issue #10 — IntentAgent uses the intent_classify route when registered
# ---------------------------------------------------------------------------

class TestIntentRoute(unittest.TestCase):
    def test_intent_classify_route_overrides_heuristic(self):
        from agents.intent_agent import IntentAgent
        from agents.agent_base import AgentContext, CancellationToken
        from agents.model_router import ModelRouter

        router = ModelRouter()
        # rank=10 wins over rank=100 (the heuristic fallback).
        router.register(
            "intent_classify", "stub_ml",
            lambda text, **_kw: {
                "label": "command",
                "confidence": 0.99,
                "is_question": False,
                "is_imperative": True,
                "expects_facts": True,
                "ends_open": False,
                "polarity": "neutral",
                "topic_hint": "",
            },
            rank=10,
        )

        agent = IntentAgent(model_router=router)
        ctx = AgentContext(
            user_text="open the door",
            turn_id="t1",
            conversation_id="c1",
            user_profile="",
            response_threshold=0.0,
            cancel_token=CancellationToken(turn_id="t1"),
            metadata={},
        )
        out = agent.run(ctx)
        self.assertEqual(out["label"], "command")
        self.assertEqual(out["confidence"], 0.99)


# ---------------------------------------------------------------------------
# Issue #16 — WebSocketBackend & CompositeBackend
# ---------------------------------------------------------------------------

class TestWebSocketBackend(unittest.TestCase):
    def test_buffers_when_broadcaster_is_missing(self):
        from error_handler import (
            ErrorReport,
            ErrorReportFactory,
            ErrorSeverity,
            WebSocketBackend,
        )
        backend = WebSocketBackend(None)
        report = ErrorReportFactory.create(
            ErrorSeverity.ERROR, "general", "boom",
        )
        backend.emit(report)
        # Nothing crashes, message is buffered.
        captured: list[dict] = []
        backend.set_broadcaster(captured.append)
        self.assertEqual(len(captured), 1)
        payload = captured[0]
        self.assertEqual(payload["type"], "log_entry")
        self.assertIn("boom", payload["text"])
        self.assertIn("[ERROR]", payload["text"])

    def test_emit_uses_broadcaster_directly_when_wired(self):
        from error_handler import (
            ErrorReportFactory,
            ErrorSeverity,
            WebSocketBackend,
        )
        captured: list[dict] = []
        backend = WebSocketBackend(captured.append)
        backend.emit(ErrorReportFactory.create(
            ErrorSeverity.WARNING, "general", "ping",
        ))
        self.assertEqual(len(captured), 1)
        self.assertIn("ping", captured[0]["text"])

    def test_handler_attach_websocket_broadcaster_is_idempotent(self):
        from error_handler import ReviaErrorHandler

        ReviaErrorHandler.reset_instance()
        try:
            handler = ReviaErrorHandler.get_instance()
            captured: list[dict] = []
            ws1 = handler.attach_websocket_broadcaster(captured.append)
            ws2 = handler.attach_websocket_broadcaster(captured.append)
            # Same backend instance is reused, not duplicated.
            self.assertIs(ws1, ws2)

            handler.error("general", "live")
            self.assertTrue(any("live" in p["text"] for p in captured))
        finally:
            ReviaErrorHandler.reset_instance()

    def test_buffered_lines_replay_on_broadcaster_attach(self):
        from error_handler import ReviaErrorHandler

        ReviaErrorHandler.reset_instance()
        try:
            handler = ReviaErrorHandler.get_instance()
            handler.attach_websocket_broadcaster(None)  # buffer-only
            handler.error("general", "early-startup-error")
            captured: list[dict] = []
            handler.attach_websocket_broadcaster(captured.append)
            self.assertTrue(any("early-startup-error" in p["text"] for p in captured))
        finally:
            ReviaErrorHandler.reset_instance()


if __name__ == "__main__":
    unittest.main()
