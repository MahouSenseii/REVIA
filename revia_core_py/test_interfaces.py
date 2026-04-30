"""Unit tests for V2.4 InterfaceRouter + built-in interfaces.

Run from inside ``revia_core_py``::

    python -m unittest test_interfaces -v
"""
from __future__ import annotations

import threading
import time
import unittest
from dataclasses import dataclass, field
from typing import Any

from interfaces import (
    Interface,
    InterfaceContext,
    InterfaceDecision,
    InterfaceRouter,
    LogInterface,
    NotificationInterface,
    TextChatInterface,
    VisionInterface,
    VoiceInterface,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class _FakeFinal:
    text: str = "Hello world"
    emotion_label: str = "neutral"
    confidence: float = 0.9
    prosody: dict[str, Any] = field(default_factory=dict)
    voice_style: dict[str, Any] = field(default_factory=dict)


def _ctx(text="Hello world", intent_label="chat", **kw) -> InterfaceContext:
    final = _FakeFinal(
        text=text,
        emotion_label=kw.pop("emotion", "neutral"),
        prosody=kw.pop("prosody", {}),
        voice_style=kw.pop("voice_style", {}),
    )
    return InterfaceContext(
        final=final,
        user_text=kw.pop("user_text", "user said hi"),
        intent={"label": intent_label, **kw.pop("intent_extra", {})},
        metadata=kw.pop("metadata", {"turn_id": "t-1"}),
    )


# ---------------------------------------------------------------------------
# TextChatInterface
# ---------------------------------------------------------------------------

class TestTextChatInterface(unittest.TestCase):

    def test_delivers_when_enabled(self):
        captured: list[tuple[str, dict]] = []
        iface = TextChatInterface(
            broadcast_fn=lambda channel, payload: captured.append((channel, payload)),
        )
        d = iface.execute(_ctx(text="Welcome"))
        self.assertTrue(d.delivered)
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0][0], "chat:text")
        self.assertEqual(captured[0][1]["text"], "Welcome")

    def test_skips_when_disabled(self):
        iface = TextChatInterface(broadcast_fn=lambda *a, **k: None, enabled=False)
        d = iface.execute(_ctx())
        self.assertFalse(d.delivered)
        self.assertEqual(d.skipped_reason, "disabled")

    def test_skips_empty_text(self):
        iface = TextChatInterface(broadcast_fn=lambda *a, **k: None)
        d = iface.execute(_ctx(text=""))
        self.assertFalse(d.delivered)
        self.assertEqual(d.skipped_reason, "empty_text")


# ---------------------------------------------------------------------------
# VoiceInterface
# ---------------------------------------------------------------------------

class TestVoiceInterface(unittest.TestCase):

    def test_delivered_with_callback_and_enabled(self):
        spoken: list[dict] = []

        def speak(text, rate, pitch_shift_st, emotion):
            spoken.append({"text": text, "rate": rate,
                           "pitch_shift_st": pitch_shift_st, "emotion": emotion})
            return True

        iface = VoiceInterface(speak_fn=speak, enabled=True)
        ctx = _ctx(prosody={"speech_rate": 1.05, "pitch_shift_st": 0.5})
        d = iface.execute(ctx)
        self.assertTrue(d.delivered)
        self.assertEqual(len(spoken), 1)
        self.assertAlmostEqual(spoken[0]["rate"], 1.05)
        self.assertEqual(d.payload["emotion"], "neutral")

    def test_skipped_when_no_callback(self):
        iface = VoiceInterface(speak_fn=None, enabled=True)
        d = iface.execute(_ctx())
        self.assertFalse(d.delivered)
        self.assertEqual(d.skipped_reason, "no_tts_callback")

    def test_skipped_for_system_log_intent(self):
        iface = VoiceInterface(speak_fn=lambda *a, **k: True, enabled=True)
        d = iface.execute(_ctx(intent_label="system_log"))
        self.assertFalse(d.delivered)
        self.assertIn("skips_voice", d.skipped_reason)

    def test_disabled_by_default(self):
        iface = VoiceInterface(speak_fn=lambda *a, **k: True)
        self.assertFalse(iface.enabled)
        d = iface.execute(_ctx())
        self.assertFalse(d.delivered)
        self.assertEqual(d.skipped_reason, "disabled")


# ---------------------------------------------------------------------------
# VisionInterface
# ---------------------------------------------------------------------------

class TestVisionInterface(unittest.TestCase):

    def test_dispatches_emotion_and_tone(self):
        seen: dict = {}

        def avatar(emotion, tone, text):
            seen["emotion"] = emotion
            seen["tone"] = tone

        iface = VisionInterface(avatar_fn=avatar, enabled=True)
        ctx = _ctx(emotion="Joy", prosody={"tone": "playful"})
        d = iface.execute(ctx)
        self.assertTrue(d.delivered)
        self.assertEqual(seen["emotion"], "Joy")
        self.assertEqual(seen["tone"], "playful")

    def test_skipped_without_callback(self):
        iface = VisionInterface(avatar_fn=None, enabled=True)
        d = iface.execute(_ctx())
        self.assertFalse(d.delivered)
        self.assertEqual(d.skipped_reason, "no_avatar_callback")


# ---------------------------------------------------------------------------
# LogInterface
# ---------------------------------------------------------------------------

class TestLogInterface(unittest.TestCase):

    def test_logs_even_for_empty_text(self):
        captured: list[str] = []
        iface = LogInterface(log_fn=captured.append, enabled=True)
        d = iface.execute(_ctx(text=""))
        self.assertTrue(d.delivered)
        self.assertEqual(len(captured), 1)
        self.assertIn("[Interface]", captured[0])

    def test_disabled_log_skips(self):
        captured: list[str] = []
        iface = LogInterface(log_fn=captured.append, enabled=False)
        d = iface.execute(_ctx())
        self.assertFalse(d.delivered)
        self.assertEqual(captured, [])


# ---------------------------------------------------------------------------
# NotificationInterface
# ---------------------------------------------------------------------------

class TestNotificationInterface(unittest.TestCase):

    def test_dispatches_only_for_trigger_intent(self):
        sent: list[dict] = []

        def notify(title, body):
            sent.append({"title": title, "body": body})
            return True

        iface = NotificationInterface(notify_fn=notify, enabled=True)
        # Trigger intent.
        d_ok = iface.execute(_ctx(intent_label="emotional_share"))
        # Non-trigger intent.
        d_skip = iface.execute(_ctx(intent_label="chat"))
        self.assertTrue(d_ok.delivered)
        self.assertFalse(d_skip.delivered)
        self.assertEqual(len(sent), 1)
        self.assertIn("not_in_trigger_set", d_skip.skipped_reason)

    def test_skipped_without_callback(self):
        iface = NotificationInterface(notify_fn=None, enabled=True)
        d = iface.execute(_ctx(intent_label="emotional_share"))
        self.assertFalse(d.delivered)
        self.assertEqual(d.skipped_reason, "no_notify_callback")


# ---------------------------------------------------------------------------
# Custom interface that raises -> surfaced as decision.error
# ---------------------------------------------------------------------------

class _FailingInterface(Interface):
    name = "failing"
    kind = "system"

    def deliver(self, ctx):
        raise RuntimeError("boom")


class _SlowInterface(Interface):
    name = "slow"
    kind = "system"

    def __init__(self, delay_s: float):
        super().__init__(enabled=True)
        self._delay = float(delay_s)

    def accept(self, ctx):
        return True, ""

    def deliver(self, ctx):
        time.sleep(self._delay)
        return {"slept": self._delay}


# ---------------------------------------------------------------------------
# InterfaceRouter
# ---------------------------------------------------------------------------

class TestInterfaceRouter(unittest.TestCase):

    def test_dispatch_runs_all_in_parallel(self):
        captured: list[tuple[str, dict]] = []
        text_iface = TextChatInterface(
            broadcast_fn=lambda c, p: captured.append((c, p)),
        )
        log_lines: list[str] = []
        log_iface = LogInterface(log_fn=log_lines.append)
        router = InterfaceRouter([text_iface, log_iface])

        out = router.dispatch(_ctx(text="hi"))
        router.shutdown()
        self.assertEqual(out.delivered_count(), 2)
        self.assertEqual(len(captured), 1)
        self.assertEqual(len(log_lines), 1)

    def test_failure_in_one_does_not_block_others(self):
        log_lines: list[str] = []
        router = InterfaceRouter([
            _FailingInterface(),
            LogInterface(log_fn=log_lines.append),
        ])
        out = router.dispatch(_ctx())
        router.shutdown()
        # Two decisions, one delivered (log), one failed (failing).
        self.assertEqual(len(out.decisions), 2)
        self.assertEqual(out.delivered_count(), 1)
        failed = [d for d in out.decisions if d.interface == "failing"]
        self.assertEqual(len(failed), 1)
        self.assertIn("RuntimeError", failed[0].error)

    def test_set_enabled_toggles_channel(self):
        iface = TextChatInterface(broadcast_fn=lambda *a, **k: None)
        router = InterfaceRouter([iface])
        self.assertTrue(router.set_enabled("text_chat", False))
        out = router.dispatch(_ctx())
        router.shutdown()
        decision = out.decisions[0]
        self.assertFalse(decision.delivered)
        self.assertEqual(decision.skipped_reason, "disabled")

    def test_set_enabled_unknown_channel_returns_false(self):
        router = InterfaceRouter([])
        self.assertFalse(router.set_enabled("nonexistent", True))
        router.shutdown()

    def test_status_payload(self):
        router = InterfaceRouter([
            TextChatInterface(broadcast_fn=lambda *a, **k: None),
            VoiceInterface(speak_fn=None),
        ])
        st = router.status()
        router.shutdown()
        names = {entry["name"] for entry in st["interfaces"]}
        self.assertEqual(names, {"text_chat", "voice"})
        self.assertEqual(st["count"], 2)
        self.assertTrue(st["parallel"])

    def test_parallel_dispatch_is_actually_parallel(self):
        # Two slow interfaces each sleeping 100ms.  Parallel total < 200ms.
        router = InterfaceRouter(
            [_SlowInterface(0.1), _SlowInterface(0.1)],
            parallel=True,
            per_interface_timeout_s=2.0,
        )
        # rename to keep unique names
        router.remove("slow")
        s1 = _SlowInterface(0.1); s1.name = "slow_a"
        s2 = _SlowInterface(0.1); s2.name = "slow_b"
        router.add(s1)
        router.add(s2)
        t0 = time.monotonic()
        out = router.dispatch(_ctx())
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        router.shutdown()
        self.assertEqual(out.delivered_count(), 2)
        # Allow generous slack but ensure we're nowhere near serial 200ms.
        self.assertLess(elapsed_ms, 180)

    def test_per_interface_timeout(self):
        router = InterfaceRouter(
            [_SlowInterface(0.5)],   # 500ms task
            parallel=True,
            per_interface_timeout_s=0.05,   # 50ms cap
        )
        out = router.dispatch(_ctx())
        router.shutdown()
        decision = out.decisions[0]
        self.assertFalse(decision.delivered)
        self.assertIn("timeout", decision.error)

    def test_add_remove_get_names(self):
        router = InterfaceRouter([])
        router.add(TextChatInterface(broadcast_fn=lambda *a, **k: None))
        self.assertIn("text_chat", router.names())
        self.assertIsNotNone(router.get("text_chat"))
        removed = router.remove("text_chat")
        self.assertIsNotNone(removed)
        self.assertNotIn("text_chat", router.names())
        router.shutdown()

    def test_dispatch_output_to_dict(self):
        router = InterfaceRouter([
            LogInterface(log_fn=lambda *_a, **_k: None),
        ])
        out = router.dispatch(_ctx())
        router.shutdown()
        d = out.to_dict()
        self.assertEqual(d["delivered_count"], 1)
        self.assertEqual(d["total"], 1)
        self.assertEqual(len(d["decisions"]), 1)
        self.assertIn("elapsed_ms", d)


# ---------------------------------------------------------------------------
# InterfaceContext property accessors (works on dataclass + dict alike)
# ---------------------------------------------------------------------------

class TestInterfaceContextAccessors(unittest.TestCase):

    def test_accepts_dict_final(self):
        ctx = InterfaceContext(final={
            "text": "hi from dict",
            "emotion_label": "Joy",
            "prosody": {"speech_rate": 1.1},
            "voice_style": {"tone": "playful"},
        })
        self.assertEqual(ctx.text, "hi from dict")
        self.assertEqual(ctx.emotion_label, "Joy")
        self.assertEqual(ctx.prosody["speech_rate"], 1.1)
        self.assertEqual(ctx.voice_style["tone"], "playful")

    def test_accepts_dataclass_final(self):
        ctx = InterfaceContext(final=_FakeFinal(
            text="from dataclass",
            emotion_label="Calm",
            prosody={"pitch_shift_st": -1.0},
            voice_style={"tone": "warm"},
        ))
        self.assertEqual(ctx.text, "from dataclass")
        self.assertEqual(ctx.emotion_label, "Calm")
        self.assertEqual(ctx.prosody["pitch_shift_st"], -1.0)


if __name__ == "__main__":
    unittest.main()
