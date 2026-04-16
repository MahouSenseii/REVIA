"""Targeted conversation runtime tests for response pacing behavior."""

import unittest

from conversation_runtime import (
    BehaviorController,
    ReadinessSnapshot,
    ReviaState,
    TriggerKind,
    TriggerRequest,
    TriggerSource,
)


def _ready_snapshot() -> ReadinessSnapshot:
    return ReadinessSnapshot(
        startup_phase="Ready",
        startup_complete=True,
        checks={},
        blocking_reasons=[],
        ready=True,
        can_start_conversation=True,
        can_auto_initiate=True,
    )


class TestResponsePacing(unittest.TestCase):
    def setUp(self):
        self.behavior = BehaviorController(
            log_fn=lambda _msg: None,
            response_cooldown_s=10.0,
        )
        self.ready = _ready_snapshot()

    def _user_trigger(self) -> TriggerRequest:
        return TriggerRequest(
            source=TriggerSource.USER_MESSAGE.value,
            kind=TriggerKind.RESPONSE.value,
            reason="manual user message",
            text="hello",
        )

    def test_user_response_allowed_during_cooldown(self):
        self.behavior.start_cooldown("response")
        decision = self.behavior.evaluate(
            self._user_trigger(),
            self.ready,
            ReviaState.COOLDOWN.value,
        )
        self.assertTrue(decision.allowed)

    def test_user_response_allowed_while_previous_output_is_speaking(self):
        decision = self.behavior.evaluate(
            self._user_trigger(),
            self.ready,
            ReviaState.SPEAKING.value,
        )
        self.assertTrue(decision.allowed)

    def test_user_response_still_blocked_while_request_is_thinking(self):
        decision = self.behavior.evaluate(
            self._user_trigger(),
            self.ready,
            ReviaState.THINKING.value,
        )
        self.assertFalse(decision.allowed)
        self.assertIn("state=Thinking", decision.reason)


if __name__ == "__main__":
    unittest.main()
