import unittest

from PySide6.QtCore import QObject, QCoreApplication, Signal

from conversation_starter import ConversationStarter


class _Decision:
    def __init__(self, allowed):
        self.allowed = allowed


class _BehaviorStub:
    def should_initiate_conversation(self, **_kwargs):
        return _Decision(True)


class _ClientStub:
    def __init__(self):
        self.calls = []

    def get_status_snapshot(self):
        return {"conversation_readiness": {"can_auto_initiate": True}}

    def send_proactive(self, **kwargs):
        self.calls.append(kwargs)


class _EventBusStub(QObject):
    telemetry_updated = Signal(dict)
    log_entry = Signal(str)


class TestConversationStarter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QCoreApplication.instance() or QCoreApplication([])

    def setUp(self):
        self.client = _ClientStub()
        self.event_bus = _EventBusStub()
        self.behavior = _BehaviorStub()
        self.starter = ConversationStarter(
            self.client,
            self.event_bus,
            self.behavior,
        )
        self.starter.enable(interval_ms=300000)

    def tearDown(self):
        self.starter.cleanup()

    def test_user_activity_prevents_startup_from_arming(self):
        self.starter.record_user_activity()
        self.starter._maybe_schedule_startup(
            {"conversation_readiness": {"can_auto_initiate": True}}
        )

        self.assertTrue(self.starter._startup_cancelled_by_user)
        self.assertFalse(self.starter._startup_scheduled)

    def test_canceled_startup_token_cannot_trigger_proactive_send(self):
        self.starter._maybe_schedule_startup(
            {"conversation_readiness": {"can_auto_initiate": True}}
        )
        token = self.starter._startup_token

        self.assertTrue(self.starter._startup_scheduled)

        self.starter.record_user_activity()
        self.starter._trigger(
            source="Startup",
            reason="startup warmup complete",
            startup=True,
            startup_token=token,
        )

        self.assertEqual(self.client.calls, [])


if __name__ == "__main__":
    unittest.main()
