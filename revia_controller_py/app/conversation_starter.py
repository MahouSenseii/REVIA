"""ConversationStarter manages deliberate autonomous chat triggers."""

import time

from PySide6.QtCore import QObject, QTimer


class ConversationStarter(QObject):
    def __init__(self, client, event_bus, behavior_controller, interval_ms=300_000, parent=None):
        super().__init__(parent)
        self._client = client
        self._event_bus = event_bus
        self._behavior = behavior_controller
        self._interval_ms = interval_ms
        self._enabled = False
        self._last_activity = time.monotonic()
        self._startup_delay_ms = 4000
        self._startup_scheduled = False
        self._startup_sent = False
        self._startup_cancelled_by_user = False
        self._startup_token = 0

        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._on_tick)

        self._event_bus.telemetry_updated.connect(self._on_telemetry)

    def enable(self, interval_ms=None):
        if interval_ms is not None:
            self._interval_ms = interval_ms
            self._timer.setInterval(interval_ms)
        self._enabled = True
        self._timer.start()

    def disable(self):
        self._enabled = False
        self._timer.stop()

    def cleanup(self):
        """Stop the timer and clean up resources."""
        try:
            self._timer.stop()
        except Exception:
            pass

    @property
    def is_enabled(self):
        return self._enabled

    def record_user_activity(self):
        self._last_activity = time.monotonic()
        if not self._startup_sent:
            self._startup_cancelled_by_user = True
            self._startup_scheduled = False
            self._startup_token += 1

    def greet_on_startup(self, delay_ms=4_000):
        self._startup_delay_ms = int(delay_ms)
        self._maybe_schedule_startup()

    def _on_tick(self):
        if not self._enabled:
            return
        idle_s = time.monotonic() - self._last_activity
        if idle_s >= (self._interval_ms / 1000.0):
            self._trigger(
                source="IdleTimer",
                reason="idle autonomous conversation",
                force=False,
            )

    def _on_telemetry(self, data):
        if not isinstance(data, dict):
            return
        self._maybe_schedule_startup(data)

    def _maybe_schedule_startup(self, data=None):
        if (
            not self._enabled
            or self._startup_sent
            or self._startup_scheduled
            or self._startup_cancelled_by_user
        ):
            return
        snapshot = data if isinstance(data, dict) else self._client.get_status_snapshot()
        readiness = (snapshot.get("conversation_readiness", {}) or {})
        if not readiness.get("can_auto_initiate", False):
            return
        self._startup_scheduled = True
        self._startup_token += 1
        token = self._startup_token
        self._event_bus.log_entry.emit("[Revia] Startup autonomous line armed.")
        QTimer.singleShot(
            self._startup_delay_ms,
            lambda: self._trigger(
                source="Startup",
                reason="startup warmup complete",
                force=False,
                startup=True,
                startup_token=token,
            ),
        )

    def _trigger(self, source, reason, force=False, startup=False, startup_token=None):
        if not self._enabled:
            return
        if startup:
            if self._startup_cancelled_by_user:
                self._startup_scheduled = False
                return
            if startup_token is not None and startup_token != self._startup_token:
                self._startup_scheduled = False
                return
        decision = self._behavior.should_initiate_conversation(
            source=source,
            reason=reason,
            force=force,
            require_speech_output=True,
        )
        if not decision.allowed:
            if startup:
                self._startup_scheduled = False
            return
        self._last_activity = time.monotonic()
        if startup:
            self._startup_sent = True
            # Also clear the "pending" flag so that a future reconnect cycle
            # (e.g. after losing and regaining the LLM connection) can re-arm
            # a startup greeting if _startup_sent is reset externally.
            self._startup_scheduled = False
        self._client.send_proactive(force=force, source=source, reason=reason)
