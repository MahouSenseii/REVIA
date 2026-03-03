"""ConversationStarter — lets Revia initiate conversations proactively.

Behaviour
---------
* On startup (after a short delay) Revia sends a greeting.
* A repeating timer fires every `interval_ms` milliseconds of *user inactivity*.
  If the user has been active recently the tick is skipped until the next one.
* The host panel can call `record_user_activity()` on every user send so the
  idle window resets properly.
* The starter can be enabled/disabled at runtime.
"""
import time

from PySide6.QtCore import QObject, QTimer


class ConversationStarter(QObject):
    def __init__(self, client, event_bus, interval_ms=300_000, parent=None):
        """
        Parameters
        ----------
        client       : ControllerClient — used to call send_proactive()
        event_bus    : EventBus
        interval_ms  : milliseconds of inactivity before triggering (default 5 min)
        """
        super().__init__(parent)
        self._client = client
        self._event_bus = event_bus
        self._interval_ms = interval_ms
        self._enabled = False
        self._last_activity = time.monotonic()
        self._greeted = False  # Prevent repeated startup greetings on reconnect

        # Repeating inactivity timer
        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._on_tick)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enable(self, interval_ms=None):
        if interval_ms is not None:
            self._interval_ms = interval_ms
            self._timer.setInterval(interval_ms)
        self._enabled = True
        self._timer.start()

    def disable(self):
        self._enabled = False
        self._timer.stop()

    @property
    def is_enabled(self):
        return self._enabled

    def record_user_activity(self):
        """Call this whenever the user sends a message to reset the idle clock."""
        self._last_activity = time.monotonic()

    def greet_on_startup(self, delay_ms=4_000):
        """Send an opening greeting *delay_ms* milliseconds after startup.
        Only fires once per session regardless of reconnects.
        """
        if self._greeted:
            return
        self._greeted = True
        QTimer.singleShot(delay_ms, self._trigger)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_tick(self):
        idle_s = time.monotonic() - self._last_activity
        # Only trigger if the user has been idle for at least half the interval
        if idle_s >= (self._interval_ms / 1000) * 0.5:
            self._trigger()

    def _trigger(self):
        if not self._enabled:
            return
        if not self._client.connected:
            return
        self._client.send_proactive()
