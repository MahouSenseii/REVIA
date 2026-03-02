import json
import threading
import requests
from PySide6.QtCore import QObject, QTimer, QUrl
from PySide6.QtWebSockets import QWebSocket
from PySide6.QtNetwork import QAbstractSocket


class ControllerClient(QObject):
    BASE_URL = "http://127.0.0.1:8123"
    WS_URL = "ws://127.0.0.1:8124"

    def __init__(self, event_bus, parent=None):
        super().__init__(parent)
        self.event_bus = event_bus
        self.connected = False

        self.ws = QWebSocket()
        self.ws.connected.connect(self._on_ws_connected)
        self.ws.disconnected.connect(self._on_ws_disconnected)
        self.ws.textMessageReceived.connect(self._on_ws_message)

        self.reconnect_timer = QTimer(self)
        self.reconnect_timer.timeout.connect(self._try_connect)
        self.reconnect_timer.setInterval(3000)

        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self._poll_status)
        self.poll_timer.setInterval(2000)

    def start(self):
        self._try_connect()
        self.reconnect_timer.start()
        self.poll_timer.start()

    def stop(self):
        self.reconnect_timer.stop()
        self.poll_timer.stop()
        self.ws.close()

    def _try_connect(self):
        # Only open when socket is fully unconnected (prevents double-open during reconnect)
        if not self.connected and self.ws.state() == QAbstractSocket.UnconnectedState:
            self.ws.open(QUrl(self.WS_URL))

    def _on_ws_connected(self):
        self.connected = True
        self.event_bus.connection_changed.emit(True)

    def _on_ws_disconnected(self):
        self.connected = False
        self.event_bus.connection_changed.emit(False)

    def _on_ws_message(self, msg):
        try:
            data = json.loads(msg)
            msg_type = data.get("type", "")
            if msg_type == "telemetry_update":
                self.event_bus.telemetry_updated.emit(data.get("data", {}))
            elif msg_type == "status_update":
                self.event_bus.status_changed.emit(data.get("state", ""))
            elif msg_type == "chat_token":
                self.event_bus.chat_token.emit(data.get("token", ""))
            elif msg_type == "chat_complete":
                self.event_bus.chat_complete.emit(data.get("text", ""))
            elif msg_type == "log_entry":
                self.event_bus.log_entry.emit(data.get("text", ""))
        except Exception:
            pass

    def _poll_status(self):
        def _do():
            try:
                r = requests.get(f"{self.BASE_URL}/api/status", timeout=1)
                if r.ok:
                    self._last_poll = r.json()
                else:
                    self._last_poll = None
            except Exception:
                self._last_poll = None
        threading.Thread(target=_do, daemon=True).start()
        # Emit result on main thread after the request has time to complete.
        # Also trigger a WS reconnect attempt here (safe: runs on main thread).
        QTimer.singleShot(1200, self._emit_poll_result)

    def _emit_poll_result(self):
        data = getattr(self, "_last_poll", None)
        if data:
            # Core is reachable — attempt WS reconnect if still disconnected (main thread, safe)
            if not self.connected:
                self._try_connect()
            self.event_bus.telemetry_updated.emit({
                "system": data.get("system", {}),
                "llm": data.get("llm", {}),
            })

    def send_chat(self, text, image_b64=None):
        def _do():
            try:
                payload = {"text": text}
                if image_b64:
                    payload["image"] = image_b64
                requests.post(
                    f"{self.BASE_URL}/api/chat",
                    json=payload,
                    timeout=120,
                )
            except Exception:
                self.event_bus.log_entry.emit(
                    "[ERROR] Failed to send chat message"
                )
        threading.Thread(target=_do, daemon=True).start()

    def get_plugins(self):
        try:
            r = requests.get(f"{self.BASE_URL}/api/plugins", timeout=2)
            return r.json() if r.ok else []
        except Exception:
            return []

    def toggle_plugin(self, name, enable):
        action = "enable" if enable else "disable"
        try:
            requests.post(f"{self.BASE_URL}/api/plugins/{name}/{action}", timeout=2)
        except Exception:
            pass

    def get_neural(self):
        try:
            r = requests.get(f"{self.BASE_URL}/api/neural", timeout=2)
            return r.json() if r.ok else {}
        except Exception:
            return {}

    def toggle_neural(self, name, enable):
        action = "enable" if enable else "disable"
        try:
            requests.post(f"{self.BASE_URL}/api/neural/{name}/{action}", timeout=2)
        except Exception:
            pass

    def get_profile(self):
        try:
            r = requests.get(f"{self.BASE_URL}/api/profile", timeout=2)
            return r.json() if r.ok else {}
        except Exception:
            return {}

    def save_profile(self, data):
        try:
            requests.post(
                f"{self.BASE_URL}/api/profile", json=data, timeout=2
            )
        except Exception:
            pass
