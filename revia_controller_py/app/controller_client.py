import json
import threading
from concurrent.futures import ThreadPoolExecutor
import requests
from PySide6.QtCore import QObject, QTimer, QUrl, Signal
from PySide6.QtWebSockets import QWebSocket
from PySide6.QtNetwork import QAbstractSocket


class ControllerClient(QObject):
    BASE_URL = "http://127.0.0.1:8123"
    WS_URL = "ws://127.0.0.1:8124"
    poll_result_ready = Signal(object)

    def __init__(self, event_bus, parent=None):
        super().__init__(parent)
        self.event_bus = event_bus
        self.connected = False
        self.ws_connected = False
        self.rest_reachable = False
        self._last_status = {}
        self._poll_inflight = False
        self._request_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="revia-http",
        )

        self._session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=6,
            pool_maxsize=12,
            max_retries=0,
        )
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

        self.ws = QWebSocket()
        self.ws.connected.connect(self._on_ws_connected)
        self.ws.disconnected.connect(self._on_ws_disconnected)
        self.ws.textMessageReceived.connect(self._on_ws_message)
        self.poll_result_ready.connect(self._emit_poll_result)

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
        self.rest_reachable = False
        self.ws_connected = False
        self._set_core_reachability(False)

    def shutdown(self):
        self.stop()
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            self._executor.shutdown(wait=False)

    def _session_get(self, url, **kwargs):
        with self._request_lock:
            return self._session.get(url, **kwargs)

    def _session_post(self, url, **kwargs):
        with self._request_lock:
            return self._session.post(url, **kwargs)

    def _try_connect(self):
        # Only open when socket is fully unconnected (prevents double-open during reconnect)
        if not self.ws_connected and self.ws.state() == QAbstractSocket.UnconnectedState:
            self.ws.open(QUrl(self.WS_URL))

    def _set_core_reachability(self, reachable):
        reachable = bool(reachable)
        if self.connected == reachable:
            return
        self.connected = reachable
        self.event_bus.connection_changed.emit(reachable)

    def _on_ws_connected(self):
        self.ws_connected = True
        self._set_core_reachability(True)

    def _on_ws_disconnected(self):
        self.ws_connected = False
        if not self.rest_reachable:
            self._set_core_reachability(False)

    def _on_ws_message(self, msg):
        try:
            data = json.loads(msg)
            msg_type = data.get("type", "")
            if msg_type == "telemetry_update":
                payload = data.get("data", {}) or {}
                self._last_status = payload
                self.rest_reachable = True
                self._set_core_reachability(True)
                self.event_bus.telemetry_updated.emit(payload)
            elif msg_type == "status_update":
                payload = data.get("status", {}) or {}
                state = data.get("state", "") or payload.get("state", "")
                if payload:
                    self._last_status = payload
                    self.rest_reachable = True
                    self._set_core_reachability(True)
                    self.event_bus.telemetry_updated.emit(payload)
                self.event_bus.status_changed.emit(state)
            elif msg_type == "chat_token":
                payload = {
                    "token": data.get("token", ""),
                    "request_id": data.get("request_id", ""),
                    "turn_id": data.get("turn_id", 0),
                    "mode": data.get("mode", ""),
                }
                self.event_bus.chat_token_payload.emit(payload)
                self.event_bus.chat_token.emit(payload.get("token", ""))
            elif msg_type == "chat_complete":
                payload = {
                    "text": data.get("text", ""),
                    "request_id": data.get("request_id", ""),
                    "turn_id": data.get("turn_id", 0),
                    "mode": data.get("mode", ""),
                    "success": data.get("success", True),
                    "retryable": data.get("retryable", False),
                    "speakable": data.get("speakable", True),
                    "error_type": data.get("error_type", ""),
                    "metadata": data.get("metadata", {}) or {},
                }
                self.event_bus.chat_complete_payload.emit(payload)
                self.event_bus.chat_complete.emit(payload.get("text", ""))
            elif msg_type == "log_entry":
                self.event_bus.log_entry.emit(data.get("text", ""))
            elif msg_type == "proactive_start":
                self.event_bus.proactive_start.emit()
        except Exception:
            pass

    def _poll_status(self):
        if self._poll_inflight:
            return
        self._poll_inflight = True

        def _do():
            try:
                r = self._session_get(
                    f"{self.BASE_URL}/api/status",
                    timeout=(0.5, 1.0),
                )
                result = r.json() if r.ok else None
            except Exception:
                result = None
            self.poll_result_ready.emit(result)

        self._executor.submit(_do)

    def _emit_poll_result(self, data):
        self._poll_inflight = False
        if data:
            self.rest_reachable = True
            self._last_status = data
            self._set_core_reachability(True)
            if not self.ws_connected:
                self._try_connect()
            state = data.get("state", "")
            if state:
                self.event_bus.status_changed.emit(state)
            self.event_bus.telemetry_updated.emit(data)
        else:
            self.rest_reachable = False
            if not self.ws_connected:
                self._set_core_reachability(False)

    def get_status_snapshot(self):
        return dict(self._last_status)

    def send_chat(self, text, image_b64=None, vision_context=None, source=None, reason=None):
        def _do():
            try:
                payload = {"text": text}
                if image_b64:
                    payload["image"] = image_b64
                if vision_context:
                    payload["vision_context"] = vision_context
                if source:
                    payload["source"] = source
                if reason:
                    payload["reason"] = reason
                r = self._session_post(
                    f"{self.BASE_URL}/api/chat",
                    json=payload,
                    timeout=(1.0, 15.0),
                )
                if r.ok:
                    try:
                        data = r.json()
                    except Exception:
                        data = {}
                    if data:
                        self.event_bus.chat_request_accepted.emit({
                            "request_id": data.get("request_id", ""),
                            "turn_id": data.get("turn_id", 0),
                            "text": text,
                            "source": source or "UserMessage",
                        })
                if not r.ok:
                    try:
                        data = r.json()
                    except Exception:
                        data = {}
                    readiness = data.get("conversation_readiness", {}) or {}
                    reasons = readiness.get("blocking_reasons", []) or []
                    detail = (
                        data.get("decision", {}) or {}
                    ).get("reason") or (reasons[0] if reasons else f"HTTP {r.status_code}")
                    message = f"Conversation unavailable right now: {detail}"
                    payload = {
                        "text": message,
                        "request_id": "",
                        "turn_id": 0,
                        "mode": "ERROR_RESPONSE",
                        "success": False,
                        "retryable": True,
                        "speakable": False,
                        "error_type": "conversation_not_ready",
                        "metadata": {"detail": detail},
                    }
                    self.event_bus.chat_complete_payload.emit(payload)
                    self.event_bus.chat_complete.emit(message)
                    self.event_bus.log_entry.emit(f"[Revia] Chat blocked: {detail}")
            except Exception:
                message = (
                    "Could not reach REVIA Core. Make sure the core server is running."
                )
                payload = {
                    "text": message,
                    "request_id": "",
                    "turn_id": 0,
                    "mode": "ERROR_RESPONSE",
                    "success": False,
                    "retryable": True,
                    "speakable": False,
                    "error_type": "connection_error",
                    "metadata": {},
                }
                self.event_bus.chat_complete_payload.emit(payload)
                self.event_bus.chat_complete.emit(message)
                self.event_bus.log_entry.emit(
                    "[ERROR] Failed to send chat message"
                )
        self._executor.submit(_do)

    def get_plugins(self):
        try:
            r = self._session_get(f"{self.BASE_URL}/api/plugins", timeout=2)
            return r.json() if r.ok else []
        except Exception:
            return []

    def toggle_plugin(self, name, enable):
        action = "enable" if enable else "disable"
        try:
            self._session_post(
                f"{self.BASE_URL}/api/plugins/{name}/{action}",
                timeout=2,
            )
        except Exception:
            pass

    def get_neural(self):
        try:
            r = self._session_get(f"{self.BASE_URL}/api/neural", timeout=2)
            return r.json() if r.ok else {}
        except Exception:
            return {}

    def toggle_neural(self, name, enable):
        action = "enable" if enable else "disable"
        try:
            self._session_post(
                f"{self.BASE_URL}/api/neural/{name}/{action}",
                timeout=2,
            )
        except Exception:
            pass

    def get_profile(self):
        try:
            r = self._session_get(f"{self.BASE_URL}/api/profile", timeout=2)
            return r.json() if r.ok else {}
        except Exception:
            return {}

    def save_profile(self, data):
        try:
            self._session_post(
                f"{self.BASE_URL}/api/profile", json=data, timeout=2
            )
        except Exception:
            pass

    def get_emotion_history(self, limit=50):
        try:
            r = self._session_get(
                f"{self.BASE_URL}/api/emotions/history",
                params={"limit": limit},
                timeout=2,
            )
            return r.json() if r.ok else []
        except Exception:
            return []

    def get_current_emotion(self):
        try:
            r = self._session_get(
                f"{self.BASE_URL}/api/emotions/current",
                timeout=2,
            )
            return r.json() if r.ok else {}
        except Exception:
            return {}

    def get_docker_memory_status(self):
        try:
            r = self._session_get(
                f"{self.BASE_URL}/api/memory/docker/status", timeout=2
            )
            return r.json() if r.ok else {}
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Generic helpers (used by integrations tab and other dynamic callers)
    # ------------------------------------------------------------------

    def get(self, path: str, params=None, timeout: int = 3):
        """Synchronous GET against the core REST API. Returns parsed JSON or None."""
        try:
            r = self._session_get(
                f"{self.BASE_URL}{path}",
                params=params,
                timeout=timeout,
            )
            return r.json() if r.ok else None
        except Exception:
            return None

    def post(self, path: str, json=None, timeout: int = 5):
        """Synchronous POST against the core REST API. Returns parsed JSON or None."""
        try:
            r = self._session_post(
                f"{self.BASE_URL}{path}",
                json=json or {},
                timeout=timeout,
            )
            return r.json() if r.ok else None
        except Exception:
            return None

    def get_websearch_status(self) -> dict:
        return self.get("/api/websearch/status") or {}

    def toggle_websearch(self, enable: bool):
        action = "enable" if enable else "disable"
        self.post(f"/api/websearch/{action}")

    def send_proactive(self, force=False, source=None, reason=None):
        """Ask the core server to generate a proactive message from Revia."""
        def _do():
            try:
                payload = {"force": bool(force)}
                if source:
                    payload["source"] = source
                if reason:
                    payload["reason"] = reason
                r = self._session_post(
                    f"{self.BASE_URL}/api/proactive",
                    json=payload,
                    timeout=(1.0, 15.0),
                )
                if not r.ok:
                    try:
                        data = r.json()
                    except Exception:
                        data = {}
                    reason = (
                        data.get("decision", {}) or {}
                    ).get("reason") or f"HTTP {r.status_code}"
                    self.event_bus.log_entry.emit(
                        f"[Revia] Proactive trigger blocked: {reason}"
                    )
            except Exception:
                self.event_bus.log_entry.emit(
                    "[ERROR] Failed to trigger proactive message"
                )
        self._executor.submit(_do)

    def push_runtime_config(self, data):
        return self.post("/api/runtime/config", json=data, timeout=3)
