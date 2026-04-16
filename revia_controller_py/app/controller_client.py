import json
import os
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from itertools import count
import requests
from PySide6.QtCore import QObject, QTimer, QUrl, Signal
from PySide6.QtWebSockets import QWebSocket
from PySide6.QtNetwork import QAbstractSocket

_log = logging.getLogger(__name__)


class ControllerClient(QObject):
    BASE_URL = os.environ.get("REVIA_CORE_URL", "http://127.0.0.1:8123")
    WS_URL = os.environ.get("REVIA_CORE_WS_URL", "ws://127.0.0.1:8124")
    poll_result_ready = Signal(object)
    async_result_ready = Signal(object)

    def __init__(self, event_bus, parent=None):
        super().__init__(parent)
        self.event_bus = event_bus
        self.connected = False
        self.ws_connected = False
        self.rest_reachable = False
        self._last_status = {}
        self._poll_inflight = False
        self._executor = ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="revia-http",
        )
        self._async_callbacks = {}
        self._async_callbacks_lock = threading.Lock()
        self._async_callback_ids = count(1)

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
        self.ws.error.connect(self._on_ws_error)
        self.poll_result_ready.connect(self._emit_poll_result)
        self.async_result_ready.connect(self._dispatch_async_result)

        # Exponential backoff state for reconnection attempts.
        # Interval doubles on each failed attempt: 3s → 6s → 12s → … → 30s cap.
        self._reconnect_base_ms = 3000
        self._reconnect_max_ms = 30000
        self._reconnect_attempt = 0

        self.reconnect_timer = QTimer(self)
        self.reconnect_timer.timeout.connect(self._try_connect)
        self.reconnect_timer.setInterval(self._reconnect_base_ms)

        self.ws_connection_timer = QTimer(self)
        self.ws_connection_timer.setSingleShot(True)
        self.ws_connection_timer.setInterval(5000)
        self.ws_connection_timer.timeout.connect(self._on_ws_connection_timeout)

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
        return self._session.get(url, **kwargs)

    def _session_post(self, url, **kwargs):
        return self._session.post(url, **kwargs)

    def _build_url(self, path):
        if str(path).startswith(("http://", "https://")):
            return path
        return f"{self.BASE_URL}{path}"

    def _request(self, method, path, *, params=None, json=None, timeout=3):
        url = self._build_url(path)
        if method == "GET":
            return self._session_get(url, params=params, timeout=timeout)
        return self._session_post(url, params=params, json=json or {}, timeout=timeout)

    @staticmethod
    def _decode_json_response(response, default=None):
        if response is None or not response.ok:
            return default
        try:
            return response.json()
        except Exception:
            return default

    def _register_async_callbacks(self, on_success=None, on_error=None):
        if not callable(on_success) and not callable(on_error):
            return None
        callback_id = next(self._async_callback_ids)
        with self._async_callbacks_lock:
            self._async_callbacks[callback_id] = (on_success, on_error)
        return callback_id

    def _dispatch_async_result(self, payload):
        callback_id = payload.get("callback_id")
        if callback_id is None:
            return
        with self._async_callbacks_lock:
            on_success, on_error = self._async_callbacks.pop(
                callback_id, (None, None)
            )

        try:
            if payload.get("ok"):
                if callable(on_success):
                    on_success(payload.get("data"))
                return
            if callable(on_error):
                on_error(payload.get("error") or "request failed", payload.get("detail"))
            elif payload.get("error"):
                _log.debug(
                    "[ControllerClient] Async request failed: %s",
                    payload.get("error"),
                )
        except Exception:
            _log.exception("[ControllerClient] Async callback failed")

    def _submit_async_json(
        self,
        method,
        path,
        *,
        params=None,
        json=None,
        timeout=3,
        default=None,
        on_success=None,
        on_error=None,
    ):
        callback_id = self._register_async_callbacks(on_success, on_error)

        def _do():
            payload = {
                "callback_id": callback_id,
                "ok": False,
                "data": default,
                "error": "",
                "detail": None,
            }
            try:
                response = self._request(
                    method,
                    path,
                    params=params,
                    json=json,
                    timeout=timeout,
                )
                if response.ok:
                    payload["ok"] = True
                    payload["data"] = self._decode_json_response(
                        response, default=default
                    )
                else:
                    payload["error"] = f"HTTP {response.status_code}"
                    try:
                        payload["detail"] = response.json()
                    except Exception:
                        payload["detail"] = (response.text or "")[:200]
            except Exception as exc:
                payload["error"] = str(exc)

            if payload["callback_id"] is not None:
                self.async_result_ready.emit(payload)
            elif payload.get("error"):
                _log.debug(
                    "[ControllerClient] Async request without callback failed: %s",
                    payload.get("error"),
                )

        try:
            return self._executor.submit(_do)
        except Exception as exc:
            if callback_id is not None:
                self.async_result_ready.emit(
                    {
                        "callback_id": callback_id,
                        "ok": False,
                        "data": default,
                        "error": str(exc),
                        "detail": None,
                    }
                )
            return None

    def get_async(
        self,
        path,
        *,
        params=None,
        timeout=3,
        default=None,
        on_success=None,
        on_error=None,
    ):
        return self._submit_async_json(
            "GET",
            path,
            params=params,
            timeout=timeout,
            default=default,
            on_success=on_success,
            on_error=on_error,
        )

    def post_async(
        self,
        path,
        *,
        json=None,
        timeout=5,
        default=None,
        on_success=None,
        on_error=None,
    ):
        return self._submit_async_json(
            "POST",
            path,
            json=json,
            timeout=timeout,
            default=default,
            on_success=on_success,
            on_error=on_error,
        )

    def _try_connect(self):
        # Only open when socket is fully unconnected (prevents double-open during reconnect)
        if not self.ws_connected and self.ws.state() == QAbstractSocket.UnconnectedState:
            self.ws.open(QUrl(self.WS_URL))
            self.ws_connection_timer.start()

    def _set_core_reachability(self, reachable):
        reachable = bool(reachable)
        if self.connected == reachable:
            return
        self.connected = reachable
        self.event_bus.connection_changed.emit(reachable)

    def _on_ws_connected(self):
        self.ws_connected = True
        self.ws_connection_timer.stop()
        self._set_core_reachability(True)
        # Reset exponential backoff — connection is healthy again.
        self._reconnect_attempt = 0
        self.reconnect_timer.setInterval(self._reconnect_base_ms)

    def _on_ws_disconnected(self):
        self.ws_connected = False
        self.ws_connection_timer.stop()
        if not self.rest_reachable:
            self._set_core_reachability(False)

    def _on_ws_error(self, error):
        _log.debug("[ControllerClient] WebSocket error: %s", error)
        self.ws_connection_timer.stop()

    def _on_ws_connection_timeout(self):
        _log.debug("[ControllerClient] WebSocket connection timeout")
        if not self.ws_connected:
            self.ws.close()
            self._set_core_reachability(False)
            # Advance exponential backoff so repeated failures don't hammer the server.
            self._reconnect_attempt += 1
            new_interval = min(
                self._reconnect_base_ms * (2 ** (self._reconnect_attempt - 1)),
                self._reconnect_max_ms,
            )
            self.reconnect_timer.setInterval(new_interval)
            _log.debug(
                "[ControllerClient] Reconnect backoff: attempt=%d interval=%dms",
                self._reconnect_attempt, new_interval,
            )

    def _on_ws_message(self, msg):
        try:
            data = json.loads(msg)
        except json.JSONDecodeError as exc:
            preview = str(msg or "").replace("\r", "\\r").replace("\n", "\\n")
            _log.warning(
                "[ControllerClient] Failed to decode WS message: %s | preview=%s",
                exc,
                preview[:200],
            )
            return

        msg_type = data.get("type", "")
        try:
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
            elif msg_type == "chat_sentence":
                sentence = data.get("sentence", "")
                req_id = data.get("request_id", "")
                if sentence:
                    self.event_bus.chat_sentence.emit(sentence, req_id)
            elif msg_type == "interrupt_ack":
                self.event_bus.interrupt_ack.emit()
            elif msg_type == "log_entry":
                self.event_bus.log_entry.emit(data.get("text", ""))
            elif msg_type == "proactive_start":
                self.event_bus.proactive_start.emit()
        except Exception as exc:
            _log.exception(
                "[ControllerClient] Error handling WS message type=%s: %s",
                msg_type or "unknown",
                exc,
            )

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
            except Exception as exc:
                _log.debug("[ControllerClient] Poll status error: %s", exc)
                result = None
            self.poll_result_ready.emit(result)

        try:
            future = self._executor.submit(_do)
        except Exception as exc:
            self._poll_inflight = False
            _log.debug("[ControllerClient] Poll submit failed: %s", exc)
            return
        future.add_done_callback(self._on_poll_future_done)

    def _on_poll_future_done(self, future):
        try:
            if future.exception() is not None:
                _log.debug("[ControllerClient] Poll future exception: %s", future.exception())
        except Exception:
            pass
        finally:
            # Always reset inflight flag so polling can resume even after errors
            self._poll_inflight = False

    def _emit_poll_result(self, data):
        # Note: _poll_inflight is reset in _on_poll_future_done (single reset point)
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

    def send_interrupt(self):
        """Tell the server to stop generating tokens immediately."""
        def _do():
            try:
                r = self._session_post(
                    f"{self.BASE_URL}/api/interrupt",
                    json={},
                    timeout=(1.0, 5.0),
                )
                if r.ok:
                    _log.debug("[ControllerClient] Interrupt acknowledged by server")
                else:
                    _log.warning("[ControllerClient] Interrupt request failed: HTTP %s", r.status_code)
            except Exception as exc:
                _log.debug("[ControllerClient] Interrupt request error: %s", exc)
        self._executor.submit(_do)

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
        try:
            self._executor.submit(_do)
        except Exception as exc:
            _log.debug("[ControllerClient] Chat submit failed: %s", exc)
            message = (
                "Could not send the request because the controller worker is unavailable."
            )
            payload = {
                "text": message,
                "request_id": "",
                "turn_id": 0,
                "mode": "ERROR_RESPONSE",
                "success": False,
                "retryable": True,
                "speakable": False,
                "error_type": "controller_unavailable",
                "metadata": {},
            }
            self.event_bus.chat_complete_payload.emit(payload)
            self.event_bus.chat_complete.emit(message)
            self.event_bus.log_entry.emit(
                "[ERROR] Failed to queue chat message"
            )

    def get_plugins(self):
        return self.get("/api/plugins", timeout=2, default=[])

    def toggle_plugin(self, name, enable, *, on_success=None, on_error=None):
        action = "enable" if enable else "disable"
        error_cb = on_error
        if error_cb is None:
            error_cb = (
                lambda error, _detail=None, plugin=name, verb=action: self.event_bus.log_entry.emit(
                    f"[ERROR] Failed to {verb} plugin '{plugin}': {error}"
                )
            )
        return self.post_async(
            f"/api/plugins/{name}/{action}",
            timeout=2,
            default={},
            on_success=on_success,
            on_error=error_cb,
        )

    def get_neural(self):
        return self.get("/api/neural", timeout=2, default={})

    def toggle_neural(self, name, enable, *, on_success=None, on_error=None):
        action = "enable" if enable else "disable"
        error_cb = on_error
        if error_cb is None:
            error_cb = (
                lambda error, _detail=None, module=name, verb=action: self.event_bus.log_entry.emit(
                    f"[ERROR] Failed to {verb} neural module '{module}': {error}"
                )
            )
        return self.post_async(
            f"/api/neural/{name}/{action}",
            timeout=2,
            default={},
            on_success=on_success,
            on_error=error_cb,
        )

    def get_profile(self):
        return self.get("/api/profile", timeout=2, default={})

    def save_profile(self, data):
        return self.post_async(
            "/api/profile",
            json=data,
            timeout=2,
            default={},
            on_error=lambda error, _detail=None: self.event_bus.log_entry.emit(
                f"[ERROR] Failed to save profile: {error}"
            ),
        )

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
        return self.get("/api/memory/docker/status", timeout=2, default={})

    # ------------------------------------------------------------------
    # Generic helpers (used by integrations tab and other dynamic callers)
    # ------------------------------------------------------------------

    def get(self, path: str, params=None, timeout: int = 3, default=None):
        """Synchronous GET against the core REST API. Returns parsed JSON or None."""
        try:
            r = self._request(
                "GET",
                path,
                params=params,
                timeout=timeout,
            )
            return self._decode_json_response(r, default=default)
        except Exception:
            return default

    def post(self, path: str, json=None, timeout: int = 5, default=None):
        """Synchronous POST against the core REST API. Returns parsed JSON or None."""
        try:
            r = self._request(
                "POST",
                path,
                json=json or {},
                timeout=timeout,
            )
            return self._decode_json_response(r, default=default)
        except Exception:
            return default

    def get_websearch_status(self) -> dict:
        return self.get("/api/websearch/status", default={}) or {}

    def toggle_websearch(self, enable: bool):
        action = "enable" if enable else "disable"
        self.post_async(
            f"/api/websearch/{action}",
            default={},
            on_error=lambda error, _detail=None: self.event_bus.log_entry.emit(
                f"[ERROR] Failed to {action} web search: {error}"
            ),
        )

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
                    block_reason = (
                        data.get("decision", {}) or {}
                    ).get("reason") or f"HTTP {r.status_code}"
                    self.event_bus.log_entry.emit(
                        f"[Revia] Proactive trigger blocked: {block_reason}"
                    )
            except Exception as exc:
                self.event_bus.log_entry.emit(
                    f"[ERROR] Failed to trigger proactive message: {exc}"
                )
        self._executor.submit(_do)

    def push_runtime_config(self, data):
        return self.post_async(
            "/api/runtime/config",
            json=data,
            timeout=3,
            default={},
            on_error=lambda error, _detail=None: _log.debug(
                "[ControllerClient] Failed to push runtime config: %s", error
            ),
        )
