"""
REVIA VTube Studio Integration Service
=======================================

Connects REVIA to VTube Studio via its WebSocket API (port 8001 by default).
Provides:
  - Authentication token flow (request → approve in VTube Studio → store token)
  - Emotion → expression mapping (13 emotions → hotkey triggers + parameter sets)
  - Real-time lip sync via ParamMouthOpenY while TTS is speaking
  - Idle micro-movement (subtle head drift when not speaking)
  - Clean connect / disconnect / status API

VTube Studio API reference:
  https://github.com/DenchiSoft/VTubeStudio

Usage
-----
    from app.vtube_service import VTubeService
    svc = VTubeService(event_bus=bus, settings_path="vtube_settings.json")
    svc.connect()                       # opens WebSocket, starts auth
    svc.set_emotion("happy")            # trigger happy expression
    svc.set_speaking(True)              # start lip sync pulsing
    svc.set_speaking(False)             # stop lip sync, close mouth
    svc.disconnect()
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import threading
import time
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, Signal

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VTUBE_WS_HOST_DEFAULT = "127.0.0.1"
VTUBE_WS_PORT_DEFAULT = 8001
PLUGIN_NAME = "REVIA Neural Assistant"
PLUGIN_DEVELOPER = "REVIA AI"
API_NAME = "VTubeStudioPublicAPI"
API_VERSION = "1.0"

# How fast the mouth pulses while speaking (seconds per open/close cycle)
_LIP_SYNC_HZ = 8.0  # 8 open/close cycles per second
_LIP_SYNC_INTERVAL = 1.0 / _LIP_SYNC_HZ

# Idle micro-movement period (seconds per gentle drift cycle)
_IDLE_DRIFT_PERIOD = 3.0   # slow, barely-noticeable
_IDLE_DRIFT_AMOUNT = 0.04  # subtle ±4% parameter nudge

# Emotion → VTube Studio expression mapping.
# Each entry maps to:
#   "expression": str | None   — expression name to activate (model-specific)
#   "params": dict[str,float]  — parameter overrides while emotion is active
#   "duration": float          — seconds to hold the expression before fading
_EMOTION_MAP: dict[str, dict] = {
    "happy": {
        "expression": "happy",
        "params": {"ParamBrowLeftY": 0.3, "ParamBrowRightY": 0.3, "ParamEyeLOpen": 1.0, "ParamEyeROpen": 1.0},
        "duration": 4.0,
    },
    "excited": {
        "expression": "excited",
        "params": {"ParamBrowLeftY": 0.6, "ParamBrowRightY": 0.6, "ParamEyeLOpen": 1.2, "ParamEyeROpen": 1.2},
        "duration": 3.0,
    },
    "curious": {
        "expression": "curious",
        "params": {"ParamBrowLeftY": 0.2, "ParamBrowRightY": -0.1, "ParamAngleZ": 5.0},
        "duration": 3.0,
    },
    "sad": {
        "expression": "sad",
        "params": {"ParamBrowLeftY": -0.4, "ParamBrowRightY": -0.4, "ParamEyeLOpen": 0.6, "ParamEyeROpen": 0.6},
        "duration": 5.0,
    },
    "angry": {
        "expression": "angry",
        "params": {"ParamBrowLeftY": -0.6, "ParamBrowRightY": -0.6, "ParamEyeLOpen": 0.8, "ParamEyeROpen": 0.8},
        "duration": 3.0,
    },
    "frustrated": {
        "expression": "frustrated",
        "params": {"ParamBrowLeftY": -0.5, "ParamBrowRightY": -0.5},
        "duration": 3.0,
    },
    "fear": {
        "expression": "fear",
        "params": {"ParamBrowLeftY": 0.4, "ParamBrowRightY": 0.4, "ParamEyeLOpen": 1.3, "ParamEyeROpen": 1.3},
        "duration": 2.0,
    },
    "lonely": {
        "expression": "sad",
        "params": {"ParamBrowLeftY": -0.2, "ParamEyeLOpen": 0.7, "ParamEyeROpen": 0.7},
        "duration": 5.0,
    },
    "concerned": {
        "expression": "concerned",
        "params": {"ParamBrowLeftY": -0.1, "ParamBrowRightY": 0.1},
        "duration": 4.0,
    },
    "confident": {
        "expression": "confident",
        "params": {"ParamBrowLeftY": 0.1, "ParamBrowRightY": 0.1, "ParamEyeLOpen": 0.9, "ParamEyeROpen": 0.9},
        "duration": 4.0,
    },
    "nervous": {
        "expression": "nervous",
        "params": {"ParamBrowLeftY": 0.2, "ParamBrowRightY": -0.1},
        "duration": 3.0,
    },
    "amused": {
        "expression": "happy",
        "params": {"ParamBrowLeftY": 0.2, "ParamEyeLOpen": 0.85, "ParamEyeROpen": 0.85},
        "duration": 4.0,
    },
    "neutral": {
        "expression": None,
        "params": {},
        "duration": 0.0,
    },
}


# ---------------------------------------------------------------------------
# VTubeService
# ---------------------------------------------------------------------------

class VTubeService(QObject):
    """Qt-safe VTube Studio integration service.

    All signals are emitted from the correct thread — internal threads marshal
    via queued connections when needed.
    """

    # Public signals
    connected = Signal()
    disconnected = Signal()
    auth_pending = Signal()          # user must approve the plugin in VTube Studio
    auth_failed = Signal(str)
    status_changed = Signal(str)     # human-readable status string
    error_occurred = Signal(str)

    def __init__(
        self,
        event_bus=None,
        host: str = VTUBE_WS_HOST_DEFAULT,
        port: int = VTUBE_WS_PORT_DEFAULT,
        settings_path: str | Path | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.event_bus = event_bus
        self._host = host
        self._port = port

        # Persistent token storage
        if settings_path is None:
            settings_path = Path(__file__).resolve().parents[2] / "vtube_settings.json"
        self._settings_path = Path(settings_path)
        self._auth_token: str = self._load_token()

        # Runtime state
        self._ws = None                  # websocket.WebSocket instance
        self._connected = False
        self._lock = threading.Lock()
        self._req_id = 0

        # Lip sync thread
        self._speaking = False
        self._lip_thread: threading.Thread | None = None
        self._lip_stop = threading.Event()

        # Idle drift thread
        self._idle_thread: threading.Thread | None = None
        self._idle_stop = threading.Event()

        # Current emotion
        self._current_emotion = "neutral"
        self._emotion_timer: threading.Timer | None = None

        # Status callbacks
        self._on_status: Callable[[str], None] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def connect(self, host: str | None = None, port: int | None = None):
        """Connect to VTube Studio and authenticate. Non-blocking."""
        if host:
            self._host = host
        if port:
            self._port = port
        t = threading.Thread(target=self._connect_thread, daemon=True, name="vtube-connect")
        t.start()

    def disconnect(self):
        """Gracefully close the WebSocket connection."""
        self._speaking = False
        self._lip_stop.set()
        self._idle_stop.set()
        with self._lock:
            ws = self._ws
            self._ws = None
            self._connected = False
        if ws:
            try:
                ws.close()
            except Exception:
                pass
        self.status_changed.emit("Disconnected")
        self.disconnected.emit()
        _log.info("[VTube] Disconnected")

    @property
    def is_connected(self) -> bool:
        return self._connected

    def set_emotion(self, emotion: str):
        """Trigger an expression matching the given emotion label."""
        if not self._connected:
            return
        label = emotion.lower().strip()
        if label == self._current_emotion:
            return
        self._current_emotion = label
        mapping = _EMOTION_MAP.get(label, _EMOTION_MAP["neutral"])
        t = threading.Thread(
            target=self._apply_emotion,
            args=(mapping,),
            daemon=True,
            name="vtube-emotion",
        )
        t.start()

    def set_speaking(self, speaking: bool):
        """Start or stop lip sync animation."""
        if speaking == self._speaking:
            return
        self._speaking = speaking
        if speaking:
            self._start_lip_sync()
        else:
            self._stop_lip_sync()

    def send_parameter(self, param_id: str, value: float):
        """Directly set a single VTube Studio parameter value."""
        if not self._connected:
            return
        t = threading.Thread(
            target=self._inject_params,
            args=({param_id: value},),
            daemon=True,
            name="vtube-param",
        )
        t.start()

    # ------------------------------------------------------------------
    # Internal — connection
    # ------------------------------------------------------------------

    def _connect_thread(self):
        try:
            import websocket
        except ImportError:
            msg = "websocket-client not installed. Run: pip install websocket-client"
            _log.error("[VTube] %s", msg)
            self.error_occurred.emit(msg)
            return

        url = f"ws://{self._host}:{self._port}"
        self.status_changed.emit(f"Connecting to {url}…")
        _log.info("[VTube] Connecting to %s", url)

        try:
            ws = websocket.WebSocket()
            ws.connect(url, timeout=5)
        except Exception as exc:
            msg = f"Cannot reach VTube Studio at {url}: {exc}"
            _log.warning("[VTube] %s", msg)
            self.status_changed.emit(f"Failed: {exc}")
            self.error_occurred.emit(msg)
            return

        with self._lock:
            self._ws = ws

        # Authenticate
        ok = self._authenticate()
        if not ok:
            return

        self._connected = True
        self.status_changed.emit("Connected")
        self.connected.emit()
        _log.info("[VTube] Connected and authenticated")

        # Start idle drift
        self._start_idle_drift()

    def _authenticate(self) -> bool:
        """Run the VTube Studio authentication flow. Returns True on success."""
        if self._auth_token:
            # Try stored token first
            ok = self._auth_with_token(self._auth_token)
            if ok:
                return True
            _log.info("[VTube] Stored token rejected, requesting new token")

        # Request a new token (user must approve in VTube Studio)
        self.status_changed.emit("Waiting for VTube Studio approval…")
        self.auth_pending.emit()
        _log.info("[VTube] Requesting plugin auth token — approve in VTube Studio")

        resp = self._send_recv({
            "messageType": "AuthenticationTokenRequest",
            "data": {
                "pluginName": PLUGIN_NAME,
                "pluginDeveloper": PLUGIN_DEVELOPER,
                "pluginIcon": None,
            },
        })
        if not resp:
            self.auth_failed.emit("No response from VTube Studio")
            return False

        token = (resp.get("data") or {}).get("authenticationToken", "")
        if not token:
            msg = f"Auth token not granted: {resp.get('data', {}).get('reason', 'unknown')}"
            self.auth_failed.emit(msg)
            self.status_changed.emit(f"Auth failed: {msg}")
            return False

        self._auth_token = token
        self._save_token(token)
        return self._auth_with_token(token)

    def _auth_with_token(self, token: str) -> bool:
        resp = self._send_recv({
            "messageType": "AuthenticationRequest",
            "data": {
                "pluginName": PLUGIN_NAME,
                "pluginDeveloper": PLUGIN_DEVELOPER,
                "authenticationToken": token,
            },
        })
        if not resp:
            return False
        data = resp.get("data") or {}
        return bool(data.get("authenticated", False))

    # ------------------------------------------------------------------
    # Internal — expressions / lip sync / idle
    # ------------------------------------------------------------------

    def _apply_emotion(self, mapping: dict):
        """Send expression + parameter overrides for the given emotion mapping."""
        expression = mapping.get("expression")
        params = mapping.get("params", {})
        duration = float(mapping.get("duration", 3.0))

        # Trigger named expression if the model has it
        if expression:
            self._send_recv({
                "messageType": "ExpressionActivationRequest",
                "data": {"expressionFile": f"{expression}.exp3.json", "active": True},
            })

        # Apply parameter overrides
        if params:
            self._inject_params(params)

        # Schedule expression reset
        if self._emotion_timer:
            self._emotion_timer.cancel()
        if duration > 0 and expression:
            def _reset():
                if self._connected:
                    self._send_recv({
                        "messageType": "ExpressionActivationRequest",
                        "data": {"expressionFile": f"{expression}.exp3.json", "active": False},
                    })
            self._emotion_timer = threading.Timer(duration, _reset)
            self._emotion_timer.daemon = True
            self._emotion_timer.start()

    def _start_lip_sync(self):
        """Start a background thread that pulses ParamMouthOpenY."""
        self._lip_stop.clear()
        self._lip_thread = threading.Thread(
            target=self._lip_sync_loop,
            daemon=True,
            name="vtube-lipsync",
        )
        self._lip_thread.start()

    def _stop_lip_sync(self):
        """Stop the lip sync thread and close the mouth."""
        self._lip_stop.set()
        if self._lip_thread:
            self._lip_thread.join(timeout=0.5)
        # Close mouth
        if self._connected:
            self._inject_params({"ParamMouthOpenY": 0.0})

    def _lip_sync_loop(self):
        """Pulse ParamMouthOpenY in a sine wave while speaking."""
        t0 = time.monotonic()
        while not self._lip_stop.is_set():
            if not self._connected:
                break
            elapsed = time.monotonic() - t0
            # Sine wave: 0 → 1 → 0, range 0.2 – 0.95 to look natural
            raw = (math.sin(2 * math.pi * _LIP_SYNC_HZ * elapsed) + 1) / 2
            val = 0.2 + raw * 0.75
            # Add small random jitter for naturalness
            val = max(0.0, min(1.0, val + random.uniform(-0.05, 0.05)))
            self._inject_params({"ParamMouthOpenY": val})
            time.sleep(_LIP_SYNC_INTERVAL)

    def _start_idle_drift(self):
        """Start gentle idle head micro-movement."""
        self._idle_stop.clear()
        self._idle_thread = threading.Thread(
            target=self._idle_drift_loop,
            daemon=True,
            name="vtube-idle",
        )
        self._idle_thread.start()

    def _idle_drift_loop(self):
        """Slowly drift head angle to give a natural 'alive' feel during idle."""
        t0 = time.monotonic()
        while not self._idle_stop.is_set():
            if not self._connected:
                break
            elapsed = time.monotonic() - t0
            # Slow Lissajous-ish drift
            az = _IDLE_DRIFT_AMOUNT * math.sin(2 * math.pi * elapsed / _IDLE_DRIFT_PERIOD)
            ay = _IDLE_DRIFT_AMOUNT * 0.5 * math.sin(2 * math.pi * elapsed / (_IDLE_DRIFT_PERIOD * 1.7))
            if not self._speaking:  # don't fight mouth movement while speaking
                self._inject_params({"ParamAngleZ": az * 15, "ParamAngleY": ay * 10})
            time.sleep(0.1)

    def _inject_params(self, params: dict[str, float]):
        """Send InjectParameterDataRequest to set parameter values."""
        if not self._connected:
            return
        param_list = [{"id": k, "value": v} for k, v in params.items()]
        self._send_recv({
            "messageType": "InjectParameterDataRequest",
            "data": {
                "faceFound": False,
                "mode": "set",
                "parameterValues": param_list,
            },
        }, expect_response=False)

    # ------------------------------------------------------------------
    # Internal — WebSocket transport
    # ------------------------------------------------------------------

    def _next_req_id(self) -> str:
        self._req_id += 1
        return f"revia_{self._req_id}"

    def _send_recv(self, payload: dict, expect_response: bool = True) -> dict | None:
        """Send a VTube Studio API message and optionally read the response."""
        with self._lock:
            ws = self._ws
        if ws is None:
            return None

        msg = {
            "apiName": API_NAME,
            "apiVersion": API_VERSION,
            "requestID": self._next_req_id(),
            **payload,
        }
        try:
            ws.send(json.dumps(msg))
            if not expect_response:
                return None
            raw = ws.recv()
            return json.loads(raw)
        except Exception as exc:
            _log.debug("[VTube] send/recv error: %s", exc)
            if self._connected:
                self._connected = False
                self.status_changed.emit("Disconnected (connection lost)")
                self.disconnected.emit()
            return None

    # ------------------------------------------------------------------
    # Token persistence
    # ------------------------------------------------------------------

    def _load_token(self) -> str:
        try:
            if self._settings_path.exists():
                data = json.loads(self._settings_path.read_text())
                return str(data.get("auth_token", ""))
        except Exception:
            pass
        return ""

    def _save_token(self, token: str):
        try:
            existing = {}
            if self._settings_path.exists():
                existing = json.loads(self._settings_path.read_text())
            existing["auth_token"] = token
            self._settings_path.write_text(json.dumps(existing, indent=2))
        except Exception as exc:
            _log.warning("[VTube] Failed to save auth token: %s", exc)
