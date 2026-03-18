from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field
from enum import Enum


class ResponseMode(str, Enum):
    NORMAL_RESPONSE = "NORMAL_RESPONSE"
    ERROR_RESPONSE = "ERROR_RESPONSE"
    SYSTEM_STATUS_RESPONSE = "SYSTEM_STATUS_RESPONSE"
    TOOL_UNAVAILABLE_RESPONSE = "TOOL_UNAVAILABLE_RESPONSE"
    GREETING_RESPONSE = "GREETING_RESPONSE"
    STARTUP_RESPONSE = "STARTUP_RESPONSE"


class RequestLifecycleState(str, Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    THINKING = "THINKING"
    GENERATING = "GENERATING"
    SPEAKING = "SPEAKING"
    ERROR = "ERROR"


@dataclass
class TurnRecord:
    request_id: str
    turn_id: int
    source: str
    user_text: str
    response_mode: str
    lifecycle_state: str = RequestLifecycleState.IDLE.value
    started_at: float = field(default_factory=time.monotonic)
    metadata: dict = field(default_factory=dict)


@dataclass
class AssistantResponse:
    text: str
    response_mode: str = ResponseMode.NORMAL_RESPONSE.value
    success: bool = True
    error_type: str = ""
    retryable: bool = False
    speakable: bool = True
    commit_to_history: bool = True
    commit_to_memory: bool = True
    metadata: dict = field(default_factory=dict)

    def to_payload(self, request_id: str, turn_id: int) -> dict:
        payload = {
            "type": "chat_complete",
            "text": self.text,
            "request_id": request_id,
            "turn_id": turn_id,
            "mode": self.response_mode,
            "success": self.success,
            "retryable": self.retryable,
            "speakable": self.speakable,
            "error_type": self.error_type,
            "metadata": dict(self.metadata or {}),
        }
        return payload


class TurnManager:
    def __init__(self, log_fn):
        self._log = log_fn
        self._lock = threading.Lock()
        self._next_turn_id = 1
        self._current_request_id = ""
        self._active_turn: TurnRecord | None = None
        self._last_committed_signature = ""
        self._last_committed_user_signature = ""
        self._last_committed_mode = ""

    def start_turn(
        self,
        *,
        source: str,
        user_text: str,
        response_mode: str,
        metadata: dict | None = None,
    ) -> TurnRecord:
        started_at = time.monotonic()
        normalized_user = self._signature(user_text)
        request_id = self._make_request_id(source, normalized_user, started_at)
        with self._lock:
            record = TurnRecord(
                request_id=request_id,
                turn_id=self._next_turn_id,
                source=str(source or "unknown"),
                user_text=str(user_text or ""),
                response_mode=str(response_mode),
                lifecycle_state=RequestLifecycleState.THINKING.value,
                started_at=started_at,
                metadata=dict(metadata or {}),
            )
            self._next_turn_id += 1
            self._current_request_id = request_id
            self._active_turn = record
        self._log(
            f"Turn started | request_id={record.request_id} | turn_id={record.turn_id} "
            f"| source={record.source} | mode={record.response_mode}"
        )
        return record

    def mark_state(self, request_id: str, lifecycle_state: RequestLifecycleState | str, reason: str = ""):
        with self._lock:
            if self._current_request_id != request_id or self._active_turn is None:
                return False
            state_value = (
                lifecycle_state.value
                if isinstance(lifecycle_state, RequestLifecycleState)
                else str(lifecycle_state)
            )
            self._active_turn.lifecycle_state = state_value
        self._log(
            f"Turn state | request_id={request_id} | state={state_value}"
            + (f" | reason={reason}" if reason else "")
        )
        return True

    def is_current(self, request_id: str) -> bool:
        with self._lock:
            return bool(request_id) and request_id == self._current_request_id

    def finish_turn(self, request_id: str, *, lifecycle_state: RequestLifecycleState | str = RequestLifecycleState.IDLE, reason: str = ""):
        state_value = (
            lifecycle_state.value
            if isinstance(lifecycle_state, RequestLifecycleState)
            else str(lifecycle_state)
        )
        with self._lock:
            if self._current_request_id != request_id:
                return False
            self._current_request_id = ""
            if self._active_turn is not None:
                self._active_turn.lifecycle_state = state_value
                self._active_turn = None
        self._log(
            f"Turn finished | request_id={request_id} | state={state_value}"
            + (f" | reason={reason}" if reason else "")
        )
        return True

    def should_block_duplicate_output(self, user_text: str, response_text: str, response_mode: str) -> bool:
        resp_sig = self._signature(response_text)
        user_sig = self._signature(user_text)
        if not resp_sig:
            return False
        with self._lock:
            return (
                resp_sig == self._last_committed_signature
                and user_sig != self._last_committed_user_signature
                and str(response_mode) != ResponseMode.NORMAL_RESPONSE.value
            )

    def remember_committed_output(self, user_text: str, response_text: str, response_mode: str):
        with self._lock:
            self._last_committed_signature = self._signature(response_text)
            self._last_committed_user_signature = self._signature(user_text)
            self._last_committed_mode = str(response_mode or "")

    def snapshot(self) -> dict:
        with self._lock:
            active = self._active_turn
            return {
                "active_request_id": self._current_request_id,
                "active_turn": {
                    "request_id": active.request_id,
                    "turn_id": active.turn_id,
                    "source": active.source,
                    "response_mode": active.response_mode,
                    "lifecycle_state": active.lifecycle_state,
                    "started_at_monotonic": round(active.started_at, 3),
                } if active else None,
                "last_response_mode": self._last_committed_mode,
            }

    @staticmethod
    def _signature(value: str) -> str:
        text = " ".join(str(value or "").strip().lower().split())
        if not text:
            return ""
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _make_request_id(source: str, user_signature: str, started_at: float) -> str:
        seed = f"{source}|{user_signature}|{started_at:.9f}"
        return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]
