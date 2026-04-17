from __future__ import annotations

import os
import threading
import time
from enum import Enum

from PySide6.QtCore import QObject, QTimer, Signal


class EAssistantState(str, Enum):
    IDLE = "Idle"
    LISTENING = "Listening"
    THINKING = "Thinking"
    GENERATING = "Generating"
    SPEAKING = "Speaking"
    ERROR = "Error"


class ESTTState(str, Enum):
    DISABLED = "Disabled"
    IDLE = "Idle"
    LISTENING = "Listening"
    PROCESSING = "Processing"
    COMPLETE = "Complete"
    ERROR = "Error"


class ETTSState(str, Enum):
    DISABLED = "Disabled"
    IDLE = "Idle"
    GENERATING = "Generating"
    SPEAKING = "Speaking"
    INTERRUPTED = "Interrupted"
    COMPLETE = "Complete"
    ERROR = "Error"


class AssistantStatusManager(QObject):
    runtime_state_changed = Signal(dict)

    def __init__(
        self,
        *,
        event_bus,
        model_tab,
        voice_tab,
        vision_tab,
        filters_tab,
        memory_tab,
        system_tab,
        profile_tab,
        chat_panel,
        audio_service,
        parent=None,
    ):
        super().__init__(parent)
        self.event_bus = event_bus
        self.model_tab = model_tab
        self.voice_tab = voice_tab
        self.vision_tab = vision_tab
        self.filters_tab = filters_tab
        self.memory_tab = memory_tab
        self.system_tab = system_tab
        self.profile_tab = profile_tab
        self.chat_panel = chat_panel
        self.audio_service = audio_service
        self._last_telemetry: dict = {}
        self._telemetry_lock = threading.Lock()
        self._pending_runtime_emit = False
        self._last_runtime_signature = ""
        self._last_ui_signature = ""
        self._assistant_state = EAssistantState.IDLE
        self._request_state = EAssistantState.IDLE
        self._request_id = ""
        self._request_stage = ""
        self._assistant_error = ""
        self._request_started_at = None
        self._request_generation_started_at = None
        self._request_last_thinking_duration = 0.0
        self._request_last_generation_duration = 0.0
        self._request_last_total_duration = 0.0
        self._request_current_elapsed = 0.0
        self._stt_state = (
            ESTTState.IDLE
            if self.audio_service and self.audio_service.is_stt_available()
            else ESTTState.DISABLED
        )
        self._tts_state = ETTSState.IDLE
        self._stt_turn_started_at = None
        self._stt_listen_started_at = None
        self._stt_processing_started_at = None
        self._stt_last_listen_duration = 0.0
        self._stt_last_processing_duration = 0.0
        self._stt_last_total_duration = 0.0
        self._stt_current_elapsed = 0.0
        self._stt_error = ""
        self._tts_session_started_at = None
        self._tts_generation_started_at = None
        self._tts_playback_started_at = None
        self._tts_generation_accum = 0.0
        self._tts_playback_accum = 0.0
        self._tts_last_generation_duration = 0.0
        self._tts_last_playback_duration = 0.0
        self._tts_last_total_duration = 0.0
        self._tts_current_elapsed = 0.0
        self._tts_error = ""

        self._live_timer = QTimer(self)
        self._live_timer.setInterval(100)
        self._live_timer.timeout.connect(self._on_live_timer)

        self._wire_signals()
        self.refresh_status(emit_runtime=True)

    def _wire_signals(self):
        self.event_bus.telemetry_updated.connect(self._on_telemetry)
        self.event_bus.chat_request_accepted.connect(self._on_request_accepted)
        self.event_bus.chat_token_payload.connect(self._on_token_payload)
        self.event_bus.chat_complete_payload.connect(self._on_complete_payload)

        for signal in (
            self.model_tab.source_type.currentIndexChanged,
            self.model_tab.local_server.currentTextChanged,
            self.model_tab.local_server_url.textChanged,
            self.model_tab.local_path.textChanged,
            self.model_tab.api_provider.currentTextChanged,
            self.model_tab.api_endpoint.textChanged,
            self.model_tab.api_model.currentTextChanged,
            self.voice_tab.engine_combo.currentTextChanged,
            self.voice_tab.ptt_mode.currentTextChanged,
            self.filters_tab.nsfw_filter.toggled,
            self.filters_tab.profanity_filter.toggled,
            self.filters_tab.pii_filter.toggled,
            self.filters_tab.injection_guard.toggled,
            self.memory_tab.memory_backend.currentTextChanged,
            self.memory_tab.auto_store.toggled,
            self.system_tab.emotion_toggle.toggled,
            self.system_tab.router_toggle.toggled,
            self.system_tab.websearch_toggle.toggled,
            self.profile_tab.char_name.textChanged,
            self.profile_tab.profile_name.textChanged,
            self.chat_panel.tts_combo.currentTextChanged,
        ):
            signal.connect(lambda *_args: self.refresh_status(emit_runtime=True))

        if self.vision_tab:
            for signal in (
                self.vision_tab.vision_engine.currentTextChanged,
                self.vision_tab.auto_capture.toggled,
            ):
                signal.connect(lambda *_args: self.refresh_status(emit_runtime=True))

        self.chat_panel.vision_mode_changed.connect(
            lambda *_args: self.refresh_status(emit_runtime=True)
        )

        if self.audio_service:
            self.audio_service.stt_listening_started.connect(self._on_stt_listening_started)
            self.audio_service.stt_listening_stopped.connect(self._on_stt_listening_stopped)
            self.audio_service.stt_processing_started.connect(self._on_stt_processing_started)
            self.audio_service.stt_processing_finished.connect(self._on_stt_processing_finished)
            self.audio_service.stt_error.connect(self._on_stt_error)
            self.audio_service.status_changed.connect(self._on_audio_status_text)
            self.audio_service.tts_started.connect(self._on_fallback_tts_started)
            self.audio_service.tts_finished.connect(self._on_fallback_tts_finished)

        backend = self.voice_tab.voice_mgr.backend
        backend.synthesis_started.connect(self._on_tts_generation_started)
        backend.synthesis_finished.connect(self._on_tts_generation_finished)
        backend.playback_started.connect(self._on_tts_playback_started)
        backend.playback_finished.connect(self._on_tts_playback_finished)
        backend.playback_interrupted.connect(self._on_tts_playback_interrupted)
        backend.error_occurred.connect(self._on_tts_error)
        self.voice_tab.voice_mgr.error.connect(self._on_tts_error)
        self.voice_tab.voice_mgr.voice_changed.connect(
            lambda *_args: self.refresh_status(emit_runtime=True)
        )

        self.chat_panel.tts_session_started.connect(self._on_tts_session_started)
        self.chat_panel.tts_session_finished.connect(self._on_tts_session_finished)
        self.chat_panel.assistant_output_interrupted.connect(
            self._on_assistant_output_interrupted
        )

    def _on_telemetry(self, data):
        if isinstance(data, dict):
            with self._telemetry_lock:
                self._last_telemetry = dict(data)
            self._sync_request_state_from_telemetry(data)
        self.refresh_status(emit_runtime=True)

    def _on_request_accepted(self, payload):
        if not isinstance(payload, dict):
            return
        self._reset_request_timing()
        self._request_id = str(payload.get("request_id", "") or "")
        self._request_state = EAssistantState.THINKING
        self._request_stage = "request_accepted"
        self._assistant_error = ""
        self._log(
            f"Request queued | request_id={self._request_id or 'unknown'} | state=Thinking"
        )
        self.refresh_status(emit_runtime=True)

    def _on_token_payload(self, payload):
        if not isinstance(payload, dict):
            return
        request_id = str(payload.get("request_id", "") or "")
        if self._request_id and request_id and request_id != self._request_id:
            return
        if request_id:
            self._request_id = request_id
        self._mark_request_generation_started()
        if self._request_state != EAssistantState.GENERATING:
            self._request_state = EAssistantState.GENERATING
            self._request_stage = "llm_decode"
            self._assistant_error = ""
            self._log(
                f"Generation started | request_id={self._request_id or 'unknown'}"
            )
            self.refresh_status(emit_runtime=True)

    def _on_complete_payload(self, payload):
        if not isinstance(payload, dict):
            return
        request_id = str(payload.get("request_id", "") or "")
        if self._request_id and request_id and request_id != self._request_id:
            return

        success = bool(payload.get("success", True))
        speakable = bool(payload.get("speakable", True))
        text = str(payload.get("text", "") or "")
        error_type = str(payload.get("error_type", "") or "")
        self._finish_request_timing()
        self._request_state = EAssistantState.IDLE
        self._request_id = ""
        self._request_stage = ""

        if not success:
            self._assistant_error = error_type or "generation_error"
            self._log(
                "Assistant response error | "
                f"type={self._assistant_error or 'unknown'} | speakable={speakable}"
            )
        elif text:
            self._assistant_error = ""

        self.refresh_status(emit_runtime=True)

    def _sync_request_state_from_telemetry(self, data):
        if not isinstance(data, dict):
            return
        request_lifecycle = data.get("request_lifecycle", {}) or {}
        active_request_id = str(request_lifecycle.get("active_request_id", "") or "")
        active_turn = request_lifecycle.get("active_turn", {}) or {}
        lifecycle_state = str(active_turn.get("lifecycle_state", "") or "").upper()
        lifecycle_reason = self._extract_request_stage(active_turn)
        mapped_state = {
            "THINKING": EAssistantState.THINKING,
            "GENERATING": EAssistantState.GENERATING,
            "SPEAKING": EAssistantState.GENERATING,
            "ERROR": EAssistantState.ERROR,
        }.get(lifecycle_state)

        if active_request_id:
            self._ensure_request_timing_started()
            if self._request_id != active_request_id:
                self._request_id = active_request_id
            self._request_stage = lifecycle_reason
            if mapped_state is not None:
                if mapped_state == EAssistantState.GENERATING:
                    self._mark_request_generation_started()
                self._request_state = mapped_state
                if mapped_state == EAssistantState.ERROR:
                    self._assistant_error = lifecycle_reason or self._assistant_error or "generation_error"
                else:
                    self._assistant_error = ""
        elif self._request_state != EAssistantState.IDLE:
            self._finish_request_timing()
            self._request_state = EAssistantState.IDLE
            self._request_id = ""
            self._request_stage = ""
        else:
            self._request_stage = ""

    def _reset_request_timing(self):
        now = time.perf_counter()
        self._request_started_at = now
        self._request_generation_started_at = None
        self._request_last_thinking_duration = 0.0
        self._request_last_generation_duration = 0.0
        self._request_last_total_duration = 0.0
        self._request_current_elapsed = 0.0

    def _ensure_request_timing_started(self):
        if self._request_started_at is None:
            self._reset_request_timing()

    def _mark_request_generation_started(self):
        now = time.perf_counter()
        if self._request_started_at is None:
            self._request_started_at = now
        if self._request_generation_started_at is None:
            self._request_last_thinking_duration = max(
                0.0,
                now - self._request_started_at,
            )
            self._request_generation_started_at = now
        self._request_current_elapsed = 0.0

    def _finish_request_timing(self):
        if self._request_started_at is None:
            self._request_current_elapsed = 0.0
            return
        now = time.perf_counter()
        if self._request_generation_started_at is not None:
            self._request_last_generation_duration = max(
                0.0,
                now - self._request_generation_started_at,
            )
        else:
            self._request_last_thinking_duration = max(
                self._request_last_thinking_duration,
                now - self._request_started_at,
            )
            self._request_last_generation_duration = 0.0
        self._request_last_total_duration = max(
            0.0,
            now - self._request_started_at,
        )
        self._request_started_at = None
        self._request_generation_started_at = None
        self._request_current_elapsed = 0.0

    def _on_stt_listening_started(self):
        now = time.perf_counter()
        self._stt_error = ""
        if self._stt_turn_started_at is None:
            self._stt_turn_started_at = now
            self._stt_last_listen_duration = 0.0
            self._stt_last_processing_duration = 0.0
            self._stt_last_total_duration = 0.0
        self._stt_listen_started_at = now
        self._stt_processing_started_at = None
        self._stt_current_elapsed = 0.0
        if self._stt_state != ESTTState.LISTENING:
            self._log("STT listening started")
        self._stt_state = ESTTState.LISTENING
        self.refresh_status(emit_runtime=True)

    def _on_stt_listening_stopped(self):
        if self._stt_listen_started_at is not None:
            self._stt_last_listen_duration = max(
                0.0,
                time.perf_counter() - self._stt_listen_started_at,
            )
        self._stt_listen_started_at = None
        self._stt_current_elapsed = 0.0
        self._log(
            f"STT listening stopped | listen={self._stt_last_listen_duration:.2f}s"
        )
        if self._stt_state == ESTTState.LISTENING:
            self._stt_state = ESTTState.IDLE
        self.refresh_status(emit_runtime=True)

    def _on_stt_processing_started(self):
        now = time.perf_counter()
        if self._stt_listen_started_at is not None:
            self._stt_last_listen_duration = max(
                0.0,
                now - self._stt_listen_started_at,
            )
        self._stt_listen_started_at = None
        self._stt_processing_started_at = now
        self._stt_current_elapsed = 0.0
        self._stt_state = ESTTState.PROCESSING
        self._stt_error = ""
        self._log("STT processing started")
        self.refresh_status(emit_runtime=True)

    def _on_stt_processing_finished(self, success, error_text):
        if self._stt_processing_started_at is not None:
            self._stt_last_processing_duration = max(
                0.0,
                time.perf_counter() - self._stt_processing_started_at,
            )
        self._stt_processing_started_at = None
        if self._stt_turn_started_at is not None:
            self._stt_last_total_duration = max(
                0.0,
                time.perf_counter() - self._stt_turn_started_at,
            )
        self._stt_turn_started_at = None
        error_text = str(error_text or "").strip()
        if success:
            self._stt_error = ""
            self._stt_state = ESTTState.COMPLETE
            self._log(
                "STT processing finished | "
                f"listen={self._stt_last_listen_duration:.2f}s | "
                f"process={self._stt_last_processing_duration:.2f}s"
            )
        else:
            self._stt_error = error_text
            self._stt_state = ESTTState.ERROR if error_text else ESTTState.IDLE
            self._log(
                "STT processing finished with issue | "
                f"error={error_text or 'none'} | "
                f"listen={self._stt_last_listen_duration:.2f}s | "
                f"process={self._stt_last_processing_duration:.2f}s"
            )
        self._stt_current_elapsed = 0.0
        self.refresh_status(emit_runtime=True)

    def _on_stt_error(self, message):
        self._stt_error = str(message or "Unknown STT error")
        if self._stt_processing_started_at is not None:
            self._stt_last_processing_duration = max(
                0.0,
                time.perf_counter() - self._stt_processing_started_at,
            )
        self._stt_processing_started_at = None
        self._stt_listen_started_at = None
        if self._stt_turn_started_at is not None:
            self._stt_last_total_duration = max(
                0.0,
                time.perf_counter() - self._stt_turn_started_at,
            )
        self._stt_turn_started_at = None
        self._stt_state = ESTTState.ERROR
        self._log(f"STT error | {self._stt_error}")
        self.refresh_status(emit_runtime=True)

    def _on_audio_status_text(self, message):
        message = str(message or "").strip()
        if not message:
            return
        if "speech_recognition not installed" in message.lower():
            self._stt_state = ESTTState.DISABLED
            self._stt_error = message
            self._log(f"STT unavailable | {message}")
            self.refresh_status(emit_runtime=True)

    def _on_tts_session_started(self):
        if self._tts_session_started_at is None:
            self._tts_session_started_at = time.perf_counter()
            self._tts_generation_accum = 0.0
            self._tts_playback_accum = 0.0
            self._tts_last_generation_duration = 0.0
            self._tts_last_playback_duration = 0.0
            self._tts_last_total_duration = 0.0
            self._tts_error = ""
            self._log("TTS session started")
            self.refresh_status(emit_runtime=False)

    def _on_tts_session_finished(self, interrupted):
        interrupted = bool(interrupted)
        now = time.perf_counter()
        if self._tts_generation_started_at is not None:
            self._tts_generation_accum += max(
                0.0,
                now - self._tts_generation_started_at,
            )
            self._tts_generation_started_at = None
        if self._tts_playback_started_at is not None:
            self._tts_playback_accum += max(
                0.0,
                now - self._tts_playback_started_at,
            )
            self._tts_playback_started_at = None
        if self._tts_session_started_at is not None:
            self._tts_last_total_duration = max(
                0.0,
                now - self._tts_session_started_at,
            )
        self._tts_session_started_at = None
        self._tts_last_generation_duration = self._tts_generation_accum
        self._tts_last_playback_duration = self._tts_playback_accum
        self._tts_generation_accum = 0.0
        self._tts_playback_accum = 0.0
        self._tts_current_elapsed = 0.0
        if interrupted:
            self._tts_state = ETTSState.INTERRUPTED
            self._log(
                "TTS session interrupted | "
                f"gen={self._tts_last_generation_duration:.2f}s | "
                f"speak={self._tts_last_playback_duration:.2f}s"
            )
        elif self._tts_state != ETTSState.ERROR:
            self._tts_state = ETTSState.COMPLETE
            self._log(
                "TTS session finished | "
                f"gen={self._tts_last_generation_duration:.2f}s | "
                f"speak={self._tts_last_playback_duration:.2f}s"
            )
        self.refresh_status(emit_runtime=True)

    def _on_tts_generation_started(self):
        self._on_tts_session_started()
        self._tts_generation_started_at = time.perf_counter()
        self._tts_current_elapsed = 0.0
        self._tts_state = ETTSState.GENERATING
        self._tts_error = ""
        self._log("TTS generation started")
        self.refresh_status(emit_runtime=True)

    def _on_tts_generation_finished(self, _metrics):
        if self._tts_generation_started_at is not None:
            self._tts_generation_accum += max(
                0.0,
                time.perf_counter() - self._tts_generation_started_at,
            )
        self._tts_generation_started_at = None
        self._tts_last_generation_duration = self._tts_generation_accum
        self._tts_current_elapsed = 0.0
        self._log(
            f"TTS generation finished | gen={self._tts_last_generation_duration:.2f}s"
        )
        if self._tts_state == ETTSState.GENERATING:
            self.refresh_status(emit_runtime=True)

    def _on_tts_playback_started(self):
        self._on_tts_session_started()
        self._tts_playback_started_at = time.perf_counter()
        self._tts_current_elapsed = 0.0
        self._tts_state = ETTSState.SPEAKING
        self._log("TTS playback started")
        self.refresh_status(emit_runtime=True)

    def _on_tts_playback_finished(self):
        if self._tts_playback_started_at is not None:
            self._tts_playback_accum += max(
                0.0,
                time.perf_counter() - self._tts_playback_started_at,
            )
        self._tts_playback_started_at = None
        self._tts_last_playback_duration = self._tts_playback_accum
        self._tts_current_elapsed = 0.0
        self._log(
            f"TTS playback finished | speak={self._tts_last_playback_duration:.2f}s"
        )
        self.refresh_status(emit_runtime=False)

    def _on_tts_playback_interrupted(self):
        if self._tts_session_started_at is None and self._tts_state == ETTSState.INTERRUPTED:
            return
        if self._tts_playback_started_at is not None:
            self._tts_playback_accum += max(
                0.0,
                time.perf_counter() - self._tts_playback_started_at,
            )
        self._tts_playback_started_at = None
        self._tts_last_playback_duration = self._tts_playback_accum
        self._tts_current_elapsed = 0.0
        self._tts_state = ETTSState.INTERRUPTED
        self._log(
            f"TTS playback interrupted | speak={self._tts_last_playback_duration:.2f}s"
        )
        self.refresh_status(emit_runtime=True)

    def _on_tts_error(self, message):
        self._tts_error = str(message or "Unknown TTS error")
        now = time.perf_counter()
        if self._tts_generation_started_at is not None:
            self._tts_generation_accum += max(0.0, now - self._tts_generation_started_at)
            self._tts_generation_started_at = None
        if self._tts_playback_started_at is not None:
            self._tts_playback_accum += max(0.0, now - self._tts_playback_started_at)
            self._tts_playback_started_at = None
        if self._tts_session_started_at is not None:
            self._tts_last_total_duration = max(0.0, now - self._tts_session_started_at)
        self._tts_last_generation_duration = self._tts_generation_accum
        self._tts_last_playback_duration = self._tts_playback_accum
        self._tts_current_elapsed = 0.0
        self._tts_state = ETTSState.ERROR
        self._log(f"TTS error | {self._tts_error}")
        self.refresh_status(emit_runtime=True)

    def _on_assistant_output_interrupted(self):
        self._assistant_error = ""
        self._request_state = EAssistantState.IDLE
        self._request_id = ""
        self._request_stage = ""
        self._log("Assistant output interrupted")
        self._on_tts_playback_interrupted()
        self._on_tts_session_finished(True)

    def _on_fallback_tts_started(self):
        self._on_tts_generation_started()
        self._on_tts_playback_started()

    def _on_fallback_tts_finished(self):
        self._on_tts_playback_finished()
        self._on_tts_session_finished(False)

    def _on_live_timer(self):
        if self._request_state == EAssistantState.THINKING and self._request_started_at is not None:
            self._request_current_elapsed = max(
                0.0,
                time.perf_counter() - self._request_started_at,
            )
        elif self._request_state == EAssistantState.GENERATING and self._request_generation_started_at is not None:
            self._request_current_elapsed = max(
                0.0,
                time.perf_counter() - self._request_generation_started_at,
            )
        else:
            self._request_current_elapsed = 0.0

        if self._stt_state == ESTTState.LISTENING and self._stt_listen_started_at is not None:
            self._stt_current_elapsed = max(
                0.0,
                time.perf_counter() - self._stt_listen_started_at,
            )
        elif self._stt_state == ESTTState.PROCESSING and self._stt_processing_started_at is not None:
            self._stt_current_elapsed = max(
                0.0,
                time.perf_counter() - self._stt_processing_started_at,
            )
        else:
            self._stt_current_elapsed = 0.0

        if self._tts_state == ETTSState.GENERATING and self._tts_generation_started_at is not None:
            self._tts_current_elapsed = max(
                0.0,
                time.perf_counter() - self._tts_generation_started_at,
            )
        elif self._tts_state == ETTSState.SPEAKING and self._tts_playback_started_at is not None:
            self._tts_current_elapsed = max(
                0.0,
                time.perf_counter() - self._tts_playback_started_at,
            )
        else:
            self._tts_current_elapsed = 0.0

        self.refresh_status(emit_runtime=False)

    def refresh_status(self, emit_runtime=False):
        if emit_runtime:
            self._pending_runtime_emit = True

        if not self._is_stt_enabled():
            if self._stt_state != ESTTState.DISABLED:
                self._log("STT disabled | resetting timers")
            self._stt_state = ESTTState.DISABLED
            self._stt_current_elapsed = 0.0
            self._stt_listen_started_at = None
            self._stt_processing_started_at = None
        elif self._stt_state == ESTTState.DISABLED:
            self._stt_state = ESTTState.IDLE

        if not self._is_tts_enabled():
            if self._tts_state not in (ETTSState.DISABLED, ETTSState.IDLE):
                self._log("TTS disabled | resetting timers")
            self._tts_state = ETTSState.DISABLED
            self._tts_current_elapsed = 0.0
            self._tts_session_started_at = None
            self._tts_generation_started_at = None
            self._tts_playback_started_at = None
            self._tts_generation_accum = 0.0
            self._tts_playback_accum = 0.0
        elif self._tts_state == ETTSState.DISABLED:
            self._tts_state = ETTSState.IDLE

        snapshot = self.get_assistant_status_snapshot()
        ui_signature = self._signature(snapshot, include_elapsed=True)
        runtime_signature = self._signature(snapshot, include_elapsed=False)

        if ui_signature != self._last_ui_signature:
            self._last_ui_signature = ui_signature
            self.event_bus.assistant_status_updated.emit(snapshot)

        if runtime_signature != self._last_runtime_signature:
            self._last_runtime_signature = runtime_signature
            self._pending_runtime_emit = True
            self._log(
                "Assistant status refresh | "
                f"assistant={snapshot['assistant_state_display']} | "
                f"model={snapshot['current_model_name']} | "
                f"stt={snapshot['stt_state']} | "
                f"tts={snapshot['tts_state']}"
            )

        if self._pending_runtime_emit:
            self._pending_runtime_emit = False
            self.runtime_state_changed.emit(snapshot)

        self._update_live_timer_state()

    def get_assistant_status_snapshot(self) -> dict:
        with self._telemetry_lock:
            telemetry = dict(self._last_telemetry or {})
        emotion = telemetry.get("emotion", {}) or {}
        llm = telemetry.get("llm_connection", {}) or {}
        runtime = telemetry.get("runtime_status", {}) or {}
        source_is_online = self.model_tab.source_type.currentIndex() == 1
        local_model_name = os.path.basename(self.model_tab.local_path.text().strip())
        online_model_name = self.model_tab.api_model.currentText().strip()

        filter_enabled = any(
            (
                self.filters_tab.nsfw_filter.isChecked(),
                self.filters_tab.profanity_filter.isChecked(),
                self.filters_tab.pii_filter.isChecked(),
                self.filters_tab.injection_guard.isChecked(),
            )
        )
        if filter_enabled:
            filter_level = (
                "strict"
                if all(
                    (
                        self.filters_tab.nsfw_filter.isChecked(),
                        self.filters_tab.pii_filter.isChecked(),
                        self.filters_tab.injection_guard.isChecked(),
                    )
                )
                else "standard"
            )
        else:
            filter_level = "off"

        active_voice_name = ""
        try:
            active_voice_name = self.voice_tab.voice_mgr.active_profile.name
        except Exception:
            active_voice_name = ""

        model_name = str(
            llm.get("model")
            or runtime.get("current_model_name")
            or online_model_name
            or local_model_name
            or "None"
        )
        current_emotion = str(
            emotion.get("label")
            or runtime.get("current_emotion")
            or "Neutral"
        )
        current_persona = (
            self.profile_tab.char_name.text().strip()
            or str(runtime.get("active_persona_profile_name", "")).strip()
            or "Revia"
        )
        voice_name = (
            active_voice_name
            or str(runtime.get("current_tts_voice", "")).strip()
            or self.voice_tab.engine_combo.currentText()
        )
        vision_enabled = self._is_vision_enabled()
        vision_model = self._get_vision_model()
        vision_state = self._derive_vision_state(vision_enabled)
        thinking_timer = self._get_thinking_timer()
        stt_timer = self._get_stt_timer()
        tts_timer = self._get_tts_timer()
        tts_generation_timer = self._get_tts_generation_timer()

        self._assistant_state = self._derive_assistant_state()
        assistant_state_detail = self._get_assistant_state_detail()
        snapshot = {
            "assistant_state": self._assistant_state.value,
            "assistant_state_detail": assistant_state_detail,
            "assistant_state_display": self._format_assistant_state_display(
                self._assistant_state.value,
                assistant_state_detail,
            ),
            "model_name": model_name,
            "model_ready": str(llm.get("state", "Disconnected")) == "Ready",
            "model_state": str(llm.get("state", "Disconnected")),
            "online_enabled": bool(source_is_online or self.system_tab.websearch_toggle.isChecked()),
            "web_search_enabled": self.system_tab.websearch_toggle.isChecked(),
            "filters_enabled": filter_enabled,
            "filter_level": filter_level,
            "voice_enabled": self._is_tts_enabled(),
            "voice_name": voice_name,
            "vision_enabled": vision_enabled,
            "vision_model": vision_model,
            "vision_state": vision_state,
            "current_emotion": current_emotion,
            "current_persona": current_persona,
            "thinking_timer": round(thinking_timer, 3),
            "thinking_last_duration": round(self._request_last_thinking_duration, 3),
            "thinking_current_elapsed": round(
                self._request_current_elapsed
                if self._request_state == EAssistantState.THINKING
                else 0.0,
                3,
            ),
            "request_last_generation_duration": round(self._request_last_generation_duration, 3),
            "request_last_total_duration": round(self._request_last_total_duration, 3),
            "stt_enabled": self._is_stt_enabled(),
            "stt_state": self._stt_state.value,
            "stt_timer": round(stt_timer, 3),
            "stt_last_listen_duration": round(self._stt_last_listen_duration, 3),
            "stt_last_processing_duration": round(self._stt_last_processing_duration, 3),
            "stt_last_total_duration": round(self._stt_last_total_duration, 3),
            "stt_current_elapsed": round(self._stt_current_elapsed, 3),
            "stt_error": self._stt_error,
            "tts_enabled": self._is_tts_enabled(),
            "tts_state": self._tts_state.value,
            "tts_timer": round(tts_timer, 3),
            "tts_generation_timer": round(tts_generation_timer, 3),
            "tts_last_generation_duration": round(self._tts_last_generation_duration, 3),
            "tts_last_playback_duration": round(self._tts_last_playback_duration, 3),
            "tts_last_total_duration": round(self._tts_last_total_duration, 3),
            "tts_current_elapsed": round(self._tts_current_elapsed, 3),
            "tts_error": self._tts_error,
            "last_error": self._assistant_error,
            "active_request_id": self._request_id,
            "current_model_name": model_name,
            "active_persona_profile_name": current_persona,
            "content_filter_level": filter_level,
            "voice_output_enabled": self._is_tts_enabled(),
            "voice_input_enabled": self._is_stt_enabled(),
            "tts_engine": self.voice_tab.engine_combo.currentText(),
            "stt_mode": self.voice_tab.ptt_mode.currentText(),
            "current_tts_voice": voice_name,
            "vision_available": vision_enabled,
            "emotion_mode_enabled": self.system_tab.emotion_toggle.isChecked(),
            "tool_access_enabled": self.system_tab.websearch_toggle.isChecked(),
            "ui_state": self._assistant_state.value,
        }
        snapshot["thinking_time_text"] = self._format_thinking_time(snapshot)
        snapshot["stt_status_text"] = self.get_formatted_stt_status(snapshot)
        snapshot["stt_time_text"] = self._format_stt_time(snapshot)
        snapshot["tts_status_text"] = self.get_formatted_tts_status(snapshot)
        snapshot["tts_generation_time_text"] = self._format_tts_generation_time(snapshot)
        snapshot["tts_time_text"] = self._format_tts_time(snapshot)
        return snapshot

    def build_runtime_config_snapshot(self) -> dict:
        snapshot = self.get_assistant_status_snapshot()
        return {
            "online_enabled": snapshot["online_enabled"],
            "web_search_enabled": snapshot["web_search_enabled"],
            "safety_filter_enabled": snapshot["filters_enabled"],
            "nsfw_filter_enabled": self.filters_tab.nsfw_filter.isChecked(),
            "profanity_filter_enabled": self.filters_tab.profanity_filter.isChecked(),
            "pii_filter_enabled": self.filters_tab.pii_filter.isChecked(),
            "prompt_injection_guard_enabled": self.filters_tab.injection_guard.isChecked(),
            "content_filter_level": snapshot["content_filter_level"],
            "local_llm_enabled": self.model_tab.source_type.currentIndex() == 0,
            "local_llm_provider": (
                self.model_tab.api_provider.currentText()
                if self.model_tab.source_type.currentIndex() == 1
                else self.model_tab.local_server.currentText()
            ),
            "local_llm_endpoint": (
                self.model_tab.api_endpoint.text().strip()
                if self.model_tab.source_type.currentIndex() == 1
                else self.model_tab.local_server_url.text().strip()
            ),
            "voice_input_enabled": snapshot["voice_input_enabled"],
            "voice_output_enabled": snapshot["voice_output_enabled"],
            "current_tts_voice": snapshot["current_tts_voice"],
            "vision_enabled": snapshot["vision_enabled"],
            "vision_model": snapshot["vision_model"],
            "vision_state": snapshot["vision_state"],
            "memory_enabled": (
                self.memory_tab.memory_backend.currentText() != "None"
                and self.memory_tab.auto_store.isChecked()
            ),
            "emotion_mode_enabled": snapshot["emotion_mode_enabled"],
            "current_emotion": snapshot["current_emotion"],
            "tool_access_enabled": snapshot["tool_access_enabled"],
            "tool_modes": {
                "web_search": self.system_tab.websearch_toggle.isChecked(),
                "router": self.system_tab.router_toggle.isChecked(),
            },
            "active_persona_profile_name": snapshot["active_persona_profile_name"],
            "active_persona_profile_id": self.profile_tab.profile_name.text().strip() or "default",
            "streaming_enabled": True,
            "current_model_name": snapshot["current_model_name"],
            "fallback_model_name": "",
            "safe_mode_enabled": snapshot["filters_enabled"],
            "moderation_mode": snapshot["content_filter_level"],
            "ui_state": snapshot["assistant_state"],
            "assistant_state_detail": snapshot["assistant_state_detail"],
            "assistant_state_display": snapshot["assistant_state_display"],
            "tts_engine": snapshot["tts_engine"],
            "stt_mode": snapshot["stt_mode"],
            "assistant_state": snapshot["assistant_state"],
            "model_name": snapshot["model_name"],
            "model_ready": snapshot["model_ready"],
            "stt_enabled": snapshot["stt_enabled"],
            "stt_state": snapshot["stt_state"],
            "stt_timer": snapshot["stt_timer"],
            "stt_last_listen_duration": snapshot["stt_last_listen_duration"],
            "stt_last_processing_duration": snapshot["stt_last_processing_duration"],
            "stt_last_total_duration": snapshot["stt_last_total_duration"],
            "stt_current_elapsed": snapshot["stt_current_elapsed"],
            "stt_error": snapshot["stt_error"],
            "tts_enabled": snapshot["tts_enabled"],
            "tts_state": snapshot["tts_state"],
            "tts_timer": snapshot["tts_timer"],
            "tts_last_generation_duration": snapshot["tts_last_generation_duration"],
            "tts_last_playback_duration": snapshot["tts_last_playback_duration"],
            "tts_last_total_duration": snapshot["tts_last_total_duration"],
            "tts_current_elapsed": snapshot["tts_current_elapsed"],
            "tts_error": snapshot["tts_error"],
        }

    def get_formatted_stt_status(self, snapshot: dict | None = None) -> str:
        snap = snapshot or self.get_assistant_status_snapshot()
        state = str(snap.get("stt_state", ESTTState.DISABLED.value))
        if state == ESTTState.ERROR.value and snap.get("stt_error"):
            return f"{state} | {snap.get('stt_error')}"
        return state

    def get_formatted_tts_status(self, snapshot: dict | None = None) -> str:
        snap = snapshot or self.get_assistant_status_snapshot()
        state = str(snap.get("tts_state", ETTSState.DISABLED.value))
        if state == ETTSState.ERROR.value and snap.get("tts_error"):
            return f"{state} | {snap.get('tts_error')}"
        return state

    def _format_stt_time(self, snapshot: dict) -> str:
        return f"{float(snapshot.get('stt_timer', 0.0) or 0.0):.2f}s"

    def _format_thinking_time(self, snapshot: dict) -> str:
        return f"{float(snapshot.get('thinking_timer', 0.0) or 0.0):.2f}s"

    def _format_tts_generation_time(self, snapshot: dict) -> str:
        return f"{float(snapshot.get('tts_generation_timer', 0.0) or 0.0):.2f}s"

    def _format_tts_time(self, snapshot: dict) -> str:
        return f"{float(snapshot.get('tts_timer', 0.0) or 0.0):.2f}s"

    def _derive_assistant_state(self) -> EAssistantState:
        if self._tts_state == ETTSState.SPEAKING:
            return EAssistantState.SPEAKING
        if self._tts_state == ETTSState.GENERATING:
            return EAssistantState.GENERATING
        if self._request_state == EAssistantState.ERROR:
            return EAssistantState.ERROR
        if self._request_state in (EAssistantState.THINKING, EAssistantState.GENERATING):
            return self._request_state
        if self._stt_state in (ESTTState.LISTENING, ESTTState.PROCESSING):
            return EAssistantState.LISTENING
        if (
            self._assistant_error
            or self._stt_state == ESTTState.ERROR
            or self._tts_state == ETTSState.ERROR
        ):
            return EAssistantState.ERROR
        return EAssistantState.IDLE

    def _get_assistant_state_detail(self) -> str:
        if self._assistant_state not in (EAssistantState.THINKING, EAssistantState.GENERATING):
            return ""
        return str(self._request_stage or "").strip()

    def _update_live_timer_state(self):
        active = (
            self._request_state in (EAssistantState.THINKING, EAssistantState.GENERATING)
            or self._stt_state in (ESTTState.LISTENING, ESTTState.PROCESSING)
            or self._tts_state in (ETTSState.GENERATING, ETTSState.SPEAKING)
        )
        if active and not self._live_timer.isActive():
            self._live_timer.start()
        elif not active and self._live_timer.isActive():
            self._live_timer.stop()

    def _is_stt_enabled(self) -> bool:
        return bool(self.audio_service and self.audio_service.is_stt_available())

    def _is_tts_enabled(self) -> bool:
        return bool(self.chat_panel and self.chat_panel.is_tts_enabled())

    def _is_vision_enabled(self) -> bool:
        return bool(self._is_vision_configured() or (self.chat_panel and self.chat_panel.is_vision_enabled()))

    def _get_vision_model(self) -> str:
        if not self._is_vision_configured():
            return ""
        return str(self.vision_tab.vision_engine.currentText() or "").strip()

    def _derive_vision_state(self, enabled: bool) -> str:
        if not enabled:
            return "Off"
        if self.chat_panel and self.chat_panel.is_vision_processing():
            return "Processing"
        return "On"

    def _is_vision_configured(self) -> bool:
        if not self.vision_tab:
            return False
        engine = str(self.vision_tab.vision_engine.currentText() or "").strip()
        return bool(engine and engine.lower() != "none")

    def _get_stt_timer(self) -> float:
        if self._stt_state in (ESTTState.LISTENING, ESTTState.PROCESSING):
            return max(0.0, float(self._stt_current_elapsed or 0.0))
        return 0.0

    def _get_thinking_timer(self) -> float:
        if self._request_state == EAssistantState.THINKING:
            return max(0.0, float(self._request_current_elapsed or 0.0))
        return max(0.0, float(self._request_last_thinking_duration or 0.0))

    def _get_tts_timer(self) -> float:
        if self._tts_state in (ETTSState.GENERATING, ETTSState.SPEAKING):
            return max(0.0, float(self._tts_current_elapsed or 0.0))
        return 0.0

    def _get_tts_generation_timer(self) -> float:
        if self._tts_state == ETTSState.GENERATING:
            return max(0.0, float(self._tts_current_elapsed or 0.0))
        return max(0.0, float(self._tts_last_generation_duration or 0.0))

    def _signature(self, snapshot: dict, *, include_elapsed: bool) -> str:
        parts = [
            str(snapshot.get("assistant_state", "")),
            str(snapshot.get("assistant_state_detail", "")),
            str(snapshot.get("model_name", "")),
            str(snapshot.get("model_state", "")),
            str(snapshot.get("online_enabled", False)),
            str(snapshot.get("filters_enabled", False)),
            str(snapshot.get("filter_level", "")),
            str(snapshot.get("voice_enabled", False)),
            str(snapshot.get("voice_name", "")),
            str(snapshot.get("vision_enabled", False)),
            str(snapshot.get("vision_model", "")),
            str(snapshot.get("vision_state", "")),
            str(snapshot.get("current_emotion", "")),
            str(snapshot.get("current_persona", "")),
            f"{float(snapshot.get('thinking_last_duration', 0.0) or 0.0):.3f}",
            str(snapshot.get("stt_enabled", False)),
            str(snapshot.get("stt_state", "")),
            str(snapshot.get("stt_error", "")),
            str(snapshot.get("tts_enabled", False)),
            str(snapshot.get("tts_state", "")),
            str(snapshot.get("tts_error", "")),
            f"{float(snapshot.get('stt_last_listen_duration', 0.0) or 0.0):.3f}",
            f"{float(snapshot.get('stt_last_processing_duration', 0.0) or 0.0):.3f}",
            f"{float(snapshot.get('tts_last_generation_duration', 0.0) or 0.0):.3f}",
            f"{float(snapshot.get('tts_last_playback_duration', 0.0) or 0.0):.3f}",
        ]
        if include_elapsed:
            parts.extend(
                (
                    f"{float(snapshot.get('thinking_current_elapsed', 0.0) or 0.0):.2f}",
                    f"{float(snapshot.get('stt_current_elapsed', 0.0) or 0.0):.2f}",
                    f"{float(snapshot.get('tts_current_elapsed', 0.0) or 0.0):.2f}",
                )
            )
        return "|".join(parts)

    @staticmethod
    def _extract_request_stage(active_turn: dict) -> str:
        if not isinstance(active_turn, dict):
            return ""
        return str(active_turn.get("lifecycle_reason", "") or "").strip()

    @staticmethod
    def _format_assistant_state_display(state: str, detail: str) -> str:
        state_text = str(state or "Unknown").strip() or "Unknown"
        detail_text = str(detail or "").strip()
        if detail_text:
            return f"{state_text} | {detail_text}"
        return state_text

    def cleanup(self):
        """Stop all timers and disconnect all signals.

        Must be called before the widget tree is destroyed.  Without this,
        Qt can fire slots against already-deleted C++ objects, producing
        hard-to-reproduce RuntimeError / segfaults.
        """
        try:
            self._live_timer.stop()
            self._live_timer.timeout.disconnect(self._on_live_timer)
        except RuntimeError:
            pass

        # Event bus signals
        _bus_pairs = [
            (self.event_bus.telemetry_updated, self._on_telemetry),
            (self.event_bus.chat_request_accepted, self._on_request_accepted),
            (self.event_bus.chat_token_payload, self._on_token_payload),
            (self.event_bus.chat_complete_payload, self._on_complete_payload),
        ]
        for signal, slot in _bus_pairs:
            try:
                signal.disconnect(slot)
            except RuntimeError:
                pass

        # Audio service signals
        if self.audio_service:
            _audio_pairs = [
                (self.audio_service.stt_listening_started, self._on_stt_listening_started),
                (self.audio_service.stt_listening_stopped, self._on_stt_listening_stopped),
                (self.audio_service.stt_processing_started, self._on_stt_processing_started),
                (self.audio_service.stt_processing_finished, self._on_stt_processing_finished),
                (self.audio_service.stt_error, self._on_stt_error),
                (self.audio_service.status_changed, self._on_audio_status_text),
                (self.audio_service.tts_started, self._on_fallback_tts_started),
                (self.audio_service.tts_finished, self._on_fallback_tts_finished),
            ]
            for signal, slot in _audio_pairs:
                try:
                    signal.disconnect(slot)
                except RuntimeError:
                    pass

        # TTS backend signals
        try:
            backend = self.voice_tab.voice_mgr.backend
            _tts_pairs = [
                (backend.synthesis_started, self._on_tts_generation_started),
                (backend.synthesis_finished, self._on_tts_generation_finished),
                (backend.playback_started, self._on_tts_playback_started),
                (backend.playback_finished, self._on_tts_playback_finished),
                (backend.playback_interrupted, self._on_tts_playback_interrupted),
                (backend.error_occurred, self._on_tts_error),
            ]
            for signal, slot in _tts_pairs:
                try:
                    signal.disconnect(slot)
                except RuntimeError:
                    pass
        except Exception:
            pass

        # Chat panel TTS session signals
        _chat_pairs = [
            (self.chat_panel.tts_session_started, self._on_tts_session_started),
            (self.chat_panel.tts_session_finished, self._on_tts_session_finished),
            (self.chat_panel.assistant_output_interrupted, self._on_assistant_output_interrupted),
        ]
        for signal, slot in _chat_pairs:
            try:
                signal.disconnect(slot)
            except RuntimeError:
                pass

    def _log(self, message: str):
        self.event_bus.log_entry.emit(f"[Status] {message}")
