import base64
import logging
import queue
import re
import threading
import time
from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit, QPushButton,
    QLabel,
    QComboBox,
    QSizePolicy,
    QApplication,
)
from PySide6.QtCore import Qt, QBuffer, QIODevice, QTimer, Signal
from PySide6.QtGui import QFont, QColor

from app.ui_status import apply_status_style

logger = logging.getLogger(__name__)

_CHAT_THEME_FALLBACK = {
    "PrimaryText": "#ede9fe",
    "SecondaryText": "#8b7ab8",
    "Accent": "#a855f7",
    "AccentHover": "#c084fc",
}


def _chat_theme_tokens():
    tokens = dict(_CHAT_THEME_FALLBACK)
    app = QApplication.instance()
    raw = app.property("reviaThemeTokens") if app else None
    if isinstance(raw, dict):
        for key, value in raw.items():
            if key in tokens and QColor(str(value)).isValid():
                tokens[key] = QColor(str(value)).name()
    return tokens


def _chat_color(key):
    return _chat_theme_tokens().get(key, _CHAT_THEME_FALLBACK[key])


class ChatPanel(QFrame):
    tts_session_started = Signal()
    tts_session_finished = Signal(bool)
    assistant_output_interrupted = Signal()
    vision_mode_changed = Signal(bool)

    def __init__(self, event_bus, client, audio_service=None, continuous_audio=None, parent=None):
        super().__init__(parent)
        self.event_bus = event_bus
        self.client = client
        self.audio_service = audio_service
        self.continuous_audio = continuous_audio  # ContinuousAudioPipeline (PRD §6)
        self.camera_service = None
        self.voice_manager = None
        self.voice_tab = None
        self._sing_handler = None
        self.setObjectName("chatPanel")
        self._current_response = ""
        self._full_response = ""
        self._vision_active = False
        self._current_emotion = "neutral"
        # Sentence-level streaming TTS
        self._tts_sentence_buf = ""
        self._tts_queue = queue.Queue(maxsize=20)
        self._tts_worker_active = False
        self._tts_worker_lock = threading.Lock()
        self._awaiting_reply = False
        self._awaiting_started_at = 0.0
        self._assistant_audio_active = False
        self._pending_input_source = "UserMessage"
        self._pending_request_id = ""
        self._current_request_id = ""
        self._last_completed_text = ""
        self._ignored_request_ids = []
        self._tts_session_interrupted = False
        self._server_sentence_streaming = False  # Set True when chat_sentence events arrive
        # Each item: (text, image_b64, vision_context, stack_index, stack_total, source)
        self._outbound_queue = []
        self._OUTBOUND_QUEUE_MAX = 50  # Prevent unbounded memory growth
        self._wired_voice_backend = None
        self._activity_thinking_started_at = None
        self._activity_thinking_duration = 0.0
        self._activity_generation_started_at = None
        self._activity_generation_duration = 0.0
        self._activity_tts_session_active = False
        self._activity_tts_generation_started_at = None
        self._activity_tts_generation_accum = 0.0
        self._activity_tts_generation_duration = 0.0
        self._activity_tts_playback_started_at = None
        self._activity_tts_playback_accum = 0.0
        self._activity_tts_playback_duration = 0.0
        self._activity_waiting_for_tts = False
        self._activity_interrupted = False
        self._activity_summary = "Ready"
        self._activity_timer = QTimer(self)
        self._activity_timer.setInterval(100)
        self._activity_timer.timeout.connect(self._refresh_activity_indicator)

        # Hard-timeout watchdog: if the backend stays in Thinking for longer than
        # this, force-clear reply state so the UI never stays stuck indefinitely.
        # 60s is generous enough for slow models but prevents the UI from
        # appearing frozen for extended periods.
        self._thinking_watchdog = QTimer(self)
        self._thinking_watchdog.setSingleShot(True)
        self._thinking_watchdog.setInterval(60_000)  # 60 seconds
        self._thinking_watchdog.timeout.connect(self._on_thinking_timeout)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.chat_display = QTextEdit()
        self.chat_display.setObjectName("chatDisplay")
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(QFont("Segoe UI", 10))
        layout.addWidget(self.chat_display, stretch=1)

        self.activity_label = QLabel("Ready")
        self.activity_label.setObjectName("chatActivity")
        self.activity_label.setFont(QFont("Segoe UI", 9))
        self.activity_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.activity_label.setProperty("activityState", "idle")
        apply_status_style(self.activity_label, role="muted")
        self.activity_label.setContentsMargins(10, 6, 10, 6)
        layout.addWidget(self.activity_label)

        input_row = QFrame()
        input_row.setObjectName("chatInputRow")
        input_row.setMinimumHeight(48)
        input_row.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row_layout = QHBoxLayout(input_row)
        row_layout.setContentsMargins(8, 6, 8, 6)
        row_layout.setSpacing(6)

        self.mic_btn = QPushButton("\U0001f3a4")
        self.mic_btn.setObjectName("micBtn")
        self.mic_btn.setMinimumSize(34, 34)
        self.mic_btn.setMaximumSize(42, 42)
        self.mic_btn.setCheckable(True)
        self.mic_btn.setToolTip("Toggle Listening (click to start/stop)")
        self.mic_btn.toggled.connect(self._on_mic_toggled)
        row_layout.addWidget(self.mic_btn)

        self.vision_btn = QPushButton("\U0001f441")
        self.vision_btn.setObjectName("visionBtn")
        self.vision_btn.setMinimumSize(34, 34)
        self.vision_btn.setMaximumSize(42, 42)
        self.vision_btn.setCheckable(True)
        self.vision_btn.setToolTip("Toggle live vision context for messages")
        self.vision_btn.toggled.connect(self._on_vision_toggled)
        row_layout.addWidget(self.vision_btn)

        self.tts_combo = QComboBox()
        self.tts_combo.addItems(["pyttsx3 (Fast)", "Qwen3-TTS (Quality)", "Off"])
        self.tts_combo.setMinimumWidth(112)
        self.tts_combo.setMaximumWidth(190)
        self.tts_combo.setToolTip("TTS engine for AI responses")
        self.tts_combo.currentTextChanged.connect(self._on_tts_engine_changed)
        row_layout.addWidget(self.tts_combo)

        self.input_field = QLineEdit()
        self.input_field.setObjectName("chatInput")
        self.input_field.setPlaceholderText("Type a message...")
        self.input_field.setFont(QFont("Segoe UI", 10))
        row_layout.addWidget(self.input_field, stretch=1)

        self.send_btn = QPushButton("Send")
        self.send_btn.setObjectName("sendBtn")
        self.send_btn.setMinimumSize(58, 34)
        self.send_btn.setMaximumWidth(86)
        row_layout.addWidget(self.send_btn)

        layout.addWidget(input_row)

        self.send_btn.clicked.connect(self._send)
        self.input_field.returnPressed.connect(self._send)
        self.tts_session_finished.connect(self._on_tts_session_finished)
        self.event_bus.chat_request_accepted.connect(self._on_request_accepted)
        self.event_bus.chat_token_payload.connect(self._on_token_payload)
        self.event_bus.chat_complete_payload.connect(self._on_complete_payload)
        self.event_bus.proactive_start.connect(self._on_proactive_start)
        self.event_bus.telemetry_updated.connect(self._on_telemetry)
        self.event_bus.chat_sentence.connect(self._on_chat_sentence)
        self.event_bus.interrupt_ack.connect(self._on_interrupt_ack)

        # Wire up audio service
        if self.audio_service:
            self.audio_service.speech_recognized.connect(
                self._on_speech_recognized
            )
            self.audio_service.tts_started.connect(self._on_fallback_tts_started)
            self.audio_service.tts_finished.connect(self._on_fallback_tts_finished)

        # Wire up ContinuousAudioPipeline (PRD section 6) for barge-in detection
        if self.continuous_audio:
            self.continuous_audio.interruption_detected.connect(
                self._on_barge_in_detected
            )
            self.continuous_audio.speech_onset.connect(
                self._on_continuous_speech_onset
            )

        # Inline status tracking (Fix 4 - thinking/status inline in chat)
        self._inline_status_active = False
        self._inline_status_start = 0  # document char position before status block

        self._conversation_starter = None
        self._behavior_controller = None

    def set_camera_service(self, cam):
        self.camera_service = cam

    def set_conversation_starter(self, cs):
        self._conversation_starter = cs

    def set_behavior_controller(self, controller):
        self._behavior_controller = controller

    def set_voice_manager(self, vm):
        self.voice_manager = vm
        if vm:
            # Sync combo with backend engine (read from VoiceManager, not backend directly)
            eng = vm.active_backend_name
            if "qwen" in eng.lower():
                self.tts_combo.setCurrentText("Qwen3-TTS (Quality)")
            else:
                self.tts_combo.setCurrentText("pyttsx3 (Fast)")
            # Keep combo in sync when backend changes via voice_tab or other sources
            if hasattr(vm, "backend_changed"):
                vm.backend_changed.connect(self._on_voice_manager_backend_changed)
            backend = getattr(vm, "backend", None)
            if backend is not None and backend is not self._wired_voice_backend:
                backend.synthesis_started.connect(self._on_tts_generation_started)
                backend.synthesis_finished.connect(self._on_tts_generation_finished)
                backend.playback_started.connect(self._on_tts_playback_started)
                backend.playback_finished.connect(self._on_tts_playback_finished)
                backend.playback_interrupted.connect(self._on_tts_playback_interrupted)
                self._wired_voice_backend = backend

    def set_voice_tab(self, voice_tab):
        self.voice_tab = voice_tab

    def set_sing_handler(self, handler):
        self._sing_handler = handler
        if handler and hasattr(handler, "set_reply_callback"):
            def _reply(text):
                QTimer.singleShot(
                    0,
                    lambda msg=str(text or ""): self._append_system_note(msg),
                )
            handler.set_reply_callback(_reply)

    def _on_voice_manager_backend_changed(self, engine_id: str) -> None:
        """Keep tts_combo label in sync when the backend changes externally."""
        if engine_id == "qwen3-tts":
            target = "Qwen3-TTS (Quality)"
        else:
            target = "pyttsx3 (Fast)"
        if self.tts_combo.currentText() != target:
            self.tts_combo.blockSignals(True)
            self.tts_combo.setCurrentText(target)
            self.tts_combo.blockSignals(False)

    def _on_barge_in_detected(self, fragment):
        """User spoke while TTS was playing — interrupt output.

        IMPORTANT: This slot may fire from the ContinuousAudioPipeline's
        background thread, so Qt widget calls must be marshalled to the GUI
        thread via QTimer.singleShot(0, ...).
        """
        logger.info("[ChatPanel] Barge-in detected: %s", fragment[:50])
        if self._assistant_audio_active:
            # Stop TTS playback immediately (thread-safe - only touches locks)
            self._interrupt_output()
            # Also tell the server to abort token generation
            try:
                self.client.send_interrupt()
            except Exception:
                pass
            # Marshal the Qt widget update to the GUI thread
            safe_fragment = fragment[:30]
            QTimer.singleShot(0, lambda sf=safe_fragment: self.chat_display.append(
                self._fmt_system(f'Barge-in: &ldquo;{sf}...&rdquo;')
            ))

    def _on_continuous_speech_onset(self):
        """ContinuousAudio detected user starting to speak."""
        logger.debug("[ChatPanel] Continuous VAD: speech onset")

    def _notify_continuous_audio_tts_state(self, is_speaking: bool):
        """Tell ContinuousAudioPipeline whether TTS is actively playing."""
        if self.continuous_audio:
            try:
                self.continuous_audio.notify_revia_speaking(is_speaking)
            except Exception:
                pass

    def _on_tts_engine_changed(self, text):
        """Route engine selection through VoiceManager to keep ownership clear."""
        if text == "Off":
            return
        engine_id = "qwen3-tts" if "Qwen3-TTS" in text else "pyttsx3"
        if self.voice_manager:
            # Use VoiceManager.set_backend so backend_changed signal fires
            self.voice_manager.set_backend(engine_id)

    def _capture_frame_b64(self):
        if not self.camera_service or not self.camera_service.is_active():
            return None
        pixmap = self.camera_service.snapshot()
        if pixmap is None:
            return None
        buf = QBuffer()
        buf.open(QIODevice.WriteOnly)
        pixmap.save(buf, "JPEG", 85)
        return base64.b64encode(buf.data().data()).decode("ascii")

    def _on_speech_recognized(self, text):
        self._pending_input_source = "VoiceInput"
        self.input_field.setText(text)
        self.chat_display.append(self._fmt_speech(text))
        # Auto-send the recognized speech
        self._send()
        # Turn off mic if not in always-listening mode
        if self.audio_service and not self.audio_service._always_listening:
            self.mic_btn.setChecked(False)

    def _on_chat_sentence(self, sentence, request_id):
        """Server emitted a complete sentence — queue it for TTS immediately.

        This enables sentence-level streaming: TTS starts speaking the first
        sentence while the LLM is still generating the rest.
        """
        # Mark that server provides sentence events - disables old local extraction.
        # If fallback buffering already captured text before the first sentence
        # event arrived, drop it so we do not replay the same text later.
        if not self._server_sentence_streaming and self._tts_sentence_buf:
            self._tts_sentence_buf = ""
        self._server_sentence_streaming = True
        if not self.voice_manager or not sentence.strip():
            return
        tts_mode = self.tts_combo.currentText()
        if tts_mode == "Off":
            return
        # Only queue if this sentence belongs to the current request
        if request_id and self._current_request_id and request_id != self._current_request_id:
            return
        try:
            self._tts_queue.put_nowait((sentence, self._current_emotion))
            self._ensure_tts_worker()
        except queue.Full:
            logger.warning("[ChatPanel] TTS queue full, dropping sentence")

    def _drain_tts_queue(self, max_items=200):
        """Drain pending TTS items and keep Queue bookkeeping balanced."""
        drained = 0
        for _ in range(max_items):
            try:
                self._tts_queue.get_nowait()
            except queue.Empty:
                break
            else:
                self._tts_queue.task_done()
                drained += 1
        if drained >= max_items and not self._tts_queue.empty():
            logger.warning("[ChatPanel] TTS queue drain hit safety cap (%d)", max_items)
        return drained

    def _interrupt_output(self):
        """Stop TTS playback and clear the sentence queue.

        Thread-safe: can be called from the audio thread (barge-in) or the
        GUI thread. Qt signal emissions are marshalled via QTimer.singleShot.
        """
        # Clear any pending inline status block in the chat stream
        QTimer.singleShot(0, self._clear_inline_status)
        # Set interrupted flag first so the TTS worker stops picking up new items
        with self._tts_worker_lock:
            self._tts_session_interrupted = True
            self._assistant_audio_active = False
        # Drain the queue - get_nowait() is atomic; safety cap prevents infinite loop
        drained = self._drain_tts_queue()
        # Stop any active playback
        if self.voice_manager and hasattr(self.voice_manager, 'backend'):
            try:
                self.voice_manager.backend.stop_output()
            except Exception:
                pass
        # Marshal Qt-only calls to the GUI thread to avoid cross-thread crashes
        QTimer.singleShot(0, lambda: self._notify_continuous_audio_tts_state(False))
        QTimer.singleShot(0, self.assistant_output_interrupted.emit)
        logger.info("[ChatPanel] Output interrupted, drained %d queued TTS items", drained)

    def _on_interrupt_ack(self):
        """Server acknowledged interrupt — clear TTS queue."""
        self._interrupt_output()

    def _on_proactive_start(self):
        """Revia is about to speak unprompted — show a subtle indicator."""
        self.chat_display.append(self._fmt_system("Revia initiates..."))

    def _handle_local_command(self, text):
        if not re.match(r"^!sing(?:\s|$)", text or "", re.IGNORECASE):
            return False
        self.input_field.clear()
        self.chat_display.append(self._fmt_user(text))
        if not self._sing_handler:
            self._append_system_note("Sing mode is not enabled yet.")
            return True
        args = re.sub(r"^!sing(?:\s+)?", "", text, flags=re.IGNORECASE).strip()
        try:
            reply = self._sing_handler.handle(args, "local_chat")
        except Exception as exc:
            logger.error("[ChatPanel] !sing command error: %s", exc)
            reply = "Something went wrong with !sing."
        if reply:
            self._append_system_note(reply)
        return True

    def _send(self):
        text = self.input_field.text().strip()
        if not text or len(text) > 10000:
            return
        if self._handle_local_command(text):
            self._pending_input_source = "UserMessage"
            return
        source = self._pending_input_source or "UserMessage"
        reason = "voice input response" if source == "VoiceInput" else "manual user message"
        if self.is_assistant_speaking() and not self._awaiting_reply:
            self.interrupt_assistant_output()
            # Also tell the server to stop generating so tokens don't pile up
            try:
                self.client.send_interrupt()
            except Exception:
                pass
            self._append_system_note("Interrupted previous speech for new message")
        if self._behavior_controller and not self._awaiting_reply:
            decision = self._behavior_controller.should_respond(
                source=source,
                reason=reason,
                require_voice_input=(source == "VoiceInput"),
                require_speech_output=False,
            )
            if not decision.allowed:
                self._append_system_note(
                    f"Conversation unavailable: {decision.reason}"
                )
                self._pending_input_source = "UserMessage"
                return
        if self._conversation_starter:
            self._conversation_starter.record_user_activity()

        stacked_questions = self._split_stacked_questions(text)
        stack_total = len(stacked_questions)

        image_b64 = None
        vision_context = None
        if self._vision_active:
            image_b64 = self._capture_frame_b64()
            if self.camera_service:
                vision_context = self.camera_service.build_live_context()
            if image_b64:
                self.chat_display.append(
                    self._fmt_user(text, note="[+ vision frame]")
                )
            else:
                self.chat_display.append(
                    self._fmt_user(text, note="[vision ctx]")
                )
        else:
            self.chat_display.append(self._fmt_user(text))
        if stack_total > 1:
            self.chat_display.append(
                self._fmt_system(f"Stack detected: {stack_total} questions — answering one-by-one")
            )

        self.input_field.clear()
        if self._awaiting_reply:
            for idx, q in enumerate(stacked_questions, start=1):
                if len(self._outbound_queue) >= self._OUTBOUND_QUEUE_MAX:
                    logger.warning("[ChatPanel] Outbound queue full (%d), dropping remaining", self._OUTBOUND_QUEUE_MAX)
                    break
                self._outbound_queue.append(
                    (q, image_b64, vision_context, idx, stack_total, source)
                )
            self.chat_display.append(
                self._fmt_system("Queued — sending after current reply")
            )
            self._pending_input_source = "UserMessage"
            return

        first = stacked_questions[0]
        self._start_request(
            first,
            image_b64=image_b64,
            vision_context=vision_context,
            source=source,
            reason=reason,
        )
        if stack_total > 1:
            for idx, q in enumerate(stacked_questions[1:], start=2):
                if len(self._outbound_queue) >= self._OUTBOUND_QUEUE_MAX:
                    logger.warning("[ChatPanel] Outbound queue full (%d), dropping remaining", self._OUTBOUND_QUEUE_MAX)
                    break
                self._outbound_queue.append(
                    (q, image_b64, vision_context, idx, stack_total, source)
                )
        self._pending_input_source = "UserMessage"

    def _split_stacked_questions(self, text):
        src = str(text or "").strip()
        if not src:
            return []

        # Multiline lists are treated as independent items if 2+ lines look like prompts/questions.
        raw_lines = [ln.strip() for ln in re.split(r"[\r\n]+", src) if ln.strip()]
        cleaned_lines = [
            re.sub(r"^[\-\*\d\.\)\]\s]+", "", ln).strip()
            for ln in raw_lines
        ]
        line_like_questions = [
            ln for ln in cleaned_lines
            if ln and (ln.endswith("?") or len(ln.split()) >= 4)
        ]
        if len(line_like_questions) >= 2:
            return line_like_questions[:8]

        # Single-line multi-question split by '?' boundaries.
        if src.count("?") >= 2:
            segments = []
            for m in re.finditer(r"[^?]+\?", src):
                seg = m.group(0).strip()
                seg = re.sub(r"^[,;:\-\s]+", "", seg).strip()
                if len(seg) >= 4:
                    segments.append(seg)
            if len(segments) >= 2:
                return segments[:8]

        return [src]

    def _start_request(self, text, image_b64=None, vision_context=None, source="UserMessage", reason="manual user message"):
        self._awaiting_reply = True
        self._awaiting_started_at = time.monotonic()
        self._pending_request_id = ""
        self._current_request_id = ""
        self._current_response = ""
        self._full_response = ""
        self._tts_sentence_buf = ""
        self._server_sentence_streaming = False  # Reset for each new request
        drained = self._drain_tts_queue()
        if drained:
            logger.debug("[ChatPanel] Cleared %d stale TTS items before request start", drained)
        self.client.send_chat(
            text,
            image_b64=image_b64,
            vision_context=vision_context,
            source=source,
            reason=reason,
        )

    def _on_request_accepted(self, payload):
        if not isinstance(payload, dict):
            return
        self._pending_request_id = str(payload.get("request_id", "") or "")
        self._begin_activity_request()

    def _remember_ignored_request(self, request_id):
        rid = str(request_id or "").strip()
        if not rid or rid in self._ignored_request_ids:
            return
        self._ignored_request_ids.append(rid)
        if len(self._ignored_request_ids) > 32:
            del self._ignored_request_ids[:-32]

    def _on_token_payload(self, payload):
        if not isinstance(payload, dict):
            return
        token = str(payload.get("token", "") or "")
        request_id = str(payload.get("request_id", "") or "")
        if not token:
            return
        if request_id and request_id in self._ignored_request_ids:
            self.event_bus.log_entry.emit(
                f"[Revia] Ignored token payload for canceled request {request_id}"
            )
            return
        if self._current_request_id and request_id and request_id != self._current_request_id:
            self.event_bus.log_entry.emit(
                f"[Revia] Ignored stale token payload for request {request_id}"
            )
            return
        if self._pending_request_id and request_id and request_id != self._pending_request_id and not self._current_request_id:
            self.event_bus.log_entry.emit(
                f"[Revia] Ignored out-of-order token payload for request {request_id}"
            )
            return
        self._start_activity_generation()
        if request_id and not self._current_request_id:
            self._current_request_id = request_id
        if not self._current_response:
            self.chat_display.append(self._fmt_revia_start())
        self._current_response += token
        self._full_response += token
        cursor = self.chat_display.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(token)
        self.chat_display.setTextCursor(cursor)
        self.chat_display.ensureCursorVisible()

        # NOTE: Sentence-level TTS is now handled by server-side chat_sentence
        # events via _on_chat_sentence(). The old local sentence extraction below
        # is kept ONLY as a fallback for pyttsx3 when server doesn't send
        # chat_sentence events (e.g., older server versions).
        # To avoid double-playback, we track whether chat_sentence events arrived.
        if not self._server_sentence_streaming:
            tts_mode = self.tts_combo.currentText()
            if tts_mode.startswith("pyttsx3") and self.voice_manager:
                self._tts_sentence_buf += token
                self._flush_tts_sentences(end_of_response=False)

    def _flush_tts_sentences(self, end_of_response=False):
        """Split buffer on sentence boundaries and queue each complete sentence with emotion."""
        buf = self._tts_sentence_buf
        while buf:
            # Find earliest sentence terminator followed by whitespace (or end if flushing)
            best = -1
            for i, ch in enumerate(buf):
                if ch in '.!?':
                    # Skip over closing quotes/parens
                    j = i + 1
                    while j < len(buf) and buf[j] in '"\')':
                        j += 1
                    at_end = j >= len(buf)
                    followed_by_space = j < len(buf) and buf[j] in ' \n\t'
                    if followed_by_space or (end_of_response and at_end):
                        best = j
                        break
            if best == -1:
                break
            sentence = buf[:best].strip()
            buf = buf[best:].lstrip()
            if len(sentence) > 2 and not sentence.startswith('['):
                # Queue (text, emotion) tuple
                self._tts_queue.put((sentence, self._current_emotion))
                self._ensure_tts_worker()
        self._tts_sentence_buf = buf

    def _ensure_tts_worker(self):
        with self._tts_worker_lock:
            if self._tts_worker_active:
                return
            self._tts_worker_active = True
            self._tts_session_interrupted = False
            self.tts_session_started.emit()
            threading.Thread(target=self._tts_worker_prefetch, daemon=True).start()

    def _tts_worker_prefetch(self):
        """Synthesize upcoming chunks while the current chunk is playing.

        Uses a ThreadPoolExecutor so up to 3 sentences are synthesized in
        parallel.  A sequence-numbered ordered buffer ensures playback always
        happens in the original sentence order even when faster-synthesizing
        chunks finish ahead of slower ones.
        """
        if not self.voice_manager:
            with self._tts_worker_lock:
                self._tts_worker_active = False
                self._assistant_audio_active = False
                self._tts_session_interrupted = False
            self.tts_session_finished.emit(False)
            return

        import heapq
        from concurrent.futures import ThreadPoolExecutor

        ready_queue = queue.Queue(maxsize=6)
        sentinel = object()
        stop_event = threading.Event()

        def _interrupted():
            if stop_event.is_set():
                return True
            with self._tts_worker_lock:
                return bool(self._tts_session_interrupted)

        def _coerce_item(item):
            if isinstance(item, tuple) and len(item) == 2:
                return str(item[0] or "").strip(), item[1] or self._current_emotion
            return str(item or "").strip(), self._current_emotion

        def _put_ready(payload):
            while not _interrupted():
                try:
                    ready_queue.put(payload, timeout=0.25)
                    return True
                except queue.Full:
                    continue
            return False

        # --- parallel synthesis state ---
        heap_lock = threading.Lock()
        ordered_heap = []      # min-heap of (seq_no, sentence, emotion, wav, error)
        play_seq = [0]         # next expected sequence number for playback
        pending_count = [0]    # futures not yet resolved
        heap_event = threading.Event()

        def _synth_loop():
            """Submit synthesis jobs to the thread pool; put results in order."""
            MAX_PARALLEL = 3
            executor = ThreadPoolExecutor(
                max_workers=MAX_PARALLEL, thread_name_prefix="revia-synth"
            )
            submit_seq = [0]
            idle_ticks = 0

            def _do_synth(sentence, emotion, seq):
                wav_path = None
                error = ""
                try:
                    if sentence and hasattr(self.voice_manager, "synthesize_to_wav"):
                        wav_path, info = self.voice_manager.synthesize_to_wav(
                            sentence, emotion=emotion
                        )
                        if not wav_path:
                            error = str(info or "")
                    elif sentence:
                        error = "voice prefetch unavailable"
                except Exception as exc:
                    error = str(exc)
                    logger.error("[ChatPanel] TTS synthesis error: %s", exc)
                return seq, sentence, emotion, wav_path, error

            def _on_future_done(future, seq):
                try:
                    result = future.result()
                except Exception as exc:
                    result = (seq, "", self._current_emotion, None, str(exc))
                finally:
                    self._tts_queue.task_done()
                with heap_lock:
                    heapq.heappush(ordered_heap, result)
                    pending_count[0] -= 1
                heap_event.set()

            def _flush_ordered():
                """Push any in-order results from the heap to ready_queue."""
                while True:
                    with heap_lock:
                        if not ordered_heap or ordered_heap[0][0] != play_seq[0]:
                            break
                        result = heapq.heappop(ordered_heap)
                    play_seq[0] += 1
                    _seq, sentence, emotion, wav_path, error = result
                    if sentence and not _interrupted():
                        if not _put_ready((sentence, emotion, wav_path, error)):
                            return  # interrupted

            try:
                while not _interrupted():
                    try:
                        item = self._tts_queue.get(timeout=0.15)
                    except queue.Empty:
                        if not self._awaiting_reply:
                            idle_ticks += 1
                            if idle_ticks >= 6:
                                break
                        else:
                            idle_ticks = 0
                        _flush_ordered()
                        heap_event.wait(timeout=0.1)
                        heap_event.clear()
                        continue

                    idle_ticks = 0
                    sentence, emotion = _coerce_item(item)
                    seq = submit_seq[0]
                    submit_seq[0] += 1

                    with heap_lock:
                        pending_count[0] += 1
                    future = executor.submit(_do_synth, sentence, emotion, seq)
                    future.add_done_callback(lambda f, s=seq: _on_future_done(f, s))

                    _flush_ordered()

                # Drain: wait for all in-flight futures to finish, then flush
                while pending_count[0] > 0 and not _interrupted():
                    heap_event.wait(timeout=0.5)
                    heap_event.clear()
                    _flush_ordered()
                _flush_ordered()

            finally:
                executor.shutdown(wait=False)

            _put_ready(sentinel)

        synth_thread = threading.Thread(
            target=_synth_loop,
            daemon=True,
            name="revia-tts-synth",
        )
        synth_thread.start()

        interrupted = False
        try:
            while True:
                if _interrupted():
                    interrupted = True
                    break

                try:
                    ready = ready_queue.get(timeout=0.25)
                except queue.Empty:
                    if not synth_thread.is_alive():
                        break
                    continue

                if ready is sentinel:
                    break

                sentence, emotion, wav_path, error = ready
                if not sentence:
                    continue

                with self._tts_worker_lock:
                    if self._tts_session_interrupted:
                        interrupted = True
                        break
                    self._assistant_audio_active = True

                self._notify_continuous_audio_tts_state(True)
                self.event_bus.log_entry.emit("[Revia] TTS start")
                try:
                    if wav_path and hasattr(self.voice_manager, "play_wav_sync"):
                        self.voice_manager.play_wav_sync(wav_path)
                    else:
                        if error:
                            self.event_bus.log_entry.emit(
                                f"[Revia] TTS prefetch fallback: {error}"
                            )
                        self.voice_manager.speak_sync(sentence, emotion=emotion)
                except Exception as exc:
                    logger.error(f"TTS worker error: {exc}")
                    self.event_bus.log_entry.emit(f"[Revia] TTS worker error: {exc}")
                finally:
                    self.event_bus.log_entry.emit("[Revia] TTS end")
                    with self._tts_worker_lock:
                        self._assistant_audio_active = False
                    self._notify_continuous_audio_tts_state(False)
        finally:
            interrupted = interrupted or _interrupted()
            stop_event.set()
            if interrupted:
                try:
                    if self.voice_manager and hasattr(self.voice_manager, "backend"):
                        self.voice_manager.backend.stop_output()
                except Exception:
                    pass
                self._drain_tts_queue()
            synth_thread.join(timeout=0.5)
            with self._tts_worker_lock:
                self._assistant_audio_active = False
                self._tts_worker_active = False
                self._tts_session_interrupted = False
            self._notify_continuous_audio_tts_state(False)
            self.tts_session_finished.emit(bool(interrupted))

    def _on_telemetry(self, data):
        emotion = data.get("emotion", {}) if isinstance(data, dict) else {}
        label = str(emotion.get("label", "neutral")).strip().lower()
        if label:
            self._current_emotion = label
        if self.voice_manager and emotion:
            self.voice_manager.apply_emotion_modifiers(emotion)
        self._recover_stale_reply_state(data)

    def _on_complete_payload(self, payload):
        if not isinstance(payload, dict):
            return
        text = str(payload.get("text", "") or "")
        request_id = str(payload.get("request_id", "") or "")
        speakable = bool(payload.get("speakable", True))
        mode = str(payload.get("mode", "") or "")
        success = bool(payload.get("success", True))
        error_type = str(payload.get("error_type", "") or "")
        metadata = payload.get("metadata", {}) or {}
        if request_id and request_id in self._ignored_request_ids:
            self.event_bus.log_entry.emit(
                f"[Revia] Ignored completion payload for canceled request {request_id}"
            )
            return
        if self._current_request_id and request_id and request_id != self._current_request_id:
            self.event_bus.log_entry.emit(
                f"[Revia] Ignored stale completion payload for request {request_id}"
            )
            return
        if self._pending_request_id and request_id and request_id != self._pending_request_id and not self._current_request_id:
            self.event_bus.log_entry.emit(
                f"[Revia] Ignored out-of-order completion payload for request {request_id}"
            )
            return
        if request_id and not self._current_request_id:
            self._current_request_id = request_id

        # This completion belongs to the active request, so the watchdog can
        # be safely disarmed now.
        self._thinking_watchdog.stop()
        self._finish_activity_request()
        # Always clear any lingering inline status block before rendering output
        self._clear_inline_status()

        if (not text) and (not success) and error_type != "interrupted":
            text = str(
                metadata.get("fallback_text")
                or "Uh... something's wrong. Someone tell my operator he messed up."
            ).strip()

        if text and text == self._last_completed_text and mode != "NORMAL_RESPONSE" and success:
            self.event_bus.log_entry.emit(
                f"[Revia] Duplicate completion text suppressed for request {request_id or 'unknown'}"
            )
            self._activity_waiting_for_tts = False
            self._finalize_activity_summary()
            self._clear_reply_tracking()
            self._schedule_next_queued_request(delay_ms=250)
            return

        if not self._current_response and text:
            self.chat_display.append(self._fmt_revia(text))
        self._current_response = ""
        self._full_response = ""
        self.chat_display.append("")
        if text:
            self._last_completed_text = text

        tts_mode = self.tts_combo.currentText()
        if tts_mode != "Off" and speakable:
            if self.voice_manager:
                if tts_mode.startswith("pyttsx3"):
                    if self._server_sentence_streaming:
                        self._tts_sentence_buf = ""
                    else:
                        # Only use the local fallback buffer when the server did
                        # not stream sentence boundaries for this reply.
                        self._flush_tts_sentences(end_of_response=True)
                        remaining = self._tts_sentence_buf.strip()
                        self._tts_sentence_buf = ""
                        if remaining and not remaining.startswith('['):
                            # Queue (text, emotion) tuple
                            self._tts_queue.put((remaining, self._current_emotion))
                            self._ensure_tts_worker()
                else:
                    # For Qwen quality mode, synthesize once per completed reply
                    # only when sentence streaming is unavailable.
                    clean = (text or "").strip()
                    self._tts_sentence_buf = ""
                    if clean and not clean.startswith('[') and not self._server_sentence_streaming:
                        # Queue (text, emotion) tuple
                        self._tts_queue.put((clean, self._current_emotion))
                        self._ensure_tts_worker()
            elif self.audio_service:
                clean = (text or "").strip()
                if clean and not clean.startswith("["):
                    self.audio_service.speak(clean)
        else:
            self._tts_sentence_buf = ""

        self._activity_waiting_for_tts = bool(
            tts_mode != "Off" and speakable and (self.voice_manager or self.audio_service)
        )
        if error_type == "interrupted":
            self._activity_interrupted = True
            self._activity_waiting_for_tts = False
            self._finalize_activity_summary()
        elif not self._activity_waiting_for_tts:
            self._finalize_activity_summary()
        else:
            QTimer.singleShot(350, self._finalize_activity_summary_if_ready)

        self._clear_reply_tracking()
        self._schedule_next_queued_request()

    def interrupt_assistant_output(self):
        self._tts_sentence_buf = ""
        self._activity_interrupted = True
        # Disarm the watchdog - interrupt is an intentional user action
        self._thinking_watchdog.stop()
        # Always remove any pending "Thinking..." / inline status bubble immediately so
        # the user sees a clean slate before the next response starts rendering.
        self._clear_inline_status()
        # Stop playback FIRST so speak_sync() unblocks before we touch worker state
        if self.voice_manager:
            try:
                self.voice_manager.backend.stop_output()
            except Exception as e:
                logger.warning(f"Error stopping voice output: {e}")
        drained = self._drain_tts_queue()
        # Set all flags under the lock atomically to avoid racing the TTS worker
        with self._tts_worker_lock:
            self._assistant_audio_active = False
            self._tts_session_interrupted = True
            # NOTE: we do NOT set _tts_worker_active = False here. The TTS worker
            # thread will see _tts_session_interrupted and exit on its own. Setting
            # it to False while the thread is still alive would allow a second worker
            # to start concurrently, causing audio device contention crashes.
        self._pending_request_id = ""
        self._current_request_id = ""
        self._notify_continuous_audio_tts_state(False)
        if drained:
            logger.debug("[ChatPanel] Cleared %d queued TTS items during manual interrupt", drained)
        self.assistant_output_interrupted.emit()

    def _begin_activity_request(self):
        now = time.monotonic()
        self._activity_thinking_started_at = now
        self._activity_thinking_duration = 0.0
        self._activity_generation_started_at = None
        self._activity_generation_duration = 0.0
        self._activity_tts_session_active = False
        self._activity_tts_generation_started_at = None
        self._activity_tts_generation_accum = 0.0
        self._activity_tts_generation_duration = 0.0
        self._activity_tts_playback_started_at = None
        self._activity_tts_playback_accum = 0.0
        self._activity_tts_playback_duration = 0.0
        self._activity_waiting_for_tts = False
        self._activity_interrupted = False
        # Inject inline thinking indicator into the chat stream
        self._insert_inline_status("Thinking...")
        # Arm the hard-timeout watchdog - stops automatically when reply arrives
        self._thinking_watchdog.start()
        self._refresh_activity_indicator()

    def _start_activity_generation(self):
        if self._activity_generation_started_at is not None:
            return
        now = time.monotonic()
        if self._activity_thinking_started_at is not None:
            self._activity_thinking_duration = max(
                0.0,
                now - self._activity_thinking_started_at,
            )
            self._activity_thinking_started_at = None
        self._activity_generation_started_at = now
        # Disarm the watchdog - we have a live response stream
        self._thinking_watchdog.stop()
        # Clear the inline thinking indicator - response content follows immediately
        self._clear_inline_status()
        self._refresh_activity_indicator()

    def _finish_activity_request(self):
        now = time.monotonic()
        if self._activity_generation_started_at is not None:
            self._activity_generation_duration = max(
                0.0,
                now - self._activity_generation_started_at,
            )
            self._activity_generation_started_at = None
        elif self._activity_thinking_started_at is not None:
            self._activity_thinking_duration = max(
                0.0,
                now - self._activity_thinking_started_at,
            )
            self._activity_thinking_started_at = None
        self._refresh_activity_indicator()

    def _ensure_tts_activity_session(self):
        if not self._activity_tts_session_active:
            self._activity_tts_session_active = True
            self._activity_tts_generation_started_at = None
            self._activity_tts_generation_accum = 0.0
            self._activity_tts_generation_duration = 0.0
            self._activity_tts_playback_started_at = None
            self._activity_tts_playback_accum = 0.0
            self._activity_tts_playback_duration = 0.0

    def _on_tts_generation_started(self):
        self._ensure_tts_activity_session()
        if self._activity_tts_generation_started_at is None:
            self._activity_tts_generation_started_at = time.monotonic()
        self._refresh_activity_indicator()

    def _on_tts_generation_finished(self, _metrics=None):
        if self._activity_tts_generation_started_at is not None:
            self._activity_tts_generation_accum += max(
                0.0,
                time.monotonic() - self._activity_tts_generation_started_at,
            )
            self._activity_tts_generation_started_at = None
        self._activity_tts_generation_duration = self._activity_tts_generation_accum
        self._refresh_activity_indicator()

    def _on_tts_playback_started(self):
        self._ensure_tts_activity_session()
        if self._activity_tts_playback_started_at is None:
            self._activity_tts_playback_started_at = time.monotonic()
        self._refresh_activity_indicator()

    def _on_tts_playback_finished(self):
        if self._activity_tts_playback_started_at is not None:
            self._activity_tts_playback_accum += max(
                0.0,
                time.monotonic() - self._activity_tts_playback_started_at,
            )
            self._activity_tts_playback_started_at = None
        self._activity_tts_playback_duration = self._activity_tts_playback_accum
        self._refresh_activity_indicator()

    def _on_tts_playback_interrupted(self):
        self._activity_interrupted = True
        self._on_tts_playback_finished()

    def _on_fallback_tts_started(self):
        # The lightweight fallback path only exposes a start/end speaking window,
        # so treat it as playback timing rather than inventing a separate synth span.
        self._on_tts_playback_started()

    def _on_fallback_tts_finished(self):
        self._on_tts_playback_finished()
        self._finish_tts_activity_session(interrupted=False)

    def _finish_tts_activity_session(self, interrupted=False):
        now = time.monotonic()
        if self._activity_tts_generation_started_at is not None:
            self._activity_tts_generation_accum += max(
                0.0,
                now - self._activity_tts_generation_started_at,
            )
            self._activity_tts_generation_started_at = None
        if self._activity_tts_playback_started_at is not None:
            self._activity_tts_playback_accum += max(
                0.0,
                now - self._activity_tts_playback_started_at,
            )
            self._activity_tts_playback_started_at = None
        self._activity_tts_generation_duration = self._activity_tts_generation_accum
        self._activity_tts_playback_duration = self._activity_tts_playback_accum
        self._activity_tts_session_active = False
        if interrupted:
            self._activity_interrupted = True
        self._activity_waiting_for_tts = False
        self._finalize_activity_summary()

    def _on_tts_session_finished(self, interrupted):
        self._finish_tts_activity_session(interrupted=bool(interrupted))

    def _finalize_activity_summary_if_ready(self):
        if self._activity_waiting_for_tts and self._activity_tts_session_active:
            return
        self._activity_waiting_for_tts = False
        self._finalize_activity_summary()

    def _finalize_activity_summary(self):
        parts = []
        if self._activity_thinking_duration > 0.0:
            parts.append(f"Thinking {self._activity_thinking_duration:.1f}s")
        if self._activity_generation_duration > 0.0:
            parts.append(f"Generating {self._activity_generation_duration:.1f}s")
        if self._activity_tts_generation_duration > 0.0:
            parts.append(f"TTS Gen {self._activity_tts_generation_duration:.1f}s")
        if self._activity_tts_playback_duration > 0.0:
            parts.append(f"Speaking {self._activity_tts_playback_duration:.1f}s")
        if not parts:
            self._activity_summary = "Interrupted" if self._activity_interrupted else "Ready"
        else:
            summary = " | ".join(parts)
            if self._activity_interrupted:
                summary = f"Interrupted | {summary}"
            self._activity_summary = summary
        self._refresh_activity_indicator()

    def _current_activity_text(self):
        if self._activity_tts_playback_started_at is not None:
            return f"Speaking {max(0.0, time.monotonic() - self._activity_tts_playback_started_at):.1f}s", True
        if self._activity_tts_generation_started_at is not None:
            return f"TTS Gen {max(0.0, time.monotonic() - self._activity_tts_generation_started_at):.1f}s", True
        if self._activity_generation_started_at is not None:
            return f"Generating {max(0.0, time.monotonic() - self._activity_generation_started_at):.1f}s", True
        if self._activity_thinking_started_at is not None:
            return f"Thinking {max(0.0, time.monotonic() - self._activity_thinking_started_at):.1f}s", True
        return self._activity_summary, False

    def _refresh_activity_indicator(self):
        text, active = self._current_activity_text()
        text = str(text or "Ready")
        self.activity_label.setText(text)
        self.activity_label.setToolTip(text)
        self.activity_label.setProperty("activityState", "active" if active else "idle")
        apply_status_style(
            self.activity_label,
            role="warning" if active else "muted",
        )
        if active:
            if not self._activity_timer.isActive():
                self._activity_timer.start()
        elif self._activity_timer.isActive():
            self._activity_timer.stop()

    @staticmethod
    def _activity_style(*, active: bool) -> str:
        return ""

    def _recover_stale_reply_state(self, data):
        if not isinstance(data, dict) or not self._awaiting_reply:
            return
        request_lifecycle = data.get("request_lifecycle", {}) or {}
        active_request_id = str(request_lifecycle.get("active_request_id", "") or "")
        active_turn = request_lifecycle.get("active_turn", {}) or {}
        runtime_state = str(data.get("state", "") or "")
        known_request_id = self._current_request_id or self._pending_request_id
        awaiting_age_s = (
            time.monotonic() - self._awaiting_started_at
            if self._awaiting_started_at > 0.0
            else 0.0
        )

        if known_request_id and active_request_id and active_request_id != known_request_id:
            self.event_bus.log_entry.emit(
                f"[Revia] Clearing stale reply state for request {known_request_id} "
                f"(active request is {active_request_id})"
            )
            self._clear_reply_tracking()
            self._schedule_next_queued_request(delay_ms=250)
            return

        if known_request_id and not active_request_id and runtime_state in ("Idle", "Cooldown", "Error"):
            self.event_bus.log_entry.emit(
                f"[Revia] Clearing stale reply state for completed request {known_request_id}"
            )
            self._clear_reply_tracking()
            self._schedule_next_queued_request(delay_ms=250)
            return

        if not known_request_id and not active_request_id and runtime_state in ("Idle", "Cooldown", "Error") and awaiting_age_s >= 5.0:
            self.event_bus.log_entry.emit(
                "[Revia] Clearing stale reply state after timeout with no active request"
            )
            self._clear_reply_tracking()
            self._schedule_next_queued_request(delay_ms=250)

    def _clear_reply_tracking(self):
        # Disarm the watchdog - reply is being resolved (success, error, or stale recovery)
        self._thinking_watchdog.stop()
        # Always clear any lingering inline status bubble - this method is called
        # on every exit path (completion, stale recovery, timeout) so doing it here
        # guarantees the indicator never gets orphaned.
        self._clear_inline_status()
        self._awaiting_reply = False
        self._awaiting_started_at = 0.0
        self._pending_request_id = ""
        self._current_request_id = ""
        self._current_response = ""
        self._full_response = ""
        self._tts_sentence_buf = ""

    def _on_thinking_timeout(self):
        """Called when the backend stays in Thinking for too long with no response.

        Force-clears all reply state and shows a user-visible note so the
        conversation doesn't appear permanently frozen.
        """
        if not self._awaiting_reply:
            return  # Already resolved — nothing to do
        logger.warning("[ChatPanel] Thinking watchdog fired — no response after timeout, clearing state")
        self._remember_ignored_request(self._current_request_id)
        self._remember_ignored_request(self._pending_request_id)
        try:
            self.client.send_interrupt()
        except Exception as exc:
            logger.warning("[ChatPanel] Failed to interrupt timed-out request: %s", exc)
        self._clear_inline_status()
        self._append_system_note("Response timed out — backend took too long. Try again.")
        self._finish_activity_request()
        self._clear_reply_tracking()
        self._schedule_next_queued_request(delay_ms=900)

    def _schedule_next_queued_request(self, delay_ms=900):
        if not self._outbound_queue:
            return
        nxt_text, nxt_img, nxt_ctx, nxt_i, nxt_n, nxt_source = self._outbound_queue.pop(0)
        if nxt_n > 1:
            self.chat_display.append(
                self._fmt_system(f"Continuing question {nxt_i}/{nxt_n}")
            )
        QTimer.singleShot(
            delay_ms,
            lambda _t=nxt_text, _i=nxt_img, _c=nxt_ctx, _s=nxt_source: self._start_request(
                _t, image_b64=_i, vision_context=_c, source=_s, reason="queued user message",
            ),
        )

    def _on_vision_toggled(self, active):
        self._vision_active = active
        self.vision_mode_changed.emit(bool(active))
        if active:
            self.vision_btn.setToolTip("Vision ON - every message includes a fresh frame")
            self.input_field.setPlaceholderText(
                "Vision is live - ask about what the camera sees..."
            )
        else:
            self.vision_btn.setToolTip("Toggle live vision context for messages")
            self.input_field.setPlaceholderText("Type a message...")

    def _is_always_listening_mode(self):
        try:
            return "always" in self.voice_tab.ptt_mode.currentText().lower()
        except Exception:
            return False

    def sync_mic_state_from_audio(self):
        """Reflect externally-started STT in the mic button without toggling it."""
        active = bool(self.audio_service and self.audio_service.is_listening())
        if self.mic_btn.isChecked() == active:
            return
        self.mic_btn.blockSignals(True)
        self.mic_btn.setChecked(active)
        self.mic_btn.blockSignals(False)
        if active:
            self.mic_btn.setToolTip("Listening... (click to stop)")
        else:
            self.mic_btn.setToolTip("Toggle Listening (click to start/stop)")

    def _on_mic_toggled(self, active):
        if active:
            self.mic_btn.setToolTip("Listening... (click to stop)")
            self.chat_display.append(
                self._fmt_system("🎤 Listening — speak now")
            )
            if self.audio_service:
                started = self.audio_service.start_listening(
                    always=self._is_always_listening_mode()
                )
                if started is False:
                    self.mic_btn.blockSignals(True)
                    self.mic_btn.setChecked(False)
                    self.mic_btn.blockSignals(False)
                    self.mic_btn.setToolTip("Toggle Listening (click to start/stop)")
        else:
            self.mic_btn.setToolTip("Toggle Listening (click to start/stop)")
            if self.audio_service:
                self.audio_service.stop_listening()

    def is_tts_enabled(self):
        return self.tts_combo.currentText() != "Off"

    def is_vision_enabled(self):
        return bool(self._vision_active)

    def is_vision_processing(self):
        return bool(self._vision_active and self._awaiting_reply)

    def is_assistant_speaking(self):
        return bool(
            self._awaiting_reply
            or self._tts_worker_active
            or self._assistant_audio_active
            or (not self._tts_queue.empty())
        )

    # Anime message format helpers
    @staticmethod
    def _fmt_user(text: str, note: str = "") -> str:
        primary = _chat_color("PrimaryText")
        secondary = _chat_color("SecondaryText")
        accent = _chat_color("Accent")
        note_html = (
            f' <span style="color:{secondary};font-size:8px;">{note}</span>'
            if note else ""
        )
        return (
            '<table width="100%" cellpadding="0" cellspacing="0"'
            ' style="margin:6px 0 3px 0;">'
            '<tr><td align="right" style="padding:6px 10px;">'
            f'<span style="color:{accent};font-weight:600;font-size:11px;">&#9670; You</span>'
            f'&nbsp;&nbsp;<span style="color:{primary};font-size:11px;line-height:1.5;">{text}</span>{note_html}'
            '</td></tr></table>'
        )

    @staticmethod
    def _fmt_revia_start() -> str:
        accent = _chat_color("AccentHover")
        return f'<span style="color:{accent};font-weight:600;font-size:11px;">&#10022; Revia</span> '

    @staticmethod
    def _fmt_revia(text: str) -> str:
        primary = _chat_color("PrimaryText")
        accent = _chat_color("AccentHover")
        return (
            '<table width="100%" cellpadding="0" cellspacing="0"'
            ' style="margin:6px 0 3px 0;">'
            '<tr><td align="left" style="padding:6px 10px;">'
            f'<span style="color:{accent};font-weight:600;font-size:11px;">&#10022; Revia</span>'
            f'&nbsp;&nbsp;<span style="color:{primary};font-size:11px;line-height:1.5;">{text}</span>'
            '</td></tr></table>'
        )

    @staticmethod
    def _fmt_system(text: str) -> str:
        secondary = _chat_color("SecondaryText")
        return (
            f'<span style="color:{secondary};font-size:9px;font-style:italic;">'
            f'&#11015; {text}</span>'
        )

    @staticmethod
    def _fmt_speech(text: str) -> str:
        accent = _chat_color("AccentHover")
        return (
            f'<span style="color:{accent};font-size:9px;font-style:italic;">'
            f'&#127908; &ldquo;{text}&rdquo;</span>'
        )

    def _append_system_note(self, text):
        self.chat_display.append(self._fmt_system(text))

    # Inline chat status (Fix 4)

    def _insert_inline_status(self, text: str) -> None:
        """Append a transient status bubble to the chat stream.

        Saves the document position so the block can be replaced or removed
        when the assistant transitions to the next state.
        """
        if self._inline_status_active:
            self._clear_inline_status()
        # Record position at current document end (before append inserts a new block)
        cursor = self.chat_display.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self._inline_status_start = cursor.position()
        self.chat_display.append(self._fmt_status_inline(text))
        self._inline_status_active = True
        self.chat_display.ensureCursorVisible()

    def _clear_inline_status(self) -> None:
        """Remove the transient inline status block from the chat stream."""
        if not self._inline_status_active:
            return
        cursor = self.chat_display.textCursor()
        cursor.setPosition(self._inline_status_start)
        cursor.movePosition(cursor.MoveOperation.End, cursor.MoveMode.KeepAnchor)
        cursor.removeSelectedText()
        self._inline_status_active = False
        self._inline_status_start = 0

    @staticmethod
    def _fmt_status_inline(text: str) -> str:
        secondary = _chat_color("SecondaryText")
        return (
            f'<span style="color:{secondary};font-size:9px;font-style:italic;">'
            f'&#10227; {text}</span>'
        )
