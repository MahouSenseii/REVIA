import base64
import logging
import queue
import re
import threading
from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit, QPushButton,
    QComboBox,
)
from PySide6.QtCore import Qt, QBuffer, QIODevice, QTimer, Signal
from PySide6.QtGui import QFont, QPixmap

logger = logging.getLogger(__name__)


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
        self._assistant_audio_active = False
        self._pending_input_source = "UserMessage"
        self._pending_request_id = ""
        self._current_request_id = ""
        self._last_completed_text = ""
        self._tts_session_interrupted = False
        self._server_sentence_streaming = False  # Set True when chat_sentence events arrive
        # Each item: (text, image_b64, vision_context, stack_index, stack_total, source)
        self._outbound_queue = []
        self._OUTBOUND_QUEUE_MAX = 50  # Prevent unbounded memory growth

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.chat_display = QTextEdit()
        self.chat_display.setObjectName("chatDisplay")
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(QFont("Consolas", 10))
        layout.addWidget(self.chat_display, stretch=1)

        input_row = QFrame()
        input_row.setObjectName("chatInputRow")
        input_row.setFixedHeight(48)
        row_layout = QHBoxLayout(input_row)
        row_layout.setContentsMargins(8, 6, 8, 6)
        row_layout.setSpacing(6)

        self.mic_btn = QPushButton("\U0001f3a4")
        self.mic_btn.setObjectName("micBtn")
        self.mic_btn.setFixedSize(36, 36)
        self.mic_btn.setCheckable(True)
        self.mic_btn.setToolTip("Toggle Listening (click to start/stop)")
        self.mic_btn.toggled.connect(self._on_mic_toggled)
        row_layout.addWidget(self.mic_btn)

        self.vision_btn = QPushButton("\U0001f441")
        self.vision_btn.setObjectName("visionBtn")
        self.vision_btn.setFixedSize(36, 36)
        self.vision_btn.setCheckable(True)
        self.vision_btn.setToolTip("Toggle live vision context for messages")
        self.vision_btn.toggled.connect(self._on_vision_toggled)
        row_layout.addWidget(self.vision_btn)

        self.tts_combo = QComboBox()
        self.tts_combo.addItems(["pyttsx3 (Fast)", "Qwen3-TTS (Quality)", "Off"])
        self.tts_combo.setFixedWidth(145)
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
        self.send_btn.setFixedSize(70, 36)
        row_layout.addWidget(self.send_btn)

        layout.addWidget(input_row)

        self.send_btn.clicked.connect(self._send)
        self.input_field.returnPressed.connect(self._send)
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

        # Wire up ContinuousAudioPipeline (PRD §6) for barge-in detection
        if self.continuous_audio:
            self.continuous_audio.interruption_detected.connect(
                self._on_barge_in_detected
            )
            self.continuous_audio.speech_onset.connect(
                self._on_continuous_speech_onset
            )

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
        # Sync combo with backend engine
        if vm:
            eng = vm.backend.engine_name
            if "qwen" in eng.lower():
                self.tts_combo.setCurrentText("Qwen3-TTS (Quality)")
            elif eng == "pyttsx3":
                self.tts_combo.setCurrentText("pyttsx3 (Fast)")

    def _on_barge_in_detected(self, fragment):
        """User spoke while TTS was playing — interrupt output."""
        logger.info("[ChatPanel] Barge-in detected: %s", fragment[:50])
        if self._assistant_audio_active:
            self._interrupt_output()
            self.chat_display.append(
                '<span style="color:#f59e0b;font-size:9px;">'
                f'[Barge-in: "{fragment[:30]}..."]</span>'
            )

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
        if text == "Off":
            return
        engine_id = "qwen3-tts" if "Qwen3-TTS" in text else "pyttsx3"
        if self.voice_manager:
            self.voice_manager.backend.set_engine(engine_id)

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
        self.chat_display.append(
            '<span style="color:#dc3250;font-size:9px;">'
            f'[Speech: "{text}"]</span>'
        )
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
        # Mark that server provides sentence events — disables old local extraction
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

    def _interrupt_output(self):
        """Stop TTS playback and clear the sentence queue."""
        # Set interrupted flag first so the TTS worker stops picking up new items
        with self._tts_worker_lock:
            self._tts_session_interrupted = True
            self._assistant_audio_active = False
        # Drain the queue — get_nowait() is atomic; safety cap prevents infinite loop
        drained = 0
        for _ in range(200):  # Safety cap: queue maxsize is 20, 200 is generous
            try:
                self._tts_queue.get_nowait()
                self._tts_queue.task_done()
                drained += 1
            except queue.Empty:
                break
        # Stop any active playback
        if self.voice_manager and hasattr(self.voice_manager, 'backend'):
            try:
                self.voice_manager.backend.stop_output()
            except Exception:
                pass
        self._notify_continuous_audio_tts_state(False)
        self.assistant_output_interrupted.emit()
        logger.info("[ChatPanel] Output interrupted, TTS queue cleared")

    def _on_interrupt_ack(self):
        """Server acknowledged interrupt — clear TTS queue."""
        self._interrupt_output()

    def _on_proactive_start(self):
        """Revia is about to speak unprompted — show a subtle indicator."""
        self.chat_display.append(
            '<span style="color:#a78bfa;font-size:9px;">'
            '[Revia initiates...]</span>'
        )

    def _send(self):
        text = self.input_field.text().strip()
        if not text or len(text) > 10000:
            return
        source = self._pending_input_source or "UserMessage"
        reason = "voice input response" if source == "VoiceInput" else "manual user message"
        if self.is_assistant_speaking() and not self._awaiting_reply:
            self.interrupt_assistant_output()
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
                    '<span style="color:#00d4ff;font-weight:bold;">You:</span> '
                    f'{text} <span style="color:#888;font-size:9px;">'
                    '[+ live vision frame/context]</span>'
                )
            else:
                self.chat_display.append(
                    '<span style="color:#00d4ff;font-weight:bold;">You:</span> '
                    f'{text} <span style="color:#cc3040;font-size:9px;">'
                    '[vision context only]</span>'
                )
        else:
            self.chat_display.append(
                f'<span style="color:#00d4ff;font-weight:bold;">You:</span> {text}'
            )
        if stack_total > 1:
            self.chat_display.append(
                '<span style="color:#9ca3af;font-size:9px;">'
                f'[Question stack detected: {stack_total}. Answering one-by-one.]</span>'
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
                '<span style="color:#9ca3af;font-size:9px;">'
                '[Queued: sending after current reply]</span>'
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
        self._pending_request_id = ""
        self._current_request_id = ""
        self._current_response = ""
        self._full_response = ""
        self._tts_sentence_buf = ""
        self._server_sentence_streaming = False  # Reset for each new request
        # Drain any leftover TTS queue from previous response (fix TOCTOU race)
        while True:
            try:
                self._tts_queue.get_nowait()
            except queue.Empty:
                break
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

    def _on_token_payload(self, payload):
        if not isinstance(payload, dict):
            return
        token = str(payload.get("token", "") or "")
        request_id = str(payload.get("request_id", "") or "")
        if not token:
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
        if request_id and not self._current_request_id:
            self._current_request_id = request_id
        if not self._current_response:
            self.chat_display.append(
                '<span style="color:#a78bfa;font-weight:bold;">Revia:</span> '
            )
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
            threading.Thread(target=self._tts_worker, daemon=True).start()

    def _tts_worker(self):
        """Background thread: consumes sentence queue and speaks each one in order."""
        while True:
            try:
                item = self._tts_queue.get(timeout=1.0)
                # Unpack (text, emotion) tuple
                if isinstance(item, tuple) and len(item) == 2:
                    sentence, emotion = item
                else:
                    # Legacy: handle plain text items
                    sentence = item
                    emotion = self._current_emotion
                with self._tts_worker_lock:
                    self._assistant_audio_active = True
                # Notify ContinuousAudio that TTS is playing (enables barge-in detection)
                self._notify_continuous_audio_tts_state(True)
                self.event_bus.log_entry.emit("[Revia] TTS start")
                self.voice_manager.speak_sync(
                    sentence, emotion=emotion
                )
                self.event_bus.log_entry.emit("[Revia] TTS end")
                self._tts_queue.task_done()
            except queue.Empty:
                with self._tts_worker_lock:
                    if self._tts_queue.empty():
                        was_interrupted = self._tts_session_interrupted
                        self._assistant_audio_active = False
                        self._tts_worker_active = False
                        self._tts_session_interrupted = False
                        # Notify ContinuousAudio that TTS stopped
                        self._notify_continuous_audio_tts_state(False)
                        if not was_interrupted:
                            self.tts_session_finished.emit(False)
                        return
            except Exception as exc:
                logger.error(f"TTS worker error: {exc}")
                self.event_bus.log_entry.emit(f"[Revia] TTS worker error: {exc}")
                with self._tts_worker_lock:
                    was_interrupted = self._tts_session_interrupted
                    self._assistant_audio_active = False
                    self._tts_worker_active = False
                    self._tts_session_interrupted = False
                if not was_interrupted:
                    self.tts_session_finished.emit(False)
                return
            finally:
                with self._tts_worker_lock:
                    if self._tts_queue.empty():
                        self._assistant_audio_active = False

    def _on_telemetry(self, data):
        emotion = data.get("emotion", {}) if isinstance(data, dict) else {}
        label = str(emotion.get("label", "neutral")).strip().lower()
        if label:
            self._current_emotion = label
        if self.voice_manager and emotion:
            self.voice_manager.apply_emotion_modifiers(emotion)

    def _on_complete_payload(self, payload):
        if not isinstance(payload, dict):
            return
        text = str(payload.get("text", "") or "")
        request_id = str(payload.get("request_id", "") or "")
        speakable = bool(payload.get("speakable", True))
        mode = str(payload.get("mode", "") or "")
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

        if text and text == self._last_completed_text and mode != "NORMAL_RESPONSE":
            self.event_bus.log_entry.emit(
                f"[Revia] Duplicate completion text suppressed for request {request_id or 'unknown'}"
            )
            self._awaiting_reply = False
            self._pending_request_id = ""
            self._current_request_id = ""
            if self._outbound_queue:
                nxt_text, nxt_img, nxt_ctx, _nxt_i, _nxt_n, nxt_source = self._outbound_queue.pop(0)
                # Bind variables via default args to avoid closure capture bugs
                QTimer.singleShot(
                    250,
                    lambda _t=nxt_text, _i=nxt_img, _c=nxt_ctx, _s=nxt_source: self._start_request(
                        _t, image_b64=_i, vision_context=_c, source=_s, reason="queued user message",
                    ),
                )
            return

        if not self._current_response and text:
            self.chat_display.append(
                f'<span style="color:#a78bfa;font-weight:bold;">Revia:</span> {text}'
            )
        self._current_response = ""
        self._full_response = ""
        self.chat_display.append("")
        if text:
            self._last_completed_text = text

        tts_mode = self.tts_combo.currentText()
        if tts_mode != "Off" and speakable:
            if self.voice_manager:
                if tts_mode.startswith("pyttsx3"):
                    # Flush any remaining partial sentence from the streaming buffer
                    self._flush_tts_sentences(end_of_response=True)
                    remaining = self._tts_sentence_buf.strip()
                    self._tts_sentence_buf = ""
                    if remaining and not remaining.startswith('['):
                        # Queue (text, emotion) tuple
                        self._tts_queue.put((remaining, self._current_emotion))
                        self._ensure_tts_worker()
                else:
                    # For Qwen quality mode, synthesize once per completed reply
                    clean = (text or "").strip()
                    self._tts_sentence_buf = ""
                    if clean and not clean.startswith('['):
                        # Queue (text, emotion) tuple
                        self._tts_queue.put((clean, self._current_emotion))
                        self._ensure_tts_worker()
            elif self.audio_service:
                clean = (text or "").strip()
                if clean and not clean.startswith("["):
                    self.audio_service.speak(clean)
        else:
            self._tts_sentence_buf = ""

        self._awaiting_reply = False
        self._pending_request_id = ""
        self._current_request_id = ""
        if self._outbound_queue:
            nxt_text, nxt_img, nxt_ctx, nxt_i, nxt_n, nxt_source = self._outbound_queue.pop(0)
            if nxt_n > 1:
                self.chat_display.append(
                    '<span style="color:#9ca3af;font-size:9px;">'
                    f'[Continuing stacked question {nxt_i}/{nxt_n}]</span>'
                )
            # Bind variables via default args to avoid closure capture bugs
            QTimer.singleShot(
                900,
                lambda _t=nxt_text, _i=nxt_img, _c=nxt_ctx, _s=nxt_source: self._start_request(
                    _t, image_b64=_i, vision_context=_c, source=_s, reason="queued user message",
                ),
            )

    def interrupt_assistant_output(self):
        self._tts_sentence_buf = ""
        while True:
            try:
                self._tts_queue.get_nowait()
            except queue.Empty:
                break
        self._assistant_audio_active = False
        self._pending_request_id = ""
        self._current_request_id = ""
        self._tts_session_interrupted = True
        with self._tts_worker_lock:
            self._tts_worker_active = False
        if self.voice_manager:
            try:
                self.voice_manager.stop_output()
            except Exception as e:
                logger.warning(f"Error stopping voice output: {e}")
        self.assistant_output_interrupted.emit()

    def _on_vision_toggled(self, active):
        self._vision_active = active
        self.vision_mode_changed.emit(bool(active))
        if active:
            self.vision_btn.setToolTip("Vision ON - every message includes a fresh frame")
            self.vision_btn.setStyleSheet(
                "background: rgba(0, 180, 220, 0.3); "
                "border: 2px solid rgba(0, 180, 220, 0.6); "
                "border-radius: 18px;"
            )
            self.input_field.setPlaceholderText(
                "Vision is live - ask about what the camera sees..."
            )
        else:
            self.vision_btn.setToolTip("Toggle live vision context for messages")
            self.vision_btn.setStyleSheet("")
            self.input_field.setPlaceholderText("Type a message...")

    def _on_mic_toggled(self, active):
        if active:
            self.mic_btn.setToolTip("Listening... (click to stop)")
            self.mic_btn.setStyleSheet(
                "background: rgba(220, 50, 80, 0.3); "
                "border: 2px solid rgba(220, 50, 80, 0.6); "
                "border-radius: 18px;"
            )
            self.chat_display.append(
                '<span style="color:#dc3250;font-size:9px;">'
                '[Listening... speak now]</span>'
            )
            if self.audio_service:
                self.audio_service.start_listening()
        else:
            self.mic_btn.setToolTip("Toggle Listening (click to start/stop)")
            self.mic_btn.setStyleSheet("")
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

    def _append_system_note(self, text):
        self.chat_display.append(
            '<span style="color:#9ca3af;font-size:9px;">'
            f'[{text}]</span>'
        )
