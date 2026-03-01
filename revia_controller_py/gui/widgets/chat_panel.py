import base64
import queue
import threading
from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit, QPushButton,
    QComboBox,
)
from PySide6.QtCore import Qt, QBuffer, QIODevice
from PySide6.QtGui import QFont, QPixmap


class ChatPanel(QFrame):
    def __init__(self, event_bus, client, audio_service=None, parent=None):
        super().__init__(parent)
        self.event_bus = event_bus
        self.client = client
        self.audio_service = audio_service
        self.camera_service = None
        self.voice_manager = None
        self.setObjectName("chatPanel")
        self._current_response = ""
        self._full_response = ""
        self._vision_active = False
        # Sentence-level streaming TTS
        self._tts_sentence_buf = ""
        self._tts_queue = queue.Queue()
        self._tts_worker_active = False

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
        self.vision_btn.setToolTip("Attach camera snapshot to next message")
        self.vision_btn.toggled.connect(self._on_vision_toggled)
        row_layout.addWidget(self.vision_btn)

        self.tts_combo = QComboBox()
        self.tts_combo.addItems(["Qwen3-TTS", "pyttsx3", "Off"])
        self.tts_combo.setFixedWidth(100)
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
        self.event_bus.chat_token.connect(self._on_token)
        self.event_bus.chat_complete.connect(self._on_complete)

        # Wire up audio service
        if self.audio_service:
            self.audio_service.speech_recognized.connect(
                self._on_speech_recognized
            )

    def set_camera_service(self, cam):
        self.camera_service = cam

    def set_voice_manager(self, vm):
        self.voice_manager = vm
        # Sync combo with backend engine
        if vm:
            eng = vm.backend.engine_name
            if "qwen" in eng.lower():
                self.tts_combo.setCurrentText("Qwen3-TTS")
            elif eng == "pyttsx3":
                self.tts_combo.setCurrentText("pyttsx3")

    def _on_tts_engine_changed(self, text):
        if text == "Off":
            return
        engine_id = "qwen3-tts" if text == "Qwen3-TTS" else "pyttsx3"
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

    def _send(self):
        text = self.input_field.text().strip()
        if not text:
            return

        image_b64 = None
        if self._vision_active:
            image_b64 = self._capture_frame_b64()
            if image_b64:
                self.chat_display.append(
                    '<span style="color:#00d4ff;font-weight:bold;">You:</span> '
                    f'{text} <span style="color:#888;font-size:9px;">'
                    '[+ camera snapshot]</span>'
                )
            else:
                self.chat_display.append(
                    '<span style="color:#00d4ff;font-weight:bold;">You:</span> '
                    f'{text} <span style="color:#cc3040;font-size:9px;">'
                    '[camera not available]</span>'
                )
            self.vision_btn.setChecked(False)
        else:
            self.chat_display.append(
                f'<span style="color:#00d4ff;font-weight:bold;">You:</span> {text}'
            )

        self.input_field.clear()
        self._current_response = ""
        self._full_response = ""
        self._tts_sentence_buf = ""
        # Drain any leftover TTS queue from previous response
        while not self._tts_queue.empty():
            try:
                self._tts_queue.get_nowait()
            except queue.Empty:
                break
        self.client.send_chat(text, image_b64=image_b64)

    def _on_token(self, token):
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

        # Sentence-streaming TTS: queue completed sentences immediately
        tts_mode = self.tts_combo.currentText()
        if tts_mode != "Off" and self.voice_manager:
            self._tts_sentence_buf += token
            self._flush_tts_sentences(end_of_response=False)

    def _flush_tts_sentences(self, end_of_response=False):
        """Split buffer on sentence boundaries and queue each complete sentence."""
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
                self._tts_queue.put(sentence)
                if not self._tts_worker_active:
                    self._tts_worker_active = True
                    threading.Thread(target=self._tts_worker, daemon=True).start()
        self._tts_sentence_buf = buf

    def _tts_worker(self):
        """Background thread: consumes sentence queue and speaks each one in order."""
        while True:
            try:
                sentence = self._tts_queue.get(timeout=3.0)
                self.voice_manager.speak_sync(sentence)
                self._tts_queue.task_done()
            except queue.Empty:
                break
        self._tts_worker_active = False

    def _on_complete(self, text):
        if not self._current_response:
            self.chat_display.append(
                f'<span style="color:#a78bfa;font-weight:bold;">Revia:</span> {text}'
            )
        self._current_response = ""
        self._full_response = ""
        self.chat_display.append("")

        tts_mode = self.tts_combo.currentText()
        if tts_mode != "Off":
            if self.voice_manager:
                # Flush any remaining partial sentence from the streaming buffer
                self._flush_tts_sentences(end_of_response=True)
                remaining = self._tts_sentence_buf.strip()
                self._tts_sentence_buf = ""
                if remaining and not remaining.startswith('['):
                    self._tts_queue.put(remaining)
                    if not self._tts_worker_active:
                        self._tts_worker_active = True
                        threading.Thread(target=self._tts_worker, daemon=True).start()
            elif self.audio_service:
                clean = (text or "").strip()
                if clean and not clean.startswith("["):
                    self.audio_service.speak(clean)

    def _on_vision_toggled(self, active):
        self._vision_active = active
        if active:
            self.vision_btn.setToolTip("Vision ON - next message includes snapshot")
            self.vision_btn.setStyleSheet(
                "background: rgba(0, 180, 220, 0.3); "
                "border: 2px solid rgba(0, 180, 220, 0.6); "
                "border-radius: 18px;"
            )
            self.input_field.setPlaceholderText(
                "Ask about what the camera sees..."
            )
        else:
            self.vision_btn.setToolTip("Attach camera snapshot to next message")
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
