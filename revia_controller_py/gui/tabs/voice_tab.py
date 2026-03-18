"""Voice Management tab for REVIA -- Qwen3-TTS integration with 3 generation modes."""
import sys
import re
import shutil
import threading
import importlib.util
from pathlib import Path
from PySide6.QtWidgets import (
    QScrollArea, QWidget, QVBoxLayout, QFormLayout, QHBoxLayout,
    QLabel, QComboBox, QGroupBox, QDoubleSpinBox, QCheckBox,
    QPushButton, QProgressBar, QListWidget, QTextEdit,
    QLineEdit, QTabWidget, QFileDialog, QInputDialog, QMessageBox,
)
from PySide6.QtCore import Qt, QTimer, QProcess
from PySide6.QtGui import QFont

from app.voice_profile import VoiceProfile, VoiceMode
from app.voice_manager import VoiceManager
from app.tts_backend import QWEN_SPEAKERS, QWEN_LANGUAGES, QWEN_MODEL_SIZES

QWEN_MODELS = {
    "Base 0.6B (Clone)": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "Base 1.7B (Clone)": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "CustomVoice 0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "CustomVoice 1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "VoiceDesign 1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
}

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


class VoiceTab(QScrollArea):
    def __init__(self, event_bus, client, audio_service=None, parent=None):
        super().__init__(parent)
        self.event_bus = event_bus
        self.client = client
        self.audio_service = audio_service
        self.voice_mgr = VoiceManager(self)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        header = QLabel("Voice Management")
        header.setObjectName("tabHeader")
        header.setFont(QFont("Segoe UI", 12, QFont.Bold))
        layout.addWidget(header)

        # ── Qwen3-TTS Server ──
        srv_group = QGroupBox("Qwen3-TTS Server")
        srv_group.setObjectName("settingsGroup")
        sv = QFormLayout(srv_group)
        sv.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self.qwen_url = QLineEdit("http://localhost:8000")
        self.qwen_url.textChanged.connect(
            lambda t: self.voice_mgr.backend.set_qwen_server(t)
        )
        sv.addRow("URL:", self.qwen_url)

        self.qwen_model_combo = QComboBox()
        self.qwen_model_combo.addItems(list(QWEN_MODELS.keys()))
        self.qwen_model_combo.setCurrentText("Base 0.6B (Clone)")
        sv.addRow("Model:", self.qwen_model_combo)

        btn_row = QHBoxLayout()
        self.start_tts_btn = QPushButton("Start")
        self.start_tts_btn.setObjectName("primaryBtn")
        self.start_tts_btn.clicked.connect(self._start_tts_server)
        btn_row.addWidget(self.start_tts_btn)
        self.stop_tts_btn = QPushButton("Stop")
        self.stop_tts_btn.setObjectName("secondaryBtn")
        self.stop_tts_btn.setEnabled(False)
        self.stop_tts_btn.clicked.connect(self._stop_tts_server)
        btn_row.addWidget(self.stop_tts_btn)
        self.engine_combo = QComboBox()
        self.engine_combo.addItems(["Qwen3-TTS", "pyttsx3"])
        self.engine_combo.setCurrentText("pyttsx3")
        self.engine_combo.currentTextChanged.connect(self._on_engine_changed)
        btn_row.addWidget(self.engine_combo)
        sv.addRow("", btn_row)

        self.tts_server_status = QLabel("Not running")
        self.tts_server_status.setFont(QFont("Consolas", 8))
        self.tts_server_status.setObjectName("metricLabel")
        self.tts_server_status.setWordWrap(True)
        sv.addRow("Server:", self.tts_server_status)

        self.voice_mgr.backend.set_engine("pyttsx3")
        self.voice_mgr.backend.set_qwen_server("http://localhost:8000")
        self._tts_process = None
        self._tts_ready_timer = None
        self._tts_last_lines: list[str] = []
        self._tts_launcher_tmp = None  # temp wrapper script for CPU-only builds
        layout.addWidget(srv_group)

        # ── 3 Mode Tabs (matching Qwen3-TTS demo) ──
        self.mode_tabs = QTabWidget()
        self.mode_tabs.addTab(self._build_design_tab(), "Voice Design")
        self.mode_tabs.addTab(self._build_clone_tab(), "Voice Clone (Base)")
        self.mode_tabs.addTab(self._build_custom_tab(), "TTS (CustomVoice)")
        layout.addWidget(self.mode_tabs)

        # ── Voice Library ──
        lib_group = QGroupBox("Voice Library")
        lib_group.setObjectName("settingsGroup")
        lib_layout = QVBoxLayout(lib_group)
        lib_layout.setSpacing(4)

        self.voice_list = QListWidget()
        self.voice_list.setMaximumHeight(110)
        self.voice_list.currentTextChanged.connect(self._on_voice_selected)
        lib_layout.addWidget(self.voice_list)

        lib_btns = QHBoxLayout()
        for text, slot in [
            ("Set Active", self._set_active), ("Rename", self._rename_voice),
            ("Delete", self._delete_voice), ("Set Default", self._set_default),
        ]:
            b = QPushButton(text)
            b.setObjectName("secondaryBtn")
            b.setFixedHeight(26)
            b.clicked.connect(slot)
            lib_btns.addWidget(b)
        lib_layout.addLayout(lib_btns)

        lib_btns2 = QHBoxLayout()
        for text, slot in [
            ("Import", self._import_voice), ("Export", self._export_voice),
            ("Open Folder", self._open_folder), ("Play Voice", self._play_selected),
        ]:
            b = QPushButton(text)
            b.setObjectName("secondaryBtn")
            b.setFixedHeight(26)
            b.clicked.connect(slot)
            lib_btns2.addWidget(b)
        lib_layout.addLayout(lib_btns2)

        self.lib_meta = QLabel("Select a voice to see details")
        self.lib_meta.setFont(QFont("Consolas", 8))
        self.lib_meta.setObjectName("metricLabel")
        self.lib_meta.setWordWrap(True)
        lib_layout.addWidget(self.lib_meta)
        layout.addWidget(lib_group)

        # ── STT Input ──
        stt_group = QGroupBox("Speech-to-Text (Input)")
        stt_group.setObjectName("settingsGroup")
        stf = QFormLayout(stt_group)
        self.input_device = QComboBox()
        self.input_device.addItem("Default Microphone")
        self._populate_input_devices()
        stf.addRow("Input Device:", self.input_device)
        self.ptt_mode = QComboBox()
        self.ptt_mode.addItems([
            "Toggle (click to start/stop)", "Push-to-Talk (hold)",
            "Always Listening (VAD)",
        ])
        stf.addRow("Activation:", self.ptt_mode)
        layout.addWidget(stt_group)

        # ── Mic Test ──
        mic_group = QGroupBox("Microphone Test")
        mic_group.setObjectName("settingsGroup")
        ml = QVBoxLayout(mic_group)
        self.vol_bar = QProgressBar()
        self.vol_bar.setRange(0, 100)
        self.vol_bar.setValue(0)
        self.vol_bar.setTextVisible(False)
        self.vol_bar.setFixedHeight(18)
        self.vol_bar.setStyleSheet(
            "QProgressBar{border:1px solid #555;border-radius:4px;background:#1a1a2e;}"
            "QProgressBar::chunk{background:qlineargradient("
            "x1:0,y1:0,x2:1,y2:0,stop:0 #00cc44,stop:0.7 #cccc00,stop:1 #cc3040);"
            "border-radius:3px;}"
        )
        ml.addWidget(self.vol_bar)
        mic_row = QHBoxLayout()
        self.mic_test_btn = QPushButton("Test Microphone")
        self.mic_test_btn.setObjectName("secondaryBtn")
        self.mic_test_btn.setCheckable(True)
        self.mic_test_btn.toggled.connect(self._toggle_mic_test)
        mic_row.addWidget(self.mic_test_btn)
        self.mic_level_label = QLabel("Level: --")
        self.mic_level_label.setFont(QFont("Consolas", 9))
        mic_row.addWidget(self.mic_level_label, stretch=1)
        ml.addLayout(mic_row)
        layout.addWidget(mic_group)

        # ── Latency Metrics ──
        lat_group = QGroupBox("TTS Latency")
        lat_group.setObjectName("settingsGroup")
        ll = QHBoxLayout(lat_group)
        self.synth_lbl = QLabel("Synth: --")
        self.synth_lbl.setFont(QFont("Consolas", 9))
        self.synth_lbl.setObjectName("metricLabel")
        ll.addWidget(self.synth_lbl)
        self.dur_lbl = QLabel("Duration: --")
        self.dur_lbl.setFont(QFont("Consolas", 9))
        self.dur_lbl.setObjectName("metricLabel")
        ll.addWidget(self.dur_lbl)
        self.rtf_lbl = QLabel("RTF: --")
        self.rtf_lbl.setFont(QFont("Consolas", 9))
        self.rtf_lbl.setObjectName("metricLabel")
        ll.addWidget(self.rtf_lbl)
        layout.addWidget(lat_group)

        # ── Status ──
        self.status_label = QLabel("Status: Ready")
        self.status_label.setFont(QFont("Consolas", 8))
        self.status_label.setObjectName("metricLabel")
        layout.addWidget(self.status_label)

        layout.addStretch()
        self.setWidget(container)

        # Wiring
        self.voice_mgr.metrics_updated.connect(self._on_metrics)
        self.voice_mgr.error.connect(
            lambda e: self._set_status(f"Error: {e}", "#cc3040")
        )
        self.event_bus.connection_changed.connect(self._on_core_connection)
        self._refresh_library()

    # ══════════════════════════════════════════════════════════════
    # Voice Design Tab
    # ══════════════════════════════════════════════════════════════
    def _build_design_tab(self):
        w = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(8, 8, 8, 8)
        vl.setSpacing(6)

        title = QLabel("Create Custom Voice with Natural Language")
        title.setFont(QFont("Segoe UI", 10, QFont.Bold))
        vl.addWidget(title)

        fl = QFormLayout()

        self.design_text = QTextEdit()
        self.design_text.setMaximumHeight(70)
        self.design_text.setPlaceholderText(
            "Hello! Welcome to Text-to-Speech system. "
            "This is a demo of our TTS capabilities."
        )
        fl.addRow("Text to Synthesize:", self.design_text)

        self.design_lang = QComboBox()
        self.design_lang.addItems(QWEN_LANGUAGES)
        fl.addRow("Language:", self.design_lang)

        self.design_desc = QTextEdit()
        self.design_desc.setMaximumHeight(60)
        self.design_desc.setPlaceholderText(
            "e.g. Speak in an incredulous tone, but with a hint of "
            "panic beginning to creep into your voice."
        )
        fl.addRow("Voice Description:", self.design_desc)

        vl.addLayout(fl)

        btn_row = QHBoxLayout()
        gen_btn = QPushButton("Generate with Custom Voice")
        gen_btn.setObjectName("primaryBtn")
        gen_btn.clicked.connect(self._run_design)
        btn_row.addWidget(gen_btn)
        play_btn = QPushButton("Play")
        play_btn.setObjectName("secondaryBtn")
        play_btn.clicked.connect(lambda: self._play_last("design"))
        btn_row.addWidget(play_btn)
        save_btn = QPushButton("Save to Library")
        save_btn.setObjectName("secondaryBtn")
        save_btn.clicked.connect(self._save_design)
        btn_row.addWidget(save_btn)
        vl.addLayout(btn_row)

        self.design_status = QLabel("")
        self.design_status.setFont(QFont("Consolas", 8))
        self.design_status.setObjectName("metricLabel")
        self.design_status.setWordWrap(True)
        vl.addWidget(self.design_status)

        vl.addStretch()
        self._last_design_wav = None
        return w

    # ══════════════════════════════════════════════════════════════
    # Voice Clone (Base) Tab
    # ══════════════════════════════════════════════════════════════
    def _build_clone_tab(self):
        w = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(8, 8, 8, 8)
        vl.setSpacing(6)

        title = QLabel("Clone Voice from Reference Audio")
        title.setFont(QFont("Segoe UI", 10, QFont.Bold))
        vl.addWidget(title)

        # Reference audio
        ref_row = QHBoxLayout()
        self.clone_ref_path = QLineEdit()
        self.clone_ref_path.setPlaceholderText("Path to reference audio (.wav)...")
        ref_row.addWidget(self.clone_ref_path, stretch=1)
        browse_btn = QPushButton("Browse")
        browse_btn.setObjectName("browseBtn")
        browse_btn.clicked.connect(self._browse_ref_audio)
        ref_row.addWidget(browse_btn)
        rec_btn = QPushButton("Record 5s")
        rec_btn.setObjectName("secondaryBtn")
        rec_btn.clicked.connect(self._record_ref_audio)
        ref_row.addWidget(rec_btn)
        vl.addLayout(ref_row)

        fl = QFormLayout()

        self.clone_ref_text = QTextEdit()
        self.clone_ref_text.setMaximumHeight(50)
        self.clone_ref_text.setPlaceholderText(
            "Enter the exact text spoken in the reference audio..."
        )
        fl.addRow("Reference Text:", self.clone_ref_text)

        self.clone_target_text = QTextEdit()
        self.clone_target_text.setMaximumHeight(60)
        self.clone_target_text.setPlaceholderText(
            "Enter the text you want the cloned voice to speak..."
        )
        fl.addRow("Target Text:", self.clone_target_text)

        row = QHBoxLayout()
        self.clone_lang = QComboBox()
        self.clone_lang.addItems(QWEN_LANGUAGES)
        row.addWidget(QLabel("Language:"))
        row.addWidget(self.clone_lang)
        self.clone_model_size = QComboBox()
        self.clone_model_size.addItems(QWEN_MODEL_SIZES)
        self.clone_model_size.setCurrentText("1.7B")
        row.addWidget(QLabel("Model Size:"))
        row.addWidget(self.clone_model_size)
        fl.addRow(row)

        self.clone_xvec = QCheckBox(
            "Use x-vector only (No reference text needed, but lower quality)"
        )
        fl.addRow("", self.clone_xvec)

        vl.addLayout(fl)

        btn_row = QHBoxLayout()
        gen_btn = QPushButton("Clone & Generate")
        gen_btn.setObjectName("primaryBtn")
        gen_btn.clicked.connect(self._run_clone)
        btn_row.addWidget(gen_btn)
        play_btn = QPushButton("Play")
        play_btn.setObjectName("secondaryBtn")
        play_btn.clicked.connect(lambda: self._play_last("clone"))
        btn_row.addWidget(play_btn)
        save_btn = QPushButton("Save to Library")
        save_btn.setObjectName("secondaryBtn")
        save_btn.clicked.connect(self._save_clone)
        btn_row.addWidget(save_btn)
        vl.addLayout(btn_row)

        self.clone_status = QLabel("")
        self.clone_status.setFont(QFont("Consolas", 8))
        self.clone_status.setObjectName("metricLabel")
        self.clone_status.setWordWrap(True)
        vl.addWidget(self.clone_status)

        vl.addStretch()
        self._last_clone_wav = None
        return w

    # ══════════════════════════════════════════════════════════════
    # TTS (CustomVoice) Tab
    # ══════════════════════════════════════════════════════════════
    def _build_custom_tab(self):
        w = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(8, 8, 8, 8)
        vl.setSpacing(6)

        title = QLabel("Text-to-Speech with Predefined Speakers")
        title.setFont(QFont("Segoe UI", 10, QFont.Bold))
        vl.addWidget(title)

        fl = QFormLayout()

        self.custom_text = QTextEdit()
        self.custom_text.setMaximumHeight(70)
        self.custom_text.setPlaceholderText(
            "Hello! Welcome to Text-to-Speech system. "
            "This is a demo of our TTS capabilities."
        )
        fl.addRow("Text to Synthesize:", self.custom_text)

        row1 = QHBoxLayout()
        self.custom_lang = QComboBox()
        self.custom_lang.addItems(QWEN_LANGUAGES)
        row1.addWidget(QLabel("Language:"))
        row1.addWidget(self.custom_lang)
        self.custom_speaker = QComboBox()
        self.custom_speaker.addItems(QWEN_SPEAKERS)
        self.custom_speaker.setCurrentText("Ryan")
        row1.addWidget(QLabel("Speaker:"))
        row1.addWidget(self.custom_speaker)
        fl.addRow(row1)

        self.custom_style = QTextEdit()
        self.custom_style.setMaximumHeight(50)
        self.custom_style.setPlaceholderText(
            "e.g. Speak in a cheerful and energetic tone..."
        )
        fl.addRow("Style Instruction\n(Optional):", self.custom_style)

        self.custom_model_size = QComboBox()
        self.custom_model_size.addItems(QWEN_MODEL_SIZES)
        self.custom_model_size.setCurrentText("1.7B")
        fl.addRow("Model Size:", self.custom_model_size)

        vl.addLayout(fl)

        btn_row = QHBoxLayout()
        gen_btn = QPushButton("Generate Speech")
        gen_btn.setObjectName("primaryBtn")
        gen_btn.clicked.connect(self._run_custom)
        btn_row.addWidget(gen_btn)
        play_btn = QPushButton("Play")
        play_btn.setObjectName("secondaryBtn")
        play_btn.clicked.connect(lambda: self._play_last("custom"))
        btn_row.addWidget(play_btn)
        save_btn = QPushButton("Save to Library")
        save_btn.setObjectName("secondaryBtn")
        save_btn.clicked.connect(self._save_custom)
        btn_row.addWidget(save_btn)
        vl.addLayout(btn_row)

        self.custom_status = QLabel("")
        self.custom_status.setFont(QFont("Consolas", 8))
        self.custom_status.setObjectName("metricLabel")
        self.custom_status.setWordWrap(True)
        vl.addWidget(self.custom_status)

        vl.addStretch()
        self._last_custom_wav = None
        return w

    # ══════════════════════════════════════════════════════════════
    # Generation actions (threaded)
    # ══════════════════════════════════════════════════════════════

    def _run_design(self):
        text = self.design_text.toPlainText().strip()
        desc = self.design_desc.toPlainText().strip()
        lang = self.design_lang.currentText()
        if not text:
            self.design_status.setText("Enter text to synthesize")
            return
        if not desc:
            self.design_status.setText("Enter a voice description")
            return
        self.design_status.setText("Generating...")
        self.design_status.setStyleSheet("color: #ccaa00;")

        def _do():
            wav, metrics = self.voice_mgr.generate_design(text, desc, lang)
            QTimer.singleShot(0, lambda: self._on_design_done(wav, metrics))
        threading.Thread(target=_do, daemon=True).start()

    def _on_design_done(self, wav, metrics):
        if wav:
            self._last_design_wav = wav
            self.design_status.setText(f"Generated: {wav}")
            self.design_status.setStyleSheet("color: #00aa40;")
            self.voice_mgr.play_wav(wav)
        else:
            self.design_status.setText(f"Failed: {metrics}")
            self.design_status.setStyleSheet("color: #cc3040;")

    def _run_clone(self):
        ref = self.clone_ref_path.text().strip()
        ref_text = self.clone_ref_text.toPlainText().strip()
        target = self.clone_target_text.toPlainText().strip()
        lang = self.clone_lang.currentText()
        xvec = self.clone_xvec.isChecked()
        if not ref:
            self.clone_status.setText("Select reference audio first")
            return
        if not target:
            self.clone_status.setText("Enter target text")
            return
        if not xvec and not ref_text:
            self.clone_status.setText(
                "Enter reference text or enable x-vector only"
            )
            return
        self.clone_status.setText("Cloning...")
        self.clone_status.setStyleSheet("color: #ccaa00;")

        def _do():
            wav, metrics = self.voice_mgr.generate_clone(
                target, ref, ref_text, lang, xvec
            )
            QTimer.singleShot(0, lambda: self._on_clone_done(wav, metrics))
        threading.Thread(target=_do, daemon=True).start()

    def _on_clone_done(self, wav, metrics):
        if wav:
            self._last_clone_wav = wav
            self.clone_status.setText(f"Generated: {wav}")
            self.clone_status.setStyleSheet("color: #00aa40;")
            self.voice_mgr.play_wav(wav)
        else:
            self.clone_status.setText(f"Failed: {metrics}")
            self.clone_status.setStyleSheet("color: #cc3040;")

    def _run_custom(self):
        text = self.custom_text.toPlainText().strip()
        lang = self.custom_lang.currentText()
        speaker = self.custom_speaker.currentText()
        style = self.custom_style.toPlainText().strip()
        model_size = self.custom_model_size.currentText()
        if not text:
            self.custom_status.setText("Enter text to synthesize")
            return
        self.custom_status.setText("Generating...")
        self.custom_status.setStyleSheet("color: #ccaa00;")

        def _do():
            wav, metrics = self.voice_mgr.generate_custom(
                text, lang, speaker, style, model_size
            )
            QTimer.singleShot(0, lambda: self._on_custom_done(wav, metrics))
        threading.Thread(target=_do, daemon=True).start()

    def _on_custom_done(self, wav, metrics):
        if wav:
            self._last_custom_wav = wav
            self.custom_status.setText(f"Generated: {wav}")
            self.custom_status.setStyleSheet("color: #00aa40;")
            self.voice_mgr.play_wav(wav)
        else:
            self.custom_status.setText(f"Failed: {metrics}")
            self.custom_status.setStyleSheet("color: #cc3040;")

    def _play_last(self, mode):
        wav = getattr(self, f"_last_{mode}_wav", None)
        if wav and Path(wav).exists():
            self.voice_mgr.play_wav(wav)

    # ══════════════════════════════════════════════════════════════
    # Save to Library
    # ══════════════════════════════════════════════════════════════

    def _save_design(self):
        wav = self._last_design_wav
        if not wav or not Path(wav).exists():
            self.design_status.setText("Generate a voice first")
            return
        name, ok = QInputDialog.getText(self, "Save Voice", "Voice name:")
        if not ok or not name.strip():
            return
        p = self.voice_mgr.create_design_profile(
            name.strip(),
            self.design_desc.toPlainText().strip(),
            self.design_lang.currentText(),
        )
        voice_dir = self.voice_mgr.save_voice(p)
        dest = voice_dir / "generated.wav"
        shutil.copy2(wav, str(dest))
        p.generated_wav = str(dest)
        p.save(voice_dir)
        self._refresh_library()
        self.design_status.setText(f"Saved: {name.strip()}")

    def _save_clone(self):
        wav = self._last_clone_wav
        if not wav or not Path(wav).exists():
            self.clone_status.setText("Generate a clone first")
            return
        name, ok = QInputDialog.getText(self, "Save Voice", "Voice name:")
        if not ok or not name.strip():
            return
        ref = self.clone_ref_path.text().strip()
        p = self.voice_mgr.create_clone_profile(
            name.strip(), ref,
            self.clone_ref_text.toPlainText().strip(),
            self.clone_lang.currentText(),
            self.clone_xvec.isChecked(),
        )
        voice_dir = self.voice_mgr.save_voice(p)
        dest = voice_dir / "generated.wav"
        shutil.copy2(wav, str(dest))
        p.generated_wav = str(dest)
        # Also copy reference audio
        if ref and Path(ref).exists():
            ref_dest = voice_dir / "ref.wav"
            shutil.copy2(ref, str(ref_dest))
            p.reference_audio = str(ref_dest)
        p.save(voice_dir)
        self._refresh_library()
        self.clone_status.setText(f"Saved: {name.strip()}")

    def _save_custom(self):
        wav = self._last_custom_wav
        if not wav or not Path(wav).exists():
            self.custom_status.setText("Generate speech first")
            return
        name, ok = QInputDialog.getText(self, "Save Voice", "Voice name:")
        if not ok or not name.strip():
            return
        p = self.voice_mgr.create_custom_profile(
            name.strip(),
            self.custom_speaker.currentText(),
            self.custom_style.toPlainText().strip(),
            self.custom_lang.currentText(),
            self.custom_model_size.currentText(),
        )
        voice_dir = self.voice_mgr.save_voice(p)
        dest = voice_dir / "generated.wav"
        shutil.copy2(wav, str(dest))
        p.generated_wav = str(dest)
        p.save(voice_dir)
        self._refresh_library()
        self.custom_status.setText(f"Saved: {name.strip()}")

    # ══════════════════════════════════════════════════════════════
    # Voice Library actions
    # ══════════════════════════════════════════════════════════════

    def _refresh_library(self):
        self.voice_mgr.library.load_all()
        self.voice_list.clear()
        for name in self.voice_mgr.library.list_names():
            p = self.voice_mgr.library.get(name)
            suffix = " *" if p and p.is_default else ""
            wav_tag = " [WAV]" if p and p.has_wav() else ""
            self.voice_list.addItem(f"{name}{suffix}{wav_tag}")
        if self.voice_list.count() == 0:
            self.lib_meta.setText("No voices saved yet. Generate and save one above.")

    def _on_voice_selected(self, text):
        name = text.replace(" *", "").replace(" [WAV]", "").strip()
        p = self.voice_mgr.library.get(name)
        if p:
            meta = (
                f"Mode: {p.mode.value} | Lang: {p.language} | "
                f"Created: {p.created[:10]}"
            )
            if p.mode == VoiceMode.CUSTOM:
                meta += f" | Speaker: {p.speaker_id}"
            if p.has_wav():
                meta += f"\nWAV: {p.generated_wav}"
            else:
                meta += "\n(No generated WAV)"
            self.lib_meta.setText(meta)

    def _set_active(self):
        cur = self.voice_list.currentItem()
        if not cur:
            return
        name = cur.text().replace(" *", "").replace(" [WAV]", "").strip()
        self.voice_mgr.set_active_voice(name)
        self._set_status(f"Active voice: {name}", "#00aa40")

    def _rename_voice(self):
        cur = self.voice_list.currentItem()
        if not cur:
            return
        old = cur.text().replace(" *", "").replace(" [WAV]", "").strip()
        new, ok = QInputDialog.getText(self, "Rename", "New name:", text=old)
        if ok and new.strip():
            self.voice_mgr.library.rename(old, new.strip())
            self._refresh_library()

    def _delete_voice(self):
        cur = self.voice_list.currentItem()
        if not cur:
            return
        name = cur.text().replace(" *", "").replace(" [WAV]", "").strip()
        reply = QMessageBox.question(
            self, "Delete Voice", f"Delete '{name}'?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.voice_mgr.library.delete(name)
            self._refresh_library()

    def _set_default(self):
        cur = self.voice_list.currentItem()
        if not cur:
            return
        name = cur.text().replace(" *", "").replace(" [WAV]", "").strip()
        self.voice_mgr.library.set_default(name)
        self._refresh_library()
        self._set_status(f"Default voice: {name}", "#00aa40")

    def _import_voice(self):
        d = QFileDialog.getExistingDirectory(self, "Import Voice Directory")
        if d:
            p = self.voice_mgr.library.import_profile(d)
            if p:
                self._refresh_library()
                self._set_status(f"Imported: {p.name}", "#00aa40")

    def _export_voice(self):
        cur = self.voice_list.currentItem()
        if not cur:
            return
        name = cur.text().replace(" *", "").replace(" [WAV]", "").strip()
        d = QFileDialog.getExistingDirectory(self, "Export To")
        if d:
            dest = Path(d) / name
            self.voice_mgr.library.export_profile(name, dest)
            self._set_status(f"Exported to: {dest}", "#00aa40")

    def _open_folder(self):
        self.voice_mgr.library.open_folder()

    def _play_selected(self):
        cur = self.voice_list.currentItem()
        if not cur:
            return
        name = cur.text().replace(" *", "").replace(" [WAV]", "").strip()
        p = self.voice_mgr.library.get(name)
        if p and p.has_wav():
            self.voice_mgr.play_wav(p.generated_wav)
        else:
            self._set_status("No WAV file for this voice", "#cc3040")

    # ══════════════════════════════════════════════════════════════
    # Clone tab helpers
    # ══════════════════════════════════════════════════════════════

    def _browse_ref_audio(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Reference Audio", "",
            "Audio (*.wav *.mp3 *.flac *.ogg);;All Files (*)",
        )
        if path:
            self.clone_ref_path.setText(path)

    def _record_ref_audio(self):
        self.clone_status.setText("Recording 5 seconds...")
        self.clone_status.setStyleSheet("color: #dc3250;")

        def _do():
            try:
                import sounddevice as sd
                import numpy as np
                import wave
                sr = 22050
                audio = sd.rec(int(sr * 5), samplerate=sr, channels=1, dtype="int16")
                sd.wait()
                out = Path(__file__).resolve().parents[2] / "voices" / "_temp_ref.wav"
                out.parent.mkdir(parents=True, exist_ok=True)
                with wave.open(str(out), "w") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sr)
                    wf.writeframes(audio.tobytes())
                QTimer.singleShot(0, lambda: self._on_rec_done(str(out)))
            except Exception as e:
                QTimer.singleShot(0, lambda: self._on_rec_done(None, str(e)))
        threading.Thread(target=_do, daemon=True).start()

    def _on_rec_done(self, path, error=None):
        if error:
            self.clone_status.setText(f"Recording failed: {error}")
            self.clone_status.setStyleSheet("color: #cc3040;")
        elif path:
            self.clone_ref_path.setText(path)
            self.clone_status.setText("Recording saved")
            self.clone_status.setStyleSheet("color: #00aa40;")

    # ══════════════════════════════════════════════════════════════
    # Engine / Devices / Metrics
    # ══════════════════════════════════════════════════════════════

    def _on_engine_changed(self, text):
        if "pyttsx3" in text.lower():
            self.voice_mgr.backend.set_engine("pyttsx3")
        else:
            self.voice_mgr.backend.set_engine("qwen3-tts")

    def _populate_input_devices(self):
        try:
            import sounddevice as sd
            for i, d in enumerate(sd.query_devices()):
                if d["max_input_channels"] > 0:
                    self.input_device.addItem(f"{d['name']} (#{i})", userData=i)
        except Exception:
            pass

    def _toggle_mic_test(self, active):
        if not self.audio_service:
            self.mic_level_label.setText("No audio service")
            self.mic_test_btn.setChecked(False)
            return
        if active:
            self.audio_service.volume_level.connect(self._on_volume)
            self.audio_service._start_volume_monitor()
            self.mic_test_btn.setText("Stop Test")
        else:
            self.audio_service._stop_volume_monitor()
            try:
                self.audio_service.volume_level.disconnect(self._on_volume)
            except Exception:
                pass
            self.vol_bar.setValue(0)
            self.mic_level_label.setText("Level: --")
            self.mic_test_btn.setText("Test Microphone")

    def _on_volume(self, level):
        pct = int(level * 100)
        self.vol_bar.setValue(pct)
        self.mic_level_label.setText(f"Level: {pct}%")

    def _on_metrics(self, metrics):
        self.synth_lbl.setText(f"Synth: {metrics.synthesis_time}s")
        self.dur_lbl.setText(f"Duration: {metrics.audio_duration}s")
        self.rtf_lbl.setText(f"RTF: {metrics.realtime_factor}x")

    def _set_status(self, text, color="#808898"):
        self.status_label.setText(f"Status: {text}")
        self.status_label.setStyleSheet(f"color: {color};")

    def _on_core_connection(self, connected):
        if connected:
            self._set_status("Core online", "#00aa40")

    # ══════════════════════════════════════════════════════════════
    # Qwen3-TTS Local Server Management
    # ══════════════════════════════════════════════════════════════

    def _start_tts_server(self):
        if self._tts_process and self._tts_process.state() != QProcess.NotRunning:
            self.tts_server_status.setText("Already running")
            return

        model_key = self.qwen_model_combo.currentText()
        model_id = QWEN_MODELS.get(model_key, "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
        port = "8000"

        self._kill_port(int(port))

        # Resolve to the project venv Python when available so the TTS
        # subprocess always has the correct torch/CUDA packages regardless
        # of which interpreter launched the GUI.
        exe = self._resolve_python_exe()

        # Diagnostic: confirm which interpreter and torch will be used.
        print(f"[TTS-SRV] GUI Python    : {sys.executable}")
        print(f"[TTS-SRV] Subprocess exe: {exe}")
        try:
            import torch as _t
            print(f"[TTS-SRV] GUI torch     : {_t.__version__}")
            print(f"[TTS-SRV] GUI CUDA built: {_t.version.cuda}")
            try:
                print(f"[TTS-SRV] GUI CUDA avail: {_t.cuda.is_available()}")
            except Exception as _ce:
                print(f"[TTS-SRV] GUI CUDA avail: ERROR – {_ce}")
        except Exception as _te:
            print(f"[TTS-SRV] GUI torch check failed: {_te}")

        self._tts_process = QProcess(self)
        self._tts_process.setProcessChannelMode(QProcess.MergedChannels)
        self._tts_process.readyReadStandardOutput.connect(self._on_tts_output)
        self._tts_process.finished.connect(self._on_tts_finished)

        from PySide6.QtCore import QProcessEnvironment
        env = QProcessEnvironment.systemEnvironment()
        sox_dir = str(Path.home() / "sox" / "sox-14.4.2")
        env.insert("PATH", sox_dir + ";" + env.value("PATH", ""))

        qwen_module = self._resolve_qwen_module()
        if not qwen_module:
            self.tts_server_status.setStyleSheet("color: #cc3333;")
            self.tts_server_status.setText(
                "Qwen3-TTS module not found. Install package, then retry."
            )
            self.start_tts_btn.setEnabled(True)
            self.stop_tts_btn.setEnabled(False)
            return

        args, device_label = self._build_tts_server_args(qwen_module, model_id, port)

        # If using CPU, hide all GPUs from the subprocess so PyTorch/CUDA
        # can't accidentally try to initialise a CUDA device and crash.
        if device_label == "CPU":
            env.insert("CUDA_VISIBLE_DEVICES", "")

        self._tts_process.setProcessEnvironment(env)
        self.tts_server_status.setText(f"Loading {model_key} ({device_label})...")
        self.tts_server_status.setStyleSheet("color: #ccaa00;")
        self.start_tts_btn.setEnabled(False)
        self.stop_tts_btn.setEnabled(True)

        print(f"[TTS-SRV] Qwen module   : {qwen_module}")
        print(f"[TTS-SRV] Starting: {exe} {' '.join(args)}")
        self._tts_process.start(exe, args)

        url = f"http://localhost:{port}"
        self.qwen_url.setText(url)
        self.voice_mgr.backend.set_qwen_server(url)

        # Poll every 5s to check if server is ready
        self._tts_poll_count = 0
        self._tts_ready_timer = QTimer(self)
        self._tts_ready_timer.timeout.connect(self._poll_tts_ready)
        self._tts_ready_timer.start(5000)

    @staticmethod
    def _resolve_python_exe() -> str:
        """Return the best available Python executable for the TTS subprocess.

        Preference order:
        1. <project_root>/.venv/Scripts/python.exe  (Windows venv)
        2. <project_root>/.venv/bin/python           (Linux/macOS venv)
        3. sys.executable                             (whatever launched the GUI)

        Using the project venv explicitly makes the TTS server immune to the
        interpreter used by the IDE or the OS-level 'python' on PATH.
        """
        # voice_tab.py lives at <root>/revia_controller_py/gui/tabs/voice_tab.py
        project_root = Path(__file__).resolve().parents[3]
        candidates = [
            project_root / ".venv" / "Scripts" / "python.exe",  # Windows
            project_root / ".venv" / "bin" / "python",           # Linux/macOS
        ]
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
        return sys.executable

    def _resolve_qwen_module(self):
        """Resolve the Qwen TTS demo module across known package layouts."""
        candidates = [
            "qwen_tts.cli.demo",
            "qwen_tts_cli.demo",
        ]
        for mod in candidates:
            try:
                if importlib.util.find_spec(mod) is not None:
                    return mod
            except (ModuleNotFoundError, ValueError):
                continue
        return ""

    def _build_tts_server_args(self, qwen_module: str, model_id: str, port: str):
        """Build launch args and prefer CUDA when available."""
        device, is_cpu_only = self._detect_tts_device()
        model_args = [model_id, "--port", port, "--ip", "0.0.0.0", "--no-flash-attn"]

        if is_cpu_only:
            # PyTorch is not compiled with CUDA.  Any call to torch.cuda.*
            # raises AssertionError unconditionally — CUDA_VISIBLE_DEVICES
            # cannot help here.  Run qwen_tts via a small wrapper script
            # that monkey-patches torch.cuda.is_available() to return False
            # safely before the module is imported.
            launcher = self._create_tts_launcher(qwen_module)
            self._tts_launcher_tmp = launcher
            args = [launcher] + model_args
        else:
            self._tts_launcher_tmp = None
            args = ["-m", qwen_module] + model_args
            if self._qwen_cli_supports_flag(qwen_module, "--device"):
                args.extend(["--device", device])

        return args, device.upper()

    def _create_tts_launcher(self, qwen_module: str) -> str:
        """Write a temp launcher that patches torch.cuda before running qwen_tts.

        On CPU-only PyTorch builds torch.cuda.is_available() raises
        AssertionError instead of returning False.  This wrapper intercepts
        that error so qwen_tts falls back to CPU without crashing.

        Returns the path to the temp script (caller is responsible for
        deleting it via self._tts_launcher_tmp when the server stops).
        """
        import tempfile
        code = (
            "import sys\n"
            "try:\n"
            "    import torch as _t\n"
            "    _orig = _t.cuda.is_available\n"
            "    def _safe():\n"
            "        try:\n"
            "            return _orig()\n"
            "        except (AssertionError, RuntimeError):\n"
            "            return False\n"
            "    _t.cuda.is_available = _safe\n"
            "except Exception:\n"
            "    pass\n"
            "import runpy\n"
            f"runpy.run_module({qwen_module!r}, run_name='__main__', alter_sys=True)\n"
        )
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix="_tts_launch.py", delete=False, prefix="revia_"
        )
        tmp.write(code)
        tmp.close()
        return tmp.name

    def _detect_tts_device(self):
        """Auto-select CUDA when available, otherwise force CPU.

        Returns:
            (device_str, is_cpu_only_build) where is_cpu_only_build=True
            means PyTorch was not compiled with CUDA at all (as opposed to
            simply having no GPU attached).  The distinction matters: for a
            CPU-only build, torch.cuda.* raises AssertionError rather than
            returning a safe False, so the subprocess needs special handling.
        """
        try:
            import torch
            # torch.cuda.is_available() raises AssertionError on CPU-only
            # torch builds ("Torch not compiled with CUDA enabled"), so
            # wrap it separately to avoid that surfacing to the user.
            try:
                if torch.cuda.is_available():
                    return "cuda", False
                return "cpu", False   # CUDA compiled but no GPU attached
            except (AssertionError, RuntimeError):
                return "cpu", True    # CUDA not compiled into this torch
        except Exception:
            pass
        return "cpu", False

    def _qwen_cli_supports_flag(self, qwen_module: str, flag: str) -> bool:
        """Check whether qwen_tts demo CLI accepts a given option."""
        try:
            import subprocess
            result = subprocess.run(
                [self._resolve_python_exe(), "-m", qwen_module, "--help"],
                capture_output=True,
                text=True,
                timeout=8,
            )
            return flag in ((result.stdout or "") + (result.stderr or ""))
        except Exception:
            return False

    def _poll_tts_ready(self):
        """Check if Gradio server is actually accepting connections."""
        self._tts_poll_count += 1
        port = 8000
        try:
            import requests
            r = requests.get(f"http://localhost:{port}/gradio_api/info", timeout=2)
            if r.ok:
                self.tts_server_status.setText(f"Running on :{port}")
                self.tts_server_status.setStyleSheet("color: #00aa40;")
                self._tts_ready_timer.stop()
                return
        except Exception:
            pass
        # Update status with progress dots
        dots = "." * (self._tts_poll_count % 4)
        model_key = self.qwen_model_combo.currentText()
        self.tts_server_status.setText(f"Loading {model_key}{dots}")

    def _stop_tts_server(self):
        if self._tts_ready_timer:
            self._tts_ready_timer.stop()
        if self._tts_process and self._tts_process.state() != QProcess.NotRunning:
            self._tts_process.kill()
            self._tts_process.waitForFinished(3000)
        self._kill_port(8000)
        # Clean up any temp launcher script written for CPU-only builds.
        if self._tts_launcher_tmp:
            try:
                import os as _os
                _os.unlink(self._tts_launcher_tmp)
            except OSError:
                pass
            self._tts_launcher_tmp = None
        self.tts_server_status.setText("Stopped")
        self.tts_server_status.setStyleSheet("")
        self.start_tts_btn.setEnabled(True)
        self.stop_tts_btn.setEnabled(False)

    def _on_tts_output(self):
        if not self._tts_process:
            return
        data = self._tts_process.readAllStandardOutput().data().decode(
            "utf-8", errors="replace"
        )
        for line in data.strip().split("\n"):
            line = self._clean_tts_log_line(line)
            if not line:
                continue
            print(f"[TTS-SRV] {line}")
            self._append_tts_log_line(line)
            if "Running on" in line:
                self.tts_server_status.setText("Running on :8000")
                self.tts_server_status.setStyleSheet("color: #00aa40;")
                if self._tts_ready_timer:
                    self._tts_ready_timer.stop()

    def _on_tts_finished(self, exit_code, exit_status):
        # Drain any remaining buffered output before updating the UI
        if self._tts_process:
            remaining = self._tts_process.readAllStandardOutput().data().decode(
                "utf-8", errors="replace"
            )
            for line in remaining.strip().split("\n"):
                line = self._clean_tts_log_line(line)
                if line:
                    print(f"[TTS-SRV] {line}")
                    self._append_tts_log_line(line)

        print(f"[TTS-SRV] Exited: code={exit_code}")
        if self._tts_ready_timer:
            self._tts_ready_timer.stop()
        self.start_tts_btn.setEnabled(True)
        self.stop_tts_btn.setEnabled(False)

        if exit_code == 0:
            self.tts_server_status.setText("Stopped")
            self.tts_server_status.setStyleSheet("")
        else:
            # Show the last error line in the status label so the user can
            # diagnose the crash without opening the console.
            error_hint = ""
            for line in reversed(self._tts_last_lines):
                if line:
                    error_hint = line[:120]
                    break
            self.tts_server_status.setStyleSheet("color: #cc3333;")
            if error_hint:
                self.tts_server_status.setText(
                    f"Exited ({exit_code}): {error_hint}"
                )
            else:
                self.tts_server_status.setText(
                    f"Exited ({exit_code}) — check console for details"
                )

    def _clean_tts_log_line(self, line: str) -> str:
        """Normalize process output for UI display by removing ANSI escapes."""
        return ANSI_ESCAPE_RE.sub("", line).strip()

    def _append_tts_log_line(self, line: str):
        """Store meaningful process output lines for crash hints."""
        self._tts_last_lines.append(line)
        if len(self._tts_last_lines) > 20:
            self._tts_last_lines.pop(0)

    def _kill_port(self, port):
        try:
            import subprocess
            r = subprocess.run(
                ["netstat", "-ano"], capture_output=True, text=True
            )
            for line in r.stdout.split("\n"):
                if f":{port}" in line and "LISTENING" in line:
                    pid = line.split()[-1]
                    if pid.isdigit() and int(pid) > 0:
                        subprocess.run(
                            ["taskkill", "/F", "/PID", pid],
                            capture_output=True,
                        )
        except Exception:
            pass
