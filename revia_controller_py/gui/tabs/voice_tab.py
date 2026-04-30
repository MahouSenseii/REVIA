"""Voice Management tab for REVIA -- Qwen3-TTS integration with 3 generation modes."""
import sys
import re
import json
import os
import shutil
import threading
import logging
import importlib.util
from pathlib import Path
from PySide6.QtWidgets import (
    QScrollArea, QWidget, QVBoxLayout, QFormLayout, QHBoxLayout,
    QLabel, QComboBox, QCheckBox,
    QPushButton, QProgressBar, QListWidget, QTextEdit,
    QLineEdit, QTabWidget, QFileDialog, QInputDialog, QMessageBox,
    QToolButton, QFrame, QListWidgetItem, QStyle,
)
from PySide6.QtCore import Qt, QTimer, QProcess, Signal, QSize
from PySide6.QtGui import QFont, QPainter, QColor

from app.voice_profile import VoiceMode
from app.voice_manager import VoiceManager
from app.tts_backend import QWEN_SPEAKERS, QWEN_LANGUAGES, QWEN_MODEL_SIZES
from app.ui_status import apply_status_style, clear_status_role
from gui.widgets.settings_card import SettingsCard

logger = logging.getLogger(__name__)

QWEN_MODELS = {
    "Base 0.6B (Clone)": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "Base 1.7B (Clone)": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "CustomVoice 0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "CustomVoice 1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "VoiceDesign 1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
}

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
QWEN_TTS_BIND_HOST = (
    os.environ.get("REVIA_QWEN_TTS_HOST", "127.0.0.1").strip() or "127.0.0.1"
)
QWEN_TTS_DEFAULT_MODEL_SIZE = "0.6B"


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _standard_icon(widget, name: str, fallback: str = "SP_FileIcon"):
    standard = getattr(QStyle, name, None)
    if standard is None and hasattr(QStyle, "StandardPixmap"):
        standard = getattr(QStyle.StandardPixmap, name, None)
    if standard is None:
        standard = getattr(QStyle, fallback, None)
    if standard is None and hasattr(QStyle, "StandardPixmap"):
        standard = getattr(QStyle.StandardPixmap, fallback, None)
    if standard is None:
        return None
    return widget.style().standardIcon(standard)


class CollapsibleSection(QWidget):
    def __init__(self, title, content, expanded=True, parent=None):
        super().__init__(parent)
        self.content = content
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.toggle = QToolButton()
        self.toggle.setText(title)
        self.toggle.setCheckable(True)
        self.toggle.setChecked(expanded)
        self.toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.toggle.clicked.connect(self._on_toggled)
        layout.addWidget(self.toggle)

        self.content.setVisible(expanded)
        layout.addWidget(self.content)

    def _on_toggled(self, checked):
        self.toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self.content.setVisible(checked)


class StatusCard(QFrame):
    def __init__(self, title, value="--", role="muted", parent=None):
        super().__init__(parent)
        self.setObjectName("voiceStatusCard")
        self.setFrameShape(QFrame.StyledPanel)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(2)
        self.title_lbl = QLabel(title)
        self.title_lbl.setObjectName("metricLabel")
        self.title_lbl.setFont(QFont("Segoe UI", 8, QFont.Bold))
        self.value_lbl = QLabel(value)
        self.value_lbl.setObjectName("metricLabel")
        self.value_lbl.setWordWrap(True)
        layout.addWidget(self.title_lbl)
        layout.addWidget(self.value_lbl)
        self.set_value(value, role)

    def set_value(self, value, role="muted"):
        self.value_lbl.setText(str(value or "--"))
        apply_status_style(self.value_lbl, role=role)


class Stepper(QWidget):
    def __init__(self, steps, parent=None):
        super().__init__(parent)
        self._steps = list(steps)
        self._labels = []
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        for step in self._steps:
            lbl = QLabel(step)
            lbl.setObjectName("metricLabel")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setMinimumHeight(24)
            lbl.setFrameShape(QFrame.StyledPanel)
            apply_status_style(lbl, role="muted")
            layout.addWidget(lbl)
            self._labels.append(lbl)

    def set_state(self, active=None, complete=None, failed=None):
        complete = set(complete or [])
        failed = set(failed or [])
        for step, lbl in zip(self._steps, self._labels):
            if step in failed:
                role = "error"
            elif step == active:
                role = "warning"
            elif step in complete:
                role = "success"
            else:
                role = "muted"
            apply_status_style(lbl, role=role)


class WaveformPreview(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._samples = []
        self._label = "No audio generated yet"
        self.setMinimumHeight(58)

    def clear(self, label="No audio generated yet"):
        self._samples = []
        self._label = label
        self.update()

    def set_audio(self, wav_path):
        self._label = str(wav_path or "")
        self._samples = self._read_samples(wav_path)
        self.update()

    def _read_samples(self, wav_path):
        try:
            import wave
            path = Path(wav_path)
            if not path.exists():
                return []
            with wave.open(str(path), "rb") as wf:
                channels = max(1, wf.getnchannels())
                width = wf.getsampwidth()
                frames = wf.readframes(min(wf.getnframes(), 44100 * 8))
            if not frames or width not in (1, 2, 4):
                return []
            step = max(width * channels, int(len(frames) / 96) or width * channels)
            values = []
            max_amp = 1
            for offset in range(0, len(frames) - width, step):
                chunk = frames[offset:offset + width]
                if width == 1:
                    amp = abs(chunk[0] - 128)
                    max_amp = 128
                elif width == 2:
                    amp = abs(int.from_bytes(chunk, "little", signed=True))
                    max_amp = 32768
                else:
                    amp = abs(int.from_bytes(chunk, "little", signed=True))
                    max_amp = 2147483648
                values.append(min(1.0, amp / max_amp))
            return values[:96]
        except Exception as exc:
            logger.debug("Could not build waveform preview: %s", exc)
            return []

    def paintEvent(self, event):
        painter = QPainter(self)
        rect = self.rect().adjusted(1, 1, -1, -1)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(rect, QColor(20, 18, 32))
        painter.setPen(QColor(78, 66, 110))
        painter.drawRoundedRect(rect, 6, 6)
        if not self._samples:
            painter.setPen(QColor(150, 140, 180))
            painter.drawText(rect, Qt.AlignCenter, self._label)
            return
        mid = rect.center().y()
        bar_w = max(2, rect.width() // max(1, len(self._samples)))
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(176, 78, 255))
        x = rect.left() + 6
        usable_h = max(12, rect.height() - 14)
        for sample in self._samples:
            h = max(4, int(sample * usable_h))
            painter.drawRoundedRect(x, mid - h // 2, max(2, bar_w - 2), h, 2, 2)
            x += bar_w


class VoiceCardWidget(QFrame):
    def __init__(self, profile, is_active, is_default, callbacks, parent=None):
        super().__init__(parent)
        self.profile = profile
        self.setObjectName("voiceCard")
        self.setFrameShape(QFrame.StyledPanel)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)

        top = QHBoxLayout()
        title = QLabel(profile.name)
        title.setFont(QFont("Segoe UI", 9, QFont.Bold))
        top.addWidget(title, stretch=1)
        badges = []
        if is_active:
            badges.append("Active")
        if is_default:
            badges.append("Default")
        badges.append(profile.mode.value.title())
        badge = QLabel(" | ".join(badges))
        badge.setObjectName("metricLabel")
        apply_status_style(badge, role="success" if is_active else "accent")
        top.addWidget(badge)
        layout.addLayout(top)

        wav_state = "WAV ready" if profile.has_wav() else "No WAV"
        meta = QLabel(f"{profile.language} | {wav_state}")
        meta.setObjectName("metricLabel")
        apply_status_style(meta, role="success" if profile.has_wav() else "warning")
        layout.addWidget(meta)

        buttons = QHBoxLayout()
        for label, callback in (
            ("Play", callbacks["play"]),
            ("Set Active", callbacks["active"]),
            ("Delete", callbacks["delete"]),
        ):
            btn = QPushButton(label)
            btn.setObjectName("secondaryBtn")
            icon_name = {
                "Play": "SP_MediaPlay",
                "Set Active": "SP_DialogApplyButton",
                "Delete": "SP_TrashIcon",
            }.get(label)
            icon = _standard_icon(btn, icon_name, fallback="SP_DialogDiscardButton")
            if icon is not None:
                btn.setIcon(icon)
                btn.setIconSize(QSize(15, 15))
            btn.setToolTip(label)
            btn.clicked.connect(lambda _checked=False, cb=callback, name=profile.name: cb(name))
            if label == "Play":
                btn.setEnabled(profile.has_wav())
            buttons.addWidget(btn)
        buttons.addStretch()
        layout.addLayout(buttons)


class VoiceTab(QScrollArea):
    design_done = Signal(object, object)
    clone_done = Signal(object, object)
    custom_done = Signal(object, object)

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
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        header = QLabel("Voice Management")
        header.setObjectName("tabHeader")
        header.setFont(QFont("Segoe UI", 12, QFont.Bold))
        layout.addWidget(header)

        self.toast_label = QLabel("")
        self.toast_label.setObjectName("toastLabel")
        self.toast_label.setWordWrap(True)
        self.toast_label.setVisible(False)
        layout.addWidget(self.toast_label)
        self._toast_timer = QTimer(self)
        self._toast_timer.setSingleShot(True)
        self._toast_timer.timeout.connect(lambda: self.toast_label.setVisible(False))

        # Qwen3-TTS Server
        srv_card = SettingsCard("Qwen3-TTS Server", subtitle="Server config & backend", icon="S")
        sv = QFormLayout()
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
        self._tts_cuda_skip_reason = ""
        self._tts_cuda_warning_logged = False
        self._startup_backend = self._preferred_startup_backend()
        self._qwen_activation_pending = self._startup_backend == "qwen3-tts"
        self.engine_combo.setCurrentText(
            "Qwen3-TTS" if self._startup_backend == "qwen3-tts" else "pyttsx3"
        )
        self.engine_combo.currentTextChanged.connect(self._on_engine_changed)
        btn_row.addWidget(self.engine_combo)
        sv.addRow("", btn_row)

        self.tts_server_status = QLabel("Not running")
        self.tts_server_status.setFont(QFont("Consolas", 8))
        self.tts_server_status.setObjectName("metricLabel")
        self.tts_server_status.setWordWrap(True)
        sv.addRow("Server:", self.tts_server_status)

        # Active backend indicator - always visible, single source of truth
        startup_label = (
            "Qwen3-TTS pending"
            if self._qwen_activation_pending
            else "pyttsx3"
        )
        startup_health = "Starting" if self._qwen_activation_pending else "Ready"
        self.active_backend_lbl = QLabel(
            f"Active: {startup_label}  |  Fallback: pyttsx3  |  {startup_health}"
        )
        self.active_backend_lbl.setObjectName("metricLabel")
        self.active_backend_lbl.setFont(QFont("Consolas", 8))
        self.active_backend_lbl.setWordWrap(True)
        sv.addRow("Backend:", self.active_backend_lbl)

        # Keep live speech on the fallback until the Qwen server is reachable.
        self.voice_mgr.backend.set_engine("pyttsx3")
        self.voice_mgr.backend.set_qwen_server("http://localhost:8000")
        # Wire backend_changed so the indicator stays in sync
        self.voice_mgr.backend_changed.connect(self._on_backend_changed)
        self._tts_process = None
        self._tts_ready_timer = None
        self._tts_last_lines: list[str] = []
        self._tts_launcher_tmp = None  # temp wrapper script for CPU-only builds
        self._tts_last_device_label = ""
        self._tts_cuda_failure_seen = False
        self._tts_cuda_retry_attempted = False
        # Capability snapshot from the running Qwen3-TTS server.  Updated
        # by the backend's ``capabilities_changed`` signal.  When the
        # active server only supports a subset of modes (Base/Clone vs
        # CustomVoice vs VoiceDesign), the unsupported tab labels show
        # ``(N/A)`` and the active one shows ``(active)``.
        self._caps_variant = "unknown"
        self._caps_label = "Probing..."
        self._caps_modes: list[str] = []
        self._caps_api_names: list[str] = []
        builder_content = QWidget()
        builder_layout = QVBoxLayout(builder_content)
        builder_layout.setContentsMargins(0, 0, 0, 0)
        builder_layout.setSpacing(6)
        srv_card.add_layout(sv)
        builder_layout.addWidget(srv_card)

        card_row = QHBoxLayout()
        self.voice_card = StatusCard("Voice", "Fallback", "warning")
        self.mode_card = StatusCard("Mode", "Select a mode", "muted")
        self.save_card = StatusCard("Save", "Auto-save waiting", "muted")
        self.output_card = StatusCard("Output", "System Default", "muted")
        for card in (self.voice_card, self.mode_card, self.save_card, self.output_card):
            card_row.addWidget(card)
        builder_layout.addLayout(card_row)

        # Active Voice / Server Capabilities panel.  Always visible,
        # tells the user *which voice* speaks, *which mode* is active,
        # and *what the server can do* without scrolling.
        active_card = SettingsCard("Active Voice & Server Capabilities", subtitle="Current voice & server info", icon="A")
        ag = QFormLayout()
        ag.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self.active_voice_lbl = QLabel("Active Voice: (none) - using fallback")
        self.active_voice_lbl.setObjectName("metricLabel")
        self.active_voice_lbl.setFont(QFont("Consolas", 8))
        self.active_voice_lbl.setWordWrap(True)
        ag.addRow("Voice:", self.active_voice_lbl)

        self.caps_lbl = QLabel("Server: probing...")
        self.caps_lbl.setObjectName("metricLabel")
        self.caps_lbl.setFont(QFont("Consolas", 8))
        self.caps_lbl.setWordWrap(True)
        ag.addRow("Capabilities:", self.caps_lbl)

        # Live "what's happening now" indicator (synthesis + playback).
        self.activity_lbl = QLabel("Idle")
        self.activity_lbl.setObjectName("metricLabel")
        self.activity_lbl.setFont(QFont("Consolas", 8))
        self.activity_lbl.setWordWrap(True)
        ag.addRow("Activity:", self.activity_lbl)
        active_card.add_layout(ag)
        builder_layout.addWidget(active_card)

        self.workflow_stepper = Stepper([
            "Name", "Configure", "Generate", "Auto-save", "Test", "Active",
        ])
        self.workflow_stepper.set_state(active="Name")
        builder_layout.addWidget(self.workflow_stepper)

        self.activity_timeline = Stepper([
            "Server", "Generating", "Saved", "Playing",
        ])
        self.activity_timeline.set_state(active="Server")
        builder_layout.addWidget(self.activity_timeline)

        # 3 Mode Tabs (matching Qwen3-TTS demo)
        self.mode_tabs = QTabWidget()
        self.mode_tabs.addTab(self._build_design_tab(), "Voice Design")
        self.mode_tabs.addTab(self._build_clone_tab(), "Voice Clone (Base)")
        self.mode_tabs.addTab(self._build_custom_tab(), "TTS (CustomVoice)")
        self.mode_tabs.setMinimumHeight(360)
        builder_layout.addWidget(self.mode_tabs)
        layout.addWidget(CollapsibleSection("Voice Builder", builder_content, expanded=True))

        # Voice Library
        lib_card = SettingsCard("Voice Library", subtitle="Manage saved voices", icon="V")
        lib_layout = QVBoxLayout()
        lib_layout.setSpacing(4)

        self.voice_list = QListWidget()
        self.voice_list.setMaximumHeight(260)
        self.voice_list.currentItemChanged.connect(self._on_voice_item_selected)
        lib_layout.addWidget(self.voice_list)

        lib_btns = QHBoxLayout()
        self.library_action_buttons = {}
        for text, slot in [
            ("Set Active", self._set_active), ("Rename", self._rename_voice),
            ("Delete", self._delete_voice), ("Set Default", self._set_default),
        ]:
            b = QPushButton(text)
            b.setObjectName("secondaryBtn")
            b.setMinimumHeight(26)
            b.setMaximumHeight(34)
            b.clicked.connect(slot)
            self.library_action_buttons[text] = b
            lib_btns.addWidget(b)
        lib_layout.addLayout(lib_btns)

        lib_btns2 = QHBoxLayout()
        for text, slot in [
            ("Import", self._import_voice), ("Export", self._export_voice),
            ("Open Folder", self._open_folder), ("Play Voice", self._play_selected),
        ]:
            b = QPushButton(text)
            b.setObjectName("secondaryBtn")
            b.setMinimumHeight(26)
            b.setMaximumHeight(34)
            b.clicked.connect(slot)
            self.library_action_buttons[text] = b
            lib_btns2.addWidget(b)
        lib_layout.addLayout(lib_btns2)

        voices_path = str(self.voice_mgr.library.base_dir)
        self.lib_meta = QLabel(f"Saved in: {voices_path}\nSelect a voice to see details")
        self.lib_meta.setFont(QFont("Consolas", 8))
        self.lib_meta.setObjectName("metricLabel")
        self.lib_meta.setWordWrap(True)
        lib_layout.addWidget(self.lib_meta)
        lib_card.add_layout(lib_layout)
        layout.addWidget(CollapsibleSection("Voice Library", lib_card, expanded=False))

        # STT Input
        input_content = QWidget()
        input_layout = QVBoxLayout(input_content)
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.setSpacing(6)
        stt_card = SettingsCard("Speech-to-Text (Input)", subtitle="STT input settings", icon="I")
        stf = QFormLayout()
        self.input_device = QComboBox()
        self.input_device.addItem("Default Microphone")
        self._populate_input_devices()
        self.input_device.currentIndexChanged.connect(self._on_input_device_changed)
        stf.addRow("Input Device:", self.input_device)
        self.ptt_mode = QComboBox()
        self.ptt_mode.addItems([
            "Toggle (click to start/stop)", "Push-to-Talk (hold)",
            "Always Listening (VAD)",
        ])
        stf.addRow("Activation:", self.ptt_mode)
        stt_card.add_layout(stf)
        input_layout.addWidget(stt_card)

        # Mic Test
        mic_card = SettingsCard("Microphone Test", subtitle="Test mic levels", icon="M")
        ml = QVBoxLayout()
        self.vol_bar = QProgressBar()
        self.vol_bar.setRange(0, 100)
        self.vol_bar.setValue(0)
        self.vol_bar.setTextVisible(False)
        self.vol_bar.setMinimumHeight(18)
        self.vol_bar.setMaximumHeight(28)
        self._apply_vol_bar_style()
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
        mic_card.add_layout(ml)
        input_layout.addWidget(mic_card)
        layout.addWidget(CollapsibleSection("Speech Input", input_content, expanded=False))

        # Latency Metrics
        diagnostics_content = QWidget()
        diagnostics_layout = QVBoxLayout(diagnostics_content)
        diagnostics_layout.setContentsMargins(0, 0, 0, 0)
        diagnostics_layout.setSpacing(6)
        lat_card = SettingsCard("TTS Latency", subtitle="Synthesis timing metrics", icon="L")
        ll = QHBoxLayout()
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
        lat_card.add_layout(ll)
        diagnostics_layout.addWidget(lat_card)
        layout.addWidget(CollapsibleSection("Diagnostics", diagnostics_content, expanded=False))

        # Status
        self.status_label = QLabel("Status: Ready")
        self.status_label.setFont(QFont("Consolas", 8))
        self.status_label.setObjectName("metricLabel")
        layout.addWidget(self.status_label)

        # TTS Output Device (Revia's voice), intentionally kept at the
        # bottom so the voice-building flow stays focused.
        out_card = SettingsCard("TTS Output", subtitle="Output device selection", icon="O")
        ov = QFormLayout()
        ov.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.output_device = QComboBox()
        self.output_device.addItem("System Default", userData=None)
        self._populate_output_devices()
        self.output_device.currentIndexChanged.connect(self._on_output_device_changed)
        ov.addRow("Output Device:", self.output_device)
        self.output_route_lbl = QLabel("Revia Voice -> System Default")
        self.output_route_lbl.setObjectName("metricLabel")
        self.output_route_lbl.setWordWrap(True)
        ov.addRow("Route:", self.output_route_lbl)
        out_btn_row = QHBoxLayout()
        self.output_test_btn = QPushButton("Test Output")
        self.output_test_btn.setObjectName("secondaryBtn")
        self.output_test_btn.clicked.connect(self._test_output_device)
        out_btn_row.addWidget(self.output_test_btn)
        self.output_refresh_btn = QPushButton("Refresh")
        self.output_refresh_btn.setObjectName("secondaryBtn")
        self.output_refresh_btn.clicked.connect(self._refresh_output_devices)
        out_btn_row.addWidget(self.output_refresh_btn)
        out_btn_row.addStretch()
        ov.addRow("", out_btn_row)
        out_card.add_layout(ov)
        layout.addWidget(CollapsibleSection("TTS Output Check", out_card, expanded=False))

        layout.addStretch()
        self.setWidget(container)

        # Wiring
        self.voice_mgr.metrics_updated.connect(self._on_metrics)
        self.voice_mgr.error.connect(
            lambda e: self._set_status(f"Error: {e}", role="error")
        )
        self.voice_mgr.status.connect(self._on_voice_status)
        self.design_done.connect(self._on_design_done)
        self.clone_done.connect(self._on_clone_done)
        self.custom_done.connect(self._on_custom_done)
        # React to TTS server capability changes + synthesis lifecycle so
        # the "Active Voice & Server Capabilities" panel stays accurate.
        backend = self.voice_mgr.backend
        backend.capabilities_changed.connect(self._on_capabilities_changed)
        backend.synthesis_started.connect(self._on_synthesis_started)
        backend.synthesis_finished.connect(self._on_synthesis_finished_ui)
        backend.error_occurred.connect(self._on_synthesis_error)
        # Mode-mismatch warnings come through status_updated; surface
        # them in the activity line so the user sees why nothing played.
        backend.status_updated.connect(self._on_backend_status_for_activity)
        backend.playback_started.connect(self._on_playback_started)
        backend.playback_finished.connect(self._on_playback_finished)
        backend.playback_interrupted.connect(self._on_playback_interrupted)
        self.voice_mgr.voice_changed.connect(self._on_voice_changed)
        self.mode_tabs.currentChanged.connect(self._on_mode_tab_changed)
        self._refresh_active_voice_label()
        self._refresh_capabilities_ui()
        self.event_bus.connection_changed.connect(self._on_core_connection)
        self.event_bus.ui_theme_changed.connect(self._on_theme_changed)
        self._refresh_library()
        self._apply_button_icons()
        self._refresh_visual_status()

        # Defer the "fetch persisted TTS output device" hit to let the core
        # come up first. If the core is already up this returns instantly;
        # if it's still booting, get_tts_output() returns the default shape
        # and we simply leave the combobox on "System Default".
        QTimer.singleShot(500, self._load_saved_output_device)

    # Voice Design Tab
    def _build_design_tab(self):
        w = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(6, 6, 6, 6)
        vl.setSpacing(4)

        fl = QFormLayout()
        fl.setSpacing(4)

        self.design_name = QLineEdit()
        self.design_name.setPlaceholderText("Name for this saved voice")
        fl.addRow("Voice Name:", self.design_name)

        self.design_text = QTextEdit()
        self.design_text.setMaximumHeight(90)
        self.design_text.setPlaceholderText(
            "Hello! Welcome to Text-to-Speech system. "
            "This is a demo of our TTS capabilities."
        )
        fl.addRow("Text:", self.design_text)

        self.design_lang = QComboBox()
        self.design_lang.addItems(QWEN_LANGUAGES)
        fl.addRow("Language:", self.design_lang)

        self.design_desc = QTextEdit()
        self.design_desc.setMaximumHeight(80)
        self.design_desc.setPlaceholderText(
            "e.g. Speak in an incredulous tone, but with a hint of "
            "panic beginning to creep into your voice."
        )
        fl.addRow("Voice Desc:", self.design_desc)

        vl.addLayout(fl)

        btn_row = QHBoxLayout()
        self.design_generate_btn = QPushButton("Generate with Custom Voice")
        self.design_generate_btn.setObjectName("primaryBtn")
        self.design_generate_btn.clicked.connect(self._run_design)
        btn_row.addWidget(self.design_generate_btn)
        self.design_play_btn = QPushButton("Play")
        self.design_play_btn.setObjectName("secondaryBtn")
        self.design_play_btn.setEnabled(False)
        self.design_play_btn.clicked.connect(lambda: self._play_last("design"))
        btn_row.addWidget(self.design_play_btn)
        self.design_delete_btn = QPushButton("Delete Voice")
        self.design_delete_btn.setObjectName("secondaryBtn")
        self.design_delete_btn.setEnabled(False)
        self.design_delete_btn.clicked.connect(lambda: self._delete_current_voice("design"))
        btn_row.addWidget(self.design_delete_btn)
        vl.addLayout(btn_row)

        self.design_progress = QProgressBar()
        self.design_progress.setRange(0, 0)
        self.design_progress.setTextVisible(False)
        self.design_progress.setVisible(False)
        vl.addWidget(self.design_progress)

        self.design_status = QLabel("")
        self.design_status.setFont(QFont("Consolas", 8))
        self.design_status.setObjectName("metricLabel")
        self.design_status.setWordWrap(True)
        vl.addWidget(self.design_status)

        self.design_waveform = WaveformPreview()
        vl.addWidget(self.design_waveform)

        vl.addStretch()
        self._last_design_wav = None
        return w

    # Voice Clone (Base) Tab
    def _build_clone_tab(self):
        w = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(6, 6, 6, 6)
        vl.setSpacing(4)

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
        fl.setSpacing(4)

        self.clone_name = QLineEdit()
        self.clone_name.setPlaceholderText("Name for this saved voice")
        fl.addRow("Voice Name:", self.clone_name)

        self.clone_ref_text = QTextEdit()
        self.clone_ref_text.setMaximumHeight(60)
        self.clone_ref_text.setPlaceholderText(
            "Enter the exact text spoken in the reference audio..."
        )
        fl.addRow("Ref Text:", self.clone_ref_text)

        self.clone_target_text = QTextEdit()
        self.clone_target_text.setMaximumHeight(76)
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
        self.clone_model_size.setCurrentText(QWEN_TTS_DEFAULT_MODEL_SIZE)
        row.addWidget(QLabel("Model Size:"))
        row.addWidget(self.clone_model_size)
        fl.addRow(row)

        self.clone_xvec = QCheckBox(
            "Use x-vector only (No reference text needed, but lower quality)"
        )
        fl.addRow("", self.clone_xvec)

        vl.addLayout(fl)

        btn_row = QHBoxLayout()
        self.clone_generate_btn = QPushButton("Clone & Generate")
        self.clone_generate_btn.setObjectName("primaryBtn")
        self.clone_generate_btn.clicked.connect(self._run_clone)
        btn_row.addWidget(self.clone_generate_btn)
        self.clone_play_btn = QPushButton("Play")
        self.clone_play_btn.setObjectName("secondaryBtn")
        self.clone_play_btn.setEnabled(False)
        self.clone_play_btn.clicked.connect(lambda: self._play_last("clone"))
        btn_row.addWidget(self.clone_play_btn)
        self.clone_delete_btn = QPushButton("Delete Voice")
        self.clone_delete_btn.setObjectName("secondaryBtn")
        self.clone_delete_btn.setEnabled(False)
        self.clone_delete_btn.clicked.connect(lambda: self._delete_current_voice("clone"))
        btn_row.addWidget(self.clone_delete_btn)
        vl.addLayout(btn_row)

        self.clone_progress = QProgressBar()
        self.clone_progress.setRange(0, 0)
        self.clone_progress.setTextVisible(False)
        self.clone_progress.setVisible(False)
        vl.addWidget(self.clone_progress)

        self.clone_status = QLabel("")
        self.clone_status.setFont(QFont("Consolas", 8))
        self.clone_status.setObjectName("metricLabel")
        self.clone_status.setWordWrap(True)
        vl.addWidget(self.clone_status)

        self.clone_waveform = WaveformPreview()
        vl.addWidget(self.clone_waveform)

        vl.addStretch()
        self._last_clone_wav = None
        return w

    # TTS (CustomVoice) Tab
    def _build_custom_tab(self):
        w = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(6, 6, 6, 6)
        vl.setSpacing(4)

        fl = QFormLayout()
        fl.setSpacing(4)

        self.custom_name = QLineEdit()
        self.custom_name.setPlaceholderText("Name for this saved voice")
        fl.addRow("Voice Name:", self.custom_name)

        self.custom_text = QTextEdit()
        self.custom_text.setMaximumHeight(90)
        self.custom_text.setPlaceholderText(
            "Hello! Welcome to Text-to-Speech system. "
            "This is a demo of our TTS capabilities."
        )
        fl.addRow("Text:", self.custom_text)

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
        self.custom_style.setMaximumHeight(68)
        self.custom_style.setPlaceholderText(
            "e.g. Speak in a cheerful and energetic tone..."
        )
        fl.addRow("Style:", self.custom_style)

        self.custom_model_size = QComboBox()
        self.custom_model_size.addItems(QWEN_MODEL_SIZES)
        self.custom_model_size.setCurrentText(QWEN_TTS_DEFAULT_MODEL_SIZE)
        fl.addRow("Model Size:", self.custom_model_size)

        vl.addLayout(fl)

        btn_row = QHBoxLayout()
        self.custom_generate_btn = QPushButton("Generate Speech")
        self.custom_generate_btn.setObjectName("primaryBtn")
        self.custom_generate_btn.clicked.connect(self._run_custom)
        btn_row.addWidget(self.custom_generate_btn)
        self.custom_play_btn = QPushButton("Play")
        self.custom_play_btn.setObjectName("secondaryBtn")
        self.custom_play_btn.setEnabled(False)
        self.custom_play_btn.clicked.connect(lambda: self._play_last("custom"))
        btn_row.addWidget(self.custom_play_btn)
        self.custom_delete_btn = QPushButton("Delete Voice")
        self.custom_delete_btn.setObjectName("secondaryBtn")
        self.custom_delete_btn.setEnabled(False)
        self.custom_delete_btn.clicked.connect(lambda: self._delete_current_voice("custom"))
        btn_row.addWidget(self.custom_delete_btn)
        vl.addLayout(btn_row)

        self.custom_progress = QProgressBar()
        self.custom_progress.setRange(0, 0)
        self.custom_progress.setTextVisible(False)
        self.custom_progress.setVisible(False)
        vl.addWidget(self.custom_progress)

        self.custom_status = QLabel("")
        self.custom_status.setFont(QFont("Consolas", 8))
        self.custom_status.setObjectName("metricLabel")
        self.custom_status.setWordWrap(True)
        vl.addWidget(self.custom_status)

        self.custom_waveform = WaveformPreview()
        vl.addWidget(self.custom_waveform)

        vl.addStretch()
        self._last_custom_wav = None
        return w

    # Generation actions (threaded)

    def _set_generation_busy(self, mode, message):
        setattr(self, f"_last_{mode}_wav", None)
        getattr(self, f"{mode}_progress").setVisible(True)
        getattr(self, f"{mode}_generate_btn").setEnabled(False)
        getattr(self, f"{mode}_play_btn").setEnabled(False)
        getattr(self, f"{mode}_delete_btn").setEnabled(False)
        getattr(self, f"{mode}_waveform").clear("Generating audio...")
        status = getattr(self, f"{mode}_status")
        status.setText(message)
        apply_status_style(status, role="warning")
        self._set_activity(message, role="warning")
        self.workflow_stepper.set_state(
            active="Generate",
            complete={"Name", "Configure"},
        )
        self.activity_timeline.set_state(active="Generating", complete={"Server"})
        self.save_card.set_value("Generating", "warning")
        self._show_toast(message, role="warning")

    def _set_generation_ready(self, mode, name, wav):
        getattr(self, f"{mode}_progress").setVisible(False)
        getattr(self, f"{mode}_generate_btn").setEnabled(True)
        getattr(self, f"{mode}_play_btn").setEnabled(True)
        getattr(self, f"{mode}_delete_btn").setEnabled(True)
        getattr(self, f"{mode}_waveform").set_audio(wav)
        status = getattr(self, f"{mode}_status")
        status.setText(f"Done. Auto-saved to Library as '{name}'.\nSaved WAV: {wav}")
        apply_status_style(status, role="success")
        self._set_activity(f"Generation complete. '{name}' was auto-saved.", role="success")
        self.workflow_stepper.set_state(
            active="Test",
            complete={"Name", "Configure", "Generate", "Auto-save", "Active"},
        )
        self.activity_timeline.set_state(active="Playing", complete={"Server", "Generating", "Saved"})
        self.save_card.set_value(f"Saved: {name}", "success")
        self._show_toast(f"Voice auto-saved: {name}", role="success")
        self._refresh_visual_status()

    def _set_generation_failed(self, mode, error):
        getattr(self, f"{mode}_progress").setVisible(False)
        getattr(self, f"{mode}_generate_btn").setEnabled(True)
        getattr(self, f"{mode}_play_btn").setEnabled(False)
        getattr(self, f"{mode}_delete_btn").setEnabled(False)
        getattr(self, f"{mode}_waveform").clear("No audio saved")
        status = getattr(self, f"{mode}_status")
        status.setText(f"Failed: {error}")
        apply_status_style(status, role="error")
        self._set_activity(f"Generation failed: {error}", role="error")
        self.workflow_stepper.set_state(failed={"Generate"})
        self.activity_timeline.set_state(failed={"Generating"})
        self.save_card.set_value("Save failed", "error")
        self._show_toast(f"Generation failed: {error}", role="error")

    def _run_design(self):
        text = self.design_text.toPlainText().strip()
        desc = self.design_desc.toPlainText().strip()
        lang = self.design_lang.currentText()
        if not self._voice_name_from_field(self.design_name, self.design_status, "generating"):
            return
        if not text:
            self.design_status.setText("Enter text to synthesize")
            apply_status_style(self.design_status, role="warning")
            return
        if not desc:
            self.design_status.setText("Enter a voice description")
            apply_status_style(self.design_status, role="warning")
            return
        self._set_generation_busy("design", "Generating Voice Design audio...")

        def _do():
            wav, metrics = self.voice_mgr.generate_design(text, desc, lang)
            self.design_done.emit(wav, metrics)
        threading.Thread(target=_do, daemon=True).start()

    def _on_design_done(self, wav, metrics):
        if wav:
            try:
                name, saved_wav = self._save_generated_voice("design", wav)
            except Exception as exc:
                self._last_design_wav = wav
                self._set_generation_failed("design", f"Generated, but auto-save failed: {exc}")
                return
            self._last_design_wav = str(saved_wav)
            self._set_generation_ready("design", name, saved_wav)
            self.voice_mgr.play_wav(saved_wav)
        else:
            self._set_generation_failed("design", metrics)

    def _run_clone(self):
        ref = self.clone_ref_path.text().strip()
        ref_text = self.clone_ref_text.toPlainText().strip()
        target = self.clone_target_text.toPlainText().strip()
        lang = self.clone_lang.currentText()
        model_size = self.clone_model_size.currentText()
        xvec = self.clone_xvec.isChecked()
        if not self._voice_name_from_field(self.clone_name, self.clone_status, "generating"):
            return
        if not ref:
            self.clone_status.setText("Select reference audio first")
            apply_status_style(self.clone_status, role="warning")
            return
        if not target:
            self.clone_status.setText("Enter target text")
            apply_status_style(self.clone_status, role="warning")
            return
        if not xvec and not ref_text:
            self.clone_status.setText(
                "Enter reference text or enable x-vector only"
            )
            apply_status_style(self.clone_status, role="warning")
            return
        self._set_generation_busy("clone", "Cloning voice and generating audio...")

        def _do():
            wav, metrics = self.voice_mgr.generate_clone(
                target, ref, ref_text, lang, xvec, model_size=model_size
            )
            self.clone_done.emit(wav, metrics)
        threading.Thread(target=_do, daemon=True).start()

    def _on_clone_done(self, wav, metrics):
        if wav:
            try:
                name, saved_wav = self._save_generated_voice("clone", wav)
            except Exception as exc:
                self._last_clone_wav = wav
                self._set_generation_failed("clone", f"Generated, but auto-save failed: {exc}")
                return
            self._last_clone_wav = str(saved_wav)
            self._set_generation_ready("clone", name, saved_wav)
            self.voice_mgr.play_wav(saved_wav)
        else:
            self._set_generation_failed("clone", metrics)

    def _run_custom(self):
        text = self.custom_text.toPlainText().strip()
        lang = self.custom_lang.currentText()
        speaker = self.custom_speaker.currentText()
        style = self.custom_style.toPlainText().strip()
        model_size = self.custom_model_size.currentText()
        if not self._voice_name_from_field(self.custom_name, self.custom_status, "generating"):
            return
        if not text:
            self.custom_status.setText("Enter text to synthesize")
            apply_status_style(self.custom_status, role="warning")
            return
        self._set_generation_busy("custom", "Generating CustomVoice audio...")

        def _do():
            wav, metrics = self.voice_mgr.generate_custom(
                text, lang, speaker, style, model_size
            )
            self.custom_done.emit(wav, metrics)
        threading.Thread(target=_do, daemon=True).start()

    def _on_custom_done(self, wav, metrics):
        if wav:
            try:
                name, saved_wav = self._save_generated_voice("custom", wav)
            except Exception as exc:
                self._last_custom_wav = wav
                self._set_generation_failed("custom", f"Generated, but auto-save failed: {exc}")
                return
            self._last_custom_wav = str(saved_wav)
            self._set_generation_ready("custom", name, saved_wav)
            self.voice_mgr.play_wav(saved_wav)
        else:
            self._set_generation_failed("custom", metrics)

    def _play_last(self, mode):
        wav = getattr(self, f"_last_{mode}_wav", None)
        if wav and Path(wav).exists():
            self.voice_mgr.play_wav(wav)

    # Auto-save / delete

    def _voice_name_from_field(self, field, status_label, action="saving"):
        name = field.text().strip()
        if name:
            if not any(ch.isalnum() for ch in name):
                status_label.setText("Voice name must contain at least one letter or number")
                apply_status_style(status_label, role="warning")
                field.setFocus()
                return ""
            return name
        status_label.setText(f"Enter a voice name before {action}")
        apply_status_style(status_label, role="warning")
        field.setFocus()
        return ""

    def _save_generated_voice(self, mode, wav):
        wav_path = Path(wav)
        if not wav_path.exists():
            raise FileNotFoundError(str(wav))
        if mode == "design":
            name = self.design_name.text().strip()
            p = self.voice_mgr.create_design_profile(
                name,
                self.design_desc.toPlainText().strip(),
                self.design_lang.currentText(),
            )
        elif mode == "clone":
            name = self.clone_name.text().strip()
            ref = self.clone_ref_path.text().strip()
            p = self.voice_mgr.create_clone_profile(
                name, ref,
                self.clone_ref_text.toPlainText().strip(),
                self.clone_lang.currentText(),
                self.clone_xvec.isChecked(),
                self.clone_model_size.currentText(),
            )
        else:
            name = self.custom_name.text().strip()
            p = self.voice_mgr.create_custom_profile(
                name,
                self.custom_speaker.currentText(),
                self.custom_style.toPlainText().strip(),
                self.custom_lang.currentText(),
                self.custom_model_size.currentText(),
            )
        voice_dir = self.voice_mgr.save_voice(p)
        dest = voice_dir / "generated.wav"
        shutil.copy2(str(wav_path), str(dest))
        p.generated_wav = str(dest)
        if mode == "clone":
            ref = self.clone_ref_path.text().strip()
            if ref and Path(ref).exists():
                ref_dest = voice_dir / "ref.wav"
                shutil.copy2(ref, str(ref_dest))
                p.reference_audio = str(ref_dest)
        p.save(voice_dir)
        self.voice_mgr.set_active_voice(name)
        self._refresh_library()
        return name, dest

    def _delete_current_voice(self, mode):
        field = getattr(self, f"{mode}_name")
        status = getattr(self, f"{mode}_status")
        name = self._voice_name_from_field(field, status, "deleting")
        if not name:
            return
        if not self.voice_mgr.library.get(name):
            status.setText(f"No saved voice named '{name}'")
            apply_status_style(status, role="warning")
            getattr(self, f"{mode}_delete_btn").setEnabled(False)
            return
        reply = QMessageBox.question(
            self,
            "Delete Voice",
            f"Delete '{name}' from the Voice Library?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self.voice_mgr.delete_voice(name)
        setattr(self, f"_last_{mode}_wav", None)
        getattr(self, f"{mode}_play_btn").setEnabled(False)
        getattr(self, f"{mode}_delete_btn").setEnabled(False)
        getattr(self, f"{mode}_waveform").clear("Deleted from library")
        self._refresh_library()
        status.setText(f"Deleted: {name}")
        apply_status_style(status, role="success")
        self._set_activity(f"Deleted voice '{name}' from the library.", role="success")
        self.save_card.set_value("Deleted", "warning")
        self._show_toast(f"Deleted voice: {name}", role="warning")
        self._refresh_visual_status()

    # Voice Library actions

    def _refresh_library(self):
        self.voice_mgr.library.load_all()
        selected_name = self._selected_voice_name()
        active_name = getattr(self.voice_mgr.active_profile, "name", "")
        self.voice_list.clear()
        names = self.voice_mgr.library.list_names()
        if not names:
            item = QListWidgetItem("No voices yet - generate a named voice above.")
            item.setData(Qt.UserRole, "")
            item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
            self.voice_list.addItem(item)
            self.lib_meta.setText(
                f"Saved in: {self.voice_mgr.library.base_dir}\n"
                "No voices saved yet. New generations auto-save here."
            )
            self._refresh_library_buttons()
            self._refresh_visual_status()
            return

        callbacks = {
            "play": self._play_voice_name,
            "active": self._set_active_voice_name,
            "delete": self._confirm_delete_voice_name,
        }
        current_item = None
        for name in names:
            p = self.voice_mgr.library.get(name)
            if not p:
                continue
            item = QListWidgetItem()
            item.setData(Qt.UserRole, name)
            card = VoiceCardWidget(
                p,
                is_active=(active_name == name),
                is_default=bool(p.is_default),
                callbacks=callbacks,
            )
            self.voice_list.addItem(item)
            item.setSizeHint(card.sizeHint())
            self.voice_list.setItemWidget(item, card)
            if selected_name == name:
                current_item = item
        if current_item is None and self.voice_list.count() > 0:
            current_item = self.voice_list.item(0)
        if current_item is not None:
            self.voice_list.setCurrentItem(current_item)
            self._show_voice_meta(str(current_item.data(Qt.UserRole) or ""))
        self._refresh_library_buttons()
        self._refresh_visual_status()

    def _selected_voice_name(self):
        cur = self.voice_list.currentItem()
        if not cur:
            return ""
        name = cur.data(Qt.UserRole)
        if name:
            return str(name)
        text = cur.text().replace(" *", "").replace(" [WAV]", "").strip()
        return text if self.voice_mgr.library.get(text) else ""

    def _refresh_library_buttons(self):
        buttons = getattr(self, "library_action_buttons", {})
        selected = self._selected_voice_name()
        profile = self.voice_mgr.library.get(selected) if selected else None
        for label, btn in buttons.items():
            if label in {"Import", "Open Folder"}:
                btn.setEnabled(True)
            elif label == "Play Voice":
                btn.setEnabled(bool(profile and profile.has_wav()))
            else:
                btn.setEnabled(bool(profile))

    def _show_voice_meta(self, name):
        p = self.voice_mgr.library.get(name)
        if not p:
            self.lib_meta.setText(
                f"Saved in: {self.voice_mgr.library.base_dir}\n"
                "Select a saved voice to see details."
            )
            return

        meta = (
            f"Mode: {p.mode.value} | Lang: {p.language} | "
            f"Created: {p.created[:10]}"
        )
        if p.mode == VoiceMode.CUSTOM:
            meta += f" | Speaker: {p.speaker_id}"
        meta += f"\nFolder: {self.voice_mgr.library.get_voice_dir(name)}"
        if p.has_wav():
            meta += f"\nWAV: {p.generated_wav}"
        else:
            meta += "\n(No generated WAV)"
        self.lib_meta.setText(meta)

    def _on_voice_item_selected(self, current, _previous=None):
        name = str(current.data(Qt.UserRole) or "") if current else ""
        self._show_voice_meta(name)
        self._refresh_library_buttons()

    def _on_voice_selected(self, text):
        name = text.replace(" *", "").replace(" [WAV]", "").strip()
        self._show_voice_meta(name)

    def _set_active(self):
        name = self._selected_voice_name()
        if not name:
            return
        self._set_active_voice_name(name)

    def _set_active_voice_name(self, name):
        self.voice_mgr.set_active_voice(name)
        self.workflow_stepper.set_state(
            active=None,
            complete={"Name", "Configure", "Generate", "Auto-save", "Test", "Active"},
        )
        self._refresh_library()
        self._set_status(f"Active voice: {name}", role="success")
        self._show_toast(f"Active voice: {name}", role="success")
        self._refresh_visual_status()

    def _rename_voice(self):
        old = self._selected_voice_name()
        if not old:
            return
        new, ok = QInputDialog.getText(self, "Rename", "New name:", text=old)
        if ok and new.strip():
            try:
                self.voice_mgr.library.rename(old, new.strip())
            except ValueError as exc:
                self._set_status(str(exc), role="error")
                return
            self._refresh_library()
            self._show_toast(f"Renamed voice: {new.strip()}", role="success")

    def _delete_voice(self):
        name = self._selected_voice_name()
        if not name:
            return
        self._confirm_delete_voice_name(name)

    def _confirm_delete_voice_name(self, name):
        reply = QMessageBox.question(
            self, "Delete Voice", f"Delete '{name}'?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self._delete_voice_name(name)

    def _delete_voice_name(self, name):
        if self.voice_mgr.delete_voice(name):
            for mode in ("design", "clone", "custom"):
                field = getattr(self, f"{mode}_name", None)
                if field is not None and field.text().strip() == name:
                    setattr(self, f"_last_{mode}_wav", None)
                    getattr(self, f"{mode}_play_btn").setEnabled(False)
                    getattr(self, f"{mode}_delete_btn").setEnabled(False)
                    getattr(self, f"{mode}_waveform").clear("Deleted from library")
            self._refresh_library()
            self._set_status(f"Deleted voice: {name}", role="success")
            self.save_card.set_value("Deleted", "warning")
            self._set_activity(f"Deleted voice '{name}' from the library.", role="success")
            self._show_toast(f"Deleted voice: {name}", role="warning")
            self._refresh_visual_status()

    def _set_default(self):
        name = self._selected_voice_name()
        if not name:
            return
        self.voice_mgr.library.set_default(name)
        self._refresh_library()
        self._set_status(f"Default voice: {name}", role="success")
        self._show_toast(f"Default voice: {name}", role="success")

    def _import_voice(self):
        d = QFileDialog.getExistingDirectory(self, "Import Voice Directory")
        if d:
            p = self.voice_mgr.library.import_profile(d)
            if p:
                self._refresh_library()
                self._set_status(f"Imported: {p.name}", role="success")
                self._show_toast(f"Imported voice: {p.name}", role="success")

    def _export_voice(self):
        name = self._selected_voice_name()
        if not name:
            return
        d = QFileDialog.getExistingDirectory(self, "Export To")
        if d:
            dest = Path(d) / name
            self.voice_mgr.library.export_profile(name, dest)
            self._set_status(f"Exported to: {dest}", role="success")
            self._show_toast(f"Exported voice: {name}", role="success")

    def _open_folder(self):
        self.voice_mgr.library.open_folder()

    def _play_selected(self):
        name = self._selected_voice_name()
        if not name:
            return
        self._play_voice_name(name)

    def _play_voice_name(self, name):
        p = self.voice_mgr.library.get(name)
        if p and p.has_wav():
            self.voice_mgr.play_wav(p.generated_wav)
            self._set_activity(f"Playing saved voice: '{name}'", role="success")
        else:
            self._set_status("No WAV file for this voice", role="error")
            self._show_toast("No WAV file for this voice", role="error")

    # Clone tab helpers

    def _browse_ref_audio(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Reference Audio", "",
            "Audio (*.wav *.mp3 *.flac *.ogg);;All Files (*)",
        )
        if path:
            self.clone_ref_path.setText(path)

    def _record_ref_audio(self):
        self.clone_status.setText("Recording 5 seconds...")
        apply_status_style(self.clone_status, role="error")

        def _do():
            try:
                import sounddevice as sd
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
            apply_status_style(self.clone_status, role="error")
        elif path:
            self.clone_ref_path.setText(path)
            self.clone_status.setText("Recording saved")
            apply_status_style(self.clone_status, role="success")

    # Engine / Devices / Metrics

    def _preferred_startup_backend(self) -> str:
        """Default to Qwen3-TTS when the local model settings enable CUDA."""
        return "qwen3-tts" if self._cuda_enabled_for_tts_default() else "pyttsx3"

    def _cuda_enabled_for_tts_default(self) -> bool:
        """Best-effort CUDA gate for startup voice defaults."""
        self._tts_cuda_skip_reason = ""
        try:
            settings_path = Path(__file__).resolve().parents[3] / "model_settings.json"
            if settings_path.is_file():
                data = json.loads(settings_path.read_text(encoding="utf-8"))
                backend = str(data.get("local_backend", "")).strip().upper()
                gpu_layers = int(data.get("srv_gpu_layers", data.get("gpu_layers", 0)) or 0)
                if backend == "CUDA" and gpu_layers != 0:
                    device, is_cpu_only = self._detect_tts_device()
                    if device == "cuda":
                        return True
                    existing_reason = getattr(self, "_tts_cuda_skip_reason", "")
                    if is_cpu_only:
                        self._tts_cuda_skip_reason = existing_reason or (
                            "TTS PyTorch cannot run CUDA on this GPU; install a "
                            "compatible CUDA-enabled torch build in the project .venv "
                            "for Qwen3-TTS GPU startup"
                        )
                    else:
                        self._tts_cuda_skip_reason = existing_reason or "TTS PyTorch cannot see a CUDA GPU"
                    if not getattr(self, "_tts_cuda_warning_logged", False):
                        self._tts_cuda_warning_logged = True
                        logger.warning(
                            "Model settings enable CUDA, but Qwen3-TTS cannot use CUDA: %s",
                            self._tts_cuda_skip_reason,
                        )
                    return False
        except Exception as exc:
            logger.debug("Error reading model_settings CUDA state: %s", exc)

        try:
            import torch
            if not getattr(getattr(torch, "version", None), "cuda", None):
                self._tts_cuda_skip_reason = "TTS PyTorch is CPU-only"
                return False
            available = bool(torch.cuda.is_available())
            if not available:
                self._tts_cuda_skip_reason = "TTS PyTorch cannot see a CUDA GPU"
            return available
        except Exception as exc:
            logger.debug("Error probing CUDA for TTS default: %s", exc)
            self._tts_cuda_skip_reason = str(exc)
            return False

    def _on_engine_changed(self, text):
        """Route engine selection through VoiceManager (single source of truth)."""
        engine_id = "pyttsx3" if "pyttsx3" in text.lower() else "qwen3-tts"
        if engine_id == "qwen3-tts":
            if self._qwen_server_ready(8000):
                self._qwen_activation_pending = False
                self._activate_qwen_backend(8000)
                return
            self._qwen_activation_pending = True
            self.voice_mgr.backend.set_qwen_server(
                self.qwen_url.text().strip() or "http://localhost:8000"
            )
            self._update_active_backend_label()
            self._set_status(
                "Starting Qwen3-TTS; using fallback until ready",
                role="warning",
            )
            if (
                not self._tts_process
                or self._tts_process.state() == QProcess.NotRunning
            ):
                self._start_tts_server()
            return
        self._qwen_activation_pending = False
        self.voice_mgr.set_backend(engine_id)
        self._update_active_backend_label()

    def _on_backend_changed(self, engine_id: str) -> None:
        """Sync UI when VoiceManager emits backend_changed."""
        self._update_active_backend_label()
        if hasattr(self, "activity_timeline"):
            if engine_id == "qwen3-tts":
                self.activity_timeline.set_state(active=None, complete={"Server"})
            else:
                self.activity_timeline.set_state(active="Server")
        if hasattr(self, "mode_card"):
            self._refresh_visual_status()
        # Keep engine_combo in sync if changed programmatically
        label = self.voice_mgr.active_backend_label
        if self.engine_combo.currentText() != label:
            self.engine_combo.blockSignals(True)
            self.engine_combo.setCurrentText(label)
            self.engine_combo.blockSignals(False)

    def _update_active_backend_label(self) -> None:
        """Refresh the active-backend indicator label."""
        if hasattr(self, "active_backend_lbl"):
            label = self.voice_mgr.active_backend_label
            fallback = self.voice_mgr.fallback_backend_name
            if (
                getattr(self, "_qwen_activation_pending", False)
                and self.voice_mgr.active_backend_name != "qwen3-tts"
            ):
                self.active_backend_lbl.setText(
                    f"Active: {label}  |  Preferred: Qwen3-TTS  |  "
                    f"Fallback: {fallback}  |  Starting"
                )
                apply_status_style(self.active_backend_lbl, role="warning")
                if hasattr(self, "mode_card"):
                    self._refresh_visual_status()
                return
            ready = self.voice_mgr.is_backend_ready()
            health = "Ready" if ready else "Unavailable"
            role = "success" if ready else "warning"
            self.active_backend_lbl.setText(
                f"Active: {label}  |  Fallback: {fallback}  |  {health}"
            )
            apply_status_style(self.active_backend_lbl, role=role)
            if hasattr(self, "mode_card"):
                self._refresh_visual_status()

    def _populate_input_devices(self):
        try:
            import sounddevice as sd
            for i, d in enumerate(sd.query_devices()):
                if d["max_input_channels"] > 0:
                    self.input_device.addItem(f"{d['name']} (#{i})", userData=i)
        except Exception as e:
            logger.warning(f"Error populating input devices: {e}")

    def _populate_output_devices(self):
        """Fill the output-device combobox with PortAudio sinks."""
        try:
            import sounddevice as sd
            for i, d in enumerate(sd.query_devices()):
                if d.get("max_output_channels", 0) > 0:
                    self.output_device.addItem(
                        f"{d['name']} (#{i})", userData=i
                    )
        except Exception as e:
            logger.warning(f"Error populating output devices: {e}")

    def _refresh_output_devices(self):
        """Re-enumerate output devices (e.g. after plugging in headphones)."""
        # Remember the current selection so we can restore it after refresh.
        current = self.output_device.currentData()
        try:
            self.output_device.blockSignals(True)
            self.output_device.clear()
            self.output_device.addItem("System Default", userData=None)
            self._populate_output_devices()
            # Restore by userData match
            for idx in range(self.output_device.count()):
                if self.output_device.itemData(idx) == current:
                    self.output_device.setCurrentIndex(idx)
                    break
        finally:
            self.output_device.blockSignals(False)
        self._refresh_output_route()
        self._show_toast("Output devices refreshed", role="success")

    def _on_output_device_changed(self, _index=None):
        """User picked a different output device - apply + persist."""
        data = self.output_device.currentData()
        label = self.output_device.currentText()
        backend = getattr(self.voice_mgr, "backend", None)
        if backend is not None and hasattr(backend, "set_output_device"):
            try:
                backend.set_output_device(data, label=label)
            except Exception as e:
                logger.warning(f"Error applying output device: {e}")

        # Persist to the core via the controller client, if available.
        client = getattr(self, "client", None)
        if client is not None and hasattr(client, "set_tts_output"):
            try:
                payload = {"device": data, "label": label}
                client.set_tts_output(payload)
            except Exception as e:
                logger.debug(f"Error persisting output device: {e}")
        self._refresh_output_route()
        self._show_toast(f"Output route: {label}", role="success")

    def _load_saved_output_device(self):
        """Fetch the persisted TTS output device from the core and select it."""
        client = getattr(self, "client", None)
        if client is None or not hasattr(client, "get_tts_output"):
            self._refresh_output_route()
            return
        try:
            saved = client.get_tts_output() or {}
        except Exception as e:
            logger.debug(f"Could not fetch saved TTS output device: {e}")
            self._refresh_output_route()
            return
        device = saved.get("device")
        if device in (None, "", -1):
            self._refresh_output_route()
            return  # already on "System Default"

        # Find the matching combobox entry by userData. If no match (user
        # unplugged the device or selected by name), fall back to string match
        # on the label so we at least show *something* meaningful.
        match_idx = -1
        for idx in range(self.output_device.count()):
            if self.output_device.itemData(idx) == device:
                match_idx = idx
                break
        if match_idx < 0:
            saved_label = str(saved.get("label") or "")
            if saved_label:
                for idx in range(self.output_device.count()):
                    if self.output_device.itemText(idx) == saved_label:
                        match_idx = idx
                        break

        if match_idx >= 0:
            # Push through the backend without sending a redundant save back
            # to the core (we're restoring what the core just told us).
            self.output_device.blockSignals(True)
            try:
                self.output_device.setCurrentIndex(match_idx)
            finally:
                self.output_device.blockSignals(False)
            backend = getattr(self.voice_mgr, "backend", None)
            if backend is not None and hasattr(backend, "set_output_device"):
                try:
                    backend.set_output_device(
                        self.output_device.currentData(),
                        label=self.output_device.currentText(),
                    )
                except Exception as e:
                    logger.warning(f"Error applying saved output device: {e}")
        else:
            logger.info(
                f"Saved TTS output device {device!r} not present on this system"
            )
        self._refresh_output_route()

    def _test_output_device(self):
        """Speak a short phrase through the currently selected output device."""
        backend = getattr(self.voice_mgr, "backend", None)
        if backend is None:
            self._set_status("No TTS backend available for test", role="warning")
            return
        phrase = "Output device test. If you can hear this, Revia is routed correctly."
        try:
            # Prefer the profile-aware path if there is an active profile so
            # the test actually uses Qwen3 if it's up. Otherwise fall through
            # to the pyttsx3 rendering which also honors the device setting.
            vp = getattr(self.voice_mgr, "active_profile", None)
            if vp is not None and hasattr(backend, "speak_from_profile"):
                backend.speak_from_profile(phrase, vp, emotion="neutral")
            else:
                import threading
                threading.Thread(
                    target=backend._speak_pyttsx3,
                    args=(phrase, 1.0, 1.0),
                    daemon=True,
                ).start()
            self._set_status(
                f"Testing output: {self.output_device.currentText()}",
                role="success",
            )
            self._show_toast(f"Testing output: {self.output_device.currentText()}", role="success")
        except Exception as e:
            logger.warning(f"Output test failed: {e}")
            self._set_status(f"Output test failed: {e}", role="error")
            self._show_toast(f"Output test failed: {e}", role="error")

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
            except Exception as e:
                logger.debug(f"Error disconnecting volume signal: {e}")
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

    def _show_toast(self, text, role="muted", timeout_ms=4200):
        if not hasattr(self, "toast_label"):
            return
        clean = str(text or "").strip()
        if not clean:
            self.toast_label.setVisible(False)
            return
        if len(clean) > 180:
            clean = clean[:177] + "..."
        self.toast_label.setText(clean)
        apply_status_style(self.toast_label, role=role)
        self.toast_label.setVisible(True)
        timer = getattr(self, "_toast_timer", None)
        if timer is not None:
            timer.start(timeout_ms)

    def _set_button_icon(self, button, icon_name, fallback="SP_FileIcon", tooltip=None):
        icon = _standard_icon(button, icon_name, fallback=fallback)
        if icon is not None:
            button.setIcon(icon)
            button.setIconSize(QSize(16, 16))
        if tooltip:
            button.setToolTip(tooltip)

    def _apply_button_icons(self):
        icon_specs = [
            (self.start_tts_btn, "SP_MediaPlay", "Start Qwen3-TTS"),
            (self.stop_tts_btn, "SP_MediaStop", "Stop Qwen3-TTS"),
            (self.design_generate_btn, "SP_DialogApplyButton", "Generate and auto-save"),
            (self.clone_generate_btn, "SP_DialogApplyButton", "Clone, generate, and auto-save"),
            (self.custom_generate_btn, "SP_DialogApplyButton", "Generate and auto-save"),
            (self.design_play_btn, "SP_MediaPlay", "Play generated voice"),
            (self.clone_play_btn, "SP_MediaPlay", "Play generated voice"),
            (self.custom_play_btn, "SP_MediaPlay", "Play generated voice"),
            (self.design_delete_btn, "SP_TrashIcon", "Delete saved voice"),
            (self.clone_delete_btn, "SP_TrashIcon", "Delete saved voice"),
            (self.custom_delete_btn, "SP_TrashIcon", "Delete saved voice"),
            (self.output_test_btn, "SP_MediaPlay", "Test selected output device"),
            (self.output_refresh_btn, "SP_BrowserReload", "Refresh output devices"),
            (self.mic_test_btn, "SP_MediaPlay", "Test microphone input"),
        ]
        for button, icon_name, tooltip in icon_specs:
            self._set_button_icon(
                button,
                icon_name,
                fallback="SP_DialogDiscardButton" if "Trash" in icon_name else "SP_FileIcon",
                tooltip=tooltip,
            )
        for label, btn in getattr(self, "library_action_buttons", {}).items():
            icon_name = {
                "Set Active": "SP_DialogApplyButton",
                "Rename": "SP_FileDialogDetailedView",
                "Delete": "SP_TrashIcon",
                "Set Default": "SP_DialogApplyButton",
                "Import": "SP_DirOpenIcon",
                "Export": "SP_DialogSaveButton",
                "Open Folder": "SP_DirOpenIcon",
                "Play Voice": "SP_MediaPlay",
            }.get(label, "SP_FileIcon")
            self._set_button_icon(
                btn,
                icon_name,
                fallback="SP_DialogDiscardButton" if label == "Delete" else "SP_FileIcon",
                tooltip=label,
            )

    def _refresh_output_route(self):
        if not hasattr(self, "output_device"):
            return
        label = self.output_device.currentText() or "System Default"
        route = f"Revia Voice -> {label}"
        if hasattr(self, "output_route_lbl"):
            self.output_route_lbl.setText(route)
            apply_status_style(self.output_route_lbl, role="success")
        if hasattr(self, "output_card"):
            self.output_card.set_value(label, "success")

    def _refresh_visual_status(self):
        prof = self.voice_mgr.active_profile
        if hasattr(self, "voice_card"):
            if prof is None:
                self.voice_card.set_value("Fallback", "warning")
            else:
                wav = "WAV ready" if prof.has_wav() else "WAV missing"
                role = "success" if prof.has_wav() else "warning"
                self.voice_card.set_value(f"{prof.name} | {wav}", role)

        if hasattr(self, "mode_card"):
            mode = self._current_mode_id() if hasattr(self, "mode_tabs") else None
            idx = self._MODE_TAB_INDEX.get(mode, -1)
            mode_label = self._MODE_TAB_BASE_LABEL.get(idx, "Mode")
            engine = self.voice_mgr.active_backend_name
            supported = set(getattr(self, "_caps_modes", []) or [])
            if engine == "pyttsx3":
                self.mode_card.set_value("Fallback engine", "warning")
            elif not supported:
                self.mode_card.set_value("Detecting modes", "warning")
            elif mode in supported:
                self.mode_card.set_value(f"{mode_label} ready", "success")
            else:
                self.mode_card.set_value(f"{mode_label} unavailable", "error")
        self._refresh_output_route()

    # ------------------------------------------------------------------
    # V4 visibility: Active voice / Server capabilities / Activity
    # ------------------------------------------------------------------

    _MODE_TAB_INDEX = {"design": 0, "clone": 1, "custom": 2}
    _MODE_TAB_BASE_LABEL = {
        0: "Voice Design",
        1: "Voice Clone (Base)",
        2: "TTS (CustomVoice)",
    }
    _TAB_TO_MODE = {0: "design", 1: "clone", 2: "custom"}

    def _refresh_active_voice_label(self):
        """Repaint the \"Active Voice\" line from the current voice profile."""
        prof = self.voice_mgr.active_profile
        if prof is None:
            self.active_voice_lbl.setText(
                "Active Voice: (none) - synthesis will use the system fallback"
            )
            apply_status_style(self.active_voice_lbl, role="warning")
            self._refresh_visual_status()
            return
        mode_str = getattr(prof.mode, "value", str(prof.mode))
        bits = [f"Active Voice: {prof.name}", f"Mode: {mode_str}"]
        if prof.language:
            bits.append(f"Lang: {prof.language}")
        if mode_str == "custom" and getattr(prof, "speaker_id", ""):
            bits.append(f"Speaker: {prof.speaker_id}")
        if prof.has_wav():
            bits.append("WAV: ready")
        else:
            bits.append("WAV: missing")
        self.active_voice_lbl.setText("  |  ".join(bits))
        role = "success" if prof.has_wav() else "warning"
        apply_status_style(self.active_voice_lbl, role=role)
        self._refresh_visual_status()

    def _on_voice_changed(self, _name):
        self._refresh_active_voice_label()
        self._refresh_capabilities_ui()
        self._refresh_library()

    def _on_capabilities_changed(self, payload):
        """Stash the latest server capabilities and repaint the UI."""
        if isinstance(payload, dict):
            self._caps_variant = str(payload.get("variant") or "unknown")
            self._caps_label = str(payload.get("label") or "")
            self._caps_modes = [str(m) for m in (payload.get("modes") or [])]
            self._caps_api_names = [str(n) for n in (payload.get("api_names") or [])]
        self._refresh_capabilities_ui()
        self._refresh_visual_status()

    def _refresh_capabilities_ui(self):
        """Repaint the capabilities label + per-tab supported badges."""
        engine = self.voice_mgr.active_backend_name
        if engine == "pyttsx3":
            self.caps_lbl.setText(
                "Server: pyttsx3 fallback engine (offline). "
                "Modes Design / Clone / CustomVoice need a Qwen3-TTS server."
            )
            apply_status_style(self.caps_lbl, role="warning")
            self._update_mode_tab_badges(supported=set(),
                                         active_mode=None)
            self._refresh_visual_status()
            return

        if not self._caps_modes and self._caps_variant in ("", "unknown"):
            self.caps_lbl.setText(
                f"Server: {self._caps_label or 'probing...'} - capabilities not yet detected"
            )
            apply_status_style(self.caps_lbl, role="warning")
            self._update_mode_tab_badges(supported=set(),
                                         active_mode=None)
            self._refresh_visual_status()
            return

        modes_pretty = ", ".join(m.title() for m in self._caps_modes) or "none"
        api_pretty = ", ".join(self._caps_api_names) or "none advertised"
        self.caps_lbl.setText(
            f"Server: {self._caps_label}.  Supports: {modes_pretty}.  "
            f"Endpoints: {api_pretty}"
        )
        apply_status_style(self.caps_lbl, role="success")
        self._update_mode_tab_badges(
            supported=set(self._caps_modes),
            active_mode=self._current_mode_id(),
        )
        self._refresh_visual_status()

    def _current_mode_id(self):
        idx = self.mode_tabs.currentIndex()
        return self._TAB_TO_MODE.get(idx)

    def _update_mode_tab_badges(self, supported, active_mode=None):
        """Decorate each mode tab with its support state.

        ``(active)`` for the selected & supported tab,
        ``(supported)`` for other supported modes,
        ``(N/A)`` for unsupported modes.
        """
        for idx, base in self._MODE_TAB_BASE_LABEL.items():
            mode = self._TAB_TO_MODE.get(idx)
            if not supported:
                badge = ""
                tip = "Qwen3-TTS capabilities have not been detected yet."
            elif mode in supported and mode == active_mode:
                badge = "  (active)"
                tip = f"{base} is supported by the current Qwen3-TTS server."
            elif mode in supported:
                badge = "  (supported)"
                tip = f"{base} is available on the current Qwen3-TTS server."
            else:
                badge = "  (N/A)"
                tip = f"{base} is not advertised by the current Qwen3-TTS server."
            self.mode_tabs.setTabText(idx, base + badge)
            self.mode_tabs.setTabToolTip(idx, tip)

    def _on_mode_tab_changed(self, _index):
        self._refresh_capabilities_ui()
        self._refresh_visual_status()

    # ---- Live activity indicator -----------------------------------

    def _set_activity(self, text, role="muted"):
        if hasattr(self, "activity_lbl"):
            self.activity_lbl.setText(text)
            apply_status_style(self.activity_lbl, role=role)

    def _on_synthesis_started(self):
        prof = self.voice_mgr.active_profile
        voice_name = prof.name if prof else "(default)"
        mode = self._current_mode_id() or "synthesis"
        self._set_activity(
            f"Synthesizing ({mode}) using '{voice_name}'...",
            role="warning",
        )
        self.activity_timeline.set_state(active="Generating", complete={"Server"})

    def _on_synthesis_finished_ui(self, metrics):
        try:
            secs = float(getattr(metrics, "synthesis_time", 0.0))
        except (TypeError, ValueError):
            secs = 0.0
        self._set_activity(
            f"Synthesis done in {secs:.2f}s - playing back...",
            role="success",
        )
        self.activity_timeline.set_state(
            active="Playing",
            complete={"Server", "Generating", "Saved"},
        )

    def _on_playback_started(self):
        prof = self.voice_mgr.active_profile
        voice_name = prof.name if prof else "(default)"
        self._set_activity(f"Playing voice: '{voice_name}'", role="success")
        self.activity_timeline.set_state(
            active="Playing",
            complete={"Server", "Generating", "Saved"},
        )

    def _on_playback_finished(self):
        self._set_activity("Idle", role="muted")
        self.activity_timeline.set_state(
            active=None,
            complete={"Server", "Generating", "Saved", "Playing"},
        )
        self.workflow_stepper.set_state(
            active=None,
            complete={"Name", "Configure", "Generate", "Auto-save", "Test", "Active"},
        )

    def _on_playback_interrupted(self):
        self._set_activity("Playback interrupted", role="warning")
        self.activity_timeline.set_state(
            active="Playing",
            complete={"Server", "Generating", "Saved"},
            failed={"Playing"},
        )
        self.workflow_stepper.set_state(
            active="Test",
            complete={"Name", "Configure", "Generate", "Auto-save", "Active"},
            failed={"Test"},
        )

    def _on_synthesis_error(self, message):
        text = str(message or "TTS error")
        # Trim very long messages so the activity line stays readable.
        if len(text) > 220:
            text = text[:217] + "..."
        self._set_activity(f"Synthesis failed: {text}", role="error")
        self.activity_timeline.set_state(failed={"Generating"})
        self._show_toast(f"Synthesis failed: {text}", role="error")

    def _on_backend_status_for_activity(self, message):
        text = str(message or "")
        low = text.lower()
        if "unavailable" in low or "is not available" in low:
            short = text if len(text) <= 220 else text[:217] + "..."
            self._set_activity(short, role="error")
            self.activity_timeline.set_state(failed={"Server"})

    def _on_voice_status(self, message):
        text = str(message or "").strip()
        if not text:
            return
        self._set_status(text, role="warning")
        self._log_voice_startup(text)
        self._refresh_visual_status()

    def _set_status(self, text, color=None, role: str = "muted"):
        """Update the status label.

        Prefer passing a semantic ``role`` (``"success"``, ``"warning"``,
        ``"error"``, ``"muted"``).  The legacy ``color`` hex arg is still
        accepted for backwards compatibility.
        """
        self.status_label.setText(f"Status: {text}")
        if color and "#" in str(color):
            apply_status_style(self.status_label, f"color: {color};")
        else:
            apply_status_style(self.status_label, role=role)

    def _on_core_connection(self, connected):
        if connected:
            self._set_status("Core online", role="success")

    def _apply_vol_bar_style(self):
        """Apply theme-aware volume bar stylesheet using current theme tokens."""
        from PySide6.QtWidgets import QApplication
        from PySide6.QtGui import QColor
        app = QApplication.instance()
        tokens = app.property("reviaThemeTokens") if app else None
        border = "#555"
        bg = "#1a1a2e"
        if isinstance(tokens, dict):
            border = tokens.get("Border", border)
            bg = tokens.get("Surface", bg)
        self.vol_bar.setStyleSheet(
            f"QProgressBar{{border:1px solid {border};border-radius:4px;background:{bg};}}"
            "QProgressBar::chunk{background:qlineargradient("
            "x1:0,y1:0,x2:1,y2:0,stop:0 #00cc44,stop:0.7 #cccc00,stop:1 #cc3040);"
            "border-radius:3px;}"
        )

    def _on_theme_changed(self, _theme_id):
        """Re-apply dynamic widget styles when theme changes."""
        self._apply_vol_bar_style()

    def auto_start_on_launch(self):
        """Start voice services on app launch when local capabilities allow it."""
        self._auto_start_stt_on_launch()
        QTimer.singleShot(350, self._auto_start_qwen_tts_on_launch)

    def _auto_start_stt_on_launch(self):
        if not self.audio_service:
            self._set_status("STT auto-start skipped: no audio service", role="warning")
            return

        check = getattr(self.audio_service, "stt_startup_check", None)
        ok, reason = check() if callable(check) else (self.audio_service.is_stt_available(), "")
        if not ok:
            self._set_status(f"STT auto-start skipped: {reason}", role="warning")
            self._log_voice_startup(f"STT auto-start skipped: {reason}")
            return

        self._on_input_device_changed(self.input_device.currentIndex())
        if self.ptt_mode.currentText() != "Always Listening (VAD)":
            self.ptt_mode.setCurrentText("Always Listening (VAD)")

        started = self.audio_service.start_listening(always=True)
        if started is False:
            self._set_status("STT auto-start failed", role="warning")
            self._log_voice_startup("STT auto-start failed")
            return

        self._set_status("STT always-listening started", role="success")
        self._log_voice_startup("STT always-listening started")

    def _auto_start_qwen_tts_on_launch(self):
        port = 8000
        qwen_requested = bool(getattr(self, "_qwen_activation_pending", False))
        if not qwen_requested and not self._cuda_enabled_for_tts_default():
            reason = self._tts_cuda_skip_reason or "CUDA not enabled"
            self._log_voice_startup(f"Qwen3-TTS auto-start skipped: {reason}")
            return

        if self._tts_process and self._tts_process.state() != QProcess.NotRunning:
            self._log_voice_startup("Qwen3-TTS already starting")
            return

        if self._qwen_server_ready(port):
            self._activate_qwen_backend(port)
            self.tts_server_status.setText(f"Running on :{port}")
            apply_status_style(self.tts_server_status, role="success")
            self._set_status("Qwen3-TTS ready on launch", role="success")
            self._log_voice_startup(f"Qwen3-TTS attached to existing server on :{port}")
            return

        if self._port_has_listener(port):
            msg = f"Port :{port} busy; Qwen3-TTS auto-start skipped"
            self.tts_server_status.setText(msg)
            apply_status_style(self.tts_server_status, role="warning")
            self._set_status(msg, role="warning")
            self._log_voice_startup(msg)
            return

        if not self._resolve_qwen_module():
            msg = "Qwen3-TTS module not found; auto-start skipped"
            self.tts_server_status.setText(msg)
            apply_status_style(self.tts_server_status, role="warning")
            self._set_status(msg, role="warning")
            self._log_voice_startup(msg)
            return

        self._qwen_activation_pending = True
        self._update_active_backend_label()
        self._set_status("Starting Qwen3-TTS on launch", role="warning")
        self._log_voice_startup("Starting Qwen3-TTS on launch")
        self._start_tts_server()

    def _on_input_device_changed(self, _index=None):
        if self.audio_service:
            self.audio_service.set_input_device(self.input_device.currentData())

    def _activate_qwen_backend(self, port=8000):
        self._qwen_activation_pending = False
        url = f"http://localhost:{port}"
        self.voice_mgr.backend.reset_qwen_connection()
        if self.qwen_url.text().strip().rstrip("/") != url:
            self.qwen_url.setText(url)
        else:
            self.voice_mgr.backend.set_qwen_server(url)
        if self.voice_mgr.active_backend_name != "qwen3-tts":
            self.voice_mgr.set_backend("qwen3-tts")
        else:
            self._update_active_backend_label()
        if hasattr(self, "activity_timeline"):
            self.activity_timeline.set_state(active=None, complete={"Server"})
        self._refresh_visual_status()

    def _select_pyttsx3_backend(self):
        self._qwen_activation_pending = False
        if self.voice_mgr.active_backend_name != "pyttsx3":
            self.voice_mgr.set_backend("pyttsx3")
        else:
            self._update_active_backend_label()
        if hasattr(self, "engine_combo") and self.engine_combo.currentText() != "pyttsx3":
            self.engine_combo.blockSignals(True)
            self.engine_combo.setCurrentText("pyttsx3")
            self.engine_combo.blockSignals(False)
        if hasattr(self, "activity_timeline"):
            self.activity_timeline.set_state(active="Server")
        self._refresh_visual_status()

    def _qwen_server_ready(self, port=8000):
        try:
            import urllib.request
            with urllib.request.urlopen(
                f"http://localhost:{port}/gradio_api/info",
                timeout=1.5,
            ) as response:
                return 200 <= int(getattr(response, "status", 200)) < 300
        except Exception:
            return False

    def _port_has_listener(self, port):
        try:
            import socket
            with socket.create_connection(("127.0.0.1", int(port)), timeout=0.5):
                return True
        except OSError:
            return False

    def _log_voice_startup(self, message):
        try:
            self.event_bus.log_entry.emit(f"[Voice] {message}")
        except Exception:
            logger.debug("[Voice] %s", message)

    # Qwen3-TTS Local Server Management

    def _start_tts_server(self, _checked=False, force_device=None):
        if self._tts_process and self._tts_process.state() != QProcess.NotRunning:
            self.tts_server_status.setText("Already running")
            return
        if force_device is None:
            self._tts_cuda_retry_attempted = False
        self.voice_mgr.backend.reset_qwen_connection()

        model_key = self.qwen_model_combo.currentText()
        model_id = QWEN_MODELS.get(model_key, "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
        port = "8000"
        self._tts_cuda_failure_seen = False
        self._tts_last_lines = []

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
                print(f"[TTS-SRV] GUI CUDA avail: ERROR - {_ce}")
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
            self._select_pyttsx3_backend()
            apply_status_style(self.tts_server_status, role="error")
            self.tts_server_status.setText(
                "Qwen3-TTS module not found. Install package, then retry."
            )
            self.start_tts_btn.setEnabled(True)
            self.stop_tts_btn.setEnabled(False)
            return

        args, device_label = self._build_tts_server_args(
            qwen_module,
            model_id,
            port,
            force_device=force_device,
        )
        self._tts_last_device_label = device_label

        # If using CPU, hide all GPUs from the subprocess so PyTorch/CUDA
        # can't accidentally try to initialise a CUDA device and crash.
        if device_label == "CPU":
            env.insert("CUDA_VISIBLE_DEVICES", "")
        else:
            if _env_flag("REVIA_QWEN_TTS_CUDA_DEBUG"):
                # Opt-in diagnostics only; this serializes CUDA work and slows synthesis.
                env.insert("CUDA_LAUNCH_BLOCKING", "1")
            else:
                env.remove("CUDA_LAUNCH_BLOCKING")
            env.insert("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        self._tts_process.setProcessEnvironment(env)
        self._qwen_activation_pending = True
        self._update_active_backend_label()
        self.tts_server_status.setText(f"Loading {model_key} ({device_label})...")
        apply_status_style(self.tts_server_status, role="warning")
        if hasattr(self, "activity_timeline"):
            self.activity_timeline.set_state(active="Server")
        self._show_toast(f"Starting Qwen3-TTS ({device_label})", role="warning")
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

    def _build_tts_server_args(self, qwen_module: str, model_id: str, port: str, force_device=None):
        """Build launch args and prefer CUDA when available."""
        override = str(force_device or "").strip().lower()
        if not override:
            override = str(os.environ.get("REVIA_QWEN_TTS_DEVICE", "")).strip().lower()
        if override in ("cpu", "cuda"):
            device, is_cpu_only = override, override == "cpu"
        else:
            device, is_cpu_only = self._detect_tts_device()
        model_args = [
            model_id,
            "--port",
            port,
            "--ip",
            QWEN_TTS_BIND_HOST,
        ]
        if self._should_disable_tts_flash_attn(device, is_cpu_only):
            model_args.append("--no-flash-attn")

        dtype = str(os.environ.get("REVIA_QWEN_TTS_DTYPE", "")).strip().lower()
        if dtype in {"bfloat16", "bf16", "float16", "fp16", "float32", "fp32"}:
            if self._qwen_cli_supports_flag(qwen_module, "--dtype"):
                model_args.extend(["--dtype", dtype])

        if is_cpu_only:
            # PyTorch is not compiled with CUDA.  Any call to torch.cuda.*
            # raises AssertionError unconditionally CUDA_VISIBLE_DEVICES
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

    def _should_disable_tts_flash_attn(self, device: str, is_cpu_only: bool) -> bool:
        """Keep FlashAttention enabled only when the local install can use it."""
        if is_cpu_only or str(device).lower().startswith("cpu"):
            return True
        if _env_flag("REVIA_QWEN_TTS_DISABLE_FLASH_ATTN"):
            return True
        if _env_flag("REVIA_QWEN_TTS_FORCE_FLASH_ATTN"):
            return False
        return not self._tts_flash_attn_available()

    @staticmethod
    def _tts_flash_attn_available() -> bool:
        try:
            return importlib.util.find_spec("flash_attn") is not None
        except (ImportError, ValueError):
            return False

    def _create_tts_launcher(self, qwen_module: str) -> str:
        """Write a temp launcher that patches torch.cuda before running qwen_tts.

        CPU-only PyTorch builds can still expose a torch.cuda namespace, but
        any accidental CUDA touch can raise "Torch not compiled with CUDA".
        This wrapper forces the CUDA probes used by Qwen to report CPU before
        the demo module imports torch.

        Returns the path to the temp script (caller is responsible for
        deleting it via self._tts_launcher_tmp when the server stops).
        """
        import tempfile
        code = (
            "import os\n"
            "import sys\n"
            "os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')\n"
            "try:\n"
            "    import torch as _t\n"
            "    def _false(*_a, **_k):\n"
            "        return False\n"
            "    def _zero(*_a, **_k):\n"
            "        return 0\n"
            "    def _none(*_a, **_k):\n"
            "        return None\n"
            "    _t.cuda.is_available = _false\n"
            "    _t.cuda.device_count = _zero\n"
            "    _t.cuda.is_initialized = _false\n"
            "    _t.cuda.current_device = _zero\n"
            "    _t.cuda.empty_cache = _none\n"
            "    _t.cuda.set_device = _none\n"
            "    if hasattr(_t, 'Tensor'):\n"
            "        _t.Tensor.cuda = lambda self, device=None, non_blocking=False, memory_format=None: self\n"
            "    if hasattr(_t, 'nn') and hasattr(_t.nn, 'Module'):\n"
            "        _t.nn.Module.cuda = lambda self, device=None: self\n"
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
            means the subprocess needs the CPU-safe launcher. That includes
            true CPU-only torch builds and CUDA builds that cannot execute on
            the installed GPU architecture.
        """
        try:
            import torch
            if not getattr(getattr(torch, "version", None), "cuda", None):
                if self is not None:
                    self._tts_cuda_skip_reason = "TTS PyTorch is CPU-only"
                return "cpu", True
            # torch.cuda.is_available() raises AssertionError on CPU-only
            # torch builds ("Torch not compiled with CUDA enabled"), so
            # wrap it separately to avoid that surfacing to the user.
            try:
                if torch.cuda.is_available():
                    try:
                        import warnings
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            probe = torch.empty(1, device="cuda")
                            probe.zero_()
                            torch.cuda.synchronize()
                    except Exception as exc:
                        detail = str(exc).splitlines()[0]
                        reason = (
                            "TTS CUDA probe failed; this PyTorch build may not "
                            f"support your GPU: {detail}"
                        )
                        if self is not None:
                            self._tts_cuda_skip_reason = reason
                        logger.debug(reason)
                        return "cpu", True
                    return "cuda", False
                return "cpu", False   # CUDA compiled but no GPU attached
            except (AssertionError, RuntimeError):
                if self is not None:
                    self._tts_cuda_skip_reason = "TTS PyTorch cannot initialize CUDA"
                return "cpu", True    # CUDA not compiled into this torch
        except Exception as e:
            logger.debug(f"Error detecting CUDA: {e}")
        return "cpu", False

    def _qwen_cli_supports_flag(self, qwen_module: str, flag: str) -> bool:
        """Check whether qwen_tts demo CLI accepts a given option."""
        cache_key = (qwen_module, flag)
        cache = getattr(self, "_qwen_cli_flag_cache", None)
        if cache is None:
            cache = {}
            self._qwen_cli_flag_cache = cache
        if cache_key in cache:
            return cache[cache_key]
        try:
            spec = importlib.util.find_spec(qwen_module)
            origin = Path(getattr(spec, "origin", "") or "")
            if origin.is_file():
                supports = flag in origin.read_text(encoding="utf-8", errors="ignore")
                cache[cache_key] = supports
                return supports
        except Exception as e:
            logger.debug(f"Error checking Qwen CLI source support: {e}")
        try:
            import subprocess
            result = subprocess.run(
                [self._resolve_python_exe(), "-m", qwen_module, "--help"],
                capture_output=True,
                text=True,
                timeout=8,
            )
            supports = flag in ((result.stdout or "") + (result.stderr or ""))
            cache[cache_key] = supports
            return supports
        except Exception as e:
            logger.debug(f"Error checking Qwen CLI support: {e}")
            cache[cache_key] = False
            return False

    def _poll_tts_ready(self):
        """Check if Gradio server is actually accepting connections."""
        self._tts_poll_count += 1
        port = 8000
        try:
            import requests
            r = requests.get(f"http://localhost:{port}/gradio_api/info", timeout=2)
            if r.ok:
                self._activate_qwen_backend(port)
                self.tts_server_status.setText(f"Running on :{port}")
                apply_status_style(self.tts_server_status, role="success")
                self._tts_ready_timer.stop()
                self._set_status("Qwen3-TTS ready", role="success")
                self._show_toast("Qwen3-TTS ready", role="success")
                return
        except Exception as e:
            logger.debug(f"TTS server not ready: {e}")
        # Update status with progress dots
        dots = "." * (self._tts_poll_count % 4)
        model_key = self.qwen_model_combo.currentText()
        self.tts_server_status.setText(f"Loading {model_key}{dots}")

    def _stop_tts_server(self):
        self._qwen_activation_pending = False
        if self._tts_ready_timer:
            self._tts_ready_timer.stop()
        if self._tts_process and self._tts_process.state() != QProcess.NotRunning:
            self._tts_process.kill()
            self._tts_process.waitForFinished(3000)
        self._kill_port(8000)
        self.voice_mgr.backend.reset_qwen_connection()
        self._cleanup_tts_launcher()
        self.tts_server_status.setText("Stopped")
        clear_status_role(self.tts_server_status)
        self.start_tts_btn.setEnabled(True)
        self.stop_tts_btn.setEnabled(False)
        self._select_pyttsx3_backend()
        self._show_toast("Qwen3-TTS stopped; using fallback", role="warning")

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
            if self._is_cuda_tts_failure(line):
                self._tts_cuda_failure_seen = True
            if "Running on" in line:
                self._activate_qwen_backend(8000)
                self.tts_server_status.setText("Running on :8000")
                apply_status_style(self.tts_server_status, role="success")
                self._show_toast("Qwen3-TTS ready", role="success")
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
                    if self._is_cuda_tts_failure(line):
                        self._tts_cuda_failure_seen = True

        print(f"[TTS-SRV] Exited: code={exit_code}")
        if self._tts_ready_timer:
            self._tts_ready_timer.stop()
        self.start_tts_btn.setEnabled(True)
        self.stop_tts_btn.setEnabled(False)
        self._cleanup_tts_launcher()

        if exit_code == 0:
            self._select_pyttsx3_backend()
            self.tts_server_status.setText("Stopped")
            clear_status_role(self.tts_server_status)
            self._show_toast("Qwen3-TTS stopped; using fallback", role="warning")
        else:
            if self._should_retry_qwen_on_cpu():
                self._tts_cuda_retry_attempted = True
                self._qwen_activation_pending = True
                self._update_active_backend_label()
                msg = "Qwen3-TTS CUDA failed; retrying on CPU"
                self.tts_server_status.setText(msg)
                apply_status_style(self.tts_server_status, role="warning")
                self._set_status(msg, role="warning")
                self._show_toast(msg, role="warning")
                self._log_voice_startup(
                    "Qwen3-TTS CUDA failed with a PyTorch device-side assert; "
                    "retrying once on CPU"
                )
                QTimer.singleShot(
                    500,
                    lambda: self._start_tts_server(force_device="cpu"),
                )
                return
            self._select_pyttsx3_backend()
            # Show the last error line in the status label so the user can
            # diagnose the crash without opening the console.
            error_hint = ""
            for line in reversed(self._tts_last_lines):
                if line:
                    error_hint = line[:120]
                    break
            apply_status_style(self.tts_server_status, role="error")
            if error_hint:
                self.tts_server_status.setText(
                    f"Exited ({exit_code}): {error_hint}"
                )
            else:
                self.tts_server_status.setText(
                    f"Exited ({exit_code}) - check console for details"
                )
            self._show_toast("Qwen3-TTS exited; using fallback", role="error")

    def _clean_tts_log_line(self, line: str) -> str:
        """Normalize process output for UI display by removing ANSI escapes."""
        return ANSI_ESCAPE_RE.sub("", line).strip()

    def _append_tts_log_line(self, line: str):
        """Store meaningful process output lines for crash hints."""
        self._tts_last_lines.append(line)
        if len(self._tts_last_lines) > 20:
            self._tts_last_lines.pop(0)

    def _is_cuda_tts_failure(self, line: str) -> bool:
        text = str(line or "").lower()
        return any(
            marker in text
            for marker in (
                "device-side assert",
                "torch_use_cuda_dsa",
                "cuda error",
                "cublas",
                "cudnn",
            )
        )

    def _should_retry_qwen_on_cpu(self) -> bool:
        return (
            self._tts_last_device_label.upper() == "CUDA"
            and self._tts_cuda_failure_seen
            and not self._tts_cuda_retry_attempted
        )

    def _cleanup_tts_launcher(self):
        if self._tts_launcher_tmp:
            try:
                import os as _os
                _os.unlink(self._tts_launcher_tmp)
            except OSError:
                pass
            self._tts_launcher_tmp = None

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
        except Exception as e:
            logger.warning(f"Error killing port listener: {e}")
