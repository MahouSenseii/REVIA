"""
REVIA VTube Studio Tab
======================

Settings panel for the VTube Studio Live2D integration.

Features
--------
- Host / port configuration
- Connect / disconnect buttons with live status
- Auth approval guidance (user must click OK in VTube Studio)
- Emotion preview — click any emotion to trigger it on the avatar
- Lip sync toggle and sensitivity slider
- Idle drift toggle
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QSpinBox, QPushButton,
    QScrollArea, QCheckBox, QGroupBox, QGridLayout,
    QSizePolicy,
)
from PySide6.QtGui import QFont

from app.vtube_service import VTubeService
from app.ui_status import apply_status_style, clear_status_role
from gui.widgets.settings_card import SettingsCard


_EMOTIONS = [
    "happy", "excited", "curious", "sad", "angry",
    "frustrated", "fear", "lonely", "concerned",
    "confident", "nervous", "amused", "neutral",
]


class VtubeTab(QScrollArea):
    """VTube Studio integration settings and controls."""

    def __init__(self, event_bus, client, vtube_service: VTubeService | None = None, parent=None):
        super().__init__(parent)
        self.event_bus = event_bus
        self.client = client
        self.vtube_service = vtube_service
        self.setWidgetResizable(True)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        header = QLabel("VTube Studio")
        header.setObjectName("tabHeader")
        header.setFont(QFont("Segoe UI", 12, QFont.Bold))
        layout.addWidget(header)

        sub = QLabel("Connect Revia's emotion and speech to a Live2D avatar via VTube Studio.")
        sub.setObjectName("cardSubText")
        sub.setWordWrap(True)
        sub.setFont(QFont("Segoe UI", 9))
        layout.addWidget(sub)

        # --- Connection card ---
        conn_card = SettingsCard("Connection", subtitle="VTube Studio WebSocket", icon="V")

        conn_form = QFormLayout()

        self._host_edit = QLineEdit("127.0.0.1")
        self._host_edit.setPlaceholderText("127.0.0.1")
        conn_form.addRow("Host:", self._host_edit)

        self._port_spin = QSpinBox()
        self._port_spin.setRange(1, 65535)
        self._port_spin.setValue(8001)
        conn_form.addRow("Port:", self._port_spin)
        conn_card.add_layout(conn_form)

        conn_btn_row = QHBoxLayout()
        self._btn_connect = QPushButton("Connect")
        self._btn_connect.setObjectName("primaryBtn")
        self._btn_connect.clicked.connect(self._on_connect)
        conn_btn_row.addWidget(self._btn_connect)

        self._btn_disconnect = QPushButton("Disconnect")
        self._btn_disconnect.setObjectName("secondaryBtn")
        self._btn_disconnect.setEnabled(False)
        self._btn_disconnect.clicked.connect(self._on_disconnect)
        conn_btn_row.addWidget(self._btn_disconnect)
        conn_btn_row.addStretch()
        conn_card.add_layout(conn_btn_row)

        self._status_label = QLabel("Status: Disconnected")
        self._status_label.setObjectName("metricLabel")
        self._status_label.setFont(QFont("Consolas", 9))
        conn_card.add_widget(self._status_label)

        self._auth_hint = QLabel(
            "When prompted, open VTube Studio → Plugins → allow REVIA Neural Assistant."
        )
        self._auth_hint.setWordWrap(True)
        self._auth_hint.setFont(QFont("Segoe UI", 9))
        self._auth_hint.setObjectName("cardSubText")
        self._auth_hint.hide()
        conn_card.add_widget(self._auth_hint)

        layout.addWidget(conn_card)

        # --- Emotion preview card ---
        emo_card = SettingsCard("Emotion preview", subtitle="Click to trigger on avatar", icon="E")

        emo_info = QLabel("Fires directly on the Live2D model — useful for testing expression mappings.")
        emo_info.setFont(QFont("Segoe UI", 8))
        emo_info.setObjectName("cardSubText")
        emo_info.setWordWrap(True)
        emo_card.add_widget(emo_info)

        emo_grid = QGridLayout()
        emo_grid.setSpacing(6)
        cols = 4
        for i, emo in enumerate(_EMOTIONS):
            btn = QPushButton(emo.title())
            btn.setObjectName("secondaryBtn")
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn.clicked.connect(lambda checked=False, e=emo: self._preview_emotion(e))
            emo_grid.addWidget(btn, i // cols, i % cols)
        emo_card.add_layout(emo_grid)
        layout.addWidget(emo_card)

        # --- Behaviour card ---
        beh_card = SettingsCard("Behaviour", subtitle="Lip sync and movement", icon="B")

        self._lipsync_cb = QCheckBox("Lip sync — mouth follows TTS playback")
        self._lipsync_cb.setChecked(True)
        beh_card.add_widget(self._lipsync_cb)

        self._idle_drift_cb = QCheckBox("Idle head micro-movement (subtle breathing feel)")
        self._idle_drift_cb.setChecked(True)
        beh_card.add_widget(self._idle_drift_cb)

        self._emotion_track_cb = QCheckBox("Auto-track emotion from EmotionNet")
        self._emotion_track_cb.setChecked(True)
        self._emotion_track_cb.setToolTip(
            "Automatically trigger avatar expressions whenever Revia's emotion changes."
        )
        beh_card.add_widget(self._emotion_track_cb)

        layout.addWidget(beh_card)

        layout.addStretch()
        self.setWidget(container)

        # Wire service signals if already created
        if self.vtube_service:
            self._wire_service(self.vtube_service)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def set_vtube_service(self, service: VTubeService):
        self.vtube_service = service
        self._wire_service(service)

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _wire_service(self, service: VTubeService):
        service.connected.connect(self._on_service_connected)
        service.disconnected.connect(self._on_service_disconnected)
        service.auth_pending.connect(self._on_auth_pending)
        service.auth_failed.connect(self._on_auth_failed)
        service.status_changed.connect(self._on_service_status)

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    @Slot()
    def _on_connect(self):
        if not self.vtube_service:
            return
        host = self._host_edit.text().strip() or "127.0.0.1"
        port = self._port_spin.value()
        self._btn_connect.setEnabled(False)
        self._status_label.setText("Status: Connecting…")
        self.vtube_service.connect(host=host, port=port)

    @Slot()
    def _on_disconnect(self):
        if self.vtube_service:
            self.vtube_service.disconnect()

    @Slot(str)
    def _preview_emotion(self, emotion: str):
        if self.vtube_service and self.vtube_service.is_connected:
            self.vtube_service.set_emotion(emotion)

    # ------------------------------------------------------------------
    # Service signal handlers
    # ------------------------------------------------------------------

    @Slot()
    def _on_service_connected(self):
        self._btn_connect.setEnabled(False)
        self._btn_disconnect.setEnabled(True)
        self._auth_hint.hide()
        self._status_label.setText("Status: Connected")
        apply_status_style(self._status_label, role="success")

    @Slot()
    def _on_service_disconnected(self):
        self._btn_connect.setEnabled(True)
        self._btn_disconnect.setEnabled(False)
        self._status_label.setText("Status: Disconnected")
        clear_status_role(self._status_label)

    @Slot()
    def _on_auth_pending(self):
        self._auth_hint.show()
        self._status_label.setText("Status: Waiting for VTube Studio approval…")
        apply_status_style(self._status_label, role="warning")

    @Slot(str)
    def _on_auth_failed(self, reason: str):
        self._btn_connect.setEnabled(True)
        self._auth_hint.hide()
        self._status_label.setText(f"Status: Auth failed — {reason}")
        apply_status_style(self._status_label, role="error")

    @Slot(str)
    def _on_service_status(self, status: str):
        self._status_label.setText(f"Status: {status}")

    # ------------------------------------------------------------------
    # Property accessors used by MainWindow wiring
    # ------------------------------------------------------------------

    @property
    def lipsync_enabled(self) -> bool:
        return self._lipsync_cb.isChecked()

    @property
    def emotion_tracking_enabled(self) -> bool:
        return self._emotion_track_cb.isChecked()
