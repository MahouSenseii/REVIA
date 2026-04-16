from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QSizePolicy
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from app.ui_status import apply_status_style, clear_status_role


class PillLabel(QLabel):
    def __init__(self, text, object_name="pill", parent=None):
        super().__init__(text, parent)
        self.setObjectName(object_name)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(28)
        self.setMinimumWidth(80)
        self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        self.setFont(QFont("Segoe UI", 9))


class TopBar(QFrame):
    def __init__(self, event_bus, parent=None):
        super().__init__(parent)
        self.event_bus = event_bus
        self.setObjectName("topBar")
        self.setMinimumHeight(44)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(8)

        self.model_pill = PillLabel("Model: ---")
        self.ram_pill = PillLabel("RAM: --- MB")
        self.vram_pill = PillLabel("VRAM: --- MB")
        layout.addWidget(self.model_pill)
        layout.addWidget(self.ram_pill)
        layout.addWidget(self.vram_pill)

        layout.addStretch()

        title = PillLabel("\u2014 REVIA \u2014", "titlePill")
        title.setFont(QFont("Segoe UI", 11, QFont.Bold))
        title.setMinimumWidth(140)
        layout.addWidget(title)

        layout.addStretch()

        self.cpu_pill = PillLabel("CPU: ---%")
        self.gpu_pill = PillLabel("GPU: ---%")
        self.health_pill = PillLabel("Health: Offline", "healthPill")
        layout.addWidget(self.cpu_pill)
        layout.addWidget(self.gpu_pill)
        layout.addWidget(self.health_pill)

        self.event_bus.telemetry_updated.connect(self._on_telemetry)

    def _on_telemetry(self, data):
        sys_data = data.get("system", {})
        llm_state = ((data.get("llm_connection", {}) or {}).get("state") or "").strip()
        readiness = data.get("conversation_readiness", {}) or {}
        runtime_state = (data.get("state") or "").strip()
        model = sys_data.get("model", "---")
        if model and model != "None":
            self.model_pill.setText(f"Model: {model}")
        else:
            self.model_pill.setText("Model: ---")

        ram_used = sys_data.get("ram_used_mb", 0)
        ram_total = sys_data.get("ram_total_mb", 0)
        if ram_total > 0:
            ram_used_gb = ram_used / 1024
            ram_total_gb = ram_total / 1024
            self.ram_pill.setText(
                f"RAM: {ram_used_gb:.1f}/{ram_total_gb:.0f} GB"
            )
        else:
            self.ram_pill.setText("RAM: ---")

        vram_used = sys_data.get("vram_used_mb", 0)
        vram_total = sys_data.get("vram_total_mb", 0)
        if vram_total > 0:
            vram_used_gb = vram_used / 1024
            vram_total_gb = vram_total / 1024
            self.vram_pill.setText(
                f"VRAM: {vram_used_gb:.1f}/{vram_total_gb:.0f} GB"
            )
        else:
            self.vram_pill.setText(f"VRAM: {vram_used} MB")

        self.cpu_pill.setText(
            f"CPU: {sys_data.get('cpu_percent', 0):.0f}%"
        )
        self.gpu_pill.setText(
            f"GPU: {sys_data.get('gpu_percent', 0):.0f}%"
        )

        if readiness.get("ready", False):
            self.set_health("Ready")
        elif llm_state == "Connecting" or runtime_state in ("Booting", "Initializing", "Cooldown", "Thinking"):
            self.set_health("Connecting")
        elif llm_state == "Error" or runtime_state == "Error":
            self.set_health("Error")
        elif llm_state == "Disconnected":
            self.set_health("Offline")

    def set_health(self, status):
        self.health_pill.setText(f"Health: {status}")
        if status in ("Online", "Ready"):
            self.health_pill.setObjectName("healthPillOnline")
            apply_status_style(self.health_pill, role="success")
        elif status == "Error":
            self.health_pill.setObjectName("healthPill")
            apply_status_style(self.health_pill, role="error")
        elif status == "Connecting":
            self.health_pill.setObjectName("healthPill")
            apply_status_style(self.health_pill, role="warning")
        else:
            self.health_pill.setObjectName("healthPill")
            clear_status_role(self.health_pill)
        self.health_pill.style().unpolish(self.health_pill)
        self.health_pill.style().polish(self.health_pill)
