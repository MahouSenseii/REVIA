from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont


class PillLabel(QLabel):
    def __init__(self, text, object_name="pill", parent=None):
        super().__init__(text, parent)
        self.setObjectName(object_name)
        self.setAlignment(Qt.AlignCenter)
        self.setFixedHeight(28)
        self.setMinimumWidth(80)
        self.setFont(QFont("Segoe UI", 9))


class TopBar(QFrame):
    def __init__(self, event_bus, parent=None):
        super().__init__(parent)
        self.event_bus = event_bus
        self.setObjectName("topBar")
        self.setFixedHeight(44)

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

    def set_health(self, status):
        self.health_pill.setText(f"Health: {status}")
        if status == "Online":
            self.health_pill.setObjectName("healthPillOnline")
        else:
            self.health_pill.setObjectName("healthPill")
        self.health_pill.style().unpolish(self.health_pill)
        self.health_pill.style().polish(self.health_pill)
