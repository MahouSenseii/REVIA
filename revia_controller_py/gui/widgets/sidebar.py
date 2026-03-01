import math
from PySide6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QLabel
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter, QColor, QBrush, QPen, QRadialGradient, QFont


class AvatarWidget(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(120, 120)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        cx, cy, r = self.width() // 2, self.height() // 2, 50

        glow = QRadialGradient(cx, cy, r + 15)
        glow.setColorAt(0, QColor(0, 212, 255, 60))
        glow.setColorAt(1, QColor(0, 212, 255, 0))
        p.setBrush(QBrush(glow))
        p.setPen(Qt.NoPen)
        p.drawEllipse(cx - r - 15, cy - r - 15, (r + 15) * 2, (r + 15) * 2)

        p.setPen(QPen(QColor(0, 212, 255, 180), 2))
        p.setBrush(QColor(20, 25, 40))
        p.drawEllipse(cx - r, cy - r, r * 2, r * 2)

        p.setPen(QColor(0, 212, 255))
        p.setFont(QFont("Segoe UI", 28, QFont.Bold))
        p.drawText(self.rect(), Qt.AlignCenter, "R")
        p.end()


class ModuleIndicator(QFrame):
    def __init__(self, label, parent=None):
        super().__init__(parent)
        self.label_text = label
        self.status = "idle"
        self.setFixedHeight(22)

    def set_status(self, status):
        self.status = status
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        colors = {
            "active": QColor(0, 220, 80),
            "idle": QColor(180, 180, 40),
            "error": QColor(220, 50, 50),
        }
        color = colors.get(self.status, QColor(100, 100, 100))
        p.setBrush(color)
        p.setPen(Qt.NoPen)
        p.drawEllipse(8, 6, 10, 10)
        p.setPen(QColor(200, 210, 230))
        p.setFont(QFont("Segoe UI", 9))
        p.drawText(24, 16, self.label_text)
        p.end()


class ActivityMeter(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(30)
        self.setMinimumHeight(140)
        self.value = 0.3
        self._phase = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._animate)
        self.timer.start(150)

    def _animate(self):
        self._phase = (self._phase + 1) % 20
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        bar_count = 15
        bar_h = max(1, (self.height() - 20) // bar_count)
        bar_w = 16
        x = (self.width() - bar_w) // 2
        for i in range(bar_count):
            y = self.height() - 10 - (i + 1) * bar_h
            intensity = math.sin((i + self._phase) * 0.5) * 0.5 + 0.5
            alpha = int(40 + intensity * 180)
            if i < int(bar_count * self.value):
                color = QColor(0, 212, 255, alpha)
            else:
                color = QColor(40, 50, 70, 60)
            p.fillRect(x, y, bar_w, bar_h - 2, color)
        p.end()


class SidebarWidget(QFrame):
    def __init__(self, event_bus, parent=None):
        super().__init__(parent)
        self.event_bus = event_bus
        self.setObjectName("sidebar")

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        layout.setContentsMargins(10, 20, 10, 20)
        layout.setSpacing(6)

        self.avatar = AvatarWidget()
        layout.addWidget(self.avatar, alignment=Qt.AlignCenter)

        name = QLabel("Revia")
        name.setObjectName("sidebarName")
        name.setAlignment(Qt.AlignCenter)
        name.setFont(QFont("Segoe UI", 16, QFont.Bold))
        layout.addWidget(name)

        role = QLabel("Neural Assistant")
        role.setObjectName("sidebarRole")
        role.setAlignment(Qt.AlignCenter)
        layout.addWidget(role)

        status_row = QHBoxLayout()
        status_row.setAlignment(Qt.AlignCenter)
        self.status_dot = QLabel("\u25cf")
        self.status_dot.setStyleSheet("color: #dc3250; font-size: 10px;")
        self.status_label = QLabel("Offline")
        self.status_label.setObjectName("sidebarStatus")
        status_row.addWidget(self.status_dot)
        status_row.addWidget(self.status_label)
        layout.addLayout(status_row)

        layout.addSpacing(15)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setObjectName("sidebarSep")
        layout.addWidget(sep)

        layout.addSpacing(8)

        mod_label = QLabel("MODULES")
        mod_label.setObjectName("sidebarSection")
        layout.addWidget(mod_label)

        self.mod_vision = ModuleIndicator("Vision")
        self.mod_stt = ModuleIndicator("Mic / STT")
        self.mod_tts = ModuleIndicator("TTS")
        self.mod_memory = ModuleIndicator("Memory")
        for m in [self.mod_vision, self.mod_stt, self.mod_tts, self.mod_memory]:
            layout.addWidget(m)

        layout.addStretch()

        self.activity = ActivityMeter()
        layout.addWidget(self.activity, alignment=Qt.AlignCenter)

        self.event_bus.connection_changed.connect(self._on_connection)
        self.event_bus.telemetry_updated.connect(self._on_telemetry)

    def _on_connection(self, connected):
        if connected:
            self.status_dot.setStyleSheet("color: #00dc50; font-size: 10px;")
            self.status_label.setText("Online")
            self.mod_stt.set_status("active")
            self.mod_tts.set_status("active")
            self.mod_memory.set_status("active")
            self.mod_vision.set_status("idle")
        else:
            self.status_dot.setStyleSheet("color: #dc3250; font-size: 10px;")
            self.status_label.setText("Offline")
            for m in [
                self.mod_vision, self.mod_stt, self.mod_tts, self.mod_memory
            ]:
                m.set_status("error")

    def _on_telemetry(self, data):
        state = data.get("state", "")
        if "listen" in state.lower():
            self.mod_stt.set_status("active")
            self.activity.value = 0.7
        elif "generat" in state.lower():
            self.activity.value = 0.9
        else:
            self.activity.value = 0.3
