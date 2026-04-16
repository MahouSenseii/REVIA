import math
from PySide6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter, QColor, QBrush, QPen, QRadialGradient, QFont

from app.ui_status import apply_status_style


class AvatarWidget(QFrame):
    """Animated anime-style avatar with pulsing violet/pink glow rings."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(84, 84)
        self.setMaximumSize(120, 120)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self._pulse = 0.0
        self._pulse_dir = 1
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(60)  # ~16 fps pulse

    def _tick(self):
        self._pulse += 0.04 * self._pulse_dir
        if self._pulse >= 1.0:
            self._pulse_dir = -1
        elif self._pulse <= 0.0:
            self._pulse_dir = 1
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        cx, cy, r = self.width() // 2, self.height() // 2, 44

        # — Outer violet glow (pulsing) —
        glow_alpha = int(25 + self._pulse * 40)
        glow = QRadialGradient(cx, cy, r + 22)
        glow.setColorAt(0.0, QColor(168, 85, 247, glow_alpha))
        glow.setColorAt(0.5, QColor(236, 72, 153, max(glow_alpha - 10, 8)))
        glow.setColorAt(1.0, QColor(168, 85, 247, 0))
        p.setBrush(QBrush(glow))
        p.setPen(Qt.NoPen)
        p.drawEllipse(cx - r - 22, cy - r - 22, (r + 22) * 2, (r + 22) * 2)

        # — Pulsing outer ring —
        ring_alpha = int(80 + self._pulse * 120)
        p.setPen(QPen(QColor(168, 85, 247, ring_alpha), 2))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(cx - r - 10, cy - r - 10, (r + 10) * 2, (r + 10) * 2)

        # — Inner pink ring —
        p.setPen(QPen(QColor(244, 114, 182, 160), 1))
        p.drawEllipse(cx - r - 4, cy - r - 4, (r + 4) * 2, (r + 4) * 2)

        # — Avatar face circle — dark gradient fill —
        face_grad = QRadialGradient(cx - 8, cy - 8, r * 1.2)
        face_grad.setColorAt(0.0, QColor(32, 18, 56))
        face_grad.setColorAt(1.0, QColor(12, 7, 22))
        p.setBrush(QBrush(face_grad))
        p.setPen(QPen(QColor(168, 85, 247, 200), 2))
        p.drawEllipse(cx - r, cy - r, r * 2, r * 2)

        # — "R" glyph with violet-to-pink gradient via linear gradient —
        p.setPen(QColor(236, 72, 153))
        p.setFont(QFont("Segoe UI", 26, QFont.Bold))
        p.drawText(self.rect(), Qt.AlignCenter, "R")

        p.end()


class ModuleIndicator(QFrame):
    """Anime-style tag badge: coloured dot + label with subtle tinted background."""

    _STATUS_COLORS = {
        "active": (QColor(45, 212, 191), QColor(45, 212, 191, 28)),   # teal dot, tinted bg
        "idle":   (QColor(168, 85, 247, 160), QColor(168, 85, 247, 18)),  # muted violet
        "error":  (QColor(244, 63, 94), QColor(244, 63, 94, 28)),     # red
    }

    def __init__(self, label, parent=None):
        super().__init__(parent)
        self.label_text = label
        self.status = "idle"
        self.setMinimumHeight(24)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_status(self, status):
        self.status = status
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        dot_color, bg_color = self._STATUS_COLORS.get(
            self.status, (QColor(100, 90, 130), QColor(100, 90, 130, 18))
        )

        # Background pill
        p.setBrush(bg_color)
        p.setPen(QPen(dot_color.lighter(130) if self.status == "active" else dot_color, 1))
        p.drawRoundedRect(4, 3, self.width() - 8, self.height() - 6, 9, 9)

        # Status dot
        p.setBrush(dot_color)
        p.setPen(Qt.NoPen)
        p.drawEllipse(11, 8, 8, 8)

        # Label
        p.setPen(QColor(220, 210, 245))
        p.setFont(QFont("Segoe UI", 9))
        p.drawText(26, 16, self.label_text)
        p.end()


class ActivityMeter(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(24)
        self.setMaximumWidth(36)
        self.setMinimumHeight(140)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
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
        bar_w = 14
        x = (self.width() - bar_w) // 2
        active_bars = int(bar_count * self.value)
        for i in range(bar_count):
            y = self.height() - 10 - (i + 1) * bar_h
            intensity = math.sin((i + self._phase) * 0.5) * 0.5 + 0.5
            alpha = int(50 + intensity * 170)
            if i < active_bars:
                # Gradient from violet (bottom) to pink (top)
                ratio = i / max(bar_count - 1, 1)
                r = int(168 + ratio * (236 - 168))
                g = int(85 - ratio * (85 - 72))
                b = int(247 - ratio * (247 - 153))
                color = QColor(r, g, b, alpha)
            else:
                color = QColor(35, 22, 55, 50)
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
        self.status_dot.setObjectName("statusDot")
        apply_status_style(self.status_dot, role="error")
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
            apply_status_style(self.status_dot, role="success")
            self.status_label.setText("Connecting")
            self.mod_vision.set_status("idle")
        else:
            apply_status_style(self.status_dot, role="error")
            self.status_label.setText("Offline")
            for m in [
                self.mod_vision, self.mod_stt, self.mod_tts, self.mod_memory
            ]:
                m.set_status("error")

    def _on_telemetry(self, data):
        state = str(data.get("state", "")).lower()
        llm = data.get("llm_connection", {}) or {}
        readiness = data.get("conversation_readiness", {}) or {}
        checks = readiness.get("checks", {}) or {}

        if readiness.get("ready", False):
            apply_status_style(self.status_dot, role="success")
            self.status_label.setText(data.get("state", "Ready"))
        elif llm.get("state") == "Connecting" or state in ("booting", "initializing", "thinking", "cooldown"):
            apply_status_style(self.status_dot, role="warning")
            self.status_label.setText(data.get("state", "Connecting"))
        elif llm.get("state") == "Error" or state == "error":
            apply_status_style(self.status_dot, role="error")
            self.status_label.setText("Error")

        if "listen" in state:
            self.mod_stt.set_status("active")
            self.activity.value = 0.7
        elif "speak" in state or "think" in state:
            self.activity.value = 0.9
        else:
            self.activity.value = 0.3

        self.mod_stt.set_status(
            "active" if (checks.get("stt", {}) or {}).get("ready", False) else "error"
        )
        self.mod_tts.set_status(
            "active" if (checks.get("tts", {}) or {}).get("ready", False) else "error"
        )
        self.mod_memory.set_status("active")
        self.mod_vision.set_status("active" if data.get("architecture", {}).get("modules", {}).get("vision", {}).get("online", False) else "idle")
