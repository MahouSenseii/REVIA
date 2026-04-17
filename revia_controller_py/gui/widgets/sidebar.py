import math
from PySide6.QtWidgets import (
    QApplication, QFrame, QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter, QColor, QBrush, QPen, QRadialGradient, QFont

from app.ui_status import apply_status_style


_THEME_FALLBACK = {
    "PrimaryBackground": "#06050f",
    "Surface": "#13112a",
    "SurfaceAlt": "#1c1838",
    "PrimaryText": "#ede9fe",
    "SecondaryText": "#8b7ab8",
    "Accent": "#a855f7",
    "AccentHover": "#c084fc",
    "Success": "#2dd4bf",
    "Error": "#f43f5e",
    "Disabled": "#4a3a6a",
}


def _theme_tokens():
    tokens = dict(_THEME_FALLBACK)
    app = QApplication.instance()
    raw = app.property("reviaThemeTokens") if app else None
    if isinstance(raw, dict):
        for key, value in raw.items():
            if key in tokens and QColor(str(value)).isValid():
                tokens[key] = str(value)
    return tokens


def _theme_color(tokens, key, alpha=None):
    color = QColor(tokens.get(key, _THEME_FALLBACK.get(key, "#000000")))
    if not color.isValid():
        color = QColor(_THEME_FALLBACK.get(key, "#000000"))
    if alpha is not None:
        color.setAlpha(alpha)
    return color


def _with_alpha(color, alpha):
    out = QColor(color)
    out.setAlpha(alpha)
    return out


def _blend(color_a, color_b, ratio, alpha=None):
    ratio = max(0.0, min(1.0, float(ratio)))
    a = QColor(color_a)
    b = QColor(color_b)
    out = QColor(
        int(a.red() + (b.red() - a.red()) * ratio),
        int(a.green() + (b.green() - a.green()) * ratio),
        int(a.blue() + (b.blue() - a.blue()) * ratio),
    )
    if alpha is not None:
        out.setAlpha(alpha)
    return out


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
        tokens = _theme_tokens()
        surface = _theme_color(tokens, "Surface")
        surface_alt = _theme_color(tokens, "SurfaceAlt")
        accent = _theme_color(tokens, "Accent")
        accent_hover = _theme_color(tokens, "AccentHover")
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        cx, cy, r = self.width() // 2, self.height() // 2, 44

        # Outer violet glow (pulsing)
        glow_alpha = int(25 + self._pulse * 40)
        glow = QRadialGradient(cx, cy, r + 22)
        glow.setColorAt(0.0, _with_alpha(accent, glow_alpha))
        glow.setColorAt(0.5, _with_alpha(accent_hover, max(glow_alpha - 10, 8)))
        glow.setColorAt(1.0, _with_alpha(accent, 0))
        p.setBrush(QBrush(glow))
        p.setPen(Qt.NoPen)
        p.drawEllipse(cx - r - 22, cy - r - 22, (r + 22) * 2, (r + 22) * 2)

        # Pulsing outer ring
        ring_alpha = int(80 + self._pulse * 120)
        p.setPen(QPen(_with_alpha(accent, ring_alpha), 2))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(cx - r - 10, cy - r - 10, (r + 10) * 2, (r + 10) * 2)

        # Inner pink ring
        p.setPen(QPen(_with_alpha(accent_hover, 160), 1))
        p.drawEllipse(cx - r - 4, cy - r - 4, (r + 4) * 2, (r + 4) * 2)

        # Avatar face circle - dark gradient fill
        face_grad = QRadialGradient(cx - 8, cy - 8, r * 1.2)
        face_grad.setColorAt(0.0, surface_alt)
        face_grad.setColorAt(1.0, surface)
        p.setBrush(QBrush(face_grad))
        p.setPen(QPen(_with_alpha(accent, 200), 2))
        p.drawEllipse(cx - r, cy - r, r * 2, r * 2)

        # "R" glyph with violet-to-pink gradient via linear gradient
        p.setPen(accent_hover)
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
        tokens = _theme_tokens()
        role_colors = {
            "active": "Success",
            "idle": "Accent",
            "error": "Error",
        }
        dot_color = _theme_color(tokens, role_colors.get(self.status, "Disabled"))
        bg_color = _with_alpha(dot_color, 28 if self.status in ("active", "error") else 18)
        border_color = (
            dot_color.darker(120)
            if _theme_color(tokens, "PrimaryBackground").lightness() > 160
            else dot_color.lighter(130)
        )
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        # Background pill
        p.setBrush(bg_color)
        p.setPen(QPen(border_color, 1))
        p.drawRoundedRect(4, 3, self.width() - 8, self.height() - 6, 9, 9)

        # Status dot
        p.setBrush(dot_color)
        p.setPen(Qt.NoPen)
        p.drawEllipse(11, 8, 8, 8)

        # Label
        p.setPen(_theme_color(tokens, "PrimaryText"))
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
        tokens = _theme_tokens()
        accent = _theme_color(tokens, "Accent")
        accent_hover = _theme_color(tokens, "AccentHover")
        disabled = _theme_color(tokens, "Disabled")
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
                color = _blend(accent, accent_hover, ratio, alpha)
            else:
                color = _with_alpha(disabled, 50)
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
