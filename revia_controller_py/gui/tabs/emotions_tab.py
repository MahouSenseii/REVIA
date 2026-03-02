import requests as _req

from PySide6.QtWidgets import (
    QScrollArea, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QGroupBox, QProgressBar, QPushButton,
    QTextEdit, QCheckBox, QSpinBox,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

# Valence → hex colour
_EMOTION_COLORS = {
    "Happy":      "#4ade80",
    "Excited":    "#34d399",
    "Curious":    "#60a5fa",
    "Neutral":    "#94a3b8",
    "Frustrated": "#f59e0b",
    "Fear":       "#a78bfa",
    "Sad":        "#818cf8",
    "Angry":      "#f87171",
}

_INJECT_HINTS = {
    "Happy":      "Match their energy with warmth and enthusiasm.",
    "Angry":      "Stay calm and understanding. Acknowledge their frustration without being defensive.",
    "Sad":        "Be gentle, empathetic, and supportive. Offer comfort.",
    "Curious":    "Be thorough, engaging, and educational in your explanation.",
    "Frustrated": "Be patient and clear. Acknowledge their concern and offer concrete help.",
    "Fear":       "Be reassuring, calm, and supportive.",
    "Excited":    "Share their enthusiasm and be energetic in your response.",
}


class EmotionsTab(QScrollArea):
    """Live emotion monitor — shows detected user emotion, VAD values,
    the exact text injected into the AI's system prompt, and emotion history."""

    def __init__(self, event_bus, client, parent=None):
        super().__init__(parent)
        self.event_bus = event_bus
        self.client = client
        self.setWidgetResizable(True)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        header = QLabel("Emotions")
        header.setObjectName("tabHeader")
        header.setFont(QFont("Segoe UI", 12, QFont.Bold))
        layout.addWidget(header)

        # ── Current emotional state ─────────────────────────────────────
        emo_group = QGroupBox("Current Emotional State (User)")
        emo_group.setObjectName("settingsGroup")
        eg = QVBoxLayout(emo_group)

        label_row = QHBoxLayout()
        self.emo_label = QLabel("Neutral")
        self.emo_label.setFont(QFont("Segoe UI", 18, QFont.Bold))
        self.emo_label.setAlignment(Qt.AlignCenter)
        label_row.addWidget(self.emo_label, stretch=1)

        self.emo_confidence = QLabel("conf: —")
        self.emo_confidence.setFont(QFont("Consolas", 9))
        self.emo_confidence.setObjectName("metricLabel")
        label_row.addWidget(self.emo_confidence)
        eg.addLayout(label_row)

        # VAD progress bars — range -100…+100 (scaled from -1…+1)
        def _make_bar(name, color):
            row = QHBoxLayout()
            lbl = QLabel(name)
            lbl.setFixedWidth(80)
            lbl.setFont(QFont("Segoe UI", 9))
            bar = QProgressBar()
            bar.setRange(-100, 100)
            bar.setValue(0)
            bar.setTextVisible(True)
            bar.setFormat(f"{name}: %v%%")
            bar.setStyleSheet(
                f"QProgressBar::chunk {{ background: {color}; border-radius: 3px; }}"
            )
            row.addWidget(lbl)
            row.addWidget(bar)
            return row, bar

        vrow, self.valence_bar   = _make_bar("Valence",   "#4ade80")
        arow, self.arousal_bar   = _make_bar("Arousal",   "#f59e0b")
        drow, self.dominance_bar = _make_bar("Dominance", "#818cf8")
        eg.addLayout(vrow)
        eg.addLayout(arow)
        eg.addLayout(drow)

        layout.addWidget(emo_group)

        # ── Emotion → AI injection ──────────────────────────────────────
        inject_group = QGroupBox("Emotion Injection to AI")
        inject_group.setObjectName("settingsGroup")
        ig = QVBoxLayout(inject_group)

        info = QLabel(
            "The text below is appended to the AI's system prompt every turn, "
            "telling it how to adapt its tone to the user's emotional state."
        )
        info.setFont(QFont("Segoe UI", 8))
        info.setWordWrap(True)
        ig.addWidget(info)

        self.inject_enabled = QCheckBox("Feed emotion context to AI  (disable = EmotionNet off)")
        self.inject_enabled.setChecked(True)
        self.inject_enabled.toggled.connect(self._toggle_injection)
        ig.addWidget(self.inject_enabled)

        self.inject_preview = QTextEdit()
        self.inject_preview.setReadOnly(True)
        self.inject_preview.setMaximumHeight(72)
        self.inject_preview.setFont(QFont("Consolas", 8))
        self.inject_preview.setPlaceholderText("No emotion injected yet…")
        ig.addWidget(self.inject_preview)

        layout.addWidget(inject_group)

        # ── Emotion history ─────────────────────────────────────────────
        hist_group = QGroupBox("Emotion History")
        hist_group.setObjectName("settingsGroup")
        hl = QVBoxLayout(hist_group)

        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(QLabel("Show last"))
        self.hist_limit = QSpinBox()
        self.hist_limit.setRange(5, 100)
        self.hist_limit.setValue(25)
        ctrl_row.addWidget(self.hist_limit)
        ctrl_row.addWidget(QLabel("entries"))
        ctrl_row.addStretch()

        refresh_btn = QPushButton("Refresh History")
        refresh_btn.setObjectName("secondaryBtn")
        refresh_btn.clicked.connect(self._refresh_history)
        ctrl_row.addWidget(refresh_btn)
        hl.addLayout(ctrl_row)

        self.hist_display = QTextEdit()
        self.hist_display.setReadOnly(True)
        self.hist_display.setMaximumHeight(200)
        self.hist_display.setFont(QFont("Consolas", 8))
        self.hist_display.setPlaceholderText("No emotion history yet…")
        hl.addWidget(self.hist_display)

        layout.addWidget(hist_group)
        layout.addStretch()
        self.setWidget(container)

        # Wire live telemetry updates
        self.event_bus.telemetry_updated.connect(self._on_telemetry)
        self.event_bus.chat_complete.connect(lambda _: self._refresh_history())

    # ── Slots ────────────────────────────────────────────────────────────

    def _on_telemetry(self, data: dict):
        emo = data.get("emotion", {})
        if not emo:
            return
        self._update_display(emo)

    def _update_display(self, emo: dict):
        label = emo.get("label", "Neutral")
        conf  = emo.get("confidence", 0.0)
        v     = emo.get("valence",   0.0)
        a     = emo.get("arousal",   0.0)
        d     = emo.get("dominance", 0.0)

        color = _EMOTION_COLORS.get(label, "#94a3b8")
        self.emo_label.setText(label)
        self.emo_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        self.emo_confidence.setText(f"conf: {conf:.0%}")

        self.valence_bar.setValue(int(v * 100))
        self.arousal_bar.setValue(int(a * 100))
        self.dominance_bar.setValue(int(d * 100))

        # Preview the actual text that will be injected
        if self.inject_enabled.isChecked() and label not in ("Neutral", "Disabled", "---", ""):
            hint = _INJECT_HINTS.get(label, "Adjust your tone to match their emotional state.")
            preview = (
                f"[Emotional context: The user currently seems {label} "
                f"(valence {v:+.2f}, confidence {conf:.0%}). {hint}]"
            )
        else:
            preview = "(No emotion text injected — emotion is Neutral or injection is disabled)"
        self.inject_preview.setPlainText(preview)

    def _toggle_injection(self, enabled: bool):
        """Enable / disable EmotionNet via the server API."""
        action = "enable" if enabled else "disable"
        try:
            _req.post(
                f"{self.client.BASE_URL}/api/neural/emotion_net/{action}",
                timeout=2,
            )
        except Exception:
            pass

    def _refresh_history(self):
        limit = self.hist_limit.value()
        try:
            r = _req.get(
                f"{self.client.BASE_URL}/api/emotions/history",
                params={"limit": limit},
                timeout=3,
            )
            if r.ok:
                history = r.json()
                if not history:
                    self.hist_display.setPlainText("No emotion history recorded yet.")
                    return
                lines = []
                for entry in reversed(history):
                    ts    = entry.get("timestamp", "")[:19]
                    lbl   = entry.get("label", "?")
                    val   = entry.get("valence",   0.0)
                    conf  = entry.get("confidence", 0.0)
                    color = _EMOTION_COLORS.get(lbl, "#94a3b8")
                    lines.append(
                        f"[{ts}]  {lbl:<12}  V:{val:+.2f}  conf:{conf:.0%}"
                    )
                self.hist_display.setPlainText("\n".join(lines))
            else:
                self.hist_display.setPlainText(f"HTTP {r.status_code}")
        except Exception as ex:
            self.hist_display.setPlainText(f"Error: {ex}")
