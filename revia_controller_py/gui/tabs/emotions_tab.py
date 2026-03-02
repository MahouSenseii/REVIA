import requests as _req

from PySide6.QtWidgets import (
    QScrollArea, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QGroupBox, QProgressBar, QPushButton,
    QTextEdit, QCheckBox, QSpinBox,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QPainter, QColor, QPen

try:
    from PySide6.QtCharts import (
        QChart, QChartView, QLineSeries, QValueAxis,
    )
    _CHARTS_AVAILABLE = True
except ImportError:
    _CHARTS_AVAILABLE = False

# Emotion → hex colour
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

_CHART_MAX_POINTS = 50


class EmotionsTab(QScrollArea):
    """Live emotion monitor — current state, VAD values, line chart history,
    and the emotion text injected into the AI's system prompt."""

    def __init__(self, event_bus, client, parent=None):
        super().__init__(parent)
        self.event_bus = event_bus
        self.client = client
        self.setWidgetResizable(True)

        self._chart_tick = 0  # increments on each data point

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

        # ── Emotion line chart ──────────────────────────────────────────
        chart_group = QGroupBox("Emotion Timeline")
        chart_group.setObjectName("settingsGroup")
        cgl = QVBoxLayout(chart_group)

        if _CHARTS_AVAILABLE:
            self._chart_view, self._emotion_series, self._x_axis = \
                self._build_chart()
            self._chart_view.setMinimumHeight(220)
            cgl.addWidget(self._chart_view)
        else:
            # Fallback: plain text history (QtCharts not installed)
            self._chart_view = None
            self._emotion_series = {}
            self._x_axis = None
            self._fallback_hist = QTextEdit()
            self._fallback_hist.setReadOnly(True)
            self._fallback_hist.setMaximumHeight(200)
            self._fallback_hist.setFont(QFont("Consolas", 8))
            self._fallback_hist.setPlaceholderText(
                "PySide6.QtCharts not available — showing text history…"
            )
            cgl.addWidget(self._fallback_hist)

        # Chart controls row
        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(QLabel("Show last"))
        self.hist_limit = QSpinBox()
        self.hist_limit.setRange(5, 100)
        self.hist_limit.setValue(25)
        ctrl_row.addWidget(self.hist_limit)
        ctrl_row.addWidget(QLabel("entries"))
        ctrl_row.addStretch()

        clear_chart_btn = QPushButton("Clear Chart")
        clear_chart_btn.setObjectName("secondaryBtn")
        clear_chart_btn.clicked.connect(self._clear_chart)
        ctrl_row.addWidget(clear_chart_btn)

        refresh_btn = QPushButton("Refresh from API")
        refresh_btn.setObjectName("secondaryBtn")
        refresh_btn.clicked.connect(self._refresh_history)
        ctrl_row.addWidget(refresh_btn)
        cgl.addLayout(ctrl_row)

        # Compact legend below the chart
        if _CHARTS_AVAILABLE:
            legend_row = QHBoxLayout()
            legend_row.setSpacing(14)
            for emo, color in _EMOTION_COLORS.items():
                dot = QLabel("●")
                dot.setStyleSheet(f"color: {color};")
                dot.setFont(QFont("Segoe UI", 10))
                lbl = QLabel(emo)
                lbl.setFont(QFont("Segoe UI", 8))
                lbl.setStyleSheet("color: #94a3b8;")
                pair = QHBoxLayout()
                pair.setSpacing(3)
                pair.addWidget(dot)
                pair.addWidget(lbl)
                legend_row.addLayout(pair)
            legend_row.addStretch()
            cgl.addLayout(legend_row)

        layout.addWidget(chart_group)

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
        layout.addStretch()
        self.setWidget(container)

        # Wire live telemetry updates
        self.event_bus.telemetry_updated.connect(self._on_telemetry)
        self.event_bus.chat_complete.connect(lambda _: self._refresh_history())

    # ── Chart builder ────────────────────────────────────────────────────

    def _build_chart(self):
        """Build and return the QChart, series dict, and x-axis."""
        chart = QChart()
        chart.setTitle("")
        chart.setAnimationOptions(QChart.NoAnimation)
        chart.legend().setVisible(False)  # we draw our own legend below

        # Dark background to match the app theme
        chart.setBackgroundVisible(False)
        chart.setPlotAreaBackgroundVisible(True)
        chart.setPlotAreaBackgroundBrush(QColor("#1a1a2e"))

        # One series per emotion
        series_map = {}
        for emotion, hex_color in _EMOTION_COLORS.items():
            series = QLineSeries()
            series.setName(emotion)
            pen = QPen(QColor(hex_color))
            pen.setWidth(2)
            series.setPen(pen)
            # Start hidden; only the active emotion gets data
            chart.addSeries(series)
            series_map[emotion] = series

        # X axis — turn index
        x_axis = QValueAxis()
        x_axis.setRange(0, _CHART_MAX_POINTS)
        x_axis.setLabelFormat("%d")
        x_axis.setLabelsColor(QColor("#64748b"))
        x_axis.setGridLineColor(QColor("#1e293b"))
        x_axis.setTitleText("Turn")
        x_axis.setTitleBrush(QColor("#64748b"))
        x_axis.setTickCount(6)
        chart.addAxis(x_axis, Qt.AlignBottom)

        # Y axis — confidence 0→1
        y_axis = QValueAxis()
        y_axis.setRange(0.0, 1.0)
        y_axis.setLabelFormat("%.1f")
        y_axis.setLabelsColor(QColor("#64748b"))
        y_axis.setGridLineColor(QColor("#1e293b"))
        y_axis.setTitleText("Confidence")
        y_axis.setTitleBrush(QColor("#64748b"))
        y_axis.setTickCount(6)
        chart.addAxis(y_axis, Qt.AlignLeft)

        for series in series_map.values():
            series.attachAxis(x_axis)
            series.attachAxis(y_axis)

        chart_view = QChartView(chart)
        chart_view.setRenderHint(QPainter.Antialiasing)
        chart_view.setStyleSheet("background: transparent; border: none;")

        return chart_view, series_map, x_axis

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

        # Add to live chart
        if _CHARTS_AVAILABLE and self._emotion_series:
            self._add_chart_point(label, conf)

        # Preview injection text
        if self.inject_enabled.isChecked() and label not in ("Neutral", "Disabled", "---", ""):
            hint = _INJECT_HINTS.get(label, "Adjust your tone to match their emotional state.")
            preview = (
                f"[Emotional context: The user currently seems {label} "
                f"(valence {v:+.2f}, confidence {conf:.0%}). {hint}]"
            )
        else:
            preview = "(No emotion text injected — emotion is Neutral or injection is disabled)"
        self.inject_preview.setPlainText(preview)

    def _add_chart_point(self, label: str, confidence: float):
        """Append a new reading to the chart, sliding window of _CHART_MAX_POINTS."""
        self._chart_tick += 1
        tick = float(self._chart_tick)

        for emo, series in self._emotion_series.items():
            val = confidence if emo == label else 0.0
            series.append(tick, val)
            # Trim to window size
            if series.count() > _CHART_MAX_POINTS:
                series.remove(0)

        # Slide x-axis window
        x_min = max(0.0, tick - _CHART_MAX_POINTS)
        x_max = x_min + _CHART_MAX_POINTS
        self._x_axis.setRange(x_min, x_max)

    def _clear_chart(self):
        """Reset all series and tick counter."""
        self._chart_tick = 0
        if _CHARTS_AVAILABLE and self._emotion_series:
            for series in self._emotion_series.values():
                series.clear()
            self._x_axis.setRange(0, _CHART_MAX_POINTS)
        elif not _CHARTS_AVAILABLE and hasattr(self, "_fallback_hist"):
            self._fallback_hist.clear()

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
        """Load history from the API and replay it into the chart."""
        limit = self.hist_limit.value()
        try:
            r = _req.get(
                f"{self.client.BASE_URL}/api/emotions/history",
                params={"limit": limit},
                timeout=3,
            )
            if not r.ok:
                return
            history = r.json()
            if not history:
                return

            if _CHARTS_AVAILABLE and self._emotion_series:
                # Clear and replay from API history
                for series in self._emotion_series.values():
                    series.clear()
                self._chart_tick = 0
                for entry in history:
                    lbl  = entry.get("label", "Neutral")
                    conf = entry.get("confidence", 0.0)
                    self._add_chart_point(lbl, conf)
            elif not _CHARTS_AVAILABLE and hasattr(self, "_fallback_hist"):
                lines = []
                for entry in reversed(history):
                    ts   = entry.get("timestamp", "")[:19]
                    lbl  = entry.get("label", "?")
                    val  = entry.get("valence",   0.0)
                    conf = entry.get("confidence", 0.0)
                    lines.append(f"[{ts}]  {lbl:<12}  V:{val:+.2f}  conf:{conf:.0%}")
                self._fallback_hist.setPlainText("\n".join(lines))
        except Exception:
            pass
