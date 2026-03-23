from collections import deque
import logging
from PySide6.QtWidgets import (
    QScrollArea, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QGroupBox, QProgressBar, QPushButton,
    QTextEdit, QCheckBox, QSpinBox,
)
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QFont, QPainter, QColor, QPen

logger = logging.getLogger(__name__)


class _BgBridge(QObject):
    """Marshals callables from a background thread to the Qt main-thread event loop."""
    _call = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._call.connect(lambda fn: fn())

try:
    from PySide6.QtCharts import (
        QChart, QChartView, QLineSeries, QValueAxis,
    )
    _CHARTS_AVAILABLE = True
except ImportError:
    _CHARTS_AVAILABLE = False


_EMOTION_COLORS = {
    "Happy": "#4ade80",
    "Excited": "#34d399",
    "Curious": "#60a5fa",
    "Neutral": "#94a3b8",
    "Frustrated": "#f59e0b",
    "Angry": "#f87171",
    "Sad": "#818cf8",
    "Fear": "#a78bfa",
    "Lonely": "#5b8def",
    "Concerned": "#fca5a5",
    "Confident": "#2dd4bf",
}

_INJECT_HINTS = {
    "Happy": "Match their energy with warmth and enthusiasm.",
    "Angry": "Stay calm and understanding. Acknowledge frustration without defensiveness.",
    "Sad": "Be gentle, empathetic, and supportive.",
    "Curious": "Be thorough, engaging, and educational.",
    "Frustrated": "Be patient and concrete. Offer clear next steps.",
    "Fear": "Be reassuring, calm, and supportive.",
    "Excited": "Share enthusiasm and be energetic.",
    "Lonely": "Use warm engagement and inclusive language so they feel heard.",
    "Concerned": "Use clear, low-ambiguity guidance and reassurance.",
    "Confident": "Match directness with concise, action-oriented responses.",
}

_CHART_MAX_POINTS = 50


class EmotionsTab(QScrollArea):
    """Live monitor for probabilistic emotion inference."""

    def __init__(self, event_bus, client, parent=None):
        super().__init__(parent)
        self.event_bus = event_bus
        self.client = client
        self._bg = _BgBridge(self)
        self.setWidgetResizable(True)

        self._chart_tick = 0
        self._prob_rows = []
        self._emotion_history = deque(maxlen=1000)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        header = QLabel("Emotions")
        header.setObjectName("tabHeader")
        header.setFont(QFont("Segoe UI", 12, QFont.Bold))
        layout.addWidget(header)

        state_group = QGroupBox("Current Emotional State (User)")
        state_group.setObjectName("settingsGroup")
        sg = QVBoxLayout(state_group)

        label_row = QHBoxLayout()
        self.emo_label = QLabel("Neutral")
        self.emo_label.setFont(QFont("Segoe UI", 18, QFont.Bold))
        self.emo_label.setAlignment(Qt.AlignCenter)
        label_row.addWidget(self.emo_label, stretch=1)

        self.emo_confidence = QLabel("conf: --")
        self.emo_confidence.setFont(QFont("Consolas", 9))
        self.emo_confidence.setObjectName("metricLabel")
        label_row.addWidget(self.emo_confidence)
        sg.addLayout(label_row)

        def _make_vad_bar(name, color):
            row = QHBoxLayout()
            name_lbl = QLabel(name)
            name_lbl.setFixedWidth(82)
            name_lbl.setFont(QFont("Segoe UI", 9))
            bar = QProgressBar()
            bar.setRange(-100, 100)
            bar.setValue(0)
            bar.setTextVisible(True)
            bar.setFormat(f"{name}: %v%%")
            bar.setStyleSheet(
                f"QProgressBar::chunk {{ background: {color}; border-radius: 3px; }}"
            )
            row.addWidget(name_lbl)
            row.addWidget(bar)
            return row, bar

        vrow, self.valence_bar = _make_vad_bar("Valence", "#4ade80")
        arow, self.arousal_bar = _make_vad_bar("Arousal", "#f59e0b")
        drow, self.dominance_bar = _make_vad_bar("Dominance", "#818cf8")
        sg.addLayout(vrow)
        sg.addLayout(arow)
        sg.addLayout(drow)
        layout.addWidget(state_group)

        probs_group = QGroupBox("Neural Inference (Top Probabilities)")
        probs_group.setObjectName("settingsGroup")
        pg = QVBoxLayout(probs_group)

        for _ in range(5):
            row = QHBoxLayout()
            lbl = QLabel("-")
            lbl.setMinimumWidth(80)
            lbl.setFont(QFont("Consolas", 9))
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setFormat("%v%%")
            row.addWidget(lbl)
            row.addWidget(bar)
            pg.addLayout(row)
            self._prob_rows.append((lbl, bar))

        self.signals_view = QTextEdit()
        self.signals_view.setReadOnly(True)
        self.signals_view.setMaximumHeight(80)
        self.signals_view.setFont(QFont("Consolas", 8))
        self.signals_view.setPlaceholderText("Signal fusion values will appear here.")
        pg.addWidget(self.signals_view)

        self.temporal_view = QTextEdit()
        self.temporal_view.setReadOnly(True)
        self.temporal_view.setMaximumHeight(72)
        self.temporal_view.setFont(QFont("Consolas", 8))
        self.temporal_view.setPlaceholderText("Temporal/behavioral context will appear here.")
        pg.addWidget(self.temporal_view)

        layout.addWidget(probs_group)

        chart_group = QGroupBox("Emotion Timeline")
        chart_group.setObjectName("settingsGroup")
        cgl = QVBoxLayout(chart_group)

        if _CHARTS_AVAILABLE:
            self._chart_view, self._emotion_series, self._x_axis = self._build_chart()
            self._chart_view.setMinimumHeight(220)
            cgl.addWidget(self._chart_view)
        else:
            self._chart_view = None
            self._emotion_series = {}
            self._x_axis = None
            self._fallback_hist = QTextEdit()
            self._fallback_hist.setReadOnly(True)
            self._fallback_hist.setMaximumHeight(200)
            self._fallback_hist.setFont(QFont("Consolas", 8))
            self._fallback_hist.setPlaceholderText(
                "PySide6.QtCharts not available. Showing text history."
            )
            cgl.addWidget(self._fallback_hist)

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

        if _CHARTS_AVAILABLE:
            legend_row = QHBoxLayout()
            legend_row.setSpacing(12)
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

        inject_group = QGroupBox("Emotion Injection to AI")
        inject_group.setObjectName("settingsGroup")
        ig = QVBoxLayout(inject_group)

        info = QLabel(
            "This context is appended to the AI system prompt so responses adapt to "
            "the inferred emotional state."
        )
        info.setFont(QFont("Segoe UI", 8))
        info.setWordWrap(True)
        ig.addWidget(info)

        self.inject_enabled = QCheckBox("Feed emotion context to AI (disable = EmotionNet off)")
        self.inject_enabled.setChecked(True)
        self.inject_enabled.toggled.connect(self._toggle_injection)
        ig.addWidget(self.inject_enabled)

        self.inject_preview = QTextEdit()
        self.inject_preview.setReadOnly(True)
        self.inject_preview.setMaximumHeight(78)
        self.inject_preview.setFont(QFont("Consolas", 8))
        self.inject_preview.setPlaceholderText("No emotion context injected yet.")
        ig.addWidget(self.inject_preview)

        layout.addWidget(inject_group)
        layout.addStretch()
        self.setWidget(container)

        self.event_bus.telemetry_updated.connect(self._on_telemetry)
        self.event_bus.chat_complete.connect(lambda _: self._refresh_history())

    def _build_chart(self):
        chart = QChart()
        chart.setTitle("")
        chart.setAnimationOptions(QChart.NoAnimation)
        chart.legend().setVisible(False)
        chart.setBackgroundVisible(False)
        chart.setPlotAreaBackgroundVisible(True)
        chart.setPlotAreaBackgroundBrush(QColor("#1a1a2e"))

        series_map = {}
        for emotion, hex_color in _EMOTION_COLORS.items():
            series = QLineSeries()
            series.setName(emotion)
            pen = QPen(QColor(hex_color))
            pen.setWidth(2)
            series.setPen(pen)
            chart.addSeries(series)
            series_map[emotion] = series

        x_axis = QValueAxis()
        x_axis.setRange(0, _CHART_MAX_POINTS)
        x_axis.setLabelFormat("%d")
        x_axis.setLabelsColor(QColor("#64748b"))
        x_axis.setGridLineColor(QColor("#1e293b"))
        x_axis.setTitleText("Turn")
        x_axis.setTitleBrush(QColor("#64748b"))
        x_axis.setTickCount(6)
        chart.addAxis(x_axis, Qt.AlignBottom)

        y_axis = QValueAxis()
        y_axis.setRange(0.0, 1.0)
        y_axis.setLabelFormat("%.1f")
        y_axis.setLabelsColor(QColor("#64748b"))
        y_axis.setGridLineColor(QColor("#1e293b"))
        y_axis.setTitleText("Probability")
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

    def _on_telemetry(self, data):
        emo = data.get("emotion", {}) if isinstance(data, dict) else {}
        if not isinstance(emo, dict) or not emo:
            return
        self._update_display(emo)

    def _update_display(self, emo):
        label = str(emo.get("label", "Neutral"))
        secondary = str(emo.get("secondary_label", "")).strip()
        conf = self._as_float(emo.get("confidence", 0.0))
        v = self._as_float(emo.get("valence", 0.0))
        a = self._as_float(emo.get("arousal", 0.0))
        d = self._as_float(emo.get("dominance", 0.0))

        color = _EMOTION_COLORS.get(label, "#94a3b8")
        self.emo_label.setText(label)
        self.emo_label.setStyleSheet(f"color: {color}; font-weight: bold;")

        if secondary and secondary != label:
            self.emo_confidence.setText(f"conf: {conf:.0%} | next: {secondary}")
        else:
            self.emo_confidence.setText(f"conf: {conf:.0%}")

        self.valence_bar.setValue(int(v * 100))
        self.arousal_bar.setValue(int(a * 100))
        self.dominance_bar.setValue(int(d * 100))

        probs = self._coerce_probabilities(emo.get("emotion_probs", {}))
        ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        if not ranked and label:
            ranked = [(label, conf)]
            probs = {label: conf}

        self._update_probability_rows(ranked)
        self._update_reasoning_views(emo)

        if _CHARTS_AVAILABLE and self._emotion_series:
            self._add_chart_point(label, conf, probs=probs)

        if self.inject_enabled.isChecked() and label not in ("Neutral", "Disabled", "---", ""):
            top_text = ", ".join(f"{name}:{prob:.0%}" for name, prob in ranked[:3]) or f"{label}:{conf:.0%}"
            hint = _INJECT_HINTS.get(label, "Adjust tone to match emotional context.")
            preview = (
                f"[Emotional context inference: top hypotheses {top_text}. "
                f"Current best read: {label} (valence {v:+.2f}, confidence {conf:.0%}). "
                f"{hint}]"
            )
        else:
            preview = "(No emotion text injected: neutral/disabled or injection off)"
        self.inject_preview.setPlainText(preview)

    @staticmethod
    def _as_float(value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _coerce_probabilities(self, probs):
        if not isinstance(probs, dict):
            return {}
        out = {}
        for key, value in probs.items():
            p = self._as_float(value, 0.0)
            if p <= 0.0:
                continue
            out[str(key)] = max(0.0, min(1.0, p))
        return out

    def _update_probability_rows(self, ranked):
        for i, (lbl, bar) in enumerate(self._prob_rows):
            if i < len(ranked):
                name, prob = ranked[i]
                color = _EMOTION_COLORS.get(name, "#94a3b8")
                lbl.setText(name)
                bar.setValue(int(prob * 100))
                bar.setStyleSheet(
                    f"QProgressBar::chunk {{ background: {color}; border-radius: 3px; }}"
                )
            else:
                lbl.setText("-")
                bar.setValue(0)
                bar.setStyleSheet("")

    def _update_reasoning_views(self, emo):
        sig = emo.get("signals", {})
        if isinstance(sig, dict) and sig:
            lines = ["Signals"]
            for k in (
                "sentiment", "positive_tone", "negative_tone", "curiosity",
                "frustration", "loneliness", "anxiety", "urgency",
                "topic_sensitivity", "importance", "signal_strength",
            ):
                if k in sig:
                    lines.append(f"{k:>16}: {self._as_float(sig[k], 0.0):.3f}")
            self.signals_view.setPlainText("\n".join(lines))
        else:
            self.signals_view.setPlainText("Signals\n(no data)")

        tmp = emo.get("temporal", {})
        if isinstance(tmp, dict) and tmp:
            lines = ["Temporal"]
            for k in (
                "assistant_streak", "unanswered_user_turns",
                "since_last_user_s", "since_last_assistant_s",
                "response_gap_norm", "user_wait_norm", "avg_gap_s", "user_ratio",
            ):
                if k in tmp:
                    val = tmp.get(k)
                    if isinstance(val, float):
                        lines.append(f"{k:>20}: {val:.3f}")
                    else:
                        lines.append(f"{k:>20}: {val}")
            self.temporal_view.setPlainText("\n".join(lines))
        else:
            self.temporal_view.setPlainText("Temporal\n(no data)")

    def _add_chart_point(self, label, confidence, probs=None):
        self._chart_tick += 1
        tick = float(self._chart_tick)
        probs = probs if isinstance(probs, dict) else {}

        for emo, series in self._emotion_series.items():
            val = self._as_float(probs.get(emo, 0.0), 0.0)
            if not probs:
                val = confidence if emo == label else 0.0
            series.append(tick, max(0.0, min(1.0, val)))
            if series.count() > _CHART_MAX_POINTS:
                series.remove(0)

        x_min = max(0.0, tick - _CHART_MAX_POINTS)
        self._x_axis.setRange(x_min, x_min + _CHART_MAX_POINTS)

    def _clear_chart(self):
        self._chart_tick = 0
        if _CHARTS_AVAILABLE and self._emotion_series:
            for series in self._emotion_series.values():
                series.clear()
            self._x_axis.setRange(0, _CHART_MAX_POINTS)
        elif not _CHARTS_AVAILABLE and hasattr(self, "_fallback_hist"):
            self._fallback_hist.clear()

    def _toggle_injection(self, enabled):
        # Fire-and-forget in background — no need to update UI from the response.
        action = "enable" if enabled else "disable"
        url = f"{self.client.BASE_URL}/api/neural/emotion_net/{action}"

        def _work():
            try:
                self.client._session_post(url, timeout=2)
            except Exception as e:
                logger.warning(f"Error toggling emotion injection: {e}")

        self.client._executor.submit(_work)

    def _refresh_history(self):
        limit = self.hist_limit.value()

        def _work():
            try:
                r = self.client._session_get(
                    f"{self.client.BASE_URL}/api/emotions/history",
                    params={"limit": limit},
                    timeout=3,
                )
                if not r.ok:
                    return
                history = r.json()
                if not history:
                    return
                self._bg._call.emit(lambda h=history: self._apply_history(h))
            except Exception as e:
                logger.warning(f"Error refreshing emotion history: {e}")

        self.client._executor.submit(_work)

    def _apply_history(self, history):
        """Apply fetched history data to chart / fallback text (main thread only)."""
        if _CHARTS_AVAILABLE and self._emotion_series:
            for series in self._emotion_series.values():
                series.clear()
            self._chart_tick = 0
            for entry in history:
                lbl = str(entry.get("label", "Neutral"))
                conf = self._as_float(entry.get("confidence", 0.0), 0.0)
                probs = self._coerce_probabilities(entry.get("emotion_probs", {}))
                self._add_chart_point(lbl, conf, probs=probs)
        elif not _CHARTS_AVAILABLE and hasattr(self, "_fallback_hist"):
            lines = []
            for entry in reversed(history):
                ts = str(entry.get("timestamp", ""))[:19]
                lbl = str(entry.get("label", "?"))
                sec = str(entry.get("secondary_label", "")).strip()
                conf = self._as_float(entry.get("confidence", 0.0), 0.0)
                val = self._as_float(entry.get("valence", 0.0), 0.0)
                if sec and sec != lbl:
                    lines.append(
                        f"[{ts}] {lbl:<11} ({conf:.0%}) alt:{sec:<11} V:{val:+.2f}"
                    )
                else:
                    lines.append(f"[{ts}] {lbl:<11} ({conf:.0%}) V:{val:+.2f}")
            self._fallback_hist.setPlainText("\n".join(lines))

