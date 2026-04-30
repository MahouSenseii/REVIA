from collections import deque
import logging
from PySide6.QtWidgets import (
    QScrollArea, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QProgressBar, QPushButton,
    QTextEdit, QCheckBox, QSpinBox,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QPainter, QColor, QPen

from app.ui_status import apply_status_style
from gui.widgets.settings_card import SettingsCard

logger = logging.getLogger(__name__)
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

        probs_card = SettingsCard(
            "Neural Inference",
            subtitle="Top probabilities",
            icon="N",
        )
        pg = QVBoxLayout()

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

        probs_card.add_layout(pg)
        layout.addWidget(probs_card)

        chart_card = SettingsCard(
            "Emotion Timeline",
            subtitle="History over time",
            icon="T",
        )
        cgl = QVBoxLayout()

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
            legend_grid = QGridLayout()
            legend_grid.setSpacing(6)
            _cols = 6  # wrap after 6 items per row
            for idx, (emo, color) in enumerate(_EMOTION_COLORS.items()):
                dot = QLabel("●")
                dot.setStyleSheet(f"color: {color};")
                dot.setFont(QFont("Segoe UI", 10))
                lbl = QLabel(emo)
                lbl.setFont(QFont("Segoe UI", 8))
                apply_status_style(lbl, role="muted")
                pair = QHBoxLayout()
                pair.setSpacing(3)
                pair.addWidget(dot)
                pair.addWidget(lbl)
                row_i, col_i = divmod(idx, _cols)
                legend_grid.addLayout(pair, row_i, col_i)
            cgl.addLayout(legend_grid)

        chart_card.add_layout(cgl)
        layout.addWidget(chart_card)

        inject_card = SettingsCard(
            "Emotion Injection to AI",
            subtitle="Context fed to the model",
            icon="I",
        )
        ig = QVBoxLayout()

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

        inject_card.add_layout(ig)
        layout.addWidget(inject_card)
        layout.addStretch()
        self.setWidget(container)

        self.event_bus.telemetry_updated.connect(self._on_telemetry)
        self.event_bus.chat_complete.connect(lambda _: self._refresh_history())
        self.event_bus.ui_theme_changed.connect(self._on_theme_changed)

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
        conf = self._as_float(emo.get("confidence", 0.0))
        v = self._as_float(emo.get("valence", 0.0))

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
        # Fire-and-forget in background - no need to update UI from the response.
        self.client.toggle_neural(
            "emotion_net",
            enabled,
            on_error=lambda error, _detail=None: logger.warning(
                "Error toggling emotion injection: %s", error
            ),
        )

    def _refresh_history(self):
        limit = self.hist_limit.value()
        self.client.get_async(
            "/api/emotions/history",
            params={"limit": limit},
            timeout=3,
            default=[],
            on_success=lambda history: self._apply_history(history or []),
            on_error=lambda error, _detail=None: logger.warning(
                "Error refreshing emotion history: %s", error
            ),
        )

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

    def _on_theme_changed(self, _theme_id):
        """Re-apply emotion legend dots and probability bars after theme switch.

        The ThemeManager clears local stylesheets during theme application,
        which removes the colored dots and progress bar chunks. Re-apply
        them using the same emotion-specific color palette (which is
        content-driven, not theme-driven).
        """
        # Re-apply legend dots
        for idx, (emo, color) in enumerate(_EMOTION_COLORS.items()):
            pair_layout = self.findChild(object, f"emo_legend_{idx}")
            # Fallback: find all QLabel children with "●" text and re-color
        # Re-apply probability bar chunk styles
        for i, (lbl, bar) in enumerate(self._prob_rows):
            if lbl.text() and lbl.text() != "-":
                color = _EMOTION_COLORS.get(lbl.text(), "#94a3b8")
                bar.setStyleSheet(
                    f"QProgressBar::chunk {{ background: {color}; border-radius: 3px; }}"
                )
        # Re-apply chart background if available
        if _CHARTS_AVAILABLE and hasattr(self, '_chart_view') and self._chart_view:
            self._chart_view.setStyleSheet("background: transparent; border: none;")
