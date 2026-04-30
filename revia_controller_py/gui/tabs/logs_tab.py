import re
import time
from datetime import datetime

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QTextEdit, QComboBox,
    QPushButton, QTableWidget, QTableWidgetItem,
    QProgressBar, QSplitter, QFrame, QSizePolicy,
)
from PySide6.QtGui import QFont, QColor, QTextCharFormat, QTextCursor, QTextBlockFormat
from PySide6.QtCore import Qt, QTimer

from app.ui_status import apply_status_style
from gui.widgets.settings_card import SettingsCard


# Log level detection + theme-aware colors

_LEVEL_PATTERNS = {
    "error": re.compile(
        r"\[(?:Core:err|ERROR|CRITICAL)\]|error[: ]|failed|crash|exception|traceback",
        re.IGNORECASE,
    ),
    "warning": re.compile(
        r"\[(?:WARNING|WARN)\]|warning[: ]|deprecated|timeout",
        re.IGNORECASE,
    ),
    "telemetry": re.compile(
        r"\[(?:Telemetry|Neural|Status)\]|emotion_|pipeline timing|span",
        re.IGNORECASE,
    ),
    "info": re.compile(
        r"\[(?:Revia|Core|LLM|Model|TTS|Memory|Search|RL)\]",
        re.IGNORECASE,
    ),
    "pipeline": re.compile(
        r"\[(?:Pipeline|vLLM|RPS|AVS|ALE|IHS|HFL)\]",
        re.IGNORECASE,
    ),
}

# Category tags extracted from log text for structured display
_CATEGORY_RE = re.compile(r"\[([A-Za-z]+)\]")


def _theme_level_colors():
    """Return level colors adapted to the current theme."""
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    tokens = app.property("reviaThemeTokens") if app else None
    base = {
        "error":     "#FF4444",
        "warning":   "#FFAA00",
        "telemetry": "#66BBFF",
        "info":      "#CCCCCC",
        "pipeline":  "#c084fc",
        "debug":     "#888888",
    }
    if isinstance(tokens, dict):
        base["error"] = tokens.get("Error", base["error"])
        base["warning"] = tokens.get("Warning", base["warning"])
        base["info"] = tokens.get("PrimaryText", base["info"])
        base["telemetry"] = tokens.get("Info", base["telemetry"])
        base["pipeline"] = tokens.get("Accent", base["pipeline"])
        base["debug"] = tokens.get("SecondaryText", base["debug"])
    return base


def _classify_log(text: str) -> str:
    """Classify a log line into error/warning/telemetry/info/pipeline/debug."""
    for level, pattern in _LEVEL_PATTERNS.items():
        if pattern.search(text):
            return level
    return "debug"


def _extract_category(text: str) -> str:
    """Extract the [Category] tag from a log line, or return 'General'."""
    m = _CATEGORY_RE.search(text)
    if m:
        return m.group(1)
    return "General"


class LogEntry:
    """Structured log entry with metadata."""
    __slots__ = ("timestamp", "level", "category", "text", "epoch")

    def __init__(self, text: str, level: str, category: str):
        self.text = text
        self.level = level
        self.category = category
        now = datetime.now()
        self.timestamp = now.strftime("%H:%M:%S")
        self.epoch = time.monotonic()


class LogsTab(QWidget):
    def __init__(self, event_bus, parent=None):
        super().__init__(parent)
        self.event_bus = event_bus

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        # ── Header row ──────────────────────────────────────────
        header_row = QHBoxLayout()
        header_row.setSpacing(8)

        header = QLabel("Logs")
        header.setObjectName("tabHeader")
        header.setFont(QFont("Segoe UI", 12, QFont.Bold))
        header_row.addWidget(header)

        header_row.addStretch()

        # Live stats indicators
        self._total_label = QLabel("0 entries")
        self._total_label.setFont(QFont("Segoe UI", 8))
        apply_status_style(self._total_label, role="muted")
        header_row.addWidget(self._total_label)

        self._error_label = QLabel("0 errors")
        self._error_label.setFont(QFont("Segoe UI", 8))
        apply_status_style(self._error_label, role="muted")
        header_row.addWidget(self._error_label)

        self._rate_label = QLabel("0/s")
        self._rate_label.setFont(QFont("Segoe UI", 8))
        apply_status_style(self._rate_label, role="muted")
        header_row.addWidget(self._rate_label)

        layout.addLayout(header_row)

        # ── Pipeline health bar ─────────────────────────────────
        health_row = QHBoxLayout()
        health_lbl = QLabel("Pipeline Health:")
        health_lbl.setFont(QFont("Segoe UI", 8))
        health_row.addWidget(health_lbl)

        self._health_bar = QProgressBar()
        self._health_bar.setRange(0, 100)
        self._health_bar.setValue(100)
        self._health_bar.setMaximumHeight(14)
        self._health_bar.setTextVisible(False)
        health_row.addWidget(self._health_bar, stretch=1)
        layout.addLayout(health_row)

        # ── Filter row ──────────────────────────────────────────
        filter_row = QHBoxLayout()
        filter_row.setSpacing(6)

        self.level_filter = QComboBox()
        self.level_filter.addItems(
            ["All", "Info", "Pipeline", "Warning", "Error", "Telemetry", "Debug"]
        )
        self.level_filter.setMaximumWidth(120)
        self.level_filter.currentTextChanged.connect(self._apply_filter)
        filter_row.addWidget(QLabel("Level:"))
        filter_row.addWidget(self.level_filter)

        self.cat_filter = QComboBox()
        self.cat_filter.addItems(["All"])
        self.cat_filter.setMaximumWidth(120)
        self.cat_filter.currentTextChanged.connect(self._apply_filter)
        filter_row.addWidget(QLabel("Source:"))
        filter_row.addWidget(self.cat_filter)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Search logs...")
        self.search.textChanged.connect(self._apply_filter)
        filter_row.addWidget(self.search, stretch=1)

        clear_btn = QPushButton("Clear")
        clear_btn.setObjectName("secondaryBtn")
        clear_btn.setMaximumWidth(60)
        clear_btn.clicked.connect(self._clear_logs)
        filter_row.addWidget(clear_btn)

        auto_scroll_cb = QLabel("Auto-scroll")
        auto_scroll_cb.setFont(QFont("Segoe UI", 8))
        filter_row.addWidget(auto_scroll_cb)

        layout.addLayout(filter_row)

        # ── Log view ───────────────────────────────────────────
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setObjectName("logView")
        self.log_view.setFont(QFont("Consolas", 9))
        self.log_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.log_view, stretch=1)

        # ── Pipeline Timing ─────────────────────────────────────
        timing_card = SettingsCard(
            "Pipeline Timing",
            subtitle="Stage durations",
            icon="P",
        )
        t = QVBoxLayout()

        self.timing_list = QTableWidget()
        self.timing_list.setColumnCount(4)
        self.timing_list.setHorizontalHeaderLabels(
            ["Stage", "Duration (ms)", "Device", "Status"]
        )
        self.timing_list.horizontalHeader().setStretchLastSection(True)
        self.timing_list.setMaximumHeight(160)
        t.addWidget(self.timing_list)

        timing_card.add_layout(t)
        layout.addWidget(timing_card)

        # ── Internal state ──────────────────────────────────────
        self._log_buffer: list[LogEntry] = []
        self._max_buffer = 3000
        self._error_count = 0
        self._warning_count = 0
        self._entries_per_window: list[float] = []  # epoch timestamps for rate calc
        self._categories: set[str] = {"All"}
        self._auto_scroll = True

        # Rate update timer
        self._rate_timer = QTimer(self)
        self._rate_timer.setInterval(2000)
        self._rate_timer.timeout.connect(self._update_rate_display)
        self._rate_timer.start()

        self.event_bus.log_entry.connect(self._add_log)
        self.event_bus.telemetry_updated.connect(self._update_timing)
        self.event_bus.ui_theme_changed.connect(self._on_theme_changed)

    # ── Theme awareness ────────────────────────────────────────

    def _on_theme_changed(self, _theme_id):
        """Re-render all log entries with updated theme colors."""
        self._apply_filter()

    # ── Log handling ────────────────────────────────────────────

    def _add_log(self, text: str):
        level = _classify_log(text)
        category = _extract_category(text)
        entry = LogEntry(text, level, category)
        self._log_buffer.append(entry)

        # Update category filter
        if category not in self._categories:
            self._categories.add(category)
            self.cat_filter.addItem(category)

        # Update stats
        if level == "error":
            self._error_count += 1
            self._error_label.setText(f"{self._error_count} errors")
            apply_status_style(self._error_label, role="error")
        elif level == "warning":
            self._warning_count += 1

        self._entries_per_window.append(entry.epoch)

        # Trim buffer
        if len(self._log_buffer) > self._max_buffer:
            self._log_buffer = self._log_buffer[-self._max_buffer:]

        self._total_label.setText(f"{len(self._log_buffer)} entries")

        # Update pipeline health: degrade on errors, recover over time
        if level == "error":
            self._health_bar.setValue(max(0, self._health_bar.value() - 15))
        elif level == "warning":
            self._health_bar.setValue(max(0, self._health_bar.value() - 5))
        elif level in ("info", "pipeline"):
            self._health_bar.setValue(min(100, self._health_bar.value() + 2))

        if not self._passes_filter(level, category, text):
            return

        self._append_entry(entry)

    def _append_entry(self, entry: LogEntry):
        """Append a structured log entry with timestamp, level badge, and text."""
        colors = _theme_level_colors()
        level_color = QColor(colors.get(entry.level, colors["debug"]))

        cursor = self.log_view.textCursor()
        cursor.movePosition(QTextCursor.End)

        # Timestamp prefix
        ts_fmt = QTextCharFormat()
        ts_fmt.setForeground(QColor(colors["debug"]))
        ts_fmt.setFontPointSize(8)
        cursor.insertText(f"{entry.timestamp} ", ts_fmt)

        # Level badge
        badge_fmt = QTextCharFormat()
        badge_fmt.setForeground(level_color)
        badge_fmt.setFontWeight(QFont.Bold)
        badge_fmt.setFontPointSize(8)
        level_tag = f"[{entry.level.upper():>9}]"
        cursor.insertText(f"{level_tag} ", badge_fmt)

        # Category tag (dimmed)
        cat_fmt = QTextCharFormat()
        cat_fmt.setForeground(QColor(colors["debug"]))
        cat_fmt.setFontPointSize(8)
        cursor.insertText(f"{entry.category:>10}  ", cat_fmt)

        # Main text
        text_fmt = QTextCharFormat()
        text_fmt.setForeground(level_color)
        if entry.level == "error":
            text_fmt.setFontWeight(QFont.Bold)
        cursor.insertText(entry.text + "\n", text_fmt)

        self.log_view.setTextCursor(cursor)
        if self._auto_scroll:
            self.log_view.ensureCursorVisible()

    def _passes_filter(self, level: str, category: str, text: str) -> bool:
        """Check if a log entry passes the current level, category, and search filters."""
        selected_level = self.level_filter.currentText().lower()
        if selected_level != "all" and selected_level != level:
            return False

        selected_cat = self.cat_filter.currentText()
        if selected_cat != "All" and category != selected_cat:
            return False

        search_text = self.search.text().strip()
        if search_text and search_text.lower() not in text.lower():
            return False

        return True

    def _apply_filter(self, *_args):
        """Re-render the log view with current filter settings."""
        self.log_view.clear()
        for entry in self._log_buffer:
            if self._passes_filter(entry.level, entry.category, entry.text):
                self._append_entry(entry)

    def _clear_logs(self):
        self.log_view.clear()
        self._log_buffer.clear()
        self._error_count = 0
        self._warning_count = 0
        self._entries_per_window.clear()
        self._health_bar.setValue(100)
        self._total_label.setText("0 entries")
        self._error_label.setText("0 errors")
        apply_status_style(self._error_label, role="muted")
        self._rate_label.setText("0/s")

    def _update_timing(self, data):
        spans = data.get("recent_spans", [])
        if not spans:
            return
        unique_stages = {}
        for s in spans:
            stage = s.get("stage", "")
            if stage:
                unique_stages[stage] = s

        self.timing_list.setRowCount(len(unique_stages))
        for i, (stage, s) in enumerate(unique_stages.items()):
            dur = s.get("duration_ms", 0)
            self.timing_list.setItem(i, 0, QTableWidgetItem(stage))
            self.timing_list.setItem(
                i, 1, QTableWidgetItem(f"{dur:.1f}")
            )
            self.timing_list.setItem(
                i, 2, QTableWidgetItem(s.get("device", "CPU"))
            )
            # Status column: flag slow stages
            if dur > 500:
                status_item = QTableWidgetItem("SLOW")
                status_item.setForeground(QColor("#FFAA00"))
            elif dur > 100:
                status_item = QTableWidgetItem("OK")
                status_item.setForeground(QColor("#66BBFF"))
            else:
                status_item = QTableWidgetItem("Fast")
                status_item.setForeground(QColor("#4ade80"))
            self.timing_list.setItem(i, 3, status_item)

    def _update_rate_display(self):
        """Calculate and display the log entry rate over the last 10s window."""
        now = time.monotonic()
        window = 10.0
        self._entries_per_window = [
            t for t in self._entries_per_window if now - t < window
        ]
        rate = len(self._entries_per_window) / window if window > 0 else 0
        self._rate_label.setText(f"{rate:.1f}/s")
