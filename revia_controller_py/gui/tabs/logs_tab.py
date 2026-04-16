import re

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QTextEdit, QComboBox, QGroupBox,
    QPushButton, QTableWidget, QTableWidgetItem,
)
from PySide6.QtGui import QFont, QColor, QTextCharFormat, QTextCursor


# ── Log level detection + colors ──

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
        r"\[(?:Revia|Core|LLM|Model|TTS)\]",
        re.IGNORECASE,
    ),
}

_LEVEL_COLORS = {
    "error":     QColor("#FF4444"),   # red
    "warning":   QColor("#FFAA00"),   # amber
    "telemetry": QColor("#66BBFF"),   # light blue
    "info":      QColor("#CCCCCC"),   # light gray
    "debug":     QColor("#888888"),   # dim gray
}


def _classify_log(text: str) -> str:
    """Classify a log line into error/warning/telemetry/info/debug."""
    for level, pattern in _LEVEL_PATTERNS.items():
        if pattern.search(text):
            return level
    return "debug"


class LogsTab(QWidget):
    def __init__(self, event_bus, parent=None):
        super().__init__(parent)
        self.event_bus = event_bus

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        header = QLabel("Logs")
        header.setObjectName("tabHeader")
        header.setFont(QFont("Segoe UI", 12, QFont.Bold))
        layout.addWidget(header)

        # Advanced logs toggle (hidden by default)
        toggle_row = QHBoxLayout()
        toggle_lbl = QLabel("Show Advanced")
        toggle_lbl.setFont(QFont("Segoe UI", 9))
        toggle_row.addWidget(toggle_lbl)
        from PySide6.QtWidgets import QCheckBox
        self.show_logs_toggle = QCheckBox("Enable Log View")
        self.show_logs_toggle.setChecked(False)
        self.show_logs_toggle.toggled.connect(self._toggle_logs)
        toggle_row.addWidget(self.show_logs_toggle)
        toggle_row.addStretch()
        layout.addLayout(toggle_row)

        # Create log view BEFORE filter row so clear button lambda can reference it
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setObjectName("logView")
        self.log_view.setFont(QFont("Consolas", 9))
        self.log_view.setVisible(False)  # Hidden by default

        filter_row = QHBoxLayout()
        self.level_filter = QComboBox()
        self.level_filter.addItems(
            ["All", "Info", "Warning", "Error", "Telemetry"]
        )
        self.level_filter.setVisible(False)  # Hidden by default
        self.level_filter.currentTextChanged.connect(self._apply_filter)
        filter_row.addWidget(QLabel("Level:"))
        filter_row.addWidget(self.level_filter)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Search logs...")
        self.search.setVisible(False)  # Hidden by default
        self.search.textChanged.connect(self._apply_filter)
        filter_row.addWidget(self.search)

        clear_btn = QPushButton("Clear")
        clear_btn.setObjectName("secondaryBtn")
        clear_btn.clicked.connect(self._clear_logs)
        filter_row.addWidget(clear_btn)

        layout.addLayout(filter_row)
        layout.addWidget(self.log_view)

        timing_group = QGroupBox("Pipeline Timing (Last Run)")
        timing_group.setObjectName("settingsGroup")
        t = QVBoxLayout(timing_group)

        self.timing_list = QTableWidget()
        self.timing_list.setColumnCount(3)
        self.timing_list.setHorizontalHeaderLabels(
            ["Stage", "Duration (ms)", "Device"]
        )
        self.timing_list.horizontalHeader().setStretchLastSection(True)
        self.timing_list.setMaximumHeight(200)
        t.addWidget(self.timing_list)

        self.timing_list.setVisible(False)  # Hidden by default
        layout.addWidget(timing_group)

        # Internal log buffer for filtering
        self._log_buffer: list[tuple[str, str]] = []  # (level, text)
        self._max_buffer = 2000

        self.event_bus.log_entry.connect(self._add_log)
        self.event_bus.telemetry_updated.connect(self._update_timing)

    def _toggle_logs(self, enabled):
        """Toggle visibility of log view and timing data."""
        self.log_view.setVisible(enabled)
        self.timing_list.setVisible(enabled)
        self.level_filter.setVisible(enabled)
        self.search.setVisible(enabled)

    def _add_log(self, text: str):
        level = _classify_log(text)
        self._log_buffer.append((level, text))

        # Trim buffer
        if len(self._log_buffer) > self._max_buffer:
            self._log_buffer = self._log_buffer[-self._max_buffer:]

        if not self.log_view.isVisible():
            return

        # Check if this line passes current filter
        if not self._passes_filter(level, text):
            return

        self._append_colored(level, text)

    def _append_colored(self, level: str, text: str):
        """Append a log line with color based on severity level."""
        color = _LEVEL_COLORS.get(level, _LEVEL_COLORS["debug"])

        cursor = self.log_view.textCursor()
        cursor.movePosition(QTextCursor.End)

        fmt = QTextCharFormat()
        fmt.setForeground(color)

        # Bold for errors
        if level == "error":
            fmt.setFontWeight(QFont.Bold)

        cursor.insertText(text + "\n", fmt)
        self.log_view.setTextCursor(cursor)
        self.log_view.ensureCursorVisible()

    def _passes_filter(self, level: str, text: str) -> bool:
        """Check if a log entry passes the current level and search filters."""
        selected = self.level_filter.currentText().lower()
        if selected != "all":
            if selected != level:
                return False

        search_text = self.search.text().strip()
        if search_text and search_text.lower() not in text.lower():
            return False

        return True

    def _apply_filter(self, *_args):
        """Re-render the log view with current filter settings."""
        if not self.log_view.isVisible():
            return

        self.log_view.clear()
        for level, text in self._log_buffer:
            if self._passes_filter(level, text):
                self._append_colored(level, text)

    def _clear_logs(self):
        self.log_view.clear()
        self._log_buffer.clear()

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
            self.timing_list.setItem(i, 0, QTableWidgetItem(stage))
            self.timing_list.setItem(
                i, 1, QTableWidgetItem(f"{s.get('duration_ms', 0):.1f}")
            )
            self.timing_list.setItem(
                i, 2, QTableWidgetItem(s.get("device", "CPU"))
            )
