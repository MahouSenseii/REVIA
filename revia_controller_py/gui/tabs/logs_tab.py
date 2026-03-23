from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QTextEdit, QComboBox, QGroupBox,
    QPushButton, QTableWidget, QTableWidgetItem,
)
from PySide6.QtGui import QFont


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

        filter_row = QHBoxLayout()
        self.level_filter = QComboBox()
        self.level_filter.addItems(
            ["All", "Info", "Warning", "Error", "Telemetry"]
        )
        self.level_filter.setVisible(False)  # Hidden by default
        filter_row.addWidget(QLabel("Level:"))
        filter_row.addWidget(self.level_filter)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Search logs...")
        self.search.setVisible(False)  # Hidden by default
        filter_row.addWidget(self.search)

        clear_btn = QPushButton("Clear")
        clear_btn.setObjectName("secondaryBtn")
        clear_btn.clicked.connect(lambda: self.log_view.clear())
        filter_row.addWidget(clear_btn)

        layout.addLayout(filter_row)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setObjectName("logView")
        self.log_view.setFont(QFont("Consolas", 9))
        self.log_view.setVisible(False)  # Hidden by default
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

        self.event_bus.log_entry.connect(self._add_log)
        self.event_bus.telemetry_updated.connect(self._update_timing)

    def _toggle_logs(self, enabled):
        """Toggle visibility of log view and timing data."""
        self.log_view.setVisible(enabled)
        self.timing_list.setVisible(enabled)
        self.level_filter.setVisible(enabled)
        self.search.setVisible(enabled)

    def _add_log(self, text):
        if self.log_view.isVisible():
            self.log_view.append(text)

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
