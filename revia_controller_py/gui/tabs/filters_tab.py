from PySide6.QtWidgets import (
    QScrollArea, QWidget, QVBoxLayout, QFormLayout,
    QLabel, QTextEdit, QGroupBox, QSpinBox, QCheckBox,
)
from PySide6.QtGui import QFont


class FiltersTab(QScrollArea):
    def __init__(self, event_bus, client, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        header = QLabel("Content Filters")
        header.setObjectName("tabHeader")
        header.setFont(QFont("Segoe UI", 12, QFont.Bold))
        layout.addWidget(header)

        group = QGroupBox("Safety Filters")
        group.setObjectName("settingsGroup")
        g = QVBoxLayout(group)

        self.nsfw_filter = QCheckBox("NSFW Content Filter")
        self.nsfw_filter.setChecked(True)
        g.addWidget(self.nsfw_filter)

        self.profanity_filter = QCheckBox("Profanity Filter")
        g.addWidget(self.profanity_filter)

        self.pii_filter = QCheckBox("PII Detection & Masking")
        self.pii_filter.setChecked(True)
        g.addWidget(self.pii_filter)

        self.injection_guard = QCheckBox("Prompt Injection Guard")
        self.injection_guard.setChecked(True)
        g.addWidget(self.injection_guard)

        layout.addWidget(group)

        bounds_group = QGroupBox("Behavior Boundaries")
        bounds_group.setObjectName("settingsGroup")
        b = QFormLayout(bounds_group)

        self.max_response_len = QSpinBox()
        self.max_response_len.setRange(50, 8192)
        self.max_response_len.setValue(1024)
        b.addRow("Max Response Length:", self.max_response_len)

        self.blocked_topics = QTextEdit()
        self.blocked_topics.setMaximumHeight(80)
        self.blocked_topics.setPlaceholderText(
            "Enter blocked topics, one per line..."
        )
        b.addRow("Blocked Topics:", self.blocked_topics)

        layout.addWidget(bounds_group)
        layout.addStretch()
        self.setWidget(container)
