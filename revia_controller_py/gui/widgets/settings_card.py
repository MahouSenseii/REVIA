from PySide6.QtWidgets import (
    QFrame, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy,
)
from PySide6.QtGui import QFont


class SettingsCard(QFrame):
    """A card-style container with a header bar and a content area."""

    def __init__(self, title, subtitle="", icon="", parent=None):
        super().__init__(parent)
        self.setObjectName("settingsCard")
        self.setFrameShape(QFrame.NoFrame)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # --- Header bar ---
        header_bar = QFrame()
        header_bar.setObjectName("cardHeaderBar")
        header_bar.setFrameShape(QFrame.NoFrame)
        header_layout = QHBoxLayout(header_bar)
        header_layout.setContentsMargins(14, 8, 14, 8)
        header_layout.setSpacing(8)

        if icon:
            icon_lbl = QLabel(icon)
            icon_lbl.setObjectName("cardIcon")
            header_layout.addWidget(icon_lbl)

        title_lbl = QLabel(title.upper())
        title_lbl.setObjectName("cardHeader")
        title_lbl.setFont(QFont("Segoe UI", 9, QFont.Bold))
        header_layout.addWidget(title_lbl)

        header_layout.addStretch()

        if subtitle:
            sub_lbl = QLabel(subtitle)
            sub_lbl.setObjectName("cardSubText")
            sub_lbl.setFont(QFont("Segoe UI", 8))
            header_layout.addWidget(sub_lbl)

        outer.addWidget(header_bar)

        # --- Body ---
        body = QWidget()
        body.setObjectName("cardBody")
        self._body_layout = QVBoxLayout(body)
        self._body_layout.setContentsMargins(14, 12, 14, 14)
        self._body_layout.setSpacing(8)
        outer.addWidget(body)

    def add_widget(self, widget):
        self._body_layout.addWidget(widget)

    def add_layout(self, layout):
        self._body_layout.addLayout(layout)

    def body_layout(self):
        return self._body_layout
