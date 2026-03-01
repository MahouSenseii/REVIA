from PySide6.QtWidgets import QFrame, QVBoxLayout, QLabel
from PySide6.QtGui import QFont


class StatusPanel(QFrame):
    def __init__(self, event_bus, parent=None):
        super().__init__(parent)
        self.event_bus = event_bus
        self.setObjectName("statusPanel")
        self.setFixedHeight(115)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 8, 14, 8)
        layout.setSpacing(3)

        header = QLabel("Assistant Status")
        header.setObjectName("panelHeader")
        header.setFont(QFont("Segoe UI", 10, QFont.Bold))
        layout.addWidget(header)

        self.lines = {}
        items = [
            ("listening", "Listening..."),
            ("processing", "Processing Command..."),
            ("vision", "Vision: Idle"),
            ("generating", "Generating Response..."),
        ]

        for key, text in items:
            lbl = QLabel(f"  \u25cf  {text}")
            lbl.setFont(QFont("Segoe UI", 9))
            layout.addWidget(lbl)
            self.lines[key] = lbl

        self.event_bus.status_changed.connect(self._on_status)
        self.event_bus.telemetry_updated.connect(self._on_telemetry)

    def _highlight(self, active_key):
        for key, lbl in self.lines.items():
            if key == active_key:
                lbl.setStyleSheet("color: #00d4ff;")
            else:
                lbl.setStyleSheet("")

    def _on_status(self, state):
        sl = state.lower()
        if "listen" in sl:
            self._highlight("listening")
        elif "process" in sl:
            self._highlight("processing")
        elif "generat" in sl:
            self._highlight("generating")
        elif "vision" in sl:
            self._highlight("vision")
        else:
            self._highlight(None)

    def _on_telemetry(self, data):
        state = data.get("state", "Idle").lower()
        if "listen" in state:
            self._highlight("listening")
        elif "process" in state:
            self._highlight("processing")
        elif "generat" in state or "answer" in state:
            self._highlight("generating")
        else:
            self._highlight(None)
