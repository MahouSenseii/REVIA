from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QVBoxLayout, QGridLayout, QLabel, QWidget,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont


class InferencePanel(QFrame):
    def __init__(self, event_bus, parent=None):
        super().__init__(parent)
        self.event_bus = event_bus
        self.setObjectName("inferencePanel")
        self.setFixedHeight(170)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(20)

        metrics_widget = QWidget()
        metrics_layout = QGridLayout(metrics_widget)
        metrics_layout.setSpacing(4)
        metrics_layout.setContentsMargins(0, 0, 0, 0)

        header = QLabel("Inference")
        header.setObjectName("panelHeader")
        header.setFont(QFont("Segoe UI", 10, QFont.Bold))
        metrics_layout.addWidget(header, 0, 0, 1, 2)

        self.metrics = {}
        labels = [
            ("latency", "Latency (ms):", "---"),
            ("tokens_sec", "Tokens/sec:", "---"),
            ("context", "Context Length:", "---"),
            ("emotion", "Emotion:", "---"),
            ("device", "Device:", "CPU"),
            ("backend", "Backend:", "---"),
        ]

        for i, (key, label, default) in enumerate(labels):
            lbl = QLabel(label)
            lbl.setFont(QFont("Segoe UI", 9))
            lbl.setObjectName("metricLabel")
            val = QLabel(default)
            val.setFont(QFont("Consolas", 9, QFont.Bold))
            val.setObjectName("metricValue")
            metrics_layout.addWidget(lbl, i + 1, 0)
            metrics_layout.addWidget(val, i + 1, 1)
            self.metrics[key] = val

        layout.addWidget(metrics_widget)

        cam_frame = QFrame()
        cam_frame.setObjectName("webcamFrame")
        cam_frame.setMinimumWidth(200)
        cam_layout = QVBoxLayout(cam_frame)
        cam_layout.setAlignment(Qt.AlignCenter)

        cam_label = QLabel("Webcam Preview")
        cam_label.setObjectName("webcamLabel")
        cam_label.setAlignment(Qt.AlignCenter)
        cam_label.setFont(QFont("Segoe UI", 9))
        cam_layout.addWidget(cam_label)

        self.cam_view = QLabel("[No Camera Feed]")
        self.cam_view.setAlignment(Qt.AlignCenter)
        self.cam_view.setObjectName("webcamPlaceholder")
        self.cam_view.setMinimumHeight(100)
        cam_layout.addWidget(self.cam_view)

        layout.addWidget(cam_frame, stretch=1)

        self.event_bus.telemetry_updated.connect(self._on_telemetry)

    def on_camera_frame(self, pixmap):
        scaled = pixmap.scaled(
            self.cam_view.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.cam_view.setPixmap(scaled)

    def _on_telemetry(self, data):
        llm = data.get("llm", {})
        emotion = data.get("emotion", {})
        system = data.get("system", {})
        spans = data.get("recent_spans", [])

        total_latency = sum(
            s.get("duration_ms", 0) for s in spans[-6:]
        ) if spans else 0

        self.metrics["latency"].setText(f"{total_latency:.1f}")
        self.metrics["tokens_sec"].setText(
            f"{llm.get('tokens_per_second', 0):.1f}"
        )
        self.metrics["context"].setText(f"{llm.get('context_length', 0)}")

        em_label = emotion.get("label", "---")
        em_conf = emotion.get("confidence", 0)
        self.metrics["emotion"].setText(f"{em_label} ({em_conf:.0%})")
        self.metrics["device"].setText(system.get("device", "CPU"))
        self.metrics["backend"].setText(system.get("backend", "---"))
