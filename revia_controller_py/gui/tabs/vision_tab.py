from PySide6.QtWidgets import (
    QScrollArea, QWidget, QVBoxLayout, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QComboBox, QGroupBox, QSpinBox, QCheckBox,
    QPushButton, QMessageBox, QTextEdit,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont


class VisionTab(QScrollArea):
    def __init__(self, event_bus, client, camera_service=None, parent=None):
        super().__init__(parent)
        self.event_bus = event_bus
        self.client = client
        self.camera_service = camera_service
        self.setWidgetResizable(True)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        header = QLabel("Vision Settings")
        header.setObjectName("tabHeader")
        header.setFont(QFont("Segoe UI", 12, QFont.Bold))
        layout.addWidget(header)

        # --- Vision Module ---
        group = QGroupBox("Vision Module")
        group.setObjectName("settingsGroup")
        g = QFormLayout(group)

        self.vision_engine = QComboBox()
        self.vision_engine.addItems(
            ["CLIP (Local)", "GPT-4 Vision API", "LLaVA", "None"]
        )
        g.addRow("Engine:", self.vision_engine)

        self.resolution = QComboBox()
        self.resolution.addItems(["640x480", "1280x720", "1920x1080"])
        g.addRow("Resolution:", self.resolution)

        self.auto_capture = QCheckBox("Auto-capture on vision queries")
        self.auto_capture.setChecked(True)
        g.addRow("", self.auto_capture)

        layout.addWidget(group)

        # --- Camera Source ---
        cam_group = QGroupBox("Camera Source")
        cam_group.setObjectName("settingsGroup")
        cg = QVBoxLayout(cam_group)

        detect_row = QHBoxLayout()
        self.camera_combo = QComboBox()
        self.camera_combo.setMinimumWidth(200)
        detect_row.addWidget(self.camera_combo, stretch=1)
        self.detect_btn = QPushButton("Detect Cameras")
        self.detect_btn.setObjectName("secondaryBtn")
        self.detect_btn.clicked.connect(self._detect_cameras)
        detect_row.addWidget(self.detect_btn)
        cg.addLayout(detect_row)

        wireless_row = QHBoxLayout()
        wireless_label = QLabel("Wireless / IP:")
        wireless_label.setFont(QFont("Segoe UI", 9))
        wireless_row.addWidget(wireless_label)
        self.wireless_url = QLineEdit()
        self.wireless_url.setPlaceholderText(
            "rtsp://user:pass@192.168.1.100:554/stream"
        )
        wireless_row.addWidget(self.wireless_url, stretch=1)
        cg.addLayout(wireless_row)

        conn_row = QHBoxLayout()
        self.connect_btn = QPushButton("Connect Camera")
        self.connect_btn.setObjectName("primaryBtn")
        self.connect_btn.clicked.connect(self._connect_camera)
        conn_row.addWidget(self.connect_btn)
        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.setObjectName("secondaryBtn")
        self.disconnect_btn.clicked.connect(self._disconnect_camera)
        self.disconnect_btn.setEnabled(False)
        conn_row.addWidget(self.disconnect_btn)
        cg.addLayout(conn_row)

        self.cam_status = QLabel("Status: Disconnected")
        self.cam_status.setObjectName("metricLabel")
        self.cam_status.setFont(QFont("Consolas", 9))
        cg.addWidget(self.cam_status)

        layout.addWidget(cam_group)

        # --- Object Identification ---
        obj_group = QGroupBox("Object Identification")
        obj_group.setObjectName("settingsGroup")
        og = QVBoxLayout(obj_group)

        obj_info = QLabel(
            "Detect and classify objects in camera frames using "
            "YOLO, CLIP, or a vision-language model."
        )
        obj_info.setFont(QFont("Segoe UI", 8))
        obj_info.setWordWrap(True)
        og.addWidget(obj_info)

        obj_form = QFormLayout()
        self.obj_engine = QComboBox()
        self.obj_engine.addItems([
            "YOLOv8 (Local)", "YOLO-World (Open Vocab)",
            "CLIP Zero-Shot", "GPT-4 Vision API", "None",
        ])
        obj_form.addRow("Engine:", self.obj_engine)

        self.obj_confidence = QSpinBox()
        self.obj_confidence.setRange(10, 100)
        self.obj_confidence.setValue(50)
        self.obj_confidence.setSuffix("%")
        obj_form.addRow("Min Confidence:", self.obj_confidence)

        self.obj_classes = QLineEdit()
        self.obj_classes.setPlaceholderText(
            "person, cat, dog, cup, phone... (blank = all)"
        )
        obj_form.addRow("Filter Classes:", self.obj_classes)
        og.addLayout(obj_form)

        self.obj_realtime = QCheckBox("Real-time detection overlay")
        self.obj_realtime.setChecked(False)
        og.addWidget(self.obj_realtime)

        self.obj_results = QTextEdit()
        self.obj_results.setReadOnly(True)
        self.obj_results.setMaximumHeight(80)
        self.obj_results.setPlaceholderText("Detection results...")
        og.addWidget(self.obj_results)

        obj_btn_row = QHBoxLayout()
        self.detect_objects_btn = QPushButton("Detect Now")
        self.detect_objects_btn.setObjectName("primaryBtn")
        self.detect_objects_btn.clicked.connect(self._detect_objects)
        obj_btn_row.addWidget(self.detect_objects_btn)
        og.addLayout(obj_btn_row)

        layout.addWidget(obj_group)

        # --- Gesture Recognition ---
        gest_group = QGroupBox("Gesture Recognition")
        gest_group.setObjectName("settingsGroup")
        gg = QVBoxLayout(gest_group)

        gest_info = QLabel(
            "Recognize hand gestures and body poses via MediaPipe "
            "or a custom model. Map gestures to actions."
        )
        gest_info.setFont(QFont("Segoe UI", 8))
        gest_info.setWordWrap(True)
        gg.addWidget(gest_info)

        gest_form = QFormLayout()
        self.gesture_engine = QComboBox()
        self.gesture_engine.addItems([
            "MediaPipe Hands", "MediaPipe Pose",
            "Custom Model", "None",
        ])
        gest_form.addRow("Engine:", self.gesture_engine)

        self.gesture_sensitivity = QSpinBox()
        self.gesture_sensitivity.setRange(1, 100)
        self.gesture_sensitivity.setValue(70)
        self.gesture_sensitivity.setSuffix("%")
        gest_form.addRow("Sensitivity:", self.gesture_sensitivity)
        gg.addLayout(gest_form)

        self.gesture_enabled = QCheckBox("Enable gesture recognition")
        self.gesture_enabled.setChecked(False)
        gg.addWidget(self.gesture_enabled)

        self.gesture_log = QTextEdit()
        self.gesture_log.setReadOnly(True)
        self.gesture_log.setMaximumHeight(80)
        self.gesture_log.setPlaceholderText("Gesture events...")
        gg.addWidget(self.gesture_log)

        gest_map_label = QLabel("Gesture Mappings:")
        gest_map_label.setFont(QFont("Segoe UI", 9, QFont.Bold))
        gg.addWidget(gest_map_label)

        mappings_form = QFormLayout()
        self.gesture_wave = QComboBox()
        self.gesture_wave.setEditable(True)
        self.gesture_wave.addItems([
            "Greet / Wave Back", "Toggle Listening", "No Action",
        ])
        mappings_form.addRow("Wave:", self.gesture_wave)

        self.gesture_thumbsup = QComboBox()
        self.gesture_thumbsup.setEditable(True)
        self.gesture_thumbsup.addItems([
            "Confirm / Yes", "Like Response", "No Action",
        ])
        mappings_form.addRow("Thumbs Up:", self.gesture_thumbsup)

        self.gesture_palm = QComboBox()
        self.gesture_palm.setEditable(True)
        self.gesture_palm.addItems([
            "Stop / Pause", "Mute Output", "No Action",
        ])
        mappings_form.addRow("Open Palm:", self.gesture_palm)

        self.gesture_point = QComboBox()
        self.gesture_point.setEditable(True)
        self.gesture_point.addItems([
            "Focus Object", "Take Snapshot", "No Action",
        ])
        mappings_form.addRow("Point:", self.gesture_point)
        gg.addLayout(mappings_form)

        layout.addWidget(gest_group)

        # --- Camera Preview ---
        preview_group = QGroupBox("Camera Preview")
        preview_group.setObjectName("settingsGroup")
        p = QVBoxLayout(preview_group)

        self.preview = QLabel("[Camera Inactive]")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumHeight(200)
        self.preview.setObjectName("webcamPlaceholder")
        p.addWidget(self.preview)

        snap_row = QHBoxLayout()
        self.snapshot_btn = QPushButton("Take Snapshot")
        self.snapshot_btn.setObjectName("secondaryBtn")
        self.snapshot_btn.clicked.connect(self._take_snapshot)
        self.snapshot_btn.setEnabled(False)
        snap_row.addWidget(self.snapshot_btn)
        p.addLayout(snap_row)

        layout.addWidget(preview_group)
        layout.addStretch()
        self.setWidget(container)

        if self.camera_service:
            self.camera_service.frame_ready.connect(self._on_frame)
            self.camera_service.status_changed.connect(self._on_cam_status)
            self.camera_service.camera_list_updated.connect(
                self._on_cameras_detected
            )

    def set_camera_service(self, svc):
        self.camera_service = svc
        svc.frame_ready.connect(self._on_frame)
        svc.status_changed.connect(self._on_cam_status)
        svc.camera_list_updated.connect(self._on_cameras_detected)

    # --- Camera ---

    def _detect_cameras(self):
        if not self.camera_service:
            return
        self.detect_btn.setEnabled(False)
        self.detect_btn.setText("Scanning...")
        from PySide6.QtCore import QTimer
        QTimer.singleShot(50, self._do_detect)

    def _do_detect(self):
        if self.camera_service:
            cameras = self.camera_service.detect_cameras(max_check=8)
            if not cameras:
                self.cam_status.setText("Status: No cameras found")
        self.detect_btn.setEnabled(True)
        self.detect_btn.setText("Detect Cameras")

    def _on_cameras_detected(self, cameras):
        self.camera_combo.clear()
        for cam in cameras:
            self.camera_combo.addItem(cam["name"], userData=cam["id"])
        if cameras:
            self.cam_status.setText(
                f"Status: {len(cameras)} camera(s) detected"
            )

    def _connect_camera(self):
        if not self.camera_service:
            QMessageBox.warning(
                self, "Camera", "Camera service not available."
            )
            return
        wireless = self.wireless_url.text().strip()
        if wireless:
            source = wireless
        elif self.camera_combo.count() > 0:
            source = self.camera_combo.currentData()
            if source is None:
                source = 0
        else:
            source = 0
        res = self.resolution.currentText()
        ok = self.camera_service.connect_camera(source, res)
        if ok:
            self.connect_btn.setEnabled(False)
            self.disconnect_btn.setEnabled(True)
            self.snapshot_btn.setEnabled(True)
        else:
            QMessageBox.warning(
                self, "Camera",
                f"Failed to open camera source: {source}",
            )

    def _disconnect_camera(self):
        if self.camera_service:
            self.camera_service.disconnect_camera()
        self.connect_btn.setEnabled(True)
        self.disconnect_btn.setEnabled(False)
        self.snapshot_btn.setEnabled(False)
        self.preview.clear()
        self.preview.setText("[Camera Inactive]")

    def _take_snapshot(self):
        if self.camera_service:
            pix = self.camera_service.snapshot()
            if pix:
                self.preview.setPixmap(
                    pix.scaled(
                        self.preview.size(),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation,
                    )
                )

    def _on_frame(self, pixmap):
        scaled = pixmap.scaled(
            self.preview.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.preview.setPixmap(scaled)

    def _on_cam_status(self, status):
        self.cam_status.setText(f"Status: {status}")
        if "Disconnected" in status or "Failed" in status:
            self.connect_btn.setEnabled(True)
            self.disconnect_btn.setEnabled(False)
            self.snapshot_btn.setEnabled(False)

    # --- Object detection (stub) ---

    def _detect_objects(self):
        engine = self.obj_engine.currentText()
        if "None" in engine:
            self.obj_results.setPlainText("No engine selected.")
            return
        self.obj_results.setPlainText(f"Running {engine}...")
        from PySide6.QtCore import QTimer
        QTimer.singleShot(500, self._do_detect_objects)

    def _do_detect_objects(self):
        engine = self.obj_engine.currentText()
        if self.camera_service and self.camera_service.is_active():
            self.obj_results.setPlainText(
                f"[{engine}] Detected objects:\n"
                f"  - person (92%)\n"
                f"  - laptop (87%)\n"
                f"  - chair (74%)\n"
                f"  - cup (68%)\n"
                f"\nNote: Connect a real {engine} model for actual detection."
            )
        else:
            self.obj_results.setPlainText(
                "Camera not connected. Connect a camera first."
            )
