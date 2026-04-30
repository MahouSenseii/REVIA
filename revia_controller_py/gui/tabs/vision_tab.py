from PySide6.QtWidgets import (
    QScrollArea, QWidget, QVBoxLayout, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QComboBox, QSpinBox, QCheckBox,
    QPushButton, QMessageBox, QTextEdit, QSizePolicy,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from gui.widgets.settings_card import SettingsCard


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
        vision_card = SettingsCard(
            "Vision Module",
            subtitle="Engine & capture mode",
            icon="V",
        )
        g = QFormLayout()

        self.vision_engine = QComboBox()
        self.vision_engine.addItems(
            ["CLIP (Local)", "GPT-4 Vision API", "LLaVA", "None"]
        )
        g.addRow("Engine:", self.vision_engine)

        self.resolution = QComboBox()
        self.resolution.addItems(["640x480", "1280x720", "1920x1080"])
        g.addRow("Resolution:", self.resolution)
        self.resolution.setVisible(False)  # Internal: auto-detected

        self.auto_capture = QCheckBox("Auto-capture on vision queries")
        self.auto_capture.setChecked(True)
        g.addRow("", self.auto_capture)
        vision_card.add_layout(g)

        layout.addWidget(vision_card)

        # --- Camera Source ---
        cam_card = SettingsCard(
            "Camera Source",
            subtitle="USB, Luxonis, or IP stream",
            icon="C",
        )

        detect_row = QHBoxLayout()
        self.camera_combo = QComboBox()
        self.camera_combo.setMinimumWidth(200)
        detect_row.addWidget(self.camera_combo, stretch=1)
        self.detect_btn = QPushButton("Detect Cameras")
        self.detect_btn.setObjectName("secondaryBtn")
        self.detect_btn.clicked.connect(self._detect_cameras)
        detect_row.addWidget(self.detect_btn)
        cam_card.add_layout(detect_row)

        wireless_row = QHBoxLayout()
        wireless_label = QLabel("Wireless / IP:")
        wireless_label.setFont(QFont("Segoe UI", 9))
        wireless_row.addWidget(wireless_label)
        self.wireless_url = QLineEdit()
        self.wireless_url.setPlaceholderText(
            "rtsp://user:pass@192.168.1.100:554/stream (leave blank for USB/Luxonis)"
        )
        wireless_row.addWidget(self.wireless_url, stretch=1)
        cam_card.add_layout(wireless_row)

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
        cam_card.add_layout(conn_row)

        self.cam_status = QLabel("Status: Disconnected")
        self.cam_status.setObjectName("metricLabel")
        self.cam_status.setFont(QFont("Consolas", 9))
        cam_card.add_widget(self.cam_status)

        layout.addWidget(cam_card)

        # --- Object Identification ---
        obj_card = SettingsCard(
            "Object Identification",
            subtitle="Detection & classification",
            icon="O",
        )

        obj_info = QLabel(
            "Detect and classify objects in camera frames using "
            "YOLO, CLIP, or a vision-language model."
        )
        obj_info.setFont(QFont("Segoe UI", 8))
        obj_info.setWordWrap(True)
        obj_info.setObjectName("cardSubText")
        obj_card.add_widget(obj_info)

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
        obj_card.add_layout(obj_form)

        self.obj_realtime = QCheckBox("Real-time detection overlay")
        self.obj_realtime.setChecked(True)
        obj_card.add_widget(self.obj_realtime)

        self.obj_results = QTextEdit()
        self.obj_results.setReadOnly(True)
        self.obj_results.setMaximumHeight(80)
        self.obj_results.setPlaceholderText("Detection results...")
        obj_card.add_widget(self.obj_results)

        obj_btn_row = QHBoxLayout()
        self.detect_objects_btn = QPushButton("Detect Now")
        self.detect_objects_btn.setObjectName("primaryBtn")
        self.detect_objects_btn.clicked.connect(self._detect_objects)
        obj_btn_row.addWidget(self.detect_objects_btn)
        obj_card.add_layout(obj_btn_row)

        layout.addWidget(obj_card)

        # --- Gesture Recognition ---
        gest_card = SettingsCard(
            "Gesture Recognition",
            subtitle="Hand & body poses",
            icon="G",
        )

        gest_info = QLabel(
            "Recognize hand gestures and body poses via MediaPipe "
            "or a custom model. Map gestures to actions."
        )
        gest_info.setFont(QFont("Segoe UI", 8))
        gest_info.setWordWrap(True)
        gest_info.setObjectName("cardSubText")
        gest_card.add_widget(gest_info)

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
        gest_card.add_layout(gest_form)

        self.gesture_enabled = QCheckBox("Enable gesture recognition")
        self.gesture_enabled.setChecked(False)
        gest_card.add_widget(self.gesture_enabled)

        self.gesture_log = QTextEdit()
        self.gesture_log.setReadOnly(True)
        self.gesture_log.setMaximumHeight(80)
        self.gesture_log.setPlaceholderText("Gesture events...")
        gest_card.add_widget(self.gesture_log)

        gest_map_label = QLabel("Gesture Mappings:")
        gest_map_label.setFont(QFont("Segoe UI", 9, QFont.Bold))
        gest_card.add_widget(gest_map_label)

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
        gest_card.add_layout(mappings_form)

        layout.addWidget(gest_card)

        # --- Camera Preview ---
        preview_card = SettingsCard(
            "Camera Preview",
            subtitle="Live feed & snapshot",
            icon="P",
        )

        self.preview = QLabel("[Camera Inactive]")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumHeight(200)
        self.preview.setMinimumWidth(0)
        self.preview.setMaximumWidth(720)
        self.preview.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.preview.setObjectName("webcamPlaceholder")
        preview_card.add_widget(self.preview)

        snap_row = QHBoxLayout()
        self.snapshot_btn = QPushButton("Take Snapshot")
        self.snapshot_btn.setObjectName("secondaryBtn")
        self.snapshot_btn.clicked.connect(self._take_snapshot)
        self.snapshot_btn.setEnabled(False)
        snap_row.addWidget(self.snapshot_btn)
        preview_card.add_layout(snap_row)

        layout.addWidget(preview_card)
        layout.addStretch()
        self.setWidget(container)

        if self.camera_service:
            self.camera_service.frame_ready.connect(self._on_frame)
            self.camera_service.status_changed.connect(self._on_cam_status)
            self.camera_service.camera_list_updated.connect(
                self._on_cameras_detected
            )
            self.camera_service.detection_updated.connect(
                self._on_detection_updated
            )

        self.obj_engine.currentTextChanged.connect(self._sync_object_detector)
        self.obj_confidence.valueChanged.connect(self._sync_object_detector)
        self.obj_classes.textChanged.connect(self._sync_object_detector)
        self.obj_realtime.toggled.connect(self._sync_object_detector)
        self._sync_object_detector()

    def set_camera_service(self, svc):
        self.camera_service = svc
        svc.frame_ready.connect(self._on_frame)
        svc.status_changed.connect(self._on_cam_status)
        svc.camera_list_updated.connect(self._on_cameras_detected)
        svc.detection_updated.connect(self._on_detection_updated)
        self._sync_object_detector()

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
            if isinstance(cam, dict):
                label = cam.get("name", "Camera")
                self.camera_combo.addItem(label, userData=cam)
            else:
                self.camera_combo.addItem(str(cam), userData=cam)
        if cameras:
            luxonis_count = sum(
                1 for cam in cameras
                if isinstance(cam, dict)
                and str(cam.get("type", "")).lower() == "luxonis"
            )
            suffix = f" ({luxonis_count} Luxonis)" if luxonis_count else ""
            self.cam_status.setText(
                f"Status: {len(cameras)} camera(s) detected{suffix}"
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
            selected = self.camera_combo.currentData()
            if isinstance(selected, dict):
                source = selected.get("source", selected.get("id", 0))
                if isinstance(source, dict):
                    source = dict(source)
                    source.setdefault("name", selected.get("name", ""))
                elif str(selected.get("type", "")).lower() == "luxonis":
                    source = {
                        "type": "luxonis",
                        "mxid": selected.get("mxid", ""),
                        "name": selected.get("name", ""),
                    }
            elif selected is None:
                source = 0
            else:
                source = selected
        else:
            source = 0
        res = self.resolution.currentText()
        ok = self.camera_service.connect_camera(source, res)
        if ok:
            self.connect_btn.setEnabled(False)
            self.disconnect_btn.setEnabled(True)
            self.snapshot_btn.setEnabled(True)
            self._sync_object_detector()
        else:
            QMessageBox.warning(
                self, "Camera",
                f"Failed to open camera source: {self._format_camera_source(source)}",
            )

    def _disconnect_camera(self):
        if self.camera_service:
            self.camera_service.configure_object_detection(
                engine=self.obj_engine.currentText(),
                enabled=False,
                confidence=self.obj_confidence.value() / 100.0,
                classes=self.obj_classes.text(),
            )
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
        if not self.camera_service or not self.camera_service.is_active():
            self.obj_results.setPlainText(
                "Camera not connected. Connect a camera first."
            )
            return
        if "None" in self.obj_engine.currentText():
            self.obj_results.setPlainText("No engine selected.")
            return
        self._sync_object_detector()
        self.obj_results.setPlainText(
            f"Running {self.obj_engine.currentText()} on current frame..."
        )
        from PySide6.QtCore import QTimer
        QTimer.singleShot(20, self._do_detect_objects)

    def _do_detect_objects(self):
        if not self.camera_service or not self.camera_service.is_active():
            self.obj_results.setPlainText(
                "Camera not connected. Connect a camera first."
            )
            return
        detections, err = self.camera_service.detect_objects_now()
        if err:
            self.obj_results.setPlainText(err)
            return
        self.obj_results.setPlainText(
            self._format_detection_text(
                self.obj_engine.currentText(),
                detections,
                realtime=self.obj_realtime.isChecked(),
            )
        )

    def _sync_object_detector(self, *_args):
        if not self.camera_service:
            return
        engine = self.obj_engine.currentText()
        enabled = self.obj_realtime.isChecked() and ("None" not in engine)
        self.camera_service.configure_object_detection(
            engine=engine,
            enabled=enabled,
            confidence=self.obj_confidence.value() / 100.0,
            classes=self.obj_classes.text(),
            frame_stride=2,
        )
        if enabled:
            self.obj_results.setPlainText(
                f"Live {engine} overlay enabled. "
                "Bounding boxes will be drawn on the camera feed."
            )
        elif "None" in engine:
            self.obj_results.setPlainText("Object detection disabled (engine: None).")
        else:
            self.obj_results.setPlainText(
                f"{engine} loaded for manual 'Detect Now'. "
                "Enable real-time overlay to run continuously."
            )

    def _on_detection_updated(self, payload):
        if not isinstance(payload, dict):
            return
        engine = payload.get("engine", self.obj_engine.currentText())
        detections = payload.get("detections", []) or []
        error = str(payload.get("error", "")).strip()
        latency = float(payload.get("latency_ms", 0.0) or 0.0)
        realtime = bool(payload.get("realtime", False))
        if error:
            self.obj_results.setPlainText(error)
            return
        self.obj_results.setPlainText(
            self._format_detection_text(
                engine, detections, latency_ms=latency, realtime=realtime
            )
        )

    @staticmethod
    def _format_detection_text(engine, detections, latency_ms=0.0, realtime=False):
        prefix = "Live" if realtime else "Manual"
        lines = [f"[{prefix} {engine}] {len(detections)} object(s)"]
        if latency_ms > 0:
            lines[0] += f" | {latency_ms:.1f} ms"
        if not detections:
            lines.append("No objects detected in the current frame.")
            return "\n".join(lines)
        for d in detections[:8]:
            label = d.get("label", "object")
            conf = float(d.get("confidence", 0.0))
            box = d.get("box", [0, 0, 0, 0])
            lines.append(
                f" - {label} ({conf * 100:.0f}%) box={box}"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_camera_source(source):
        if isinstance(source, dict):
            name = str(source.get("name", "")).strip()
            if name:
                return name
            src_type = str(source.get("type", "")).strip()
            if src_type:
                return src_type
        return str(source)
