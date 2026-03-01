import cv2
import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap


class CameraService(QObject):
    frame_ready = Signal(QPixmap)
    camera_list_updated = Signal(list)
    status_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cap = None
        self._active = False
        self._source = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._grab_frame)
        self._detected_cameras = []

    def detect_cameras(self, max_check=5):
        self._detected_cameras = []
        for i in range(max_check):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    backend = cap.getBackendName()
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    self._detected_cameras.append({
                        "id": i,
                        "name": f"Camera {i} ({backend} {w}x{h})",
                        "type": "local",
                    })
                cap.release()
        self.camera_list_updated.emit(self._detected_cameras)
        return self._detected_cameras

    def get_detected(self):
        return list(self._detected_cameras)

    def connect_camera(self, source, resolution="640x480"):
        self.disconnect_camera()
        self._source = source

        if isinstance(source, str) and source.strip():
            self._cap = cv2.VideoCapture(source)
        else:
            self._cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW)

        if self._cap and self._cap.isOpened():
            try:
                w, h = [int(x) for x in resolution.split("x")]
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            except ValueError:
                pass
            self._active = True
            self._timer.start(33)  # ~30 fps
            self.status_changed.emit("Connected")
            return True
        else:
            self.status_changed.emit("Failed to open camera")
            return False

    def disconnect_camera(self):
        self._active = False
        self._timer.stop()
        if self._cap:
            self._cap.release()
            self._cap = None
        self.status_changed.emit("Disconnected")

    def is_active(self):
        return self._active

    def snapshot(self):
        if self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            if ret:
                return self._frame_to_pixmap(frame)
        return None

    def _grab_frame(self):
        if not self._cap or not self._cap.isOpened():
            self.disconnect_camera()
            return
        ret, frame = self._cap.read()
        if ret:
            pixmap = self._frame_to_pixmap(frame)
            self.frame_ready.emit(pixmap)
        else:
            self.disconnect_camera()

    @staticmethod
    def _frame_to_pixmap(frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        return QPixmap.fromImage(img)
