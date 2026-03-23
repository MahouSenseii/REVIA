import sys
import time
import glob
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2
import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap

try:
    from ultralytics import YOLO as _YOLO
    _YOLO_AVAILABLE = True
    _YOLO_IMPORT_ERROR = ""
except Exception:
    _YOLO_AVAILABLE = False
    _YOLO = None
    _YOLO_IMPORT_ERROR = str(sys.exc_info()[1])


try:
    import depthai as _DEPTHAI
    _DEPTHAI_AVAILABLE = True
    _DEPTHAI_IMPORT_ERROR = ""
except Exception:
    _DEPTHAI_AVAILABLE = False
    _DEPTHAI = None
    _DEPTHAI_IMPORT_ERROR = str(sys.exc_info()[1])


# Module-level lock for synchronizing access to global state
_GLOBAL_STATE_LOCK = threading.Lock()


class CameraService(QObject):
    frame_ready = Signal(QPixmap)
    camera_list_updated = Signal(list)
    status_changed = Signal(str)
    detection_updated = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cap = None
        self._luxonis_pipeline = None
        self._luxonis_device = None
        self._luxonis_queue = None
        self._luxonis_mxid = ""
        self._luxonis_stream = "revia_rgb"
        self._active = False
        self._source = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._grab_frame)
        self._detected_cameras = []

        self._latest_frame = None
        self._latest_annotated = None

        self._detector_enabled = False
        self._detector_engine = "None"
        self._detector_conf = 0.5
        self._detector_filter_names = []
        self._detector_frame_stride = 2
        self._frame_count = 0

        self._detector_last_results = []
        self._detector_last_error = ""
        self._detector_latency_ms = 0.0

        self._yolo_model = None
        self._yolo_model_id = ""
        self._yolo_import_last_try = 0.0
        self._yolo_import_retry_s = 5.0
        self._yolo_import_last_msg = ""

        self._detector_lock = threading.Lock()
        self._detector_pool = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="revia-yolo",
        )
        self._detector_future = None

    def detect_cameras(self, max_check=5):
        self._detected_cameras = self._detect_local_cameras(max_check=max_check)
        self._detected_cameras.extend(self._detect_luxonis_cameras())
        self.camera_list_updated.emit(self._detected_cameras)
        return self._detected_cameras

    def get_detected(self):
        return list(self._detected_cameras)

    def _detect_local_cameras(self, max_check=5):
        found = []
        for i in range(max_check):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                continue
            ret, _ = cap.read()
            if ret:
                try:
                    backend = cap.getBackendName()
                except Exception:
                    backend = "OpenCV"
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                found.append({
                    "id": i,
                    "source": i,
                    "name": f"Camera {i} ({backend} {w}x{h})",
                    "type": "local",
                })
            cap.release()
        return found

    def _detect_luxonis_cameras(self):
        ok, err = self._ensure_depthai()
        if not ok:
            if err:
                self.status_changed.emit(err)
            return []
        try:
            devices = list(_DEPTHAI.Device.getAllAvailableDevices())
        except Exception:
            return []

        found = []
        for idx, dev in enumerate(devices, start=1):
            mxid = self._safe_luxonis_mxid(dev)
            short = mxid[-8:] if mxid else f"{idx:02d}"
            protocol = ""
            try:
                protocol_obj = getattr(dev, "protocol", None)
                protocol = getattr(protocol_obj, "name", "") or str(protocol_obj or "")
            except Exception:
                protocol = ""
            suffix = f" {protocol}" if protocol else ""
            found.append({
                "id": f"luxonis:{mxid or idx}",
                "source": {
                    "type": "luxonis",
                    "mxid": mxid,
                },
                "mxid": mxid,
                "name": f"Luxonis OAK {short}{suffix}",
                "type": "luxonis",
            })
        return found

    @staticmethod
    def _safe_luxonis_mxid(device_info):
        for method in ("getMxId", "getMxid", "getDeviceId"):
            try:
                fn = getattr(device_info, method, None)
                if callable(fn):
                    val = fn()
                    if val:
                        return str(val)
            except Exception:
                continue
        try:
            mxid = device_info.getMxId()
            if mxid:
                return str(mxid)
        except Exception:
            pass
        for key in ("deviceId", "mxid", "mx_id", "mxId"):
            try:
                val = getattr(device_info, key, "")
                if val:
                    return str(val)
            except Exception:
                continue
        return ""

    def configure_object_detection(
        self,
        engine="None",
        enabled=False,
        confidence=0.5,
        classes="",
        frame_stride=2,
    ):
        eng = str(engine or "None")
        if "None" in eng:
            enabled = False
        self._detector_enabled = bool(enabled)
        self._detector_engine = eng
        self._detector_conf = max(0.05, min(0.99, float(confidence)))
        self._detector_frame_stride = max(1, int(frame_stride))

        if isinstance(classes, str):
            parsed = [
                c.strip().lower()
                for c in classes.split(",")
                if c.strip()
            ]
        elif isinstance(classes, (list, tuple)):
            parsed = [str(c).strip().lower() for c in classes if str(c).strip()]
        else:
            parsed = []
        self._detector_filter_names = parsed

        if "YOLO" not in eng:
            self._yolo_model = None
            self._yolo_model_id = ""
            self._detector_future = None

    def connect_camera(self, source, resolution="640x480"):
        self.disconnect_camera()
        self._source = source
        self._frame_count = 0

        self._cap = None
        if isinstance(source, dict):
            src_type = str(source.get("type", "local")).lower()
            if src_type == "luxonis":
                return self._connect_luxonis_camera(
                    mxid=str(source.get("mxid", "")).strip(),
                    resolution=resolution,
                    display_name=str(source.get("name", "")).strip(),
                )
            source = source.get("source", source.get("id", 0))

        if isinstance(source, str):
            src = source.strip()
            if src.lower().startswith("luxonis:"):
                return self._connect_luxonis_camera(
                    mxid=src.split(":", 1)[1].strip(),
                    resolution=resolution,
                    display_name=src,
                )
            if src:
                self._cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)

        if self._cap is None:
            try:
                self._cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW)
            except Exception:
                self.status_changed.emit(f"Invalid camera source: {source}")
                return False

        if self._cap and self._cap.isOpened():
            w, h = self._parse_resolution(resolution)
            if w > 0 and h > 0:
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            self._active = True
            self._timer.start(33)  # ~30 fps
            self.status_changed.emit("Connected")
            return True
        if self._cap:
            self._cap.release()
            self._cap = None
        self.status_changed.emit("Failed to open camera")
        return False

    def _connect_luxonis_camera(self, mxid="", resolution="640x480", display_name=""):
        ok, err = self._ensure_depthai()
        if not ok:
            self.status_changed.emit(err)
            return False

        w, h = self._parse_resolution(resolution)
        if w <= 0 or h <= 0:
            w, h = (640, 480)

        device = None
        pipeline = None
        queue = None
        try:
            use_v3_output_queue = not hasattr(_DEPTHAI.Device, "getOutputQueue")
            if use_v3_output_queue:
                if mxid:
                    device = _DEPTHAI.Device(mxid)
                else:
                    device = _DEPTHAI.Device()
                pipeline = _DEPTHAI.Pipeline(device)
                cam = pipeline.create(_DEPTHAI.node.ColorCamera)
                cam.setPreviewSize(w, h)
                cam.setInterleaved(False)
                try:
                    cam.setColorOrder(_DEPTHAI.ColorCameraProperties.ColorOrder.BGR)
                except Exception:
                    pass
                try:
                    cam.setFps(30)
                except Exception:
                    pass
                queue = cam.preview.createOutputQueue(maxSize=4, blocking=False)
                pipeline.start()
            else:
                pipeline = _DEPTHAI.Pipeline()
                if hasattr(pipeline, "createColorCamera"):
                    cam = pipeline.createColorCamera()
                    xout = pipeline.createXLinkOut()
                else:
                    cam = pipeline.create(_DEPTHAI.node.ColorCamera)
                    xout = pipeline.create(_DEPTHAI.node.XLinkOut)

                xout.setStreamName(self._luxonis_stream)
                cam.setPreviewSize(w, h)
                cam.setInterleaved(False)
                try:
                    cam.setColorOrder(_DEPTHAI.ColorCameraProperties.ColorOrder.BGR)
                except Exception:
                    pass
                try:
                    cam.setFps(30)
                except Exception:
                    pass
                cam.preview.link(xout.input)

                if mxid:
                    device = _DEPTHAI.Device(_DEPTHAI.DeviceInfo(mxid))
                else:
                    device = _DEPTHAI.Device()
                device.startPipeline(pipeline)
                queue = device.getOutputQueue(
                    name=self._luxonis_stream,
                    maxSize=4,
                    blocking=False,
                )
        except Exception as exc:
            try:
                if pipeline is not None and hasattr(pipeline, "stop"):
                    pipeline.stop()
            except Exception:
                pass
            try:
                if device is not None:
                    device.close()
            except Exception:
                pass
            self.status_changed.emit(f"Failed to open Luxonis camera: {exc}")
            return False

        self._luxonis_pipeline = pipeline
        self._luxonis_device = device
        self._luxonis_queue = queue
        self._luxonis_mxid = mxid
        self._active = True
        self._timer.start(33)
        suffix = f" [{display_name}]" if display_name else ""
        self.status_changed.emit(f"Connected (Luxonis{suffix})")
        return True

    def disconnect_camera(self):
        self._active = False
        self._timer.stop()
        if self._cap:
            self._cap.release()
            self._cap = None
        self._luxonis_queue = None
        if self._luxonis_pipeline:
            try:
                self._luxonis_pipeline.stop()
            except Exception:
                pass
            self._luxonis_pipeline = None
        if self._luxonis_device:
            try:
                self._luxonis_device.close()
            except Exception:
                pass
            self._luxonis_device = None
        self._luxonis_mxid = ""
        self._latest_frame = None
        self._latest_annotated = None
        self._detector_last_results = []
        self._detector_last_error = ""
        self._detector_future = None
        self.status_changed.emit("Disconnected")

    def is_active(self):
        return self._active

    def snapshot(self):
        if self._latest_annotated is not None:
            return self._frame_to_pixmap(self._latest_annotated.copy())
        if self._latest_frame is not None:
            return self._frame_to_pixmap(self._latest_frame.copy())
        if self._luxonis_queue is not None:
            try:
                packet = self._luxonis_queue.tryGet()
            except Exception:
                packet = None
            frame = self._packet_to_frame(packet)
            if frame is not None:
                return self._frame_to_pixmap(frame)
        if self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            if ret:
                return self._frame_to_pixmap(frame)
        return None

    def build_live_context(self, max_objects=6):
        if not self._active:
            return "Camera is not active."
        if self._latest_frame is None:
            return "Camera is active but no frame is available yet."
        h, w = self._latest_frame.shape[:2]
        if self._detector_last_error:
            return (
                f"Live camera frame available ({w}x{h}). "
                f"Object detector error: {self._detector_last_error}"
            )
        if not self._detector_last_results:
            return (
                f"Live camera frame available ({w}x{h}). "
                "No objects detected at the moment."
            )

        top = self._detector_last_results[:max(1, int(max_objects))]
        counts = {}
        confidence = {}
        for det in top:
            label = str(det.get("label", "object"))
            counts[label] = counts.get(label, 0) + 1
            confidence[label] = max(
                confidence.get(label, 0.0),
                float(det.get("confidence", 0.0)),
            )

        parts = []
        for label, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            conf = confidence.get(label, 0.0)
            suffix = "s" if count != 1 else ""
            parts.append(f"{count} {label}{suffix} ({conf * 100:.0f}% max)")
        return (
            f"Live camera frame ({w}x{h}) sees: "
            + ", ".join(parts)
            + "."
        )

    def detect_objects_now(self):
        if self._latest_frame is None:
            return [], "No camera frame available yet."
        frame = self._latest_frame.copy()
        detections, err = self._run_detector(frame, force=True)
        if detections:
            self._latest_annotated = self._draw_detections(frame, detections)
        payload = {
            "engine": self._detector_engine,
            "detections": detections,
            "latency_ms": round(self._detector_latency_ms, 2),
            "error": err,
            "realtime": self._detector_enabled,
        }
        self.detection_updated.emit(payload)
        return detections, err

    def _grab_frame(self):
        if self._luxonis_queue is not None:
            try:
                packet = self._luxonis_queue.tryGet()
            except Exception as exc:
                self.status_changed.emit(f"Luxonis stream error: {exc}")
                self.disconnect_camera()
                return
            if packet is None:
                return
            frame = self._packet_to_frame(packet)
            if frame is None:
                return
        else:
            if not self._cap or not self._cap.isOpened():
                self.disconnect_camera()
                return
            ret, frame = self._cap.read()
            if not ret:
                self.disconnect_camera()
                return

        self._latest_frame = frame
        self._frame_count = (self._frame_count + 1) % (2**31)

        annotated = frame
        if self._detector_enabled and "None" not in self._detector_engine:
            with self._detector_lock:
                if self._detector_future is not None and self._detector_future.done():
                    try:
                        detections, err = self._detector_future.result()
                    except Exception as exc:
                        detections, err = [], f"Detector worker failed: {exc}"
                    self._detector_future = None
                    if err:
                        self._detector_last_results = []
                        if err != self._detector_last_error:
                            self._detector_last_error = err
                            self.detection_updated.emit({
                                "engine": self._detector_engine,
                                "detections": [],
                                "latency_ms": 0.0,
                                "error": err,
                                "realtime": self._detector_enabled,
                            })
                    else:
                        self._detector_last_error = ""
                        self._detector_last_results = detections
                        self.detection_updated.emit({
                            "engine": self._detector_engine,
                            "detections": detections,
                            "latency_ms": round(self._detector_latency_ms, 2),
                            "error": "",
                            "realtime": self._detector_enabled,
                        })

                should_run = (
                    (self._frame_count % self._detector_frame_stride == 0)
                    or not self._detector_last_results
                )
                if should_run and self._detector_future is None:
                    self._detector_future = self._detector_pool.submit(
                        self._run_detector,
                        frame.copy(),
                        False,
                    )

            if self._detector_last_results:
                annotated = self._draw_detections(frame.copy(), self._detector_last_results)

        self._latest_annotated = annotated
        pixmap = self._frame_to_pixmap(annotated)
        self.frame_ready.emit(pixmap)

    def _run_detector(self, frame, force=False):
        engine = self._detector_engine
        if not force and (not self._detector_enabled or "None" in engine):
            return [], ""
        if "YOLO" in engine:
            return self._detect_with_yolo(frame)
        return [], f"{engine} detection is not implemented yet."

    def _detect_with_yolo(self, frame):
        with self._detector_lock:
            ok, err = self._ensure_yolo_model()
            if not ok:
                return [], err
            if self._yolo_model is None:
                return [], "YOLO model not initialized."

            t0 = time.perf_counter()
            try:
                res = self._yolo_model.predict(
                    source=frame,
                    conf=self._detector_conf,
                    verbose=False,
                )
            except Exception as exc:
                return [], f"YOLO inference failed: {exc}"
            self._detector_latency_ms = (time.perf_counter() - t0) * 1000.0

            detections = []
            if not res:
                return detections, ""

            first = res[0]
            boxes = getattr(first, "boxes", None)
            names = getattr(first, "names", {}) or {}
            if boxes is None:
                return detections, ""

            try:
                xyxy = boxes.xyxy.cpu().numpy()
                conf = boxes.conf.cpu().numpy()
                cls = boxes.cls.cpu().numpy().astype(int)
            except Exception as exc:
                return [], f"YOLO parse failed: {exc}"

            for i in range(len(conf)):
                cid = int(cls[i])
                label = names.get(cid, str(cid)) if isinstance(names, dict) else str(cid)
                lname = label.lower()
                if self._detector_filter_names and lname not in self._detector_filter_names:
                    continue
                x1, y1, x2, y2 = [int(v) for v in xyxy[i]]
                detections.append({
                    "label": label,
                    "class_id": cid,
                    "confidence": float(conf[i]),
                    "box": [x1, y1, x2, y2],
                })

            detections.sort(key=lambda d: d["confidence"], reverse=True)
            return detections, ""

    @staticmethod
    def _packet_to_frame(packet):
        if packet is None:
            return None
        try:
            frame = packet.getCvFrame()
        except Exception:
            return None
        if isinstance(frame, np.ndarray) and frame.size > 0:
            return frame
        return None

    @staticmethod
    def _parse_resolution(resolution):
        try:
            parts = str(resolution or "").lower().split("x")
            if len(parts) != 2:
                return 0, 0
            w, h = int(parts[0]), int(parts[1])
            if w <= 0 or h <= 0:
                return 0, 0
            return w, h
        except Exception:
            return 0, 0

    def _ensure_depthai(self):
        global _DEPTHAI_AVAILABLE, _DEPTHAI, _DEPTHAI_IMPORT_ERROR
        with _GLOBAL_STATE_LOCK:
            if _DEPTHAI_AVAILABLE:
                return True, ""
            try:
                import depthai as _DEPTHAI_DYNAMIC
                _DEPTHAI = _DEPTHAI_DYNAMIC
                _DEPTHAI_AVAILABLE = True
                _DEPTHAI_IMPORT_ERROR = ""
                return True, ""
            except Exception as exc:
                _DEPTHAI_IMPORT_ERROR = str(exc)

        project_root = Path(__file__).resolve().parents[2]
        site_candidates = [project_root / ".venv" / "Lib" / "site-packages"]
        site_candidates.extend(
            Path(p)
            for p in glob.glob(
                str(project_root / ".venv" / "lib" / "python*" / "site-packages")
            )
        )
        for sp in site_candidates:
            if not sp.is_dir():
                continue
            sp_str = str(sp.resolve())
            if sp_str not in sys.path:
                sys.path.insert(0, sp_str)
            try:
                import depthai as _DEPTHAI_DYNAMIC
                _DEPTHAI = _DEPTHAI_DYNAMIC
                _DEPTHAI_AVAILABLE = True
                _DEPTHAI_IMPORT_ERROR = ""
                return True, ""
            except Exception:
                continue

        venv_py = project_root / ".venv" / "Scripts" / "python.exe"
        install_hint = "python -m pip install depthai"
        if venv_py.is_file():
            install_hint = f"\"{venv_py}\" -m pip install depthai"
        msg = (
            "Luxonis support requires depthai in this interpreter. "
            f"python={sys.executable} | error={_DEPTHAI_IMPORT_ERROR}. "
            f"Install with: {install_hint}"
        )
        return False, msg

    def _ensure_yolo_model(self):
        global _YOLO_AVAILABLE, _YOLO, _YOLO_IMPORT_ERROR
        with _GLOBAL_STATE_LOCK:
            if not _YOLO_AVAILABLE:
                now = time.monotonic()
                if (
                    self._yolo_import_last_msg
                    and (now - self._yolo_import_last_try) < self._yolo_import_retry_s
                ):
                    return False, self._yolo_import_last_msg
                self._yolo_import_last_try = now
                # Retry import at runtime so users can install ultralytics
                # without needing to fully rebuild anything.
                try:
                    from ultralytics import YOLO as _YOLO_DYNAMIC
                    _YOLO = _YOLO_DYNAMIC
                    _YOLO_AVAILABLE = True
                    _YOLO_IMPORT_ERROR = ""
                    self._yolo_import_last_msg = ""
                except Exception as exc:
                    _YOLO_IMPORT_ERROR = str(exc)
                    # Fallback: try importing from project's local .venv even if
                    # the controller was launched under a different interpreter.
                    project_root = Path(__file__).resolve().parents[2]
                    site_candidates = [
                        project_root / ".venv" / "Lib" / "site-packages",
                    ]
                    site_candidates.extend(
                        Path(p) for p in glob.glob(
                            str(project_root / ".venv" / "lib" / "python*" / "site-packages")
                        )
                    )
                    for sp in site_candidates:
                        if not sp.is_dir():
                            continue
                        sp_str = str(sp.resolve())
                        if sp_str not in sys.path:
                            sys.path.insert(0, sp_str)
                        try:
                            from ultralytics import YOLO as _YOLO_DYNAMIC
                            _YOLO = _YOLO_DYNAMIC
                            _YOLO_AVAILABLE = True
                            _YOLO_IMPORT_ERROR = ""
                            self._yolo_import_last_msg = ""
                            break
                        except Exception:
                            continue
                    if not _YOLO_AVAILABLE:
                        msg = (
                            "YOLO unavailable in this interpreter. "
                            f"python={sys.executable} | error={_YOLO_IMPORT_ERROR}. "
                            "Install into that same environment: python -m pip install ultralytics"
                        )
                        self._yolo_import_last_msg = msg
                        return False, msg

        model_id = "yolov8n.pt"
        if "YOLO-World" in self._detector_engine:
            model_id = "yolov8s-worldv2.pt"

        if self._yolo_model is not None and self._yolo_model_id == model_id:
            return True, ""

        try:
            self._yolo_model = _YOLO(model_id)
            self._yolo_model_id = model_id
            self._yolo_import_last_msg = ""
            return True, ""
        except Exception as exc:
            self._yolo_model = None
            self._yolo_model_id = ""
            return False, f"Failed to load YOLO model '{model_id}': {exc}"

    def close(self):
        """Explicitly release the camera and shut down the detector thread pool.

        Call this when the owning component is destroyed rather than relying on
        ``__del__``, which is not guaranteed to run when circular references exist.
        """
        self.disconnect_camera()
        try:
            self._detector_pool.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            # cancel_futures keyword was added in Python 3.9
            self._detector_pool.shutdown(wait=False)
        except Exception:
            pass

    def __del__(self):
        # Best-effort cleanup — always prefer calling close() explicitly.
        try:
            self._detector_pool.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            try:
                self._detector_pool.shutdown(wait=False)
            except Exception:
                pass
        except Exception:
            pass

    @staticmethod
    def _draw_detections(frame, detections):
        for d in detections:
            x1, y1, x2, y2 = d.get("box", [0, 0, 0, 0])
            conf = float(d.get("confidence", 0.0))
            label = str(d.get("label", "object"))
            cid = int(d.get("class_id", 0))
            color = (
                int((37 * (cid + 3)) % 255),
                int((17 * (cid + 7)) % 255),
                int((97 * (cid + 11)) % 255),
            )
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"{label} {conf * 100:.0f}%"
            (tw, th), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            ty = max(0, y1 - th - 6)
            cv2.rectangle(
                frame, (x1, ty), (x1 + tw + 6, ty + th + 6), color, -1
            )
            cv2.putText(
                frame,
                text,
                (x1 + 3, ty + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (10, 10, 10),
                1,
                cv2.LINE_AA,
            )
        return frame

    @staticmethod
    def _frame_to_pixmap(frame):
        if isinstance(frame, np.ndarray):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            return QPixmap.fromImage(img.copy())
        return QPixmap()
