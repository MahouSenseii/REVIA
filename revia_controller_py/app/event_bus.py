from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QPixmap


class EventBus(QObject):
    telemetry_updated = Signal(dict)
    status_changed = Signal(str)
    chat_token = Signal(str)
    chat_complete = Signal(str)
    log_entry = Signal(str)
    pipeline_timing = Signal(list)
    connection_changed = Signal(bool)
    plugins_updated = Signal(list)
    neural_updated = Signal(dict)
    camera_frame = Signal(QPixmap)
