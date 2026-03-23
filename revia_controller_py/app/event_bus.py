from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QPixmap


class EventBus(QObject):
    """
    Central event bus for application-wide signals.

    Thread-safety:
    Qt signals are inherently thread-safe in PySide6. Signal emission from any thread
    automatically queues the signal to the main Qt event loop. Signal subscriptions
    (connections) should be made from the main thread, but emissions can safely occur
    from worker threads.
    """
    telemetry_updated = Signal(dict)
    status_changed = Signal(str)
    assistant_status_updated = Signal(dict)
    chat_token = Signal(str)
    chat_token_payload = Signal(object)
    chat_complete = Signal(str)
    chat_complete_payload = Signal(object)
    chat_request_accepted = Signal(object)
    log_entry = Signal(str)
    pipeline_timing = Signal(list)
    connection_changed = Signal(bool)
    plugins_updated = Signal(list)
    neural_updated = Signal(dict)
    camera_frame = Signal(QPixmap)
    proactive_start = Signal()  # Revia is about to initiate a conversation
    chat_sentence = Signal(str, str)  # (sentence_text, request_id) — sentence-level TTS event
    interrupt_ack = Signal()  # Server acknowledged an interrupt request
