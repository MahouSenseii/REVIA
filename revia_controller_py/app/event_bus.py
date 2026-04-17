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

    Signal payload reference
    -------------------------
    telemetry_updated(dict)
        Full telemetry snapshot from the core server WebSocket. Keys include:
        "state", "llm_connection", "emotion", "request_lifecycle",
        "conversation_readiness", "behavior", "runtime_status".

    status_changed(str)
        Human-readable status string for simple status-bar display.

    assistant_status_updated(dict)
        Pre-rendered UI snapshot from AssistantStatusManager. Keys include:
        "assistant_state", "stt_state", "tts_state", "thinking_timer", etc.

    chat_token(str)
        A single raw LLM output token (streaming, in order).

    chat_token_payload(object -> dict)
        Enriched token event: {"token": str, "request_id": str, ...}.

    chat_complete(str)
        Final assembled reply text (entire response as one string).

    chat_complete_payload(object -> dict)
        Enriched completion event: {"text": str, "request_id": str,
        "success": bool, "speakable": bool, "error_type": str, ...}.

    chat_request_accepted(object -> dict)
        Server confirmed it accepted the request: {"request_id": str}.

    log_entry(str)
        Single log line for display in the log tab. Already formatted.

    pipeline_timing(list)
        List of TimingSample-like dicts from RequestTimingCollector.

    connection_changed(bool)
        True = connected to core server, False = disconnected.

    plugins_updated(list)
        List of active plugin descriptor dicts.

    neural_updated(dict)
        EmotionNet/neural analysis result: {"label": str, "score": float, ...}.

    ui_theme_changed(str)
        New theme name string (e.g. "dark", "light").

    camera_frame(QPixmap)
        Latest camera frame as a QPixmap for vision display.

    proactive_start()
        No payload. Fired just before Revia autonomously initiates a turn.

    chat_sentence(str, str)
        (sentence_text, request_id) — a complete sentence extracted from the
        LLM stream, ready for TTS synthesis. Emitted in streaming order.

    interrupt_ack()
        No payload. Server acknowledged a barge-in / interrupt request.
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
    ui_theme_changed = Signal(str)
    camera_frame = Signal(QPixmap)
    proactive_start = Signal()
    chat_sentence = Signal(str, str)  # (sentence_text, request_id)
    interrupt_ack = Signal()

    # Sing mode signals
    sing_state_changed = Signal(str)           # new state string
    sing_progress = Signal(str, int, int)      # (stage, current, total)
    sing_lyrics_update = Signal(int, str)      # (line_index, lyric_text)
    sing_queue_changed = Signal()              # queue was modified
