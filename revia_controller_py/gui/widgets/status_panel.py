from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.ui_status import apply_status_style, clear_status_role, role_from_state


class StatusField(QWidget):
    def __init__(self, label_text, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.label = QLabel(f"{label_text}:")
        self.label.setFont(QFont("Segoe UI", 9))
        self.label.setMinimumWidth(64)
        self.label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.value = QLabel("---")
        self.value.setFont(QFont("Segoe UI", 9))
        self.value.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.value.setWordWrap(False)
        self.value.setMinimumWidth(0)
        self.value.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        layout.addWidget(self.label)
        layout.addWidget(self.value, 1)

    def set_status(self, value, style=""):
        """Update the displayed value and apply a visual role.

        ``style`` accepts either:
        * A semantic role name (``"success"``, ``"warning"``, ``"error"``,
          ``"muted"``, ``"accent"``, ``"info"``) — preferred.
        * A legacy CSS string (``"color: #2dd4bf;"``).
        """
        text = str(value or "---")
        self.value.setText(text)
        self.value.setToolTip(text)
        if style:
            apply_status_style(self.value, legacy_style=style, role=style if "#" not in style else None)
        else:
            clear_status_role(self.value)


class StatusPanel(QFrame):
    def __init__(self, event_bus, parent=None):
        super().__init__(parent)
        self.event_bus = event_bus
        self._has_live_status = False
        self.setObjectName("statusPanel")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 8, 14, 8)
        layout.setSpacing(6)

        header = QLabel("Assistant Status")
        header.setObjectName("panelHeader")
        header.setFont(QFont("Segoe UI", 10, QFont.Bold))
        layout.addWidget(header)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(3)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)

        self.state_field = StatusField("State")
        self.model_field = StatusField("Model")
        self.think_time_field = StatusField("Think Time")
        self.tts_gen_field = StatusField("TTS Gen")
        self.online_field = StatusField("Online")
        self.filter_field = StatusField("Filter")
        self.voice_field = StatusField("Voice")
        self.vision_field = StatusField("Vision")
        self.stt_field = StatusField("STT")
        self.stt_time_field = StatusField("STT Time")
        self.tts_field = StatusField("TTS")
        self.tts_time_field = StatusField("TTS Time")
        self.emotion_field = StatusField("Emotion")
        self.persona_field = StatusField("Persona")

        rows = (
            (self.state_field, self.model_field),
            (self.think_time_field, self.tts_gen_field),
            (self.online_field, self.filter_field),
            (self.voice_field, self.vision_field),
            (self.stt_field, self.stt_time_field),
            (self.tts_field, self.tts_time_field),
            (self.emotion_field, self.persona_field),
        )
        for row_index, (left_field, right_field) in enumerate(rows):
            grid.addWidget(left_field, row_index, 0)
            grid.addWidget(right_field, row_index, 1)

        self.event_bus.status_changed.connect(self._on_status)
        self.event_bus.telemetry_updated.connect(self._on_telemetry)
        self.event_bus.assistant_status_updated.connect(self._on_assistant_status)

    def _on_status(self, state):
        state = str(state or "Unknown")
        self.state_field.set_status(state, self._state_style(state))

    def _on_telemetry(self, data):
        if self._has_live_status:
            return
        if not isinstance(data, dict):
            return
        runtime = data.get("runtime_status", {}) or {}
        architecture = (data.get("architecture", {}) or {}).get("modules", {}) or {}
        request_lifecycle = data.get("request_lifecycle", {}) or {}
        active_turn = request_lifecycle.get("active_turn", {}) or {}
        if not runtime:
            return
        vision_enabled = bool(
            runtime.get("vision_enabled", architecture.get("vision", {}).get("online", False))
        )
        vision_state = str(
            runtime.get("vision_state")
            or ("On" if vision_enabled else "Off")
        )
        self._apply_snapshot(
            {
                "assistant_state": str(data.get("state", "Unknown") or "Unknown"),
                "assistant_state_detail": str(active_turn.get("lifecycle_reason", "") or ""),
                "model_name": str(runtime.get("current_model_name") or "---"),
                "model_ready": bool(runtime.get("model_ready", False)),
                "model_state": str((data.get("llm_connection", {}) or {}).get("state", "Disconnected")),
                "online_enabled": bool(runtime.get("online_enabled", runtime.get("web_search_enabled"))),
                "filters_enabled": bool(runtime.get("safety_filter_enabled")),
                "filter_level": str(runtime.get("content_filter_level", "standard")),
                "voice_enabled": bool(runtime.get("voice_output_enabled")),
                "voice_name": str(runtime.get("tts_engine") or runtime.get("current_tts_voice") or "none"),
                "vision_enabled": vision_enabled,
                "vision_model": str(runtime.get("vision_model") or ""),
                "vision_state": vision_state,
                "stt_status_text": str(runtime.get("stt_state", "Disabled")),
                "stt_time_text": self._fallback_timer_text(
                    runtime=runtime,
                    timer_key="stt_timer",
                    current_key="stt_current_elapsed",
                    state_key="stt_state",
                    active_states=("Listening", "Processing"),
                ),
                "tts_status_text": str(runtime.get("tts_state", "Disabled")),
                "tts_time_text": self._fallback_timer_text(
                    runtime=runtime,
                    timer_key="tts_timer",
                    current_key="tts_current_elapsed",
                    state_key="tts_state",
                    active_states=("Generating", "Speaking"),
                ),
                "current_emotion": str(runtime.get("current_emotion", "Neutral")),
                "current_persona": str(runtime.get("active_persona_profile_name", "Revia")),
            }
        )

    def _on_assistant_status(self, snapshot):
        if not isinstance(snapshot, dict):
            return
        self._has_live_status = True
        self._apply_snapshot(snapshot)

    def _apply_snapshot(self, snapshot):
        state = str(snapshot.get("assistant_state", "Unknown") or "Unknown")
        state_display = str(
            snapshot.get("assistant_state_display")
            or self._format_state_display(state, snapshot.get("assistant_state_detail"))
        )
        model_name = str(snapshot.get("model_name", snapshot.get("current_model_name", "---")) or "---")
        model_state = str(snapshot.get("model_state", "Disconnected") or "Disconnected")
        online_enabled = bool(snapshot.get("online_enabled"))
        filters_enabled = bool(snapshot.get("filters_enabled"))
        voice_enabled = bool(snapshot.get("voice_enabled"))
        vision_state = str(snapshot.get("vision_state", "Off") or "Off")
        vision_model = str(snapshot.get("vision_model", "") or "")
        stt_status = str(snapshot.get("stt_status_text", snapshot.get("stt_state", "Disabled")) or "Disabled")
        tts_status = str(snapshot.get("tts_status_text", snapshot.get("tts_state", "Disabled")) or "Disabled")
        thinking_time = str(snapshot.get("thinking_time_text", "0.00s"))
        tts_generation_time = str(snapshot.get("tts_generation_time_text", "0.00s"))

        self.state_field.set_status(state_display, self._state_style(state))
        self.model_field.set_status(
            f"{model_name} | {model_state}",
            self._state_style(model_state),
        )
        self.think_time_field.set_status(thinking_time, "muted")
        self.tts_gen_field.set_status(tts_generation_time, "muted")
        self.online_field.set_status(
            "On" if online_enabled else "Off",
            self._binary_role(online_enabled, on_role="success"),
        )
        self.filter_field.set_status(
            self._format_toggle_detail(
                filters_enabled,
                snapshot.get("filter_level", "standard"),
            ),
            self._binary_role(filters_enabled, on_role="warning"),
        )
        self.voice_field.set_status(
            self._format_toggle_detail(
                voice_enabled,
                snapshot.get("voice_name") or "none",
            ),
            self._binary_role(voice_enabled, on_role="success"),
        )
        self.vision_field.set_status(
            self._format_vision_value(vision_state, vision_model),
            role_from_state(vision_state),
        )
        self.stt_field.set_status(stt_status, role_from_state(stt_status))
        self.stt_time_field.set_status(
            str(snapshot.get("stt_time_text", "0.00s")), "muted"
        )
        self.tts_field.set_status(tts_status, role_from_state(tts_status))
        self.tts_time_field.set_status(
            str(snapshot.get("tts_time_text", "0.00s")), "muted"
        )
        self.emotion_field.set_status(
            snapshot.get("current_emotion", "Neutral"), "accent"
        )
        self.persona_field.set_status(
            snapshot.get("current_persona", snapshot.get("active_persona_profile_name", "Revia")),
            "accent",
        )

    @staticmethod
    def _format_toggle_detail(enabled, detail):
        if enabled:
            detail_text = str(detail or "").strip()
            return f"On | {detail_text}" if detail_text else "On"
        return "Off"

    @staticmethod
    def _format_vision_value(state, model):
        normalized_state = str(state or "Off").strip()
        model_text = str(model or "").strip()
        if normalized_state == "On" and model_text:
            return f"On | {model_text}"
        return normalized_state or "Off"

    @staticmethod
    def _format_state_display(state, detail):
        state_text = str(state or "Unknown").strip() or "Unknown"
        detail_text = str(detail or "").strip()
        if detail_text:
            return f"{state_text} | {detail_text}"
        return state_text

    @staticmethod
    def _fallback_timer_text(runtime, *, timer_key, current_key, state_key, active_states):
        timer_value = runtime.get(timer_key)
        if timer_value is not None:
            try:
                return f"{float(timer_value or 0.0):.2f}s"
            except (TypeError, ValueError):
                return "0.00s"
        state = str(runtime.get(state_key, "Disabled") or "Disabled")
        current = float(runtime.get(current_key, 0.0) or 0.0)
        if state in active_states:
            return f"{current:.2f}s"
        return "0.00s"

    @staticmethod
    def _binary_role(enabled: bool, *, on_role: str) -> str:
        """Return a semantic role name based on an on/off boolean."""
        return on_role if enabled else "muted"

    @staticmethod
    def _state_style(state: str) -> str:
        """Delegate to ui_status role resolution — no hardcoded colors here."""
        return role_from_state(state)
