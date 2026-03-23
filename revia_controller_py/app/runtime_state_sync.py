from __future__ import annotations

import os

from PySide6.QtCore import QObject, QTimer


class RuntimeStateSync(QObject):
    def __init__(
        self,
        *,
        client,
        event_bus,
        model_tab,
        voice_tab,
        filters_tab,
        memory_tab,
        system_tab,
        profile_tab,
        chat_panel,
        audio_service,
        assistant_status_manager,
        parent=None,
    ):
        super().__init__(parent)
        self.client = client
        self.event_bus = event_bus
        self.model_tab = model_tab
        self.voice_tab = voice_tab
        self.filters_tab = filters_tab
        self.memory_tab = memory_tab
        self.system_tab = system_tab
        self.profile_tab = profile_tab
        self.chat_panel = chat_panel
        self.audio_service = audio_service
        self.assistant_status_manager = assistant_status_manager
        self._last_telemetry = {}

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(300)
        self._debounce.timeout.connect(self.push_now)

        self.event_bus.telemetry_updated.connect(self._on_telemetry)
        self.event_bus.connection_changed.connect(lambda _connected: self.schedule_sync())
        self.assistant_status_manager.runtime_state_changed.connect(
            lambda _snapshot: self.schedule_sync()
        )

        self._wire_signals()

    def _wire_signals(self):
        for signal in (
            self.model_tab.source_type.currentIndexChanged,
            self.model_tab.local_server.currentTextChanged,
            self.model_tab.local_server_url.textChanged,
            self.model_tab.api_provider.currentTextChanged,
            self.model_tab.api_endpoint.textChanged,
            self.model_tab.api_model.currentTextChanged,
            self.voice_tab.engine_combo.currentTextChanged,
            self.voice_tab.ptt_mode.currentTextChanged,
            self.filters_tab.nsfw_filter.toggled,
            self.filters_tab.profanity_filter.toggled,
            self.filters_tab.pii_filter.toggled,
            self.filters_tab.injection_guard.toggled,
            self.memory_tab.memory_backend.currentTextChanged,
            self.memory_tab.auto_store.toggled,
            self.system_tab.emotion_toggle.toggled,
            self.system_tab.router_toggle.toggled,
            self.system_tab.websearch_toggle.toggled,
            self.profile_tab.char_name.textChanged,
            self.profile_tab.profile_name.textChanged,
            self.chat_panel.tts_combo.currentTextChanged,
        ):
            signal.connect(lambda *_args: self.schedule_sync())

    def _on_telemetry(self, data):
        if isinstance(data, dict):
            self._last_telemetry = dict(data)
        self.schedule_sync()

    def schedule_sync(self):
        self._debounce.start()

    def collect_snapshot(self) -> dict:
        if self.assistant_status_manager is not None:
            return self.assistant_status_manager.build_runtime_config_snapshot()

        try:
            # Snapshot access must be thread-safe; Qt signal emissions from other threads
            # may modify _last_telemetry concurrently. This try-except is a safety net.
            telemetry = dict(self._last_telemetry or {})
        except (RuntimeError, RecursionError) as exc:
            # Catch Qt thread safety or recursion issues
            telemetry = {}
        emotion = telemetry.get("emotion", {}) or {}
        state = str(telemetry.get("state", "Idle") or "Idle")

        source_is_online = self.model_tab.source_type.currentIndex() == 1
        local_model_name = os.path.basename(self.model_tab.local_path.text().strip())
        online_model_name = self.model_tab.api_model.currentText().strip()

        filter_enabled = any(
            (
                self.filters_tab.nsfw_filter.isChecked(),
                self.filters_tab.profanity_filter.isChecked(),
                self.filters_tab.pii_filter.isChecked(),
                self.filters_tab.injection_guard.isChecked(),
            )
        )
        if filter_enabled:
            if all(
                (
                    self.filters_tab.nsfw_filter.isChecked(),
                    self.filters_tab.pii_filter.isChecked(),
                    self.filters_tab.injection_guard.isChecked(),
                )
            ):
                filter_level = "strict"
            else:
                filter_level = "standard"
        else:
            filter_level = "off"

        active_profile = ""
        try:
            voice_mgr = getattr(self.voice_tab, "voice_mgr", None)
            active_profile = getattr(getattr(voice_mgr, "active_profile", None), "name", "")
        except Exception:
            active_profile = ""

        return {
            "online_enabled": bool(source_is_online or self.system_tab.websearch_toggle.isChecked()),
            "web_search_enabled": self.system_tab.websearch_toggle.isChecked(),
            "safety_filter_enabled": filter_enabled,
            "nsfw_filter_enabled": self.filters_tab.nsfw_filter.isChecked(),
            "profanity_filter_enabled": self.filters_tab.profanity_filter.isChecked(),
            "pii_filter_enabled": self.filters_tab.pii_filter.isChecked(),
            "prompt_injection_guard_enabled": self.filters_tab.injection_guard.isChecked(),
            "content_filter_level": filter_level,
            "local_llm_enabled": not source_is_online,
            "local_llm_provider": (
                self.model_tab.api_provider.currentText()
                if source_is_online
                else self.model_tab.local_server.currentText()
            ),
            "local_llm_endpoint": (
                self.model_tab.api_endpoint.text().strip()
                if source_is_online
                else self.model_tab.local_server_url.text().strip()
            ),
            "voice_input_enabled": self.audio_service.is_stt_available(),
            "voice_output_enabled": self.chat_panel.is_tts_enabled(),
            "current_tts_voice": active_profile or self.voice_tab.engine_combo.currentText(),
            "memory_enabled": (
                self.memory_tab.memory_backend.currentText() != "None"
                and self.memory_tab.auto_store.isChecked()
            ),
            "emotion_mode_enabled": self.system_tab.emotion_toggle.isChecked(),
            "current_emotion": str(emotion.get("label", "Neutral") or "Neutral"),
            # tool_access_enabled reflects whether *any* tool is active, not
            # just web-search, so the core can gate tool use correctly.
            "tool_access_enabled": (
                self.system_tab.websearch_toggle.isChecked()
                or self.system_tab.router_toggle.isChecked()
            ),
            "tool_modes": {
                "web_search": self.system_tab.websearch_toggle.isChecked(),
                "router": self.system_tab.router_toggle.isChecked(),
            },
            "active_persona_profile_name": self.profile_tab.char_name.text().strip() or "Revia",
            "active_persona_profile_id": self.profile_tab.profile_name.text().strip() or "default",
            "streaming_enabled": True,
            "current_model_name": online_model_name or local_model_name or "None",
            "fallback_model_name": "",
            "safe_mode_enabled": filter_enabled,
            "moderation_mode": filter_level,
            "ui_state": state,
            "tts_engine": self.voice_tab.engine_combo.currentText(),
            "stt_mode": self.voice_tab.ptt_mode.currentText(),
        }

    def push_now(self):
        snapshot = self.collect_snapshot()
        self.client.push_runtime_config(snapshot)
