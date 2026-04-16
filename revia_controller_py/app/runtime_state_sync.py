from __future__ import annotations

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
        self._last_pushed_snapshot: dict | None = None

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

    def _on_telemetry(self, _data):
        self.schedule_sync()

    def schedule_sync(self):
        self._debounce.start()

    def collect_snapshot(self) -> dict:
        if self.assistant_status_manager is None:
            raise RuntimeError(
                "RuntimeStateSync.collect_snapshot() called with no AssistantStatusManager; "
                "this is a programmer error — always pass assistant_status_manager."
            )
        return self.assistant_status_manager.build_runtime_config_snapshot()

    # Keys whose values change every poll (timers, elapsed counters) and
    # should NOT trigger a config push by themselves.
    _VOLATILE_KEYS = frozenset({
        "stt_timer", "stt_current_elapsed", "stt_last_listen_duration",
        "stt_last_processing_duration", "stt_last_total_duration",
        "tts_timer", "tts_current_elapsed", "tts_last_generation_duration",
        "tts_last_playback_duration", "tts_last_total_duration",
    })

    def push_now(self):
        snapshot = self.collect_snapshot()
        # Compare only non-volatile keys to decide if we should POST.
        # Timer fields tick every poll and would defeat the dedup check.
        stable = {k: v for k, v in snapshot.items() if k not in self._VOLATILE_KEYS}
        if self._last_pushed_snapshot is not None:
            prev_stable = {k: v for k, v in self._last_pushed_snapshot.items()
                           if k not in self._VOLATILE_KEYS}
            if stable == prev_stable:
                return
        self._last_pushed_snapshot = snapshot
        self.client.push_runtime_config(snapshot)
