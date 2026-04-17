from __future__ import annotations

import threading


def _safe_float(value, default: float = 0.0) -> float:
    """Safely convert a value to float with a default fallback."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


class RuntimeStatusManager:
    def __init__(
        self,
        log_fn,
        *,
        llm_status_getter,
        telemetry_getter,
        profile_getter,
        memory_getter,
        web_search_enabled_getter,
        plugin_status_getter,
        integration_status_getter=None,
    ):
        self._log = log_fn
        self._llm_status_getter = llm_status_getter
        self._telemetry_getter = telemetry_getter
        self._profile_getter = profile_getter
        self._memory_getter = memory_getter
        self._web_search_enabled_getter = web_search_enabled_getter
        self._plugin_status_getter = plugin_status_getter
        self._integration_status_getter = integration_status_getter
        self._lock = threading.Lock()
        self._controller_state = {
            "online_enabled": False,
            "web_search_enabled": False,
            "safety_filter_enabled": False,
            "nsfw_filter_enabled": False,
            "profanity_filter_enabled": False,
            "pii_filter_enabled": False,
            "prompt_injection_guard_enabled": False,
            "content_filter_level": "standard",
            "tool_access_enabled": False,
            "local_llm_enabled": False,
            "local_llm_provider": "",
            "local_llm_endpoint": "",
            "voice_input_enabled": False,
            "voice_output_enabled": False,
            "current_tts_voice": "",
            "vision_enabled": False,
            "vision_model": "",
            "vision_state": "Off",
            "memory_enabled": True,
            "emotion_mode_enabled": True,
            "current_emotion": "Neutral",
            "streaming_enabled": True,
            "active_persona_profile_name": "Revia",
            "active_persona_profile_id": "revia",
            "fallback_model_name": "",
            "safe_mode_enabled": False,
            "moderation_mode": "standard",
            "tool_modes": {},
            "ui_state": "Idle",
            "tts_engine": "",
            "stt_mode": "",
            "assistant_state": "Idle",
            "model_name": "None",
            "model_ready": False,
            "stt_enabled": False,
            "stt_state": "Disabled",
            "stt_timer": 0.0,
            "stt_last_listen_duration": 0.0,
            "stt_last_processing_duration": 0.0,
            "stt_last_total_duration": 0.0,
            "stt_current_elapsed": 0.0,
            "stt_error": "",
            "tts_enabled": False,
            "tts_state": "Disabled",
            "tts_timer": 0.0,
            "tts_last_generation_duration": 0.0,
            "tts_last_playback_duration": 0.0,
            "tts_last_total_duration": 0.0,
            "tts_current_elapsed": 0.0,
            "tts_error": "",
        }

    # Whitelist of allowed runtime config keys
    _ALLOWED_CONFIG_KEYS = frozenset({
        "online_enabled", "web_search_enabled", "safety_filter_enabled",
        "tool_access_enabled", "voice_output_enabled", "memory_enabled",
        "emotion_mode_enabled", "active_persona_profile_name", "ui_state",
        "tts_engine", "stt_enabled", "vision_enabled", "continuous_audio_enabled",
        "proactive_mode", "fast_mode", "debug_mode",
    })

    def update_runtime_config(self, data: dict | None):
        if not isinstance(data, dict):
            return
        changed_keys: list[str] = []
        with self._lock:
            for key, value in data.items():
                if key in self._ALLOWED_CONFIG_KEYS:
                    if self._controller_state.get(key) != value:
                        changed_keys.append(key)
                    self._controller_state[key] = value
        # Only log when something actually changed - prevents log spam from
        # identical config pushes during startup signal bursts.
        _LOG_KEYS = {
            "online_enabled", "web_search_enabled", "safety_filter_enabled",
            "tool_access_enabled", "voice_output_enabled", "memory_enabled",
            "emotion_mode_enabled", "active_persona_profile_name", "ui_state",
        }
        if changed_keys:
            self._log(
                "Runtime config updated | "
                + ", ".join(
                    f"{key}={data[key]}"
                    for key in sorted(changed_keys)
                    if key in _LOG_KEYS
                )
            )

    def get_runtime_status(self) -> dict:
        llm = dict(self._llm_status_getter() or {})
        telemetry = dict(self._telemetry_getter() or {})
        profile = dict(self._profile_getter() or {})
        memory = dict(self._memory_getter() or {})
        plugins = dict(self._plugin_status_getter() or {})
        integrations = {}
        if self._integration_status_getter is not None:
            try:
                integrations = dict(self._integration_status_getter() or {})
            except Exception:
                integrations = {}

        with self._lock:
            state = dict(self._controller_state)

        state["web_search_enabled"] = bool(self._web_search_enabled_getter())
        state["local_llm_connected"] = llm.get("state") == "Ready"
        state["last_llm_error"] = llm.get("last_error", "") or llm.get("detail", "")
        state["current_model_name"] = llm.get("model", "None")
        state["local_llm_provider"] = (
            state.get("local_llm_provider")
            or llm.get("source", "")
        )
        state["local_llm_endpoint"] = state.get("local_llm_endpoint") or llm.get("detail", "")
        state["current_emotion"] = telemetry.get("emotion_label", state.get("current_emotion", "Neutral"))
        state["active_persona_profile_name"] = (
            profile.get("character_name")
            or state.get("active_persona_profile_name")
            or "Revia"
        )
        state["memory_backend"] = memory.get("backend", "unknown")
        state["memory_profile"] = memory.get("profile", state["active_persona_profile_name"])
        state["plugins"] = plugins
        state["integrations"] = integrations
        return state

    def get_runtime_status_summary(self) -> str:
        status = self.get_runtime_status()
        parts = [
            f"Model={status.get('current_model_name', 'None')}",
            f"LLM={'ready' if status.get('local_llm_connected') else 'not ready'}",
            f"Web={'on' if status.get('web_search_enabled') else 'off'}",
            f"Filters={'on' if status.get('safety_filter_enabled') else 'off'}",
            f"Voice={'on' if status.get('voice_output_enabled') else 'off'}",
            f"Vision={status.get('vision_state', 'Off')}",
            f"STT={status.get('stt_state', 'Disabled')}",
            f"TTS={status.get('tts_state', 'Disabled')}",
            f"Memory={'on' if status.get('memory_enabled') else 'off'}",
            f"Emotion={status.get('current_emotion', 'Neutral')}",
            f"Persona={status.get('active_persona_profile_name', 'Revia')}",
            f"State={status.get('assistant_state', status.get('ui_state', 'Idle'))}",
        ]
        return " | ".join(parts)

    def build_self_awareness_context(self, *, user_text: str = "", include_full: bool = False) -> str:
        status = self.get_runtime_status()
        low = str(user_text or "").lower()
        wants_status = self.is_status_question(low)

        lines = ["[Runtime self-awareness]"]
        if include_full or wants_status or not status.get("local_llm_connected", False):
            lines.append(
                f"- Model connection: {'online' if status.get('local_llm_connected') else 'offline'}"
                f" | model={status.get('current_model_name', 'None')}"
            )
            if status.get("last_llm_error"):
                lines.append(f"- Last LLM issue: {status.get('last_llm_error')}")

        if include_full or wants_status or ("web" in low or "online" in low):
            lines.append(
                f"- Online lookup: {'enabled' if status.get('web_search_enabled') else 'disabled'}"
            )

        if include_full or wants_status or any(k in low for k in ("filter", "safe", "moderation")):
            lines.append(
                f"- Safety filters: {'enabled' if status.get('safety_filter_enabled') else 'disabled'}"
                f" | level={status.get('content_filter_level', 'standard')}"
            )

        if include_full or wants_status or "voice" in low or "speak" in low:
            lines.append(
                f"- Voice output: {'enabled' if status.get('voice_output_enabled') else 'disabled'}"
                f" | engine={status.get('tts_engine') or status.get('current_tts_voice') or 'none'}"
            )
            lines.append(
                f"- Voice input: {'enabled' if status.get('voice_input_enabled') else 'disabled'}"
                f" | mode={status.get('stt_mode') or 'unknown'}"
            )
            lines.append(
                f"- STT status: {status.get('stt_state', 'Disabled')}"
                f" | listen={_safe_float(status.get('stt_last_listen_duration', 0.0) or 0.0):.2f}s"
                f" | process={_safe_float(status.get('stt_last_processing_duration', 0.0) or 0.0):.2f}s"
            )
            lines.append(
                f"- TTS status: {status.get('tts_state', 'Disabled')}"
                f" | generation={_safe_float(status.get('tts_last_generation_duration', 0.0) or 0.0):.2f}s"
                f" | playback={_safe_float(status.get('tts_last_playback_duration', 0.0) or 0.0):.2f}s"
            )

        if include_full or wants_status or "memory" in low:
            lines.append(
                f"- Memory: {'enabled' if status.get('memory_enabled') else 'disabled'}"
                f" | backend={status.get('memory_backend', 'unknown')}"
                f" | profile={status.get('memory_profile', 'Revia')}"
            )

        if include_full or wants_status or any(k in low for k in ("profile", "persona", "emotion")):
            lines.append(
                f"- Persona profile: {status.get('active_persona_profile_name', 'Revia')}"
            )
            lines.append(
                f"- Emotion mode: {'enabled' if status.get('emotion_mode_enabled') else 'disabled'}"
                f" | current emotion={status.get('current_emotion', 'Neutral')}"
            )

        if include_full or wants_status or any(k in low for k in ("tool", "search", "internet")):
            lines.append(
                f"- Tools: {'enabled' if status.get('tool_access_enabled') else 'disabled'}"
            )

        if include_full:
            lines.append(
                f"- Assistant state: {status.get('assistant_state', status.get('ui_state', 'Idle'))}"
            )

        lines.append(
            "Use this runtime state only when it matters. Never pretend an offline "
            "or disabled feature is available."
        )
        return "\n".join(lines)

    def build_status_reply(self, user_text: str) -> str:
        status = self.get_runtime_status()
        low = str(user_text or "").lower()
        name = status.get("active_persona_profile_name", "Revia")

        if any(k in low for k in ("filter", "safe mode", "moderation", "nsfw")):
            return (
                f"{name} here. Your safety filters are "
                f"{'enabled' if status.get('safety_filter_enabled') else 'disabled'}"
                f", with content filtering set to {status.get('content_filter_level', 'standard')}."
            )
        if any(k in low for k in ("online", "web", "search the web", "internet")):
            return (
                f"Online lookup is currently "
                f"{'enabled' if status.get('web_search_enabled') else 'disabled'}."
            )
        if any(k in low for k in ("model", "provider", "endpoint", "llm")):
            return (
                f"I am currently using {status.get('local_llm_provider') or 'the configured model backend'} "
                f"with model {status.get('current_model_name', 'None')}."
                + (
                    f" The endpoint is {status.get('local_llm_endpoint')}."
                    if status.get("local_llm_endpoint")
                    else ""
                )
                + (
                    " It is responding normally."
                    if status.get("local_llm_connected")
                    else " It is not responding right now."
                )
            )
        if "voice" in low or "speak" in low or "tts" in low:
            return (
                f"Voice output is {'enabled' if status.get('voice_output_enabled') else 'disabled'}"
                f", and my current voice engine is "
                f"{status.get('tts_engine') or status.get('current_tts_voice') or 'not set'}."
                f" TTS is currently {status.get('tts_state', 'Disabled')}"
                f", and STT is {status.get('stt_state', 'Disabled')}."
            )
        if "stt" in low or "microphone" in low or "listening" in low:
            return (
                f"STT is currently {status.get('stt_state', 'Disabled')}, "
                f"with the last listen time at "
                f"{_safe_float(status.get('stt_last_listen_duration', 0.0) or 0.0):.2f}s "
                f"and processing time at "
                f"{_safe_float(status.get('stt_last_processing_duration', 0.0) or 0.0):.2f}s."
            )
        if "memory" in low:
            return (
                f"Memory is {'enabled' if status.get('memory_enabled') else 'disabled'}"
                f", using {status.get('memory_backend', 'unknown')} for the current profile "
                f"{status.get('memory_profile', 'Revia')}."
            )
        if any(k in low for k in ("profile", "persona", "character")):
            return (
                f"I am currently using the {status.get('active_persona_profile_name', 'Revia')} profile."
            )
        if "emotion" in low:
            return (
                f"My emotion mode is {'enabled' if status.get('emotion_mode_enabled') else 'disabled'}, "
                f"and my current emotion state is {status.get('current_emotion', 'Neutral')}."
            )
        if "tool" in low:
            return (
                f"Tool access is {'enabled' if status.get('tool_access_enabled') else 'disabled'}."
            )
        return (
            f"Current runtime summary: {self.get_runtime_status_summary()}."
        )

    @staticmethod
    def is_status_question(text: str) -> bool:
        low = str(text or "").lower()
        if not low:
            return False
        probes = (
            "are filters",
            "filters on",
            "are you online",
            "which model",
            "what model",
            "what endpoint",
            "can you search the web",
            "is voice enabled",
            "is memory enabled",
            "which profile",
            "what profile",
            "what persona",
            "current settings",
            "runtime status",
            "are tools enabled",
            "what state are you in",
        )
        return any(probe in low for probe in probes)
