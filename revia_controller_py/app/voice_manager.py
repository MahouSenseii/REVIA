"""Central voice controller. Bridges profiles, library, TTS backend, and emotion.

Ownership model
---------------
* ``VoiceManager``   — runtime voice orchestration (single source of truth)
* ``QwenTTSBackend`` — implements both Qwen3-TTS (primary) and pyttsx3 (fallback)
* UI should read ``active_backend_name`` and ``active_profile`` from this class,
  never infer state from scattered widget or backend internals.
"""
from PySide6.QtCore import QObject, Signal
from .voice_profile import VoiceProfile, VoiceMode
from .voice_library import VoiceLibrary
from .tts_backend import QwenTTSBackend

# Human-readable labels for the two supported engine IDs
_ENGINE_LABELS: dict[str, str] = {
    "qwen3-tts": "Qwen3-TTS",
    "pyttsx3": "pyttsx3",
}

# Fallback order: if primary is unavailable, fall back to this engine
_FALLBACK_ENGINE: str = "pyttsx3"


class VoiceManager(QObject):
    """Orchestrates voice profiles, TTS generation, and emotion modulation.

    Key flow for consistent voice:
    1. User creates a voice via Design/Clone/Custom -> generates a WAV
    2. WAV is saved to the voice profile directory
    3. For ongoing chat TTS, the WAV is used as clone reference
       so all speech sounds the same as the generated voice

    Active backend is always readable via ``active_backend_name``.
    UI must subscribe to ``backend_changed`` to stay in sync.
    """

    voice_changed = Signal(str)
    metrics_updated = Signal(object)  # TTSMetrics
    error = Signal(str)
    # Emitted whenever the active TTS engine changes; payload is the new engine ID
    backend_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.library = VoiceLibrary()
        self.backend = QwenTTSBackend(self)
        self._active_profile = self.library.get_default()
        self._current_emotion = "neutral"

        self.backend.synthesis_finished.connect(self._on_synthesis_done)
        self.backend.error_occurred.connect(lambda e: self.error.emit(e))

    @property
    def active_profile(self):
        return self._active_profile

    # ── Backend ownership ──────────────────────────────────────────────────

    @property
    def active_backend_name(self) -> str:
        """The engine ID currently active in the TTS backend.

        Valid values: ``"qwen3-tts"``, ``"pyttsx3"``.
        This is the single source of truth for which engine is running.
        """
        return str(self.backend.engine_name or _FALLBACK_ENGINE)

    @property
    def active_backend_label(self) -> str:
        """Human-readable label for the active backend (e.g. ``"Qwen3-TTS"``)."""
        return _ENGINE_LABELS.get(self.active_backend_name, self.active_backend_name)

    @property
    def fallback_backend_name(self) -> str:
        """Engine used when the primary backend is unavailable."""
        return _FALLBACK_ENGINE

    def set_backend(self, engine_id: str) -> None:
        """Switch active TTS backend by engine ID and emit ``backend_changed``.

        Args:
            engine_id: ``"qwen3-tts"`` or ``"pyttsx3"``
        """
        if engine_id not in _ENGINE_LABELS:
            return
        self.backend.set_engine(engine_id)
        self.backend_changed.emit(engine_id)

    def is_backend_ready(self) -> bool:
        """True if the active backend server is reachable / operational."""
        return self.backend.is_ready()

    # ── Voice profiles ─────────────────────────────────────────────────────

    def set_active_voice(self, name):
        p = self.library.get(name)
        if p:
            self._active_profile = p
            self.voice_changed.emit(name)

    def speak(self, text, emotion=None):
        """Speak text using the active voice profile (async, returns immediately)."""
        emo = emotion or self._current_emotion
        # Update internal emotion state if explicitly provided
        if emotion:
            self._current_emotion = emotion
        self.backend.speak_from_profile(text, self._active_profile, emo)

    def speak_sync(self, text, emotion=None):
        """Speak text synchronously — blocks until synthesis and playback are done."""
        emo = emotion or self._current_emotion
        # Update internal emotion state if explicitly provided
        if emotion:
            self._current_emotion = emotion
        self.backend.speak_from_profile_sync(text, self._active_profile, emo)

    def synthesize_to_wav(self, text, emotion=None, output_path=None):
        """Generate speech audio for text without playing it."""
        emo = emotion or self._current_emotion
        if emotion:
            self._current_emotion = emotion
        return self.backend.synthesize_from_profile(
            text, self._active_profile, emo, output_path=output_path
        )

    def play_wav_sync(self, wav_path):
        """Play a generated WAV synchronously on the caller's worker thread."""
        self.backend.play_wav_sync(wav_path)

    def apply_emotion_modifiers(self, emotion_state):
        """Called by EmotionNet to update current emotion."""
        label = emotion_state.get("label", "neutral").lower()
        self._current_emotion = label

    # ── Generation methods (produce WAV files) ──

    def generate_design(self, text, voice_description, language="Auto",
                        output_path=None):
        return self.backend.generate_voice_design(
            text, voice_description, language, output_path
        )

    def generate_clone(self, target_text, ref_audio_path, ref_text="",
                       language="Auto", x_vector_only=False, output_path=None):
        return self.backend.generate_voice_clone(
            target_text, ref_audio_path, ref_text, language,
            x_vector_only, output_path
        )

    def generate_custom(self, text, language="Auto", speaker="Ryan",
                        style_instruction="", model_size="1.7B",
                        output_path=None):
        return self.backend.generate_custom_voice(
            text, language, speaker, style_instruction, model_size,
            output_path
        )

    def play_wav(self, wav_path):
        self.backend.play_wav(wav_path)

    def is_output_ready(self):
        return self.backend.is_ready()

    def stop_output(self):
        self.backend.stop_output()

    # ── Profile management ──

    def save_voice(self, profile):
        return self.library.save_profile(profile)

    def create_design_profile(self, name, voice_description, language="Auto"):
        p = VoiceProfile(name, VoiceMode.DESIGN)
        p.voice_description = voice_description
        p.language = language
        return p

    def create_clone_profile(self, name, ref_audio, ref_text="",
                             language="Auto", x_vector_only=False):
        p = VoiceProfile(name, VoiceMode.CLONE)
        p.reference_audio = str(ref_audio)
        p.clone_ref_text = ref_text
        p.language = language
        p.x_vector_only = x_vector_only
        return p

    def create_custom_profile(self, name, speaker="Ryan",
                              style_instruction="", language="Auto",
                              model_size="1.7B"):
        p = VoiceProfile(name, VoiceMode.CUSTOM)
        p.speaker_id = speaker
        p.style_instruction = style_instruction
        p.language = language
        p.model_size = model_size
        return p

    def get_latency_metrics(self):
        m = self.backend.last_metrics
        return {
            "synthesis_time": m.synthesis_time,
            "audio_duration": m.audio_duration,
            "realtime_factor": m.realtime_factor,
        }

    def _on_synthesis_done(self, metrics):
        self.metrics_updated.emit(metrics)
