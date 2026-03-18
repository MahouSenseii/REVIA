"""Central voice controller. Bridges profiles, library, TTS backend, and emotion."""
from pathlib import Path
from PySide6.QtCore import QObject, Signal
from .voice_profile import VoiceProfile, VoiceMode
from .voice_library import VoiceLibrary
from .tts_backend import QwenTTSBackend, TTSMetrics


class VoiceManager(QObject):
    """Orchestrates voice profiles, TTS generation, and emotion modulation.

    Key flow for consistent voice:
    1. User creates a voice via Design/Clone/Custom -> generates a WAV
    2. WAV is saved to the voice profile directory
    3. For ongoing chat TTS, the WAV is used as clone reference
       so all speech sounds the same as the generated voice
    """

    voice_changed = Signal(str)
    metrics_updated = Signal(object)  # TTSMetrics
    error = Signal(str)

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

    def set_active_voice(self, name):
        p = self.library.get(name)
        if p:
            self._active_profile = p
            self.voice_changed.emit(name)

    def speak(self, text, emotion=None):
        """Speak text using the active voice profile (async, returns immediately)."""
        emo = emotion or self._current_emotion
        self.backend.speak_from_profile(text, self._active_profile, emo)

    def speak_sync(self, text, emotion=None):
        """Speak text synchronously — blocks until synthesis and playback are done."""
        emo = emotion or self._current_emotion
        self.backend.speak_from_profile_sync(text, self._active_profile, emo)

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
