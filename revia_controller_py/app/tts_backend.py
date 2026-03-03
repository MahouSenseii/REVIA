"""TTS backend for REVIA. Supports Qwen3-TTS via gradio_client and local pyttsx3 fallback."""
import os
import time
import wave
import shutil
import struct
import threading
import tempfile
from pathlib import Path
from PySide6.QtCore import QObject, Signal


class TTSMetrics:
    def __init__(self):
        self.synthesis_time = 0.0
        self.audio_duration = 0.0
        self.realtime_factor = 0.0


# Qwen3-TTS HuggingFace Space IDs (public demo endpoints)
QWEN_SPACES = {
    "design": "Qwen/Qwen3-TTS",
    "clone": "Qwen/Qwen3-TTS",
    "custom": "Qwen/Qwen3-TTS",
}

# Predefined speakers (must match exact Gradio dropdown values)
QWEN_SPEAKERS = [
    "Aiden", "Dylan", "Eric", "Ono_anna", "Ryan",
    "Serena", "Sohee", "Uncle_fu", "Vivian",
]

# Supported languages (must match exact Gradio dropdown values)
QWEN_LANGUAGES = [
    "Auto", "Chinese", "English", "Japanese", "Korean",
    "French", "German", "Spanish", "Portuguese", "Russian",
]

QWEN_MODEL_SIZES = ["0.6B", "1.7B"]


class QwenTTSBackend(QObject):
    """TTS engine with Qwen3-TTS (via gradio_client or local qwen-tts) and pyttsx3 fallback.

    Three generation modes matching Qwen3-TTS:
    1. Voice Design:  text + voice_description + language -> WAV
    2. Voice Clone:   ref_audio + ref_text + target_text + language -> WAV
    3. CustomVoice:   text + language + speaker + style_instruction -> WAV

    Generated WAVs are saved to voice profile directories for reuse.
    For ongoing chat TTS, the saved WAV is used as clone reference so voice stays consistent.
    """

    synthesis_started = Signal()
    synthesis_finished = Signal(object)  # TTSMetrics
    error_occurred = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lock = threading.Lock()
        self._engine_name = "qwen3-tts"
        self._qwen_url = ""  # custom Qwen3-TTS server URL (Gradio)
        self._pyttsx3_voice_id = None
        self.last_metrics = TTSMetrics()

    @property
    def engine_name(self):
        return self._engine_name

    def set_engine(self, name):
        self._engine_name = name

    def set_qwen_server(self, url):
        self._qwen_url = url.rstrip("/") if url else ""

    def list_system_voices(self):
        try:
            import pyttsx3
            engine = pyttsx3.init()
            voices = engine.getProperty("voices")
            result = [
                {"id": v.id, "name": v.name, "lang": getattr(v, "languages", [])}
                for v in voices
            ]
            engine.stop()
            return result
        except Exception:
            return []

    # ── Voice Design ──
    def generate_voice_design(self, text, voice_description, language="Auto",
                              output_path=None):
        """Generate speech from natural language voice description.
        Returns (wav_path, metrics) or (None, error_str)."""
        if self._engine_name == "pyttsx3" or not self._has_gradio_client():
            return self._fallback_generate(text, output_path)

        return self._qwen_design(text, voice_description, language, output_path)

    # ── Voice Clone ──
    def generate_voice_clone(self, target_text, ref_audio_path, ref_text="",
                             language="Auto", x_vector_only=False,
                             output_path=None):
        """Clone voice from reference audio and synthesize target text.
        Returns (wav_path, metrics) or (None, error_str)."""
        if self._engine_name == "pyttsx3" or not self._has_gradio_client():
            return self._fallback_generate(target_text, output_path)

        return self._qwen_clone(target_text, ref_audio_path, ref_text,
                                language, x_vector_only, output_path)

    # ── CustomVoice (TTS with predefined speakers) ──
    def generate_custom_voice(self, text, language="Auto", speaker="Ryan",
                              style_instruction="", model_size="1.7B",
                              output_path=None):
        """Generate speech with predefined speaker + optional style instruction.
        Returns (wav_path, metrics) or (None, error_str)."""
        if self._engine_name == "pyttsx3" or not self._has_gradio_client():
            return self._fallback_generate(text, output_path)

        return self._qwen_custom(text, language, speaker, style_instruction,
                                 model_size, output_path)

    # ── Play WAV file ──
    def play_wav(self, wav_path):
        """Play a WAV file through the default audio device."""
        def _do():
            try:
                import sounddevice as sd
                import soundfile as sf
                data, sr = sf.read(str(wav_path))
                sd.play(data, sr)
                sd.wait()
            except ImportError:
                self._play_wav_winsound(wav_path)
            except Exception as e:
                self.error_occurred.emit(f"Playback error: {e}")
        threading.Thread(target=_do, daemon=True).start()

    def speak_from_profile(self, text, voice_profile, emotion="neutral"):
        """Speak chat text asynchronously (returns immediately)."""
        threading.Thread(
            target=self._speak_from_profile_impl,
            args=(text, voice_profile, emotion),
            daemon=True,
        ).start()

    def speak_from_profile_sync(self, text, voice_profile, emotion="neutral"):
        """Speak chat text synchronously — blocks until synthesis AND playback are done."""
        self._speak_from_profile_impl(text, voice_profile, emotion)

    def _speak_from_profile_impl(self, text, voice_profile, emotion="neutral"):
        """Synthesize and play text using the voice profile. Always runs synchronously."""
        if not self._has_gradio_client():
            mods = voice_profile.get_modulated(emotion)
            self._speak_pyttsx3(text, mods["speed"], mods["pitch"])
            return

        if self._engine_name == "pyttsx3":
            mods = voice_profile.get_modulated(emotion)
            self._speak_pyttsx3(text, mods["speed"], mods["pitch"])
            return

        if not voice_profile.has_wav():
            mods = voice_profile.get_modulated(emotion)
            self._speak_pyttsx3(text, mods["speed"], mods["pitch"])
            return

        try:
            from .voice_profile import _resolve_wav_path
            _wav_src = _resolve_wav_path(voice_profile.generated_wav)
            wav_result, info = self._qwen_clone(
                text,
                _wav_src,
                voice_profile.clone_ref_text or "",
                voice_profile.language or "Auto",
                not bool(voice_profile.clone_ref_text),
                None,
                voice_profile.model_size or "1.7B",
            )
        except Exception as e:
            wav_result = None

        if wav_result and Path(wav_result).exists():
            self._play_wav_blocking(wav_result)
        else:
            mods = voice_profile.get_modulated(emotion)
            self._speak_pyttsx3(text, mods["speed"], mods["pitch"])

    def _play_wav_blocking(self, wav_path):
        """Play WAV synchronously (called from background thread)."""
        try:
            import sounddevice as sd
            import soundfile as sf
            data, sr = sf.read(str(wav_path))
            sd.play(data, sr)
            sd.wait()
        except ImportError:
            self._play_wav_winsound(wav_path)
        except Exception as e:
            self.error_occurred.emit(f"Playback error: {e}")

    # ── Qwen3-TTS gradio_client calls ──

    def _has_gradio_client(self):
        try:
            from gradio_client import Client  # noqa: F401
            return True
        except ImportError:
            return False

    def _get_client(self, space_key="custom"):
        from gradio_client import Client
        url = self._qwen_url or QWEN_SPACES.get(space_key, QWEN_SPACES["custom"])
        try:
            return Client(url)
        except Exception as e:
            raise ConnectionError(f"Cannot connect to Qwen3-TTS at {url}: {e}")

    def _qwen_design(self, text, voice_description, language, output_path):
        # API: predict(text, language, voice_description, api_name="/generate_voice_design")
        #   -> (generated_audio: filepath, status: str)
        with self._lock:
            t0 = time.perf_counter()
            self.synthesis_started.emit()
            try:
                client = self._get_client("design")
                result = client.predict(
                    text,
                    language,
                    voice_description,
                    api_name="/generate_voice_design",
                )
                wav_path = self._extract_wav(result, output_path)
                m = self._build_metrics(t0, wav_path)
                self.synthesis_finished.emit(m)
                return wav_path, m
            except Exception as e:
                self.error_occurred.emit(f"Voice Design error: {e}")
                return None, str(e)

    def _qwen_clone(self, target_text, ref_audio, ref_text, language,
                     x_vector_only, output_path, model_size="1.7B"):
        with self._lock:
            t0 = time.perf_counter()
            self.synthesis_started.emit()
            try:
                from gradio_client import handle_file
                client = self._get_client("clone")
                ref = handle_file(str(ref_audio))

                # Local server (0.6B Base): /run_voice_clone
                #   params: ref_aud, ref_txt, use_xvec, text, lang_disp
                # HF Space: /generate_voice_clone
                #   params: ref_aud, ref_txt, text, lang, use_xvec, model_size
                try:
                    result = client.predict(
                        ref, ref_text or "", x_vector_only,
                        target_text, language,
                        api_name="/run_voice_clone",
                    )
                    print("[TTS] Used /run_voice_clone (local)")
                except Exception:
                    result = client.predict(
                        ref, ref_text or "", target_text, language,
                        x_vector_only, model_size,
                        api_name="/generate_voice_clone",
                    )
                    print("[TTS] Used /generate_voice_clone (HF Space)")

                wav_path = self._extract_wav(result, output_path)
                m = self._build_metrics(t0, wav_path)
                self.synthesis_finished.emit(m)
                return wav_path, m
            except Exception as e:
                self.error_occurred.emit(f"Voice Clone error: {e}")
                return None, str(e)

    def _qwen_custom(self, text, language, speaker, style_instruction,
                      model_size, output_path):
        # API: predict(text, language, speaker, instruct, model_size,
        #              api_name="/generate_custom_voice")
        #   -> (generated_audio: filepath, status: str)
        with self._lock:
            t0 = time.perf_counter()
            self.synthesis_started.emit()
            try:
                client = self._get_client("custom")
                result = client.predict(
                    text,
                    language,
                    speaker,
                    style_instruction or "",
                    model_size,
                    api_name="/generate_custom_voice",
                )
                wav_path = self._extract_wav(result, output_path)
                m = self._build_metrics(t0, wav_path)
                self.synthesis_finished.emit(m)
                return wav_path, m
            except Exception as e:
                self.error_occurred.emit(f"CustomVoice error: {e}")
                return None, str(e)

    def _extract_wav(self, result, output_path):
        """Extract WAV path from gradio result and optionally copy to output_path."""
        wav_path = None
        if isinstance(result, tuple):
            # (audio_tuple, status_text) or (filepath, ...)
            for item in result:
                if isinstance(item, str) and Path(item).exists():
                    wav_path = item
                    break
                if isinstance(item, tuple) and len(item) == 2:
                    # (sr, numpy_array) -> save to temp
                    sr, data = item
                    import numpy as np
                    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    import soundfile as sf
                    sf.write(tmp.name, np.asarray(data, dtype=np.float32), int(sr))
                    wav_path = tmp.name
                    break
        elif isinstance(result, str) and Path(result).exists():
            wav_path = result

        if wav_path and output_path:
            shutil.copy2(wav_path, str(output_path))
            return str(output_path)
        return wav_path

    def _build_metrics(self, t0, wav_path):
        elapsed = time.perf_counter() - t0
        audio_dur = 0.0
        if wav_path and Path(wav_path).exists():
            try:
                import soundfile as sf
                data, sr = sf.read(str(wav_path))
                audio_dur = len(data) / sr
            except Exception:
                pass
        m = TTSMetrics()
        m.synthesis_time = round(elapsed, 3)
        m.audio_duration = round(audio_dur, 2)
        m.realtime_factor = round(audio_dur / elapsed, 2) if elapsed > 0 else 0
        self.last_metrics = m
        return m

    # ── pyttsx3 fallback ──

    def _fallback_generate(self, text, output_path=None):
        with self._lock:
            t0 = time.perf_counter()
            self.synthesis_started.emit()
            try:
                import pyttsx3
                engine = pyttsx3.init()
                voices = engine.getProperty("voices")
                if self._pyttsx3_voice_id:
                    engine.setProperty("voice", self._pyttsx3_voice_id)
                elif voices:
                    engine.setProperty("voice", voices[0].id)
                engine.setProperty("rate", 180)

                if output_path:
                    engine.save_to_file(text, str(output_path))
                    engine.runAndWait()
                    engine.stop()
                    wav_path = str(output_path)
                else:
                    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    engine.save_to_file(text, tmp.name)
                    engine.runAndWait()
                    engine.stop()
                    wav_path = tmp.name

                m = self._build_metrics(t0, wav_path)
                self.synthesis_finished.emit(m)
                return wav_path, m
            except Exception as e:
                self.error_occurred.emit(f"TTS fallback error: {e}")
                return None, str(e)

    def _speak_pyttsx3(self, text, speed=1.0, pitch=1.0):
        with self._lock:
            t0 = time.perf_counter()
            self.synthesis_started.emit()
            try:
                import pyttsx3
                engine = pyttsx3.init()
                voices = engine.getProperty("voices")
                if self._pyttsx3_voice_id:
                    engine.setProperty("voice", self._pyttsx3_voice_id)
                elif voices:
                    engine.setProperty("voice", voices[0].id)
                engine.setProperty("rate", int(180 * speed))
                engine.say(text)
                engine.runAndWait()
                engine.stop()
                elapsed = time.perf_counter() - t0
                m = TTSMetrics()
                m.synthesis_time = round(elapsed, 3)
                m.audio_duration = round(len(text.split()) / 3.0, 2)
                m.realtime_factor = round(m.audio_duration / elapsed, 2) if elapsed > 0 else 0
                self.last_metrics = m
                self.synthesis_finished.emit(m)
            except Exception as e:
                self.error_occurred.emit(str(e))

    def _play_wav_winsound(self, wav_path):
        try:
            import winsound
            winsound.PlaySound(str(wav_path), winsound.SND_FILENAME)
        except Exception:
            pass
