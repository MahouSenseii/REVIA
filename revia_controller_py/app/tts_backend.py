"""TTS backend for REVIA. Supports Qwen3-TTS via gradio_client and local pyttsx3 fallback."""
import logging
import os
import sys
import time
import wave
import shutil
import struct
import threading
import tempfile
import urllib.request
from pathlib import Path
from PySide6.QtCore import QObject, Signal

_log = logging.getLogger(__name__)


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

# Emotion-based TTS style instructions for natural speech variation
_EMOTION_STYLE_MAP = {
    "happy": "Speak with bright, upbeat energy and a warm smile in your voice",
    "excited": "Speak with high energy, faster pace, and enthusiastic emphasis",
    "sad": "Speak softly, slowly, with a gentle melancholic tone",
    "angry": "Speak with sharp, clipped intensity and firm emphasis",
    "nervous": "Speak with slight hesitation, softer volume, and uncertain pacing",
    "amused": "Speak with a light, playful lilt and hint of laughter",
    "neutral": "Speak naturally with balanced pacing and clear tone",
}


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
    playback_started = Signal()
    playback_finished = Signal()
    playback_interrupted = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lock = threading.Lock()
        self._engine_name = "qwen3-tts"
        self._qwen_url = ""  # custom Qwen3-TTS server URL (Gradio)
        self._qwen_clients = {}
        self._pyttsx3_voice_id = None
        self._pyttsx3_engine = None
        self.last_metrics = TTSMetrics()
        self._playback_lock = threading.Lock()
        self._interrupt_requested = False
        self._playback_active = False

    @property
    def engine_name(self):
        return self._engine_name

    def set_engine(self, name):
        self._engine_name = name

    def set_qwen_server(self, url):
        self._qwen_url = url.rstrip("/") if url else ""

    def _get_emotion_style(self, emotion: str) -> str:
        """Get TTS style instruction based on current emotion."""
        return _EMOTION_STYLE_MAP.get(emotion.lower(), _EMOTION_STYLE_MAP["neutral"])

    def is_ready(self):
        if self._engine_name == "pyttsx3":
            return True
        url = (self._qwen_url or "http://localhost:8000").rstrip("/")
        try:
            with urllib.request.urlopen(url + "/gradio_api/info", timeout=1.5):
                return True
        except Exception as exc:
            _log.debug("[TTS] Server not ready: %s", exc)
            return False

    def stop_output(self):
        with self._playback_lock:
            self._interrupt_requested = True
        try:
            with self._lock:
                engine = self._pyttsx3_engine
            if engine is not None:
                engine.stop()
        except Exception as exc:
            _log.debug("[TTS] Failed to stop pyttsx3: %s", exc)
        try:
            import sounddevice as sd
            sd.stop()
        except Exception as exc:
            _log.debug("[TTS] Failed to stop sounddevice: %s", exc)
        try:
            if sys.platform == "win32":
                import winsound
                winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception as exc:
            _log.debug("[TTS] Failed to stop winsound: %s", exc)

    def _begin_playback(self):
        with self._playback_lock:
            self._playback_active = True
            self._interrupt_requested = False
        self.playback_started.emit()

    def _finish_playback(self):
        with self._playback_lock:
            interrupted = self._interrupt_requested
            self._interrupt_requested = False
            self._playback_active = False
        if interrupted:
            self.playback_interrupted.emit()
        else:
            self.playback_finished.emit()

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
        except Exception as exc:
            _log.debug("[TTS] Failed to list system voices: %s", exc)
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

    # ── Streaming synthesis ──
    def synthesize_streaming(self, text: str, emotion="neutral", on_chunk_ready=None):
        """Synthesize text sentence-by-sentence for lower latency.

        Splits text at sentence boundaries and synthesizes each sentence
        independently, calling on_chunk_ready as soon as each is done.

        Args:
            text: Full text to synthesize
            emotion: Current emotion for voice style
            on_chunk_ready: Callback(audio_data, chunk_index, total_chunks)
        """
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        for i, sentence in enumerate(sentences):
            try:
                audio_data = self._synthesize_single(sentence, emotion)
                if audio_data is not None and on_chunk_ready:
                    on_chunk_ready(audio_data, i, len(sentences))
            except Exception as e:
                _log.warning("Sentence TTS failed for chunk %d: %s", i, e)
                continue

    def _synthesize_single(self, sentence: str, emotion="neutral"):
        """Synthesize a single sentence and return audio data.

        This is a helper for sentence-level streaming. Subclasses can override
        to use their preferred TTS backend (Qwen3, pyttsx3, etc).
        """
        try:
            wav_path, _ = self.generate_custom_voice(
                sentence,
                language="Auto",
                speaker="Ryan",
                style_instruction=self._get_emotion_style(emotion),
            )
            if wav_path:
                with open(wav_path, 'rb') as f:
                    return f.read()
        except Exception as exc:
            _log.debug("[TTS] Single sentence synthesis failed: %s", exc)
        return None

    # ── Play WAV file ──
    def play_wav(self, wav_path):
        """Play a WAV file through the default audio device."""
        def _do():
            started = False
            try:
                import sounddevice as sd
                import soundfile as sf
                data, sr = sf.read(str(wav_path))
                self._begin_playback()
                started = True
                sd.play(data, sr)
                # Poll for interrupt instead of blocking on sd.wait()
                while sd.get_stream().active:
                    if self._interrupt_requested:
                        sd.stop()
                        break
                    time.sleep(0.05)
            except ImportError as exc:
                _log.debug("[TTS] sounddevice/soundfile not available, falling back to winsound: %s", exc)
                self._begin_playback()
                started = True
                self._play_wav_winsound(wav_path)
            except Exception as exc:
                _log.error("[TTS] Playback error: %s", exc)
                self.error_occurred.emit(f"Playback error: {exc}")
            finally:
                if started:
                    self._finish_playback()
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
            # Pass emotion style instruction through to voice clone
            style_instruction = self._get_emotion_style(emotion)
            wav_result, info = self._qwen_clone(
                text,
                _wav_src,
                voice_profile.clone_ref_text or "",
                voice_profile.language or "Auto",
                not bool(voice_profile.clone_ref_text),
                None,
                voice_profile.model_size or "0.6B",
                style_instruction=style_instruction,
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
        started = False
        try:
            import sounddevice as sd
            import soundfile as sf
            data, sr = sf.read(str(wav_path))
            self._begin_playback()
            started = True
            sd.play(data, sr)
            # Poll for interrupt instead of blocking on sd.wait()
            while sd.get_stream().active:
                if self._interrupt_requested:
                    sd.stop()
                    break
                time.sleep(0.05)
        except ImportError as exc:
            _log.debug("[TTS] sounddevice/soundfile not available, falling back to winsound: %s", exc)
            self._begin_playback()
            started = True
            self._play_wav_winsound(wav_path)
        except Exception as exc:
            _log.error("[TTS] Playback error: %s", exc)
            self.error_occurred.emit(f"Playback error: {exc}")
        finally:
            if started:
                self._finish_playback()

    # ── Qwen3-TTS gradio_client calls ──

    def _has_gradio_client(self):
        try:
            from gradio_client import Client  # noqa: F401
            return True
        except ImportError as exc:
            _log.debug("[TTS] gradio_client not available: %s", exc)
            return False

    def _get_client(self, space_key="custom"):
        from gradio_client import Client
        url = self._qwen_url or QWEN_SPACES.get(space_key, QWEN_SPACES["custom"])
        try:
            with self._lock:
                cached = self._qwen_clients.get(url)
                if cached is not None:
                    return cached
            # Create outside lock (network I/O), then store under lock
            client = Client(url)
            with self._lock:
                # Double-check: another thread may have created it while we waited
                existing = self._qwen_clients.get(url)
                if existing is not None:
                    return existing
                self._qwen_clients[url] = client
            return client
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
            except Exception as exc:
                _log.error("[TTS] Voice Design error: %s", exc)
                self.error_occurred.emit(f"Voice Design error: {exc}")
                return None, str(exc)

    def _qwen_clone(self, target_text, ref_audio, ref_text, language,
                     x_vector_only, output_path, model_size="1.7B",
                     style_instruction=""):
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
                # Voice clone API doesn't have a style_instruction param, but we
                # can prepend a prosody hint to the target text so Qwen3 adjusts
                # its speech style accordingly (works with instruct-capable models).
                tts_text = target_text
                if style_instruction:
                    tts_text = f"[{style_instruction}] {target_text}"
                    _log.debug("[TTS] Injecting emotion style into clone text: %s", style_instruction)
                try:
                    result = client.predict(
                        ref, ref_text or "", x_vector_only,
                        tts_text, language,
                        api_name="/run_voice_clone",
                    )
                    _log.debug("[TTS] Used /run_voice_clone (local)")
                except Exception as exc:
                    _log.debug("[TTS] /run_voice_clone failed: %s, trying /generate_voice_clone", exc)
                    result = client.predict(
                        ref, ref_text or "", tts_text, language,
                        x_vector_only, model_size,
                        api_name="/generate_voice_clone",
                    )
                    _log.debug("[TTS] Used /generate_voice_clone (HF Space)")

                wav_path = self._extract_wav(result, output_path)
                m = self._build_metrics(t0, wav_path)
                self.synthesis_finished.emit(m)
                return wav_path, m
            except Exception as exc:
                _log.error("[TTS] Voice Clone error: %s", exc)
                self.error_occurred.emit(f"Voice Clone error: {exc}")
                return None, str(exc)

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
            except Exception as exc:
                _log.error("[TTS] CustomVoice error: %s", exc)
                self.error_occurred.emit(f"CustomVoice error: {exc}")
                return None, str(exc)

    def _extract_wav(self, result, output_path):
        """Extract WAV path from gradio result and optionally copy to output_path."""
        wav_path = None
        try:
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
        except Exception as exc:
            _log.error("[TTS] Failed to extract WAV: %s", exc)
            return None

    def _build_metrics(self, t0, wav_path):
        elapsed = time.perf_counter() - t0
        audio_dur = 0.0
        if wav_path and Path(wav_path).exists():
            try:
                import soundfile as sf
                data, sr = sf.read(str(wav_path))
                audio_dur = len(data) / sr
            except Exception as exc:
                _log.debug("[TTS] Failed to read audio metrics: %s", exc)
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
            except Exception as exc:
                _log.error("[TTS] Fallback generation error: %s", exc)
                self.error_occurred.emit(f"TTS fallback error: {exc}")
                return None, str(exc)

    def _speak_pyttsx3(self, text, speed=1.0, pitch=1.0):
        with self._lock:
            t0 = time.perf_counter()
            self.synthesis_started.emit()
            started = False
            try:
                import pyttsx3
                if self._pyttsx3_engine is None:
                    self._pyttsx3_engine = pyttsx3.init()
                engine = self._pyttsx3_engine
                voices = engine.getProperty("voices")
                if self._pyttsx3_voice_id:
                    engine.setProperty("voice", self._pyttsx3_voice_id)
                elif voices:
                    engine.setProperty("voice", voices[0].id)
                # Favor lower perceived latency for conversational turns.
                engine.setProperty("rate", int(205 * speed))
                self._begin_playback()
                started = True
                engine.say(text)
                engine.runAndWait()
                elapsed = time.perf_counter() - t0
                m = TTSMetrics()
                m.synthesis_time = round(elapsed, 3)
                m.audio_duration = round(len(text.split()) / 3.0, 2)
                m.realtime_factor = round(m.audio_duration / elapsed, 2) if elapsed > 0 else 0
                self.last_metrics = m
                self.synthesis_finished.emit(m)
            except Exception as exc:
                _log.error("[TTS] pyttsx3 speak error: %s", exc)
                self._pyttsx3_engine = None
                self.error_occurred.emit(str(exc))
            finally:
                if started:
                    self._finish_playback()

    def _play_wav_winsound(self, wav_path):
        if sys.platform != "win32":
            return
        try:
            import winsound
            winsound.PlaySound(str(wav_path), winsound.SND_FILENAME)
        except Exception as exc:
            _log.debug("[TTS] winsound playback error: %s", exc)
