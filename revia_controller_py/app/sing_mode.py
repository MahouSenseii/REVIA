"""
REVIA Sing Mode — Karaoke-style singing with Qwen3-TTS voice cloning.

Pipeline:
  1. Upload WAV song → separate vocals from instrumental (Demucs / spleeter)
  2. Extract lyrics + timing from vocal track (Whisper with word timestamps)
  3. For each lyric line, synthesize Revia's voice singing via Qwen3-TTS
     with pitch-contour style instructions derived from the original vocal
  4. Mix synthesised vocal lines over the instrumental backing track
  5. Play the combined karaoke output in real-time (line-by-line streaming)

Dependencies (installed on demand):
  - demucs or spleeter: vocal/instrumental separation
  - openai-whisper: lyrics extraction with timestamps
  - librosa: pitch analysis of original vocals
  - numpy / soundfile: audio manipulation
  - scipy: resampling, signal processing
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import threading
import time
import wave
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class SingModeState(str, Enum):
    IDLE = "idle"
    SEPARATING = "separating"          # splitting vocals from instrumental
    TRANSCRIBING = "transcribing"      # extracting lyrics + timestamps
    ANALYSING = "analysing"            # pitch analysis on original vocal
    SYNTHESISING = "synthesising"      # generating Revia's singing voice
    MIXING = "mixing"                  # combining synth vocal + instrumental
    READY = "ready"                    # karaoke output prepared
    PLAYING = "playing"                # live playback in progress
    ERROR = "error"


@dataclass
class LyricLine:
    """A single lyric line with timing and pitch metadata."""
    text: str
    start_sec: float
    end_sec: float
    avg_pitch_hz: float = 0.0          # average F0 of original vocal
    energy: float = 0.5                # relative energy (0-1)
    synth_wav: str = ""                # path to Revia's synthesised version

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "start": round(self.start_sec, 3),
            "end": round(self.end_sec, 3),
            "pitch_hz": round(self.avg_pitch_hz, 1),
            "energy": round(self.energy, 2),
            "synth_wav": self.synth_wav,
        }


@dataclass
class SongAnalysis:
    """Full analysis of an uploaded song."""
    original_path: str = ""
    instrumental_path: str = ""
    vocal_path: str = ""
    lyrics: list[LyricLine] = field(default_factory=list)
    bpm: float = 0.0
    key: str = ""
    duration_sec: float = 0.0
    sample_rate: int = 44100
    karaoke_output_path: str = ""

    def to_dict(self) -> dict:
        return {
            "original": self.original_path,
            "instrumental": self.instrumental_path,
            "vocal": self.vocal_path,
            "lyrics": [l.to_dict() for l in self.lyrics],
            "bpm": round(self.bpm, 1),
            "key": self.key,
            "duration_sec": round(self.duration_sec, 2),
            "sample_rate": self.sample_rate,
            "karaoke_output": self.karaoke_output_path,
        }


# ---------------------------------------------------------------------------
# Pitch → style instruction mapping
# ---------------------------------------------------------------------------

def _pitch_to_style(pitch_hz: float, energy: float, bpm: float) -> str:
    """Convert vocal analysis into a Qwen3-TTS style instruction for singing.

    Maps the original vocal's pitch range and energy into natural-language
    style hints that nudge Qwen3-TTS towards a more musical delivery.
    """
    parts = ["Sing this line melodically"]

    # Pitch register
    if pitch_hz > 400:
        parts.append("in a high, bright register")
    elif pitch_hz > 250:
        parts.append("in a mid-range, clear register")
    elif pitch_hz > 0:
        parts.append("in a low, warm register")

    # Energy / dynamics
    if energy > 0.75:
        parts.append("with strong, powerful projection")
    elif energy > 0.4:
        parts.append("with moderate, steady volume")
    else:
        parts.append("softly and gently")

    # Tempo feel
    if bpm > 140:
        parts.append("at an upbeat, energetic pace")
    elif bpm > 100:
        parts.append("at a moderate, flowing pace")
    elif bpm > 0:
        parts.append("slowly and expressively")

    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------

def _ensure_dependency(package: str, pip_name: str | None = None) -> bool:
    """Check if a Python package is available; log if missing."""
    try:
        __import__(package)
        return True
    except ImportError:
        _log.warning(
            "[SingMode] %s not installed. Install with: pip install %s",
            package, pip_name or package,
        )
        return False


def _get_audio_duration(wav_path: str) -> float:
    """Return duration in seconds of a WAV file."""
    try:
        import soundfile as sf
        data, sr = sf.read(wav_path)
        return len(data) / sr
    except Exception:
        try:
            with wave.open(wav_path, "rb") as wf:
                return wf.getnframes() / wf.getframerate()
        except Exception:
            return 0.0


def _read_audio(path: str):
    """Read audio file, return (numpy_array, sample_rate)."""
    import soundfile as sf
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    return data, sr


def _write_audio(path: str, data, sr: int):
    """Write numpy array to WAV."""
    import soundfile as sf
    sf.write(path, data, sr, subtype="PCM_16")


def _mono(data):
    """Convert to mono if stereo."""
    import numpy as np
    if data.ndim == 2 and data.shape[1] > 1:
        return np.mean(data, axis=1)
    return data.flatten()


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

class VocalSeparator:
    """Stage 1: Separate vocals from instrumental using Demucs."""

    @staticmethod
    def separate(input_wav: str, output_dir: str) -> tuple[str, str]:
        """Returns (instrumental_path, vocal_path).

        Uses demucs (Facebook's music source separation model).
        Falls back to a simple high-pass / low-pass filter if demucs unavailable.
        """
        vocal_out = os.path.join(output_dir, "vocals.wav")
        instrumental_out = os.path.join(output_dir, "instrumental.wav")

        if _ensure_dependency("demucs"):
            return VocalSeparator._demucs_separate(input_wav, output_dir,
                                                    vocal_out, instrumental_out)

        if _ensure_dependency("spleeter"):
            return VocalSeparator._spleeter_separate(input_wav, output_dir,
                                                      vocal_out, instrumental_out)

        _log.warning("[SingMode] No separator available, using spectral fallback")
        return VocalSeparator._spectral_fallback(input_wav, vocal_out, instrumental_out)

    @staticmethod
    def _demucs_separate(input_wav, output_dir, vocal_out, instrumental_out):
        import subprocess
        result = subprocess.run(
            ["python", "-m", "demucs", "--two-stems", "vocals",
             "-o", output_dir, "--filename", "{stem}.{ext}", input_wav],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Demucs failed: {result.stderr[:500]}")

        # Demucs outputs to output_dir/htdemucs/vocals.wav and no_vocals.wav
        demucs_dir = os.path.join(output_dir, "htdemucs")
        if not os.path.exists(demucs_dir):
            # Newer demucs versions may use different output structure
            for d in Path(output_dir).rglob("vocals.wav"):
                demucs_dir = str(d.parent)
                break

        src_vocal = os.path.join(demucs_dir, "vocals.wav")
        src_instr = os.path.join(demucs_dir, "no_vocals.wav")

        if os.path.exists(src_vocal):
            shutil.copy2(src_vocal, vocal_out)
        if os.path.exists(src_instr):
            shutil.copy2(src_instr, instrumental_out)

        return instrumental_out, vocal_out

    @staticmethod
    def _spleeter_separate(input_wav, output_dir, vocal_out, instrumental_out):
        from spleeter.separator import Separator
        separator = Separator("spleeter:2stems")
        separator.separate_to_file(input_wav, output_dir)

        # Spleeter outputs to output_dir/<filename>/vocals.wav, accompaniment.wav
        stem = Path(input_wav).stem
        src_vocal = os.path.join(output_dir, stem, "vocals.wav")
        src_instr = os.path.join(output_dir, stem, "accompaniment.wav")

        if os.path.exists(src_vocal):
            shutil.copy2(src_vocal, vocal_out)
        if os.path.exists(src_instr):
            shutil.copy2(src_instr, instrumental_out)

        return instrumental_out, vocal_out

    @staticmethod
    def _spectral_fallback(input_wav, vocal_out, instrumental_out):
        """Simple spectral subtraction fallback when no ML separator is available.

        Uses center-channel extraction: vocals are typically panned center,
        so subtracting L from R isolates instruments, and vice versa.
        """
        import numpy as np
        data, sr = _read_audio(input_wav)

        if data.shape[1] >= 2:
            # Center extraction: vocals ≈ (L+R)/2, instruments ≈ (L-R)/2
            left, right = data[:, 0], data[:, 1]
            vocals = (left + right) / 2.0
            instruments = (left - right) / 2.0
        else:
            # Mono: can't separate, just copy
            vocals = data.flatten()
            instruments = data.flatten()

        _write_audio(vocal_out, vocals, sr)
        _write_audio(instrumental_out, instruments, sr)
        return instrumental_out, vocal_out


class LyricsExtractor:
    """Stage 2: Extract lyrics + timestamps from vocal track using Whisper."""

    @staticmethod
    def extract(vocal_wav: str, language: str = "en") -> list[LyricLine]:
        """Returns a list of LyricLine with text and timing."""
        if _ensure_dependency("whisper", "openai-whisper"):
            return LyricsExtractor._whisper_extract(vocal_wav, language)

        _log.warning("[SingMode] Whisper not available — returning empty lyrics")
        return []

    @staticmethod
    def _whisper_extract(vocal_wav: str, language: str) -> list[LyricLine]:
        import whisper
        model = whisper.load_model("base")
        result = model.transcribe(
            vocal_wav,
            language=language if language != "Auto" else None,
            word_timestamps=True,
        )

        lines: list[LyricLine] = []
        for segment in result.get("segments", []):
            text = segment.get("text", "").strip()
            if not text:
                continue
            lines.append(LyricLine(
                text=text,
                start_sec=segment.get("start", 0.0),
                end_sec=segment.get("end", 0.0),
            ))

        _log.info("[SingMode] Extracted %d lyric lines", len(lines))
        return lines


class PitchAnalyser:
    """Stage 3: Analyse pitch contour of the original vocal for each lyric line."""

    @staticmethod
    def analyse(vocal_wav: str, lyrics: list[LyricLine],
                sample_rate: int = 44100) -> list[LyricLine]:
        """Enrich each LyricLine with average pitch and energy."""
        if not _ensure_dependency("librosa"):
            return lyrics

        import librosa
        import numpy as np

        y, sr = librosa.load(vocal_wav, sr=sample_rate, mono=True)

        # Extract pitch contour (F0) using pyin
        f0, voiced, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"), sr=sr,
        )

        # RMS energy per frame
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        max_rms = rms.max() if rms.max() > 0 else 1.0

        hop = 512
        for line in lyrics:
            start_frame = int(line.start_sec * sr / hop)
            end_frame = int(line.end_sec * sr / hop)
            end_frame = min(end_frame, len(f0) if f0 is not None else 0)

            if f0 is not None and start_frame < end_frame:
                segment_f0 = f0[start_frame:end_frame]
                voiced_f0 = segment_f0[np.isfinite(segment_f0) & (segment_f0 > 0)]
                line.avg_pitch_hz = float(np.median(voiced_f0)) if len(voiced_f0) > 0 else 0.0

            if start_frame < len(rms):
                seg_rms = rms[start_frame:min(end_frame, len(rms))]
                line.energy = float(np.mean(seg_rms) / max_rms) if len(seg_rms) > 0 else 0.5

        _log.info("[SingMode] Pitch analysis complete")
        return lyrics


class VocalSynthesiser:
    """Stage 4: Synthesise Revia's singing voice for each lyric line."""

    def __init__(self, tts_backend):
        """
        Args:
            tts_backend: QwenTTSBackend instance for voice synthesis
        """
        self._tts = tts_backend

    def synthesise_lyrics(
        self,
        lyrics: list[LyricLine],
        voice_profile,
        bpm: float,
        output_dir: str,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[LyricLine]:
        """Synthesise each lyric line with singing-style TTS.

        Returns the lyrics list with synth_wav paths populated.
        """
        try:
            from .voice_profile import _resolve_wav_path
        except ImportError:
            from voice_profile import _resolve_wav_path

        for i, line in enumerate(lyrics):
            if not line.text.strip():
                continue

            style = _pitch_to_style(line.avg_pitch_hz, line.energy, bpm)
            out_path = os.path.join(output_dir, f"line_{i:04d}.wav")

            try:
                if voice_profile and voice_profile.has_wav():
                    ref_wav = _resolve_wav_path(voice_profile.generated_wav)
                    wav_path, _ = self._tts.generate_voice_clone(
                        target_text=line.text,
                        ref_audio_path=ref_wav,
                        ref_text=voice_profile.clone_ref_text or "",
                        language=voice_profile.language or "Auto",
                        x_vector_only=not bool(voice_profile.clone_ref_text),
                        output_path=out_path,
                    )
                else:
                    wav_path, _ = self._tts.generate_custom_voice(
                        text=line.text,
                        language="Auto",
                        speaker=getattr(voice_profile, "speaker_id", "Ryan"),
                        style_instruction=style,
                        output_path=out_path,
                    )

                if wav_path and Path(wav_path).exists():
                    line.synth_wav = wav_path
                    _log.debug("[SingMode] Synthesised line %d/%d: %s",
                               i + 1, len(lyrics), line.text[:40])
                else:
                    _log.warning("[SingMode] Synthesis returned no file for line %d", i)

            except Exception as exc:
                _log.error("[SingMode] Failed to synthesise line %d: %s", i, exc)

            if on_progress:
                on_progress(i + 1, len(lyrics))

        return lyrics


class AudioMixer:
    """Stage 5: Mix synthesised vocal lines over the instrumental track."""

    @staticmethod
    def mix(
        instrumental_path: str,
        lyrics: list[LyricLine],
        output_path: str,
        vocal_gain: float = 1.0,
        instrumental_gain: float = 0.7,
    ) -> str:
        """Mix synthesised vocals onto the instrumental backing.

        Each synth line is placed at its timestamp offset. The instrumental
        is ducked slightly when vocals are active for clarity.

        Returns path to the final mixed WAV.
        """
        import numpy as np

        instr_data, sr = _read_audio(instrumental_path)
        instr = _mono(instr_data) * instrumental_gain
        total_samples = len(instr)

        # Create vocal track (same length as instrumental)
        vocal_track = np.zeros(total_samples, dtype=np.float32)

        for line in lyrics:
            if not line.synth_wav or not Path(line.synth_wav).exists():
                continue

            try:
                line_data, line_sr = _read_audio(line.synth_wav)
                line_mono = _mono(line_data)

                # Resample if needed
                if line_sr != sr:
                    from scipy.signal import resample
                    new_len = int(len(line_mono) * sr / line_sr)
                    line_mono = resample(line_mono, new_len).astype(np.float32)

                # Time-stretch to match the original line duration if needed
                target_samples = int(line.duration * sr)
                if target_samples > 0 and len(line_mono) > 0:
                    ratio = target_samples / len(line_mono)
                    if 0.5 < ratio < 2.0 and abs(ratio - 1.0) > 0.05:
                        from scipy.signal import resample
                        line_mono = resample(line_mono, target_samples).astype(np.float32)

                # Place at correct position
                start_sample = int(line.start_sec * sr)
                end_sample = start_sample + len(line_mono)
                end_sample = min(end_sample, total_samples)
                usable = end_sample - start_sample

                if usable > 0:
                    vocal_track[start_sample:end_sample] += line_mono[:usable] * vocal_gain

            except Exception as exc:
                _log.warning("[SingMode] Failed to mix line '%s': %s",
                             line.text[:30], exc)

        # Duck instrumental where vocals are active
        duck_mask = np.abs(vocal_track) > 0.01
        if np.any(duck_mask):
            # Smooth the mask to avoid clicks
            from scipy.ndimage import uniform_filter1d
            smooth_mask = uniform_filter1d(duck_mask.astype(np.float32),
                                           size=int(sr * 0.05))
            instr *= (1.0 - 0.3 * smooth_mask)  # duck by 30% where vocals active

        # Mix
        mixed = instr + vocal_track

        # Normalise to prevent clipping
        peak = np.abs(mixed).max()
        if peak > 0.95:
            mixed *= 0.95 / peak

        _write_audio(output_path, mixed, sr)
        _log.info("[SingMode] Mixed karaoke output: %s (%.1fs)",
                  output_path, len(mixed) / sr)
        return output_path


# ---------------------------------------------------------------------------
# BPM / Key detection
# ---------------------------------------------------------------------------

def _detect_bpm(audio_path: str) -> float:
    """Estimate BPM from audio file."""
    try:
        import librosa
        y, sr = librosa.load(audio_path, sr=22050, mono=True, duration=60)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        # librosa may return array or scalar depending on version
        if hasattr(tempo, '__len__'):
            return float(tempo[0]) if len(tempo) > 0 else 120.0
        return float(tempo) if tempo > 0 else 120.0
    except Exception as exc:
        _log.debug("[SingMode] BPM detection failed: %s", exc)
        return 120.0


def _detect_key(audio_path: str) -> str:
    """Estimate musical key from audio file."""
    try:
        import librosa
        import numpy as np
        y, sr = librosa.load(audio_path, sr=22050, mono=True, duration=30)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        key_idx = np.argmax(np.mean(chroma, axis=1))
        keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        return keys[key_idx]
    except Exception as exc:
        _log.debug("[SingMode] Key detection failed: %s", exc)
        return "C"


# ---------------------------------------------------------------------------
# Main SingMode controller
# ---------------------------------------------------------------------------

class SingMode:
    """Orchestrates the full karaoke pipeline.

    Usage:
        sing = SingMode(tts_backend)
        analysis = sing.prepare("song.wav", voice_profile)
        sing.play(analysis)      # plays karaoke in real-time
        sing.stop()              # interrupt playback
    """

    def __init__(self, tts_backend):
        self._tts = tts_backend
        self._state = SingModeState.IDLE
        self._lock = threading.Lock()
        self._interrupt = threading.Event()
        self._work_dir: str | None = None
        self._current_analysis: SongAnalysis | None = None

        # Callbacks
        self.on_state_change: Callable[[SingModeState], None] | None = None
        self.on_progress: Callable[[str, int, int], None] | None = None
        self.on_lyrics_update: Callable[[int, LyricLine], None] | None = None

    @property
    def state(self) -> SingModeState:
        return self._state

    @property
    def current_analysis(self) -> SongAnalysis | None:
        return self._current_analysis

    def _set_state(self, state: SingModeState):
        self._state = state
        if self.on_state_change:
            try:
                self.on_state_change(state)
            except Exception:
                pass

    def _report_progress(self, stage: str, current: int, total: int):
        if self.on_progress:
            try:
                self.on_progress(stage, current, total)
            except Exception:
                pass

    def prepare(
        self,
        wav_path: str,
        voice_profile=None,
        language: str = "en",
    ) -> SongAnalysis:
        """Run the full preparation pipeline on an uploaded WAV.

        This is blocking and may take 30-120 seconds depending on song length
        and available hardware. Run in a background thread for UI responsiveness.

        Args:
            wav_path: Path to the uploaded WAV/audio file
            voice_profile: VoiceProfile for Revia's voice (optional)
            language: Language hint for lyrics extraction

        Returns:
            SongAnalysis with all paths and lyrics populated
        """
        self._interrupt.clear()
        self._work_dir = tempfile.mkdtemp(prefix="revia_sing_")
        analysis = SongAnalysis(original_path=wav_path)

        try:
            # Get basic audio info
            analysis.duration_sec = _get_audio_duration(wav_path)
            _log.info("[SingMode] Processing: %s (%.1fs)", wav_path, analysis.duration_sec)

            # Stage 1: Vocal separation
            self._set_state(SingModeState.SEPARATING)
            self._report_progress("Separating vocals", 0, 5)
            instr_path, vocal_path = VocalSeparator.separate(wav_path, self._work_dir)
            analysis.instrumental_path = instr_path
            analysis.vocal_path = vocal_path

            if self._interrupt.is_set():
                self._set_state(SingModeState.IDLE)
                return analysis

            # Stage 2: Lyrics extraction
            self._set_state(SingModeState.TRANSCRIBING)
            self._report_progress("Extracting lyrics", 1, 5)
            analysis.lyrics = LyricsExtractor.extract(vocal_path, language)

            if self._interrupt.is_set():
                self._set_state(SingModeState.IDLE)
                return analysis

            # Stage 3: Pitch analysis
            self._set_state(SingModeState.ANALYSING)
            self._report_progress("Analysing pitch", 2, 5)
            analysis.bpm = _detect_bpm(wav_path)
            analysis.key = _detect_key(wav_path)
            analysis.lyrics = PitchAnalyser.analyse(vocal_path, analysis.lyrics)

            if self._interrupt.is_set():
                self._set_state(SingModeState.IDLE)
                return analysis

            # Stage 4: Synthesise Revia's vocals
            self._set_state(SingModeState.SYNTHESISING)
            self._report_progress("Synthesising vocals", 3, 5)
            synth_dir = os.path.join(self._work_dir, "synth")
            os.makedirs(synth_dir, exist_ok=True)

            synthesiser = VocalSynthesiser(self._tts)

            def _synth_progress(current, total):
                self._report_progress("Synthesising vocals", current, total)
                if self._interrupt.is_set():
                    raise InterruptedError("Sing mode cancelled")

            analysis.lyrics = synthesiser.synthesise_lyrics(
                analysis.lyrics, voice_profile, analysis.bpm,
                synth_dir, on_progress=_synth_progress,
            )

            if self._interrupt.is_set():
                self._set_state(SingModeState.IDLE)
                return analysis

            # Stage 5: Mix
            self._set_state(SingModeState.MIXING)
            self._report_progress("Mixing karaoke", 4, 5)
            output_path = os.path.join(self._work_dir, "karaoke_output.wav")
            AudioMixer.mix(analysis.instrumental_path, analysis.lyrics, output_path)
            analysis.karaoke_output_path = output_path

            self._report_progress("Complete", 5, 5)
            self._set_state(SingModeState.READY)
            self._current_analysis = analysis
            _log.info("[SingMode] Karaoke ready: %s", output_path)
            return analysis

        except InterruptedError:
            self._set_state(SingModeState.IDLE)
            _log.info("[SingMode] Preparation cancelled")
            # Clean up the temporary working directory so disk space is not leaked
            # on repeated cancellations.
            if self._work_dir and os.path.exists(self._work_dir):
                try:
                    shutil.rmtree(self._work_dir)
                except Exception:
                    pass
                self._work_dir = None
            return analysis

        except Exception as exc:
            self._set_state(SingModeState.ERROR)
            _log.error("[SingMode] Preparation failed: %s", exc, exc_info=True)
            # Clean up the temporary working directory on failure to prevent
            # repeated failures from leaking disk space.
            if self._work_dir and os.path.exists(self._work_dir):
                try:
                    shutil.rmtree(self._work_dir)
                except Exception:
                    pass
                self._work_dir = None
            raise

    def prepare_async(
        self,
        wav_path: str,
        voice_profile=None,
        language: str = "en",
        callback: Callable[[SongAnalysis], None] | None = None,
    ) -> threading.Thread:
        """Run prepare() in a background thread. Returns the thread handle."""
        def _run():
            try:
                result = self.prepare(wav_path, voice_profile, language)
                if callback:
                    callback(result)
            except Exception as exc:
                _log.error("[SingMode] Async preparation failed: %s", exc)

        t = threading.Thread(target=_run, daemon=True, name="revia-sing-prepare")
        t.start()
        return t

    def play(self, analysis: SongAnalysis | None = None):
        """Play the karaoke output through the TTS backend's audio system.

        Args:
            analysis: SongAnalysis to play, or use the last prepared one
        """
        target = analysis or self._current_analysis
        if not target or not target.karaoke_output_path:
            _log.warning("[SingMode] No karaoke output to play")
            return

        if not Path(target.karaoke_output_path).exists():
            _log.error("[SingMode] Karaoke file missing: %s",
                       target.karaoke_output_path)
            return

        self._set_state(SingModeState.PLAYING)
        self._tts.play_wav(target.karaoke_output_path)

        # Start lyrics sync thread for real-time display
        if target.lyrics and self.on_lyrics_update:
            threading.Thread(
                target=self._lyrics_sync_loop,
                args=(target.lyrics,),
                daemon=True,
                name="revia-sing-lyrics",
            ).start()

    def _lyrics_sync_loop(self, lyrics: list[LyricLine]):
        """Emit lyrics updates in real-time during playback."""
        start_time = time.monotonic()
        for i, line in enumerate(lyrics):
            if self._interrupt.is_set() or self._state != SingModeState.PLAYING:
                break
            # Wait until this line's start time
            elapsed = time.monotonic() - start_time
            wait = line.start_sec - elapsed
            if wait > 0:
                time.sleep(wait)
            if self.on_lyrics_update:
                try:
                    self.on_lyrics_update(i, line)
                except Exception:
                    pass

    def stop(self):
        """Stop any in-progress preparation or playback."""
        self._interrupt.set()
        self._tts.stop_output()
        if self._state == SingModeState.PLAYING:
            self._set_state(SingModeState.READY)
        elif self._state not in (SingModeState.IDLE, SingModeState.READY):
            self._set_state(SingModeState.IDLE)

    def cleanup(self):
        """Remove temporary working directory."""
        if self._work_dir and os.path.exists(self._work_dir):
            try:
                shutil.rmtree(self._work_dir)
                _log.debug("[SingMode] Cleaned up: %s", self._work_dir)
            except Exception as exc:
                _log.warning("[SingMode] Cleanup failed: %s", exc)
            self._work_dir = None

    def save_karaoke(self, output_path: str) -> str | None:
        """Copy the karaoke output WAV to a user-specified location."""
        if not self._current_analysis or not self._current_analysis.karaoke_output_path:
            return None
        src = self._current_analysis.karaoke_output_path
        if not Path(src).exists():
            return None
        shutil.copy2(src, output_path)
        return output_path

    def get_lyrics_text(self) -> str:
        """Return formatted lyrics from current analysis."""
        if not self._current_analysis:
            return ""
        return "\n".join(
            f"[{line.start_sec:.1f}s] {line.text}"
            for line in self._current_analysis.lyrics
        )
