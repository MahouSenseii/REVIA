"""
Continuous Input Layer — PRD §6
================================
Always-on Voice Activity Detection (VAD) pipeline that:

  • Processes 30 ms audio frames continuously with < 5 ms processing latency
  • Emits speech_onset / speech_offset Qt Signals for the pipeline
  • Streams partial transcripts via partial_transcript Signal
  • Emits interruption_detected Signal when speech is detected while Revia
    is actively speaking (used by IHS to classify the interruption)
  • Exposes a thread-safe is_speaking_flag so the pipeline can gate TTS

PRD §6.2 — Frame budget
  Frame size:  30 ms  @ 16 kHz = 480 samples
  VAD budget:  < 5 ms per frame
  Onset grace: 2 consecutive speech frames before firing speech_onset
  Offset grace: 8 consecutive silence frames before firing speech_offset

All timing tolerances are derived from profile where possible.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable

try:
    from PySide6.QtCore import QObject, Signal
    _QT_AVAILABLE = True
except ImportError:
    # Allow module to be imported in non-Qt environments (unit tests, core_server)
    _QT_AVAILABLE = False
    class Signal:  # type: ignore[no-redef]
        def __init__(self, *args): pass
        def emit(self, *args): pass
        def connect(self, *args): pass
    class QObject: pass  # type: ignore[no-redef]

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# VAD constants (PRD §6.2)
# ---------------------------------------------------------------------------
_FRAME_MS           = 30          # ms per VAD frame
_SAMPLE_RATE        = 16_000      # Hz
_FRAME_SAMPLES      = int(_SAMPLE_RATE * _FRAME_MS / 1000)   # 480
_ONSET_HOLD_FRAMES  = 2           # consecutive speech frames → fire onset
_OFFSET_HOLD_FRAMES = 8           # consecutive silence frames → fire offset

# Energy-based VAD threshold (fraction of max int16)
# Production replacement: WebRTC VAD (webrtcvad) or Silero VAD.
# Silero VAD (https://github.com/snakers4/silero-vad) provides ML-based detection.
_DEFAULT_ENERGY_THRESHOLD = 0.005

# ── Barge-in anti-echo tuning ───────────────────────────────────────────────
# Without a proper AEC (acoustic echo canceller), the microphone will pick up
# Revia's own voice bleeding back through the speakers and trigger false
# barge-in. These parameters make barge-in detection conservative while TTS
# is actively playing.
_BARGE_IN_GRACE_S        = 1.20   # ignore barge-in for first 1.2 s of TTS
_BARGE_IN_HOLD_FRAMES    = 6      # 6 × 30 ms = 180 ms sustained speech to fire
_BARGE_IN_ENERGY_MULT    = 1.80   # additional energy multiplier during TTS


# ---------------------------------------------------------------------------
# ContinuousAudioPipeline
# ---------------------------------------------------------------------------

class ContinuousAudioPipeline(QObject):
    """
    PRD §6 — Continuous Input Layer

    Signals
    -------
    speech_onset           ()             User started speaking
    speech_offset          ()             User finished speaking (silence held)
    partial_transcript     (str)          Incremental STT text while speaking
    interruption_detected  (str)          Fired when speech detected during TTS;
                                          carries the utterance fragment so far
    vad_frame_processed    (float, bool)  (energy_rms, is_speech) per frame

    Usage::

        pipe = ContinuousAudioPipeline(profile_engine=pe)
        pipe.speech_onset.connect(pipeline.on_user_speech_start)
        pipe.speech_offset.connect(pipeline.on_user_speech_end)
        pipe.interruption_detected.connect(ihs.classify_interruption)
        pipe.start()
    """

    # Qt signals
    speech_onset          = Signal()
    speech_offset         = Signal()
    partial_transcript    = Signal(str)
    interruption_detected = Signal(str)
    vad_frame_processed   = Signal(float, bool)

    def __init__(self, profile_engine=None, parent=None):
        if _QT_AVAILABLE:
            super().__init__(parent)
        self._pe             = profile_engine
        self._running        = False
        self._lock           = threading.Lock()
        self._vad_thread: threading.Thread | None = None

        # State
        self._in_speech      = False
        self._speech_hold    = 0   # consecutive speech frames
        self._silence_hold   = 0   # consecutive silence frames
        self._partial_buffer: list[str] = []

        # Ambient noise tracking — protected by _ambient_lock so that any external
        # reader (e.g. a UI diagnostics thread) never races with the VAD writer.
        self._ambient_lock    = threading.Lock()
        self._ambient_samples: list[float] = []
        self._ambient_level: float = _DEFAULT_ENERGY_THRESHOLD * 0.5

        # Flag readable by the pipeline to know if TTS is active
        # Set this to True while Revia is speaking so the VAD can detect barge-in
        self._revia_speaking = threading.Event()
        # Timestamp (monotonic) when Revia most recently started speaking.
        # Used by the barge-in anti-echo grace window so that Revia's own
        # speaker output during the first _BARGE_IN_GRACE_S cannot trigger
        # a false interruption.
        self._tts_start_monotonic: float = 0.0

        # STT callback — inject a callable(bytes) -> str for partial transcription
        # When None, the pipeline only emits onset/offset events without text
        self._stt_callback: Callable[[bytes], str] | None = None

        # Audio source — inject a callable() -> bytes for testing
        # When None, falls back to pyaudio if available
        self._audio_source: Callable[[], bytes] | None = None
        self._pa_stream   = None

    # ── Public API ────────────────────────────────────────────────────────

    def set_stt_callback(self, fn: Callable[[bytes], str]) -> None:
        """Inject the STT function used to produce partial transcripts."""
        self._stt_callback = fn

    def set_audio_source(self, fn: Callable[[], bytes]) -> None:
        """
        Inject a custom audio source callable that returns one frame's worth
        of raw 16-bit PCM bytes.  Used for unit testing.
        """
        self._audio_source = fn

    def notify_revia_speaking(self, is_speaking: bool) -> None:
        """
        Called by the TTS subsystem to inform the VAD that Revia is currently
        producing speech.  When True, any detected user speech fires
        interruption_detected instead of (or in addition to) speech_onset.

        Records the monotonic start-time so the barge-in grace window
        (see _BARGE_IN_GRACE_S) can suppress false interruptions caused by
        acoustic echo during the first moments of TTS playback.
        """
        if is_speaking:
            # Only bump the start-time if transitioning from not-speaking.
            # Repeated True calls during a single utterance keep the grace
            # anchored to the actual start.
            if not self._revia_speaking.is_set():
                self._tts_start_monotonic = time.monotonic()
            self._revia_speaking.set()
        else:
            self._revia_speaking.clear()

    @property
    def is_user_speaking(self) -> bool:
        """Thread-safe read of current VAD state."""
        with self._lock:
            return self._in_speech

    def start(self) -> None:
        """Start the always-on VAD thread."""
        with self._lock:
            if self._running:
                return
            self._running = True
        self._vad_thread = threading.Thread(
            target=self._vad_loop, daemon=True, name="ContinuousVAD"
        )
        self._vad_thread.start()
        _log.info("[ContinuousAudio] VAD pipeline started (frame=%d ms)", _FRAME_MS)

    def stop(self) -> None:
        """Stop the VAD thread gracefully."""
        with self._lock:
            self._running = False
        if self._vad_thread:
            self._vad_thread.join(timeout=2.0)
        self._close_pa_stream()
        _log.info("[ContinuousAudio] VAD pipeline stopped")

    # ── VAD loop ──────────────────────────────────────────────────────────

    def _vad_loop(self) -> None:
        """
        Main VAD processing loop.  Runs in a dedicated daemon thread.

        Frame budget: < 5 ms processing per 30 ms frame (PRD §6.2).
        """
        self._init_audio_source()

        while True:
            with self._lock:
                if not self._running:
                    break

            t0         = time.monotonic()
            raw_frame  = self._read_frame()
            if raw_frame is None:
                time.sleep(_FRAME_MS / 1000)
                continue

            is_speech, energy = self._vad_classify(raw_frame)
            proc_ms    = (time.monotonic() - t0) * 1000

            if proc_ms > 5.0:
                _log.debug("[ContinuousAudio] Frame over budget: %.1f ms", proc_ms)

            # Emit per-frame signal (for diagnostics / UI meters)
            try:
                self.vad_frame_processed.emit(energy, is_speech)
            except Exception as exc:
                _log.debug("[ContinuousAudio] Signal emit error: %s", exc)

            self._update_state(is_speech, raw_frame)

    def _update_state(self, is_speech: bool, raw_frame: bytes) -> None:
        """
        State-machine: track onset/offset hold counters and emit events.

        While Revia is speaking, barge-in is subject to two anti-echo
        defenses (see _BARGE_IN_GRACE_S and _BARGE_IN_HOLD_FRAMES):

          • Grace period — no barge-in during the first 1.2 s of TTS,
            when echo from Revia's own speakers is most likely to be
            misclassified as user speech.
          • Stricter hold   — barge-in requires 6 consecutive speech
            frames (180 ms of sustained sound) rather than the 2 frames
            used for clean onset. Transient echo and keypresses are
            filtered out.

        Legitimate user barge-in (continuous speaking past 180 ms,
        after the grace window) still fires interruption_detected.
        """
        if is_speech:
            self._silence_hold = 0
            self._speech_hold += 1

            if self._in_speech:
                # Already in a speech run — stream any partial transcript
                partial = self._get_partial_text(raw_frame)
                if partial:
                    try:
                        self.partial_transcript.emit(partial)
                    except Exception:
                        pass
                return

            revia_speaking = self._revia_speaking.is_set()
            required_hold = (
                _BARGE_IN_HOLD_FRAMES if revia_speaking else _ONSET_HOLD_FRAMES
            )

            # Anti-echo grace period for barge-in.
            if revia_speaking:
                grace_elapsed = time.monotonic() - self._tts_start_monotonic
                if grace_elapsed < _BARGE_IN_GRACE_S:
                    return

            if self._speech_hold < required_hold:
                return

            with self._lock:
                self._in_speech = True

            if revia_speaking:
                _log.debug(
                    "[ContinuousAudio] barge-in detected "
                    "(hold=%d frames, grace_elapsed=%.2fs)",
                    self._speech_hold,
                    time.monotonic() - self._tts_start_monotonic,
                )
                fragment = self._get_partial_text(raw_frame)
                try:
                    self.interruption_detected.emit(fragment)
                except Exception:
                    pass
            else:
                _log.debug("[ContinuousAudio] speech_onset")
                try:
                    self.speech_onset.emit()
                except Exception:
                    pass

        else:
            self._speech_hold = 0
            if self._in_speech:
                self._silence_hold += 1
                if self._silence_hold >= _OFFSET_HOLD_FRAMES:
                    with self._lock:
                        self._in_speech    = False
                        self._silence_hold = 0
                    _log.debug("[ContinuousAudio] speech_offset")
                    try:
                        self.speech_offset.emit()
                    except Exception:
                        pass

    # ── VAD classifier ────────────────────────────────────────────────────

    def _vad_classify(self, raw_frame: bytes) -> tuple[bool, float]:
        """
        Classify a single audio frame as speech or silence.

        Production replacement: WebRTC VAD (webrtcvad) or Silero VAD.
        Current implementation: RMS energy threshold with adaptive ambient noise tracking.

        Returns (is_speech, energy_rms)
        """
        try:
            import struct
            n       = len(raw_frame) // 2
            samples = struct.unpack(f"<{n}h", raw_frame[:n * 2])
            rms     = (sum(s * s for s in samples) / n) ** 0.5
            energy  = rms / 32768.0   # normalise to [0, 1]
        except Exception:
            return False, 0.0

        self._update_ambient_noise(energy)
        is_speech = self._is_speech(energy)
        return is_speech, energy

    def _update_ambient_noise(self, rms: float):
        """Track ambient noise for adaptive VAD threshold."""
        with self._ambient_lock:
            self._ambient_samples.append(rms)
            if len(self._ambient_samples) > 100:
                self._ambient_samples.pop(0)
            self._ambient_level = sum(self._ambient_samples) / len(self._ambient_samples)

    def _is_speech(self, rms: float) -> bool:
        """Detect speech with adaptive threshold above ambient noise.

        While TTS is active the threshold is multiplied by
        _BARGE_IN_ENERGY_MULT to suppress echo from Revia's own speakers
        (acoustic leak from speaker to microphone).  Real user speech
        typically exceeds this boosted threshold; echo does not.
        """
        with self._ambient_lock:
            ambient = self._ambient_level
        # Speech must be significantly above ambient noise
        threshold = max(self._get_energy_threshold(), ambient * 2.5)
        if self._revia_speaking.is_set():
            threshold *= _BARGE_IN_ENERGY_MULT
        return rms > threshold

    # ── Audio I/O ─────────────────────────────────────────────────────────

    def _init_audio_source(self) -> None:
        """Initialise pyaudio stream if no custom source is injected."""
        if self._audio_source is not None:
            return   # custom source — skip pyaudio
        try:
            import pyaudio
            pa              = pyaudio.PyAudio()
            try:
                self._pa_stream = pa.open(
                    format            = pyaudio.paInt16,
                    channels          = 1,
                    rate              = _SAMPLE_RATE,
                    input             = True,
                    frames_per_buffer = _FRAME_SAMPLES,
                )
                _log.info("[ContinuousAudio] pyaudio stream opened at %d Hz", _SAMPLE_RATE)
            except Exception:
                pa.terminate()
                raise
        except Exception as exc:
            _log.warning(
                "[ContinuousAudio] pyaudio not available (%s). "
                "Inject set_audio_source() for custom input.",
                exc,
            )

    def _read_frame(self) -> bytes | None:
        """Read one 30 ms frame from the active audio source."""
        if self._audio_source is not None:
            try:
                return self._audio_source()
            except Exception:
                return None
        if self._pa_stream is not None:
            try:
                return self._pa_stream.read(_FRAME_SAMPLES, exception_on_overflow=False)
            except Exception:
                return None
        return None

    def _close_pa_stream(self) -> None:
        if self._pa_stream is not None:
            try:
                self._pa_stream.stop_stream()
                self._pa_stream.close()
            except Exception:
                pass
            self._pa_stream = None

    # ── Partial transcription ─────────────────────────────────────────────

    def _get_partial_text(self, raw_frame: bytes) -> str:
        """
        Call the injected STT callback if available.
        Returns empty string when no callback is wired.
        """
        if self._stt_callback is None:
            return ""
        try:
            return self._stt_callback(raw_frame) or ""
        except Exception as exc:
            _log.debug("[ContinuousAudio] STT callback error: %s", exc)
            return ""

    # ── Profile accessor ──────────────────────────────────────────────────

    def _get_energy_threshold(self) -> float:
        """
        In production a profile-tuned sensitivity could shift this threshold.
        We use interrupt_sensitivity as a proxy: higher sensitivity → lower
        energy threshold (easier to trigger VAD).
        """
        if self._pe:
            sensitivity = float(self._pe.interrupt_sensitivity)
            # Map [0.0, 1.0] sensitivity → [0.010, 0.002] threshold
            return max(0.002, 0.010 - sensitivity * 0.008)
        return _DEFAULT_ENERGY_THRESHOLD
