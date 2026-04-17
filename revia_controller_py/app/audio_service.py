import importlib.util
import threading
import numpy as np
from PySide6.QtCore import QObject, Signal, QTimer


class AudioService(QObject):
    """Handles STT (mic -> text), TTS (text -> speaker), and volume metering."""

    speech_recognized = Signal(str)   # emitted when speech is transcribed
    volume_level = Signal(float)      # 0.0-1.0 mic level for meter
    tts_started = Signal()
    tts_finished = Signal()
    status_changed = Signal(str)
    stt_listening_started = Signal()
    stt_listening_stopped = Signal()
    stt_processing_started = Signal()
    stt_processing_finished = Signal(bool, str)
    stt_error = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._listening = False
        self._always_listening = False
        self._stt_thread = None
        self._tts_engine = None
        self._tts_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._input_device_index = None
        self._output_device_name = None
        self._stt_phase = "idle"
        self._stt_recognition_slots = threading.BoundedSemaphore(2)

        # Volume monitoring
        self._vol_timer = QTimer(self)
        self._vol_timer.timeout.connect(self._emit_volume)
        self._current_volume = 0.0
        self._vol_stream = None

    def set_input_device(self, device_index):
        self._input_device_index = device_index

    def set_output_device(self, device_name):
        self._output_device_name = device_name

    # ---- STT ----

    def start_listening(self, always=False):
        if self._listening:
            return True
        ok, reason = self.stt_startup_check()
        if not ok:
            self._stt_phase = "error"
            self.stt_error.emit(reason)
            self.status_changed.emit(reason)
            return False
        self._listening = True
        self._always_listening = always
        self._stop_event.clear()
        self._stt_phase = "listening"
        self._stt_thread = threading.Thread(
            target=self._stt_loop, daemon=True
        )
        self._stt_thread.start()
        self.stt_listening_started.emit()
        self.status_changed.emit("Listening...")
        self._start_volume_monitor()
        return True

    def stop_listening(self):
        self._listening = False
        self._always_listening = False
        self._stop_event.set()
        self._stop_volume_monitor()
        if self._stt_phase == "listening":
            self.stt_listening_stopped.emit()
        elif self._stt_phase == "processing":
            self.stt_processing_finished.emit(False, "Cancelled")
        self._stt_phase = "idle"
        self.status_changed.emit("Stopped")

    def is_listening(self):
        return self._listening

    def is_stt_available(self):
        return importlib.util.find_spec("speech_recognition") is not None

    def stt_startup_check(self):
        """Return (ok, reason) for whether STT can be started now."""
        try:
            import speech_recognition as sr
        except Exception:
            return False, "speech_recognition not installed"

        try:
            names = sr.Microphone.list_microphone_names()
        except Exception as exc:
            return False, f"Microphone unavailable: {exc}"

        if not names:
            return False, "No microphone input devices found"
        return True, ""

    def _recognize_audio_async(self, audio, *, always_mode=False):
        """Transcribe one captured phrase without blocking the mic loop."""
        if not self._stt_recognition_slots.acquire(blocking=False):
            self.stt_error.emit("STT backlog full; dropped an audio chunk")
            return
        try:
            try:
                import speech_recognition as sr
            except ImportError:
                self._stt_phase = "error"
                self.stt_error.emit("speech_recognition not installed")
                self.status_changed.emit("speech_recognition not installed")
                self._listening = False
                return

            recognizer = sr.Recognizer()
            self._stt_phase = "processing"
            self.stt_processing_started.emit()
            try:
                text = recognizer.recognize_google(audio)
                text = text.strip()
                if text:
                    self.stt_processing_finished.emit(True, "")
                    self.speech_recognized.emit(text)
                    if not always_mode:
                        self._listening = False
                        self.status_changed.emit("Got speech")
                else:
                    self.stt_processing_finished.emit(False, "")
            except sr.UnknownValueError:
                self.stt_processing_finished.emit(False, "")
            except sr.RequestError as e:
                self._stt_phase = "error"
                message = f"STT API error: {e}"
                self.stt_error.emit(message)
                self.stt_processing_finished.emit(False, message)
                self.status_changed.emit(message)
            finally:
                if self._listening:
                    self._stt_phase = "listening"
                else:
                    self._stt_phase = "idle"
        finally:
            self._stt_recognition_slots.release()

    def _stt_loop(self):
        try:
            import speech_recognition as sr
        except ImportError:
            self._stt_phase = "error"
            self.stt_error.emit("speech_recognition not installed")
            self.status_changed.emit("speech_recognition not installed")
            self._listening = False
            return

        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 300
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 1.5  # Reduced from ~5 for faster STT detection

        mic_kwargs = {}
        if self._input_device_index is not None:
            mic_kwargs["device_index"] = self._input_device_index

        try:
            mic = sr.Microphone(**mic_kwargs)
        except Exception as e:
            self._stt_phase = "error"
            self.stt_error.emit(f"Mic error: {e}")
            self.status_changed.emit(f"Mic error: {e}")
            self._listening = False
            return

        _STT_MAX_CONSECUTIVE_ERRORS = 5
        _STT_BACKOFF_BASE_S = 2.0
        _STT_BACKOFF_CAP_S = 60.0
        _consecutive_errors = 0

        try:
            with mic as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                self.status_changed.emit("Listening... speak now")

                while self._listening and not self._stop_event.is_set():
                    if self._stt_phase != "listening":
                        self._stt_phase = "listening"
                        self.stt_listening_started.emit()
                    try:
                        audio = recognizer.listen(
                            source, timeout=5, phrase_time_limit=15
                        )
                    except sr.WaitTimeoutError:
                        continue
                    except Exception:
                        continue

                    if self._always_listening:
                        if self._stt_phase == "listening":
                            self.stt_listening_stopped.emit()
                        threading.Thread(
                            target=self._recognize_audio_async,
                            args=(audio,),
                            kwargs={"always_mode": True},
                            daemon=True,
                            name="revia-stt-recognize",
                        ).start()
                        self._stt_phase = "listening"
                        continue

                    # Transcribe single-shot microphone input inline.
                    try:
                        if self._stt_phase == "listening":
                            self.stt_listening_stopped.emit()
                        self._stt_phase = "processing"
                        self.stt_processing_started.emit()
                        text = recognizer.recognize_google(audio)
                        _consecutive_errors = 0  # reset on success
                        if text.strip():
                            self.speech_recognized.emit(text.strip())
                            self.stt_processing_finished.emit(True, "")
                            if not self._always_listening:
                                self._listening = False
                                self._stt_phase = "idle"
                                self.status_changed.emit("Got speech")
                                break
                            self._stt_phase = "listening"
                    except sr.UnknownValueError:
                        self.stt_processing_finished.emit(False, "")
                        self._stt_phase = "listening" if self._listening else "idle"
                    except sr.RequestError as e:
                        _consecutive_errors += 1
                        self._stt_phase = "error"
                        self.stt_error.emit(f"STT API error: {e}")
                        self.stt_processing_finished.emit(False, f"STT API error: {e}")
                        self.status_changed.emit(f"STT API error: {e}")
                        if _consecutive_errors >= _STT_MAX_CONSECUTIVE_ERRORS:
                            self.status_changed.emit(
                                f"STT: too many consecutive errors ({_consecutive_errors}), stopping"
                            )
                            self._listening = False
                            break
                        # Exponential backoff before next attempt
                        backoff = min(
                            _STT_BACKOFF_BASE_S * (2 ** (_consecutive_errors - 1)),
                            _STT_BACKOFF_CAP_S,
                        )
                        self._stop_event.wait(timeout=backoff)
                        self._stt_phase = "listening" if self._listening else "idle"
        finally:
            if not self._always_listening:
                self._listening = False
            if self._stt_phase == "listening":
                self.stt_listening_stopped.emit()
            self._stt_phase = "idle"

    # ---- TTS ----

    def speak(self, text):
        if not text:
            return
        threading.Thread(
            target=self._tts_speak, args=(text,), daemon=True
        ).start()

    _TTS_TIMEOUT_S = 30  # Maximum seconds to wait for pyttsx3 to finish speaking

    def _tts_speak(self, text):
        with self._tts_lock:
            try:
                if self._tts_engine is None:
                    import pyttsx3
                    self._tts_engine = pyttsx3.init()
                    voices = self._tts_engine.getProperty("voices")
                    if voices:
                        self._tts_engine.setProperty("voice", voices[0].id)
                    self._tts_engine.setProperty("rate", 180)
                self.tts_started.emit()
                self._tts_engine.say(text)
                # pyttsx3.runAndWait() blocks until the driver finishes but can hang
                # indefinitely on some platforms.  Run it in a daemon thread and join
                # with a timeout so we never stall the caller thread permanently.
                done = threading.Event()
                tts_engine = self._tts_engine

                def _run():
                    try:
                        tts_engine.runAndWait()
                    finally:
                        done.set()

                t = threading.Thread(target=_run, daemon=True)
                t.start()
                if not done.wait(timeout=self._TTS_TIMEOUT_S):
                    # Timed out - stop the engine and reset so the next call reinitialises.
                    try:
                        tts_engine.stop()
                    except Exception:
                        pass
                    self._tts_engine = None
                    self.status_changed.emit("TTS timed out")
                    return
                self.tts_finished.emit()
            except Exception as e:
                self._tts_engine = None  # reset so next call retries init
                self.status_changed.emit(f"TTS error: {e}")

    # ---- Volume meter ----

    def _start_volume_monitor(self):
        try:
            import sounddevice as sd
            dev = self._input_device_index
            self._vol_stream = sd.InputStream(
                device=dev, channels=1, samplerate=16000,
                blocksize=1024, callback=self._vol_callback,
            )
            self._vol_stream.start()
            self._vol_timer.start(50)
        except Exception:
            pass

    def _stop_volume_monitor(self):
        self._vol_timer.stop()
        if self._vol_stream:
            try:
                self._vol_stream.stop()
                self._vol_stream.close()
            except Exception:
                pass
            self._vol_stream = None
        self._current_volume = 0.0
        self.volume_level.emit(0.0)

    def _vol_callback(self, indata, frames, time_info, status):
        rms = float(np.sqrt(np.mean(indata ** 2)))
        self._current_volume = min(rms * 10.0, 1.0)

    def _emit_volume(self):
        self.volume_level.emit(self._current_volume)
