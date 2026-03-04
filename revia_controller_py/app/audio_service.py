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
            return
        self._listening = True
        self._always_listening = always
        self._stop_event.clear()
        self._stt_thread = threading.Thread(
            target=self._stt_loop, daemon=True
        )
        self._stt_thread.start()
        self.status_changed.emit("Listening...")
        self._start_volume_monitor()

    def stop_listening(self):
        self._listening = False
        self._always_listening = False
        self._stop_event.set()
        self._stop_volume_monitor()
        self.status_changed.emit("Stopped")

    def is_listening(self):
        return self._listening

    def _stt_loop(self):
        try:
            import speech_recognition as sr
        except ImportError:
            self.status_changed.emit("speech_recognition not installed")
            self._listening = False
            return

        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 300
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 1.5

        mic_kwargs = {}
        if self._input_device_index is not None:
            mic_kwargs["device_index"] = self._input_device_index

        try:
            mic = sr.Microphone(**mic_kwargs)
        except Exception as e:
            self.status_changed.emit(f"Mic error: {e}")
            self._listening = False
            return

        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            self.status_changed.emit("Listening... speak now")

            while self._listening and not self._stop_event.is_set():
                try:
                    audio = recognizer.listen(
                        source, timeout=5, phrase_time_limit=15
                    )
                except sr.WaitTimeoutError:
                    if self._always_listening:
                        continue
                    else:
                        continue
                except Exception:
                    continue

                # Transcribe in background
                try:
                    text = recognizer.recognize_google(audio)
                    if text.strip():
                        self.speech_recognized.emit(text.strip())
                        if not self._always_listening:
                            self._listening = False
                            self.status_changed.emit("Got speech")
                            break
                except sr.UnknownValueError:
                    pass
                except sr.RequestError as e:
                    self.status_changed.emit(f"STT API error: {e}")

        if not self._always_listening:
            self._listening = False

    # ---- TTS ----

    def speak(self, text):
        if not text:
            return
        threading.Thread(
            target=self._tts_speak, args=(text,), daemon=True
        ).start()

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
                self._tts_engine.runAndWait()
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
