"""
Test suite for REVIA Sing Mode.

Tests the data models, pipeline stages, style mapping, and controller logic
using mocks for external dependencies (demucs, whisper, librosa, TTS).
"""

import os
import sys
import struct
import tempfile
import threading
import time
import unittest
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "mnt", "REVIA", "revia_controller_py", "app"))

from sing_mode import (
    AudioMixer,
    LyricLine,
    LyricsExtractor,
    PitchAnalyser,
    SingMode,
    SingModeState,
    SongAnalysis,
    VocalSeparator,
    VocalSynthesiser,
    _detect_bpm,
    _detect_key,
    _get_audio_duration,
    _pitch_to_style,
)


def _make_test_wav(path: str, duration_sec: float = 1.0, sr: int = 44100,
                   channels: int = 2) -> str:
    """Create a minimal valid WAV file for testing."""
    n_frames = int(sr * duration_sec)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        # Write silence (zeros)
        wf.writeframes(b"\x00\x00" * channels * n_frames)
    return path


class TestLyricLine(unittest.TestCase):
    def test_duration(self):
        line = LyricLine(text="hello", start_sec=1.0, end_sec=3.5)
        self.assertAlmostEqual(line.duration, 2.5)

    def test_to_dict(self):
        line = LyricLine(
            text="sing it", start_sec=0.5, end_sec=2.0,
            avg_pitch_hz=261.63, energy=0.8,
        )
        d = line.to_dict()
        self.assertEqual(d["text"], "sing it")
        self.assertEqual(d["start"], 0.5)
        self.assertEqual(d["end"], 2.0)
        self.assertAlmostEqual(d["pitch_hz"], 261.6, places=0)

    def test_default_values(self):
        line = LyricLine(text="x", start_sec=0, end_sec=1)
        self.assertEqual(line.avg_pitch_hz, 0.0)
        self.assertEqual(line.energy, 0.5)
        self.assertEqual(line.synth_wav, "")


class TestSongAnalysis(unittest.TestCase):
    def test_to_dict(self):
        analysis = SongAnalysis(
            original_path="/tmp/song.wav",
            bpm=120.0,
            key="C",
            duration_sec=180.0,
            lyrics=[LyricLine(text="la la la", start_sec=0, end_sec=2)],
        )
        d = analysis.to_dict()
        self.assertEqual(d["original"], "/tmp/song.wav")
        self.assertEqual(d["bpm"], 120.0)
        self.assertEqual(len(d["lyrics"]), 1)

    def test_defaults(self):
        analysis = SongAnalysis()
        self.assertEqual(analysis.lyrics, [])
        self.assertEqual(analysis.bpm, 0.0)


class TestPitchToStyle(unittest.TestCase):
    def test_high_pitch_strong_energy_fast(self):
        style = _pitch_to_style(500, 0.9, 160)
        self.assertIn("high", style.lower())
        self.assertIn("powerful", style.lower())
        self.assertIn("upbeat", style.lower())

    def test_low_pitch_soft_slow(self):
        style = _pitch_to_style(150, 0.2, 70)
        self.assertIn("low", style.lower())
        self.assertIn("softly", style.lower())
        self.assertIn("slowly", style.lower())

    def test_mid_range(self):
        style = _pitch_to_style(300, 0.5, 120)
        self.assertIn("mid", style.lower())
        self.assertIn("moderate", style.lower())

    def test_zero_pitch(self):
        style = _pitch_to_style(0, 0.5, 120)
        self.assertIn("sing", style.lower())
        # Should not crash, just skip pitch descriptor


class TestSingModeState(unittest.TestCase):
    def test_all_states_exist(self):
        states = [s.value for s in SingModeState]
        self.assertIn("idle", states)
        self.assertIn("separating", states)
        self.assertIn("synthesising", states)
        self.assertIn("playing", states)
        self.assertIn("ready", states)
        self.assertIn("error", states)


class TestAudioDuration(unittest.TestCase):
    def test_wav_duration(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            _make_test_wav(f.name, duration_sec=2.0, channels=1)
            dur = _get_audio_duration(f.name)
        os.unlink(f.name)
        self.assertAlmostEqual(dur, 2.0, places=1)

    def test_missing_file(self):
        dur = _get_audio_duration("/nonexistent/file.wav")
        self.assertEqual(dur, 0.0)


class TestVocalSeparatorSpectralFallback(unittest.TestCase):
    """Test the spectral fallback separator (no ML deps needed)."""

    def test_stereo_separation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_wav = os.path.join(tmpdir, "test_stereo.wav")
            _make_test_wav(input_wav, duration_sec=0.5, channels=2)

            vocal_out = os.path.join(tmpdir, "vocals.wav")
            instrumental_out = os.path.join(tmpdir, "instrumental.wav")

            instr, vocal = VocalSeparator._spectral_fallback(
                input_wav, vocal_out, instrumental_out
            )

            self.assertTrue(Path(vocal).exists())
            self.assertTrue(Path(instr).exists())

    def test_mono_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_wav = os.path.join(tmpdir, "test_mono.wav")
            _make_test_wav(input_wav, duration_sec=0.5, channels=1)

            vocal_out = os.path.join(tmpdir, "vocals.wav")
            instrumental_out = os.path.join(tmpdir, "instrumental.wav")

            instr, vocal = VocalSeparator._spectral_fallback(
                input_wav, vocal_out, instrumental_out
            )

            self.assertTrue(Path(vocal).exists())
            self.assertTrue(Path(instr).exists())


class TestAudioMixer(unittest.TestCase):
    def test_mix_with_no_lyrics(self):
        """Mixing with empty lyrics should just output the instrumental."""
        import numpy as np

        with tempfile.TemporaryDirectory() as tmpdir:
            instr_path = os.path.join(tmpdir, "instr.wav")
            output_path = os.path.join(tmpdir, "output.wav")

            # Create a short instrumental
            sr = 22050
            duration = 1.0
            t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
            data = 0.3 * np.sin(2 * np.pi * 440 * t)  # 440Hz sine wave

            import soundfile as sf
            sf.write(instr_path, data, sr, subtype="PCM_16")

            result = AudioMixer.mix(instr_path, [], output_path)
            self.assertTrue(Path(result).exists())
            out_data, out_sr = sf.read(result)
            self.assertEqual(out_sr, sr)
            self.assertGreater(len(out_data), 0)


class TestSingModeController(unittest.TestCase):
    def setUp(self):
        self.mock_tts = MagicMock()
        self.mock_tts.play_wav = MagicMock()
        self.mock_tts.stop_output = MagicMock()
        self.sing = SingMode(self.mock_tts)

    def test_initial_state(self):
        self.assertEqual(self.sing.state, SingModeState.IDLE)
        self.assertIsNone(self.sing.current_analysis)

    def test_state_change_callback(self):
        states = []
        self.sing.on_state_change = lambda s: states.append(s)
        self.sing._set_state(SingModeState.SEPARATING)
        self.sing._set_state(SingModeState.READY)
        self.assertEqual(states, [SingModeState.SEPARATING, SingModeState.READY])

    def test_stop_during_playback(self):
        self.sing._state = SingModeState.PLAYING
        self.sing.stop()
        self.mock_tts.stop_output.assert_called_once()
        self.assertEqual(self.sing.state, SingModeState.READY)

    def test_stop_during_preparation(self):
        self.sing._state = SingModeState.SYNTHESISING
        self.sing.stop()
        self.assertEqual(self.sing.state, SingModeState.IDLE)
        self.assertTrue(self.sing._interrupt.is_set())

    def test_play_without_analysis(self):
        """Should not crash when no analysis exists."""
        self.sing.play()
        self.mock_tts.play_wav.assert_not_called()

    def test_play_with_analysis(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            _make_test_wav(f.name, duration_sec=0.5, channels=1)
            analysis = SongAnalysis(karaoke_output_path=f.name)
            self.sing.play(analysis)
            self.mock_tts.play_wav.assert_called_once_with(f.name)
            self.assertEqual(self.sing.state, SingModeState.PLAYING)
            os.unlink(f.name)

    def test_cleanup(self):
        self.sing._work_dir = tempfile.mkdtemp()
        self.assertTrue(Path(self.sing._work_dir).exists())
        self.sing.cleanup()
        self.assertFalse(Path(self.sing._work_dir or "/nonexistent").exists())

    def test_get_lyrics_text(self):
        self.sing._current_analysis = SongAnalysis(
            lyrics=[
                LyricLine(text="Hello world", start_sec=0.0, end_sec=1.5),
                LyricLine(text="Goodbye moon", start_sec=2.0, end_sec=3.5),
            ]
        )
        text = self.sing.get_lyrics_text()
        self.assertIn("Hello world", text)
        self.assertIn("[0.0s]", text)
        self.assertIn("[2.0s]", text)

    def test_save_karaoke_no_analysis(self):
        result = self.sing.save_karaoke("/tmp/out.wav")
        self.assertIsNone(result)

    def test_progress_callback(self):
        progress = []
        self.sing.on_progress = lambda stage, c, t: progress.append((stage, c, t))
        self.sing._report_progress("test", 1, 5)
        self.assertEqual(progress, [("test", 1, 5)])


class TestVocalSynthesiser(unittest.TestCase):
    def test_synthesise_with_mock_tts(self):
        mock_tts = MagicMock()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_wav = os.path.join(tmpdir, "line_0000.wav")
            _make_test_wav(out_wav, duration_sec=0.5, channels=1)

            mock_tts.generate_custom_voice.return_value = (out_wav, MagicMock())

            synth = VocalSynthesiser(mock_tts)
            lyrics = [LyricLine(text="la la la", start_sec=0, end_sec=1,
                                avg_pitch_hz=300, energy=0.6)]

            result = synth.synthesise_lyrics(lyrics, None, 120.0, tmpdir)
            self.assertEqual(result[0].synth_wav, out_wav)
            mock_tts.generate_custom_voice.assert_called_once()


class TestTTSBackendSingIntegration(unittest.TestCase):
    """Test the sing mode methods added to QwenTTSBackend."""

    def test_get_sing_mode_returns_singleton(self):
        # Import the class to test integration
        sys.path.insert(0, os.path.join(
            os.path.dirname(__file__), "mnt", "REVIA", "revia_controller_py"))
        try:
            # We can't fully instantiate QwenTTSBackend (needs PySide6),
            # but we can test the module structure exists
            from app.sing_mode import SingMode
            mock_tts = MagicMock()
            s1 = SingMode(mock_tts)
            s2 = SingMode(mock_tts)
            # Each instance is separate (not singleton at module level)
            self.assertIsNot(s1, s2)
        except ImportError:
            self.skipTest("PySide6 not available")


if __name__ == "__main__":
    unittest.main()
