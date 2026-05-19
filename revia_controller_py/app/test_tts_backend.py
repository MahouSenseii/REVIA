"""Focused tests for TTS text handling."""

import tempfile
import unittest
import wave
from unittest.mock import patch
from pathlib import Path

from tts_backend import QwenTTSBackend, _strip_leading_style_directives


def _make_test_wav(path: str) -> None:
    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(b"\x00\x00" * 160)


def _make_temp_wav(testcase: unittest.TestCase) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()
    _make_test_wav(tmp_path)
    testcase.addCleanup(lambda: Path(tmp_path).unlink(missing_ok=True))
    return tmp_path


class TestTTSStyleSanitizer(unittest.TestCase):
    def test_removes_neutral_style_hint_from_spoken_text(self):
        text = "[Speak naturally with balanced pacing and clear tone] Hello there."
        self.assertEqual(
            _strip_leading_style_directives(text),
            "Hello there.",
        )

    def test_removes_sing_style_hint_from_lyric_text(self):
        text = "[Sing this line melodically, in a high register] Lonely"
        self.assertEqual(_strip_leading_style_directives(text), "Lonely")

    def test_keeps_normal_bracketed_user_text(self):
        text = "[aside] this is part of the message"
        self.assertEqual(_strip_leading_style_directives(text), text)


class FakeQwenClient:
    def __init__(self, wav_path=None):
        self.wav_path = wav_path
        self.calls = []

    def predict(self, *args, api_name=None):
        self.calls.append((args, api_name))
        return {"path": self.wav_path}


class EndpointAwareFakeQwenClient(FakeQwenClient):
    def __init__(self, wav_path=None, endpoints=None, advertised=None):
        super().__init__(wav_path)
        self.endpoints = set(endpoints or ())
        self.advertised = set(self.endpoints if advertised is None else advertised)

    def predict(self, *args, api_name=None):
        self.calls.append((args, api_name))
        if api_name not in self.endpoints:
            raise ValueError(f"Cannot find a function with `api_name`: {api_name}.")
        return {"path": self.wav_path}

    def view_api(self, return_format="dict"):
        return {"named_endpoints": {name: {} for name in self.advertised}}


class TestQwenEndpointHandling(unittest.TestCase):
    def test_is_ready_caches_success_and_api_names(self):
        backend = QwenTTSBackend()
        backend.set_qwen_server("http://localhost:8000")
        calls = []

        class _Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return b'{"named_endpoints": {"/run_voice_clone": {}}}'

        def _urlopen(url, timeout):
            calls.append((url, timeout))
            return _Response()

        with patch("tts_backend.urllib.request.urlopen", _urlopen):
            self.assertTrue(backend.is_ready())
            self.assertTrue(backend.is_ready())
            self.assertEqual(backend._get_qwen_api_names(), {"/run_voice_clone"})

        self.assertEqual(len(calls), 1)

    def test_is_ready_caches_short_failure(self):
        backend = QwenTTSBackend()
        backend.set_qwen_server("http://localhost:8000")
        calls = []

        def _urlopen(url, timeout):
            calls.append((url, timeout))
            raise OSError("not ready")

        with patch("tts_backend.urllib.request.urlopen", _urlopen):
            self.assertFalse(backend.is_ready())
            self.assertFalse(backend.is_ready())

        self.assertEqual(len(calls), 1)

    def test_synthesis_concurrency_clamps_to_safe_range(self):
        backend = QwenTTSBackend()
        backend.set_synthesis_concurrency(0)
        self.assertEqual(backend.synthesis_concurrency, 1)
        backend.set_synthesis_concurrency(99)
        self.assertEqual(backend.synthesis_concurrency, 4)

    def test_voice_design_missing_endpoint_is_nonfatal(self):
        backend = QwenTTSBackend()
        client = FakeQwenClient()
        statuses = []
        errors = []
        backend.status_updated.connect(statuses.append)
        backend.error_occurred.connect(errors.append)
        backend._get_client = lambda _space_key="design": client
        backend._get_qwen_api_names = lambda _client=None: {
            "/run_voice_clone",
            "/save_prompt",
            "/load_prompt_and_gen",
        }

        wav, info = backend._qwen_design("hello", "warm voice", "Auto", None)

        self.assertIsNone(wav)
        self.assertIn("Voice Design is not available", info)
        self.assertEqual(client.calls, [])
        self.assertEqual(errors, [])
        self.assertTrue(statuses)

    def test_voice_design_uses_local_run_endpoint_when_available(self):
        tmp_path = _make_temp_wav(self)

        backend = QwenTTSBackend()
        client = FakeQwenClient(tmp_path)
        errors = []
        backend.error_occurred.connect(errors.append)
        backend._get_client = lambda _space_key="design": client
        backend._get_qwen_api_names = lambda _client=None: {
            "/run_voice_design",
        }

        wav, _metrics = backend._qwen_design(
            "hello",
            "warm clear voice",
            "Auto",
            None,
        )

        self.assertEqual(wav, tmp_path)
        self.assertEqual(client.calls[0][1], "/run_voice_design")
        self.assertEqual(client.calls[0][0], (
            "hello",
            "Auto",
            "warm clear voice",
        ))
        self.assertEqual(errors, [])

    def test_voice_design_uses_custom_endpoint_when_available(self):
        tmp_path = _make_temp_wav(self)

        backend = QwenTTSBackend()
        client = FakeQwenClient(tmp_path)
        statuses = []
        errors = []
        backend.status_updated.connect(statuses.append)
        backend.error_occurred.connect(errors.append)
        backend._get_client = lambda _space_key="design": client
        backend._get_qwen_api_names = lambda _client=None: {"/generate_custom_voice"}

        wav, _metrics = backend._qwen_design(
            "hello",
            "warm clear voice",
            "Auto",
            None,
        )

        self.assertEqual(wav, tmp_path)
        self.assertEqual(client.calls[0][1], "/generate_custom_voice")
        self.assertEqual(client.calls[0][0][:4], (
            "hello",
            "Auto",
            "Ryan",
            "warm clear voice",
        ))
        self.assertEqual(errors, [])
        self.assertTrue(any("CustomVoice style fallback" in s for s in statuses))

    def test_voice_design_retries_when_discovery_misses_hf_endpoint(self):
        tmp_path = _make_temp_wav(self)

        backend = QwenTTSBackend()
        backend.set_qwen_server("local-test-server")
        client = EndpointAwareFakeQwenClient(
            tmp_path,
            endpoints={"/generate_voice_design"},
            advertised=set(),
        )
        errors = []
        backend.error_occurred.connect(errors.append)
        backend._get_client = lambda _space_key="design": client

        wav, _metrics = backend._qwen_design(
            "hello",
            "warm clear voice",
            "Auto",
            None,
        )

        self.assertEqual(wav, tmp_path)
        self.assertEqual(
            [call[1] for call in client.calls],
            ["/run_voice_design", "/generate_voice_design"],
        )
        self.assertEqual(errors, [])

    def test_voice_design_refreshes_stale_endpoint_cache(self):
        tmp_path = _make_temp_wav(self)

        backend = QwenTTSBackend()
        backend.set_qwen_server("local-test-server")
        with backend._qwen_api_cache_lock:
            backend._qwen_api_cache["local-test-server"] = {"/run_voice_design"}
        client = EndpointAwareFakeQwenClient(
            tmp_path,
            endpoints={"/generate_custom_voice"},
            advertised={"/generate_custom_voice"},
        )
        statuses = []
        errors = []
        backend.status_updated.connect(statuses.append)
        backend.error_occurred.connect(errors.append)
        backend._get_client = lambda _space_key="design": client

        wav, _metrics = backend._qwen_design(
            "hello",
            "warm clear voice",
            "Auto",
            None,
        )

        self.assertEqual(wav, tmp_path)
        self.assertEqual(
            [call[1] for call in client.calls],
            ["/run_voice_design", "/generate_custom_voice"],
        )
        self.assertEqual(errors, [])
        self.assertTrue(any("CustomVoice style fallback" in s for s in statuses))

    def test_custom_voice_uses_local_run_instruct_when_available(self):
        tmp_path = _make_temp_wav(self)

        backend = QwenTTSBackend()
        client = FakeQwenClient(tmp_path)
        errors = []
        backend.error_occurred.connect(errors.append)
        backend._get_client = lambda _space_key="custom": client
        backend._get_qwen_api_names = lambda _client=None: {"/run_instruct"}

        wav, _metrics = backend._qwen_custom(
            "hello",
            "Auto",
            "Ryan",
            "say it cheerfully",
            "0.6B",
            None,
        )

        self.assertEqual(wav, tmp_path)
        self.assertEqual(client.calls[0][1], "/run_instruct")
        # Local /run_instruct takes 4 inputs (no model_size).
        self.assertEqual(client.calls[0][0], (
            "hello",
            "Auto",
            "Ryan",
            "say it cheerfully",
        ))
        self.assertEqual(errors, [])

    def test_custom_voice_falls_back_to_hf_endpoint(self):
        tmp_path = _make_temp_wav(self)

        backend = QwenTTSBackend()
        client = FakeQwenClient(tmp_path)
        backend._get_client = lambda _space_key="custom": client
        backend._get_qwen_api_names = lambda _client=None: {"/generate_custom_voice"}

        wav, _metrics = backend._qwen_custom(
            "hello",
            "Auto",
            "Ryan",
            "say it cheerfully",
            "0.6B",
            None,
        )

        self.assertEqual(wav, tmp_path)
        self.assertEqual(client.calls[0][1], "/generate_custom_voice")
        # HF Space /generate_custom_voice takes 5 inputs (with model_size).
        self.assertEqual(client.calls[0][0], (
            "hello",
            "Auto",
            "Ryan",
            "say it cheerfully",
            "0.6B",
        ))

    def test_extract_wav_accepts_gradio_file_dict(self):
        tmp_path = _make_temp_wav(self)

        backend = QwenTTSBackend()

        self.assertEqual(
            backend._extract_wav(({"path": tmp_path}, "ok"), None),
            tmp_path,
        )

    def test_extract_wav_rejects_empty_file(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()
        self.addCleanup(lambda: Path(tmp_path).unlink(missing_ok=True))

        backend = QwenTTSBackend()

        self.assertIsNone(backend._extract_wav({"path": tmp_path}, None))


if __name__ == "__main__":
    unittest.main()
