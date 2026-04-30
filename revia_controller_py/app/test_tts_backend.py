"""Focused tests for TTS text handling."""

import tempfile
import unittest
from pathlib import Path

from tts_backend import QwenTTSBackend, _strip_leading_style_directives


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
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        self.addCleanup(lambda: Path(tmp_path).unlink(missing_ok=True))

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
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        self.addCleanup(lambda: Path(tmp_path).unlink(missing_ok=True))

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
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        self.addCleanup(lambda: Path(tmp_path).unlink(missing_ok=True))

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
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        self.addCleanup(lambda: Path(tmp_path).unlink(missing_ok=True))

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
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        self.addCleanup(lambda: Path(tmp_path).unlink(missing_ok=True))

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
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        self.addCleanup(lambda: Path(tmp_path).unlink(missing_ok=True))

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
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        self.addCleanup(lambda: Path(tmp_path).unlink(missing_ok=True))

        backend = QwenTTSBackend()

        self.assertEqual(
            backend._extract_wav(({"path": tmp_path}, "ok"), None),
            tmp_path,
        )


if __name__ == "__main__":
    unittest.main()
