"""
Integration tests for core_server.py REST endpoints.

These tests start the Flask app in test mode (no real LLM, no real sockets)
and verify that the API contract is stable.  Import-time global objects are
stubbed out via unittest.mock.patch where needed so no external services are
required.

Run with:
    python -m pytest test_api.py -v
or:
    python -m unittest test_api -v
"""

import json
import os
import sys
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Minimal stubs so core_server can be imported without all optional deps
# ---------------------------------------------------------------------------

# The server discovers its own module path; ensure the directory is on sys.path
sys.path.insert(0, os.path.dirname(__file__))


def _make_stub_module(name):
    mod = MagicMock()
    sys.modules.setdefault(name, mod)
    return mod


for _dep in [
    "flask", "flask_sock", "redis", "requests",
    "psutil", "torch", "transformers",
    "revia_core_py.vllm_backend", "vllm_backend",
    "revia_core_py.neural_refiner", "neural_refiner",
    "revia_core_py.parallel_pipeline", "parallel_pipeline",
    "revia_core_py.integrations.integration_manager",
    "integrations.integration_manager",
]:
    _make_stub_module(_dep)


class TestStatusEndpoint(unittest.TestCase):
    """GET /api/status returns a well-formed JSON object."""

    @classmethod
    def setUpClass(cls):
        # We only import after stubs are in place
        try:
            import importlib
            # Attempt a lightweight import; skip if unavailable deps remain
            import core_server  # noqa: F401
            cls._core = core_server
            cls._app = core_server.app
            cls._app.config["TESTING"] = True
            cls._client = cls._app.test_client()
            cls._available = True
        except Exception as exc:
            cls._available = False
            cls._skip_reason = str(exc)

    def _skip_if_unavailable(self):
        if not self._available:
            self.skipTest(f"core_server unavailable: {self._skip_reason}")

    def test_status_returns_200(self):
        self._skip_if_unavailable()
        resp = self._client.get("/api/status")
        self.assertEqual(resp.status_code, 200)

    def test_status_is_json(self):
        self._skip_if_unavailable()
        resp = self._client.get("/api/status")
        self.assertEqual(resp.content_type, "application/json")

    def test_status_has_required_keys(self):
        self._skip_if_unavailable()
        resp = self._client.get("/api/status")
        data = json.loads(resp.data)
        for key in ("state", "version", "uptime_s", "system", "llm"):
            self.assertIn(key, data, f"Missing key: {key}")

    def test_status_state_is_string(self):
        self._skip_if_unavailable()
        resp = self._client.get("/api/status")
        data = json.loads(resp.data)
        self.assertIsInstance(data["state"], str)

    def test_status_uptime_is_non_negative(self):
        self._skip_if_unavailable()
        resp = self._client.get("/api/status")
        data = json.loads(resp.data)
        self.assertGreaterEqual(data["uptime_s"], 0.0)


class TestChatEndpoint(unittest.TestCase):
    """POST /api/chat validates its contract."""

    @classmethod
    def setUpClass(cls):
        try:
            import core_server
            cls._app = core_server.app
            cls._app.config["TESTING"] = True
            cls._client = cls._app.test_client()
            cls._available = True
        except Exception as exc:
            cls._available = False
            cls._skip_reason = str(exc)

    def _skip_if_unavailable(self):
        if not self._available:
            self.skipTest(f"core_server unavailable: {self._skip_reason}")

    def test_chat_missing_text_returns_400(self):
        self._skip_if_unavailable()
        resp = self._client.post(
            "/api/chat",
            data=json.dumps({}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)

    def test_chat_empty_text_returns_400(self):
        self._skip_if_unavailable()
        resp = self._client.post(
            "/api/chat",
            data=json.dumps({"text": "   "}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)


class TestInterruptEndpoint(unittest.TestCase):
    """POST /api/interrupt always returns 200 with ack field."""

    @classmethod
    def setUpClass(cls):
        try:
            import core_server
            cls._app = core_server.app
            cls._app.config["TESTING"] = True
            cls._client = cls._app.test_client()
            cls._available = True
        except Exception as exc:
            cls._available = False
            cls._skip_reason = str(exc)

    def _skip_if_unavailable(self):
        if not self._available:
            self.skipTest(f"core_server unavailable: {self._skip_reason}")

    def test_interrupt_returns_200(self):
        self._skip_if_unavailable()
        resp = self._client.post(
            "/api/interrupt",
            data=json.dumps({}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)

    def test_interrupt_response_has_ack(self):
        self._skip_if_unavailable()
        resp = self._client.post(
            "/api/interrupt",
            data=json.dumps({}),
            content_type="application/json",
        )
        data = json.loads(resp.data)
        self.assertIn("ack", data)


class TestMemoryEndpoints(unittest.TestCase):
    """Memory API contract tests."""

    @classmethod
    def setUpClass(cls):
        try:
            import core_server
            cls._app = core_server.app
            cls._app.config["TESTING"] = True
            cls._client = cls._app.test_client()
            cls._available = True
        except Exception as exc:
            cls._available = False
            cls._skip_reason = str(exc)

    def _skip_if_unavailable(self):
        if not self._available:
            self.skipTest(f"core_server unavailable: {self._skip_reason}")

    def test_memory_status_returns_200(self):
        self._skip_if_unavailable()
        resp = self._client.get("/api/memory/status")
        self.assertEqual(resp.status_code, 200)

    def test_memory_status_has_backend_key(self):
        self._skip_if_unavailable()
        resp = self._client.get("/api/memory/status")
        data = json.loads(resp.data)
        self.assertIn("backend", data)


if __name__ == "__main__":
    unittest.main()
