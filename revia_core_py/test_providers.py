"""Unit tests for the V2.3 ProviderRegistry + adapters.

Adapters are tested against an in-process WSGI fake so we don't depend
on any local Ollama / llama.cpp install.

Run from inside ``revia_core_py``::

    python -m unittest test_providers -v
"""
from __future__ import annotations

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from agents.model_router import ModelRouter, NoRouteAdmittedError

from runtime import (
    HardwareFingerprint,
    HardwareSnapshot,
    ModelRequirements,
    ProviderRegistry,
    RuntimeScheduler,
)
from runtime.providers import (
    LlamaCppAdapter,
    LmStudioAdapter,
    OllamaAdapter,
    OpenAIAdapter,
    OpenAICompatAdapter,
    VllmAdapter,
)
from runtime.providers.base import ProviderAdapter, ProviderError


# ---------------------------------------------------------------------------
# Test HTTP server (responds to OpenAI-compat + Ollama endpoints)
# ---------------------------------------------------------------------------

class _FakeServerHandler(BaseHTTPRequestHandler):
    # Class-level state set per-test.
    chat_response: str = "Hello from fake server"
    fail_with_status: int = 0
    last_request_body: dict[str, Any] | None = None

    def log_message(self, fmt, *args):  # silence test output
        return

    def _json(self, code: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        if self.path.endswith("/v1/models"):
            return self._json(200, {"data": [
                {"id": "fake-model-7b"}, {"id": "fake-model-13b"}
            ]})
        if self.path.endswith("/api/tags"):
            return self._json(200, {"models": [
                {"name": "llama3:8b"}, {"name": "qwen2.5:14b"}
            ]})
        return self._json(200, {"ok": True})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            _FakeServerHandler.last_request_body = json.loads(raw.decode("utf-8"))
        except Exception:
            _FakeServerHandler.last_request_body = None

        if _FakeServerHandler.fail_with_status:
            return self._json(
                _FakeServerHandler.fail_with_status,
                {"error": {"message": "boom"}},
            )

        if self.path.endswith("/v1/chat/completions"):
            return self._json(200, {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": _FakeServerHandler.chat_response,
                    },
                }],
            })
        if self.path.endswith("/api/chat"):
            return self._json(200, {
                "message": {
                    "role": "assistant",
                    "content": _FakeServerHandler.chat_response,
                },
                "done": True,
            })
        return self._json(404, {"error": "not_found"})


class _FakeServer:
    def __init__(self):
        self._httpd = HTTPServer(("127.0.0.1", 0), _FakeServerHandler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    @property
    def port(self) -> int:
        return self._httpd.server_address[1]

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def stop(self):
        try:
            self._httpd.shutdown()
        except Exception:
            pass
        try:
            self._httpd.server_close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# OpenAI-compatible adapters
# ---------------------------------------------------------------------------

class TestOpenAICompatAdapter(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.server = _FakeServer()

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()

    def setUp(self):
        _FakeServerHandler.chat_response = "fake reply"
        _FakeServerHandler.fail_with_status = 0
        _FakeServerHandler.last_request_body = None

    def _adapter(self) -> OpenAICompatAdapter:
        return OpenAICompatAdapter(base_url=self.server.base_url + "/v1",
                                    default_model="fake-model-7b")

    def test_is_available_when_models_endpoint_works(self):
        a = self._adapter()
        self.assertTrue(a.is_available())
        self.assertTrue(a.info.available)

    def test_is_available_false_when_url_dead(self):
        a = OpenAICompatAdapter(base_url="http://127.0.0.1:1/v1",
                                 probe_timeout_s=0.05)
        self.assertFalse(a.is_available())

    def test_chat_returns_text(self):
        _FakeServerHandler.chat_response = "hello"
        a = self._adapter()
        out = a.chat(
            [{"role": "user", "content": "hi"}],
            model="fake-model-7b", max_tokens=16, temperature=0.5,
        )
        self.assertEqual(out, "hello")
        body = _FakeServerHandler.last_request_body or {}
        self.assertEqual(body.get("model"), "fake-model-7b")
        self.assertEqual(body.get("max_tokens"), 16)
        self.assertAlmostEqual(body.get("temperature", 0.0), 0.5)

    def test_chat_raises_provider_error_on_5xx(self):
        _FakeServerHandler.fail_with_status = 503
        a = self._adapter()
        with self.assertRaises(ProviderError):
            a.chat([{"role": "user", "content": "hi"}])

    def test_list_models(self):
        a = self._adapter()
        ids = a.list_models()
        self.assertIn("fake-model-7b", ids)
        self.assertIn("fake-model-13b", ids)

    def test_make_handler_signature(self):
        _FakeServerHandler.chat_response = "from-handler"
        a = self._adapter()
        handler = a.make_handler(system_prompt_provider=lambda: "be concise")
        out = handler("user input", broadcast_fn=None)
        self.assertEqual(out, "from-handler")
        body = _FakeServerHandler.last_request_body or {}
        msgs = body.get("messages") or []
        self.assertGreaterEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[0]["content"], "be concise")
        self.assertEqual(msgs[-1]["role"], "user")
        self.assertEqual(msgs[-1]["content"], "user input")


# ---------------------------------------------------------------------------
# Subclass smoke checks (rank, base_url defaults, requirements scaling)
# ---------------------------------------------------------------------------

class TestAdapterSubclasses(unittest.TestCase):

    def test_llamacpp_defaults_and_requirements(self):
        a = LlamaCppAdapter()
        self.assertEqual(a.name, "llama.cpp")
        self.assertTrue(a.base_url.endswith(":8080/v1"))
        fp = HardwareFingerprint(suggested_profile="high_24gb")
        req = a.requirements(fingerprint=fp)
        self.assertGreaterEqual(req.vram_mb, 8000)
        self.assertEqual(req.cost_class, "free")
        self.assertTrue(req.prefers_gpu)

    def test_lmstudio_defaults(self):
        a = LmStudioAdapter()
        self.assertTrue(a.base_url.endswith(":1234/v1"))

    def test_vllm_defaults_and_higher_vram_budget(self):
        a = VllmAdapter()
        self.assertTrue(a.base_url.endswith(":8000/v1"))
        fp = HardwareFingerprint(suggested_profile="high_24gb")
        req = a.requirements(fingerprint=fp)
        # vLLM should budget at least as much VRAM as llama.cpp.
        llama_req = LlamaCppAdapter().requirements(fingerprint=fp)
        self.assertGreaterEqual(req.vram_mb, llama_req.vram_mb)

    def test_cpu_only_profile_zero_vram(self):
        fp = HardwareFingerprint(suggested_profile="cpu_only")
        for cls in (LlamaCppAdapter, LmStudioAdapter, VllmAdapter):
            req = cls().requirements(fingerprint=fp)
            self.assertEqual(req.vram_mb, 0)
            self.assertTrue(req.cpu_bound)


# ---------------------------------------------------------------------------
# Ollama adapter
# ---------------------------------------------------------------------------

class TestOllamaAdapter(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.server = _FakeServer()

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()

    def setUp(self):
        _FakeServerHandler.chat_response = "fake reply"
        _FakeServerHandler.fail_with_status = 0

    def test_strips_v1_suffix_from_base_url(self):
        a = OllamaAdapter(base_url="http://localhost:11434/v1")
        self.assertEqual(a.base_url, "http://localhost:11434")

    def test_chat_via_native_api(self):
        _FakeServerHandler.chat_response = "ollama hello"
        a = OllamaAdapter(base_url=self.server.base_url, default_model="llama3")
        out = a.chat([{"role": "user", "content": "hi"}], model="llama3")
        self.assertEqual(out, "ollama hello")
        body = _FakeServerHandler.last_request_body or {}
        self.assertEqual(body.get("model"), "llama3")
        opts = body.get("options") or {}
        self.assertIn("num_predict", opts)
        self.assertIn("temperature", opts)

    def test_list_models_via_api_tags(self):
        a = OllamaAdapter(base_url=self.server.base_url)
        names = a.list_models()
        self.assertIn("llama3:8b", names)
        self.assertIn("qwen2.5:14b", names)

    def test_is_available_via_api_tags(self):
        a = OllamaAdapter(base_url=self.server.base_url)
        self.assertTrue(a.is_available())


# ---------------------------------------------------------------------------
# OpenAI cloud adapter
# ---------------------------------------------------------------------------

class TestOpenAIAdapter(unittest.TestCase):

    def test_unavailable_without_api_key(self):
        a = OpenAIAdapter(base_url="https://api.openai.com/v1", api_key="")
        self.assertFalse(a.is_available())
        self.assertEqual(a.info.last_error, "missing_api_key")

    def test_available_with_api_key(self):
        a = OpenAIAdapter(api_key="sk-test-key", base_url="https://example.com/v1")
        self.assertTrue(a.is_available())

    def test_requirements_zero_vram(self):
        a = OpenAIAdapter(api_key="sk-test")
        req = a.requirements()
        self.assertEqual(req.vram_mb, 0)
        self.assertEqual(req.cost_class, "paid")
        self.assertFalse(req.prefers_gpu)


# ---------------------------------------------------------------------------
# ProviderRegistry
# ---------------------------------------------------------------------------

class _FakeAdapter(ProviderAdapter):
    name = "Fake"
    cost_class = "free"

    def __init__(self, *, available: bool = True, name: str = "Fake",
                 reply: str = "ok"):
        super().__init__(base_url="http://example.invalid/v1")
        self.name = name
        self._info.name = name
        self._available = available
        self._reply = reply
        self.calls = 0

    def is_available(self) -> bool:
        self._info.available = self._available
        return self._available

    def chat(self, messages, **kw):
        self.calls += 1
        return self._reply

    def requirements(self, fingerprint=None) -> ModelRequirements:
        return ModelRequirements(
            vram_mb=0, prefers_gpu=False, cpu_bound=False,
            cost_class=self.cost_class,
        )


class _PaidFakeAdapter(_FakeAdapter):
    cost_class = "paid"


class TestProviderRegistry(unittest.TestCase):

    def test_discover_filters_unavailable(self):
        a1 = _FakeAdapter(name="A1", available=True)
        a2 = _FakeAdapter(name="A2", available=False)
        reg = ProviderRegistry(adapters=[a1, a2])
        reg.discover()
        avail = reg.available_entries()
        self.assertEqual(len(avail), 1)
        self.assertEqual(avail[0].adapter.name, "A1")

    def test_register_chat_routes_skips_unavailable(self):
        a1 = _FakeAdapter(name="A1", available=True, reply="from-A1")
        a2 = _FakeAdapter(name="A2", available=False, reply="from-A2")
        reg = ProviderRegistry(adapters=[a1, a2])
        reg.discover()
        router = ModelRouter()
        added = reg.register_chat_routes(router=router, task_type="reason_chat")
        self.assertEqual(added, 1)
        self.assertEqual(len(router.candidates("reason_chat")), 1)
        self.assertEqual(router.candidates("reason_chat")[0].backend_name,
                         "provider:A1")

    def test_paid_provider_ranked_after_free(self):
        free = _FakeAdapter(name="LocalFree", available=True)
        paid = _PaidFakeAdapter(name="CloudPaid", available=True)
        reg = ProviderRegistry(adapters=[paid, free])
        reg.discover()
        router = ModelRouter()
        reg.register_chat_routes(router=router, task_type="reason_chat")
        # Primary (lowest rank) must be the free local one.
        primary = router.candidates("reason_chat")[0]
        self.assertEqual(primary.backend_name, "provider:LocalFree")

    def test_router_falls_back_to_secondary_when_scheduler_blocks_first(self):
        # Build two free adapters, blast the scheduler with a no-room VRAM
        # budget but only on the first route — easiest way is to make their
        # requirements differ so we can rely on a mocked scheduler instead.
        a1 = _FakeAdapter(name="Big", available=True, reply="big")
        a2 = _FakeAdapter(name="Small", available=True, reply="small")
        reg = ProviderRegistry(adapters=[a1, a2])
        reg.discover()
        router = ModelRouter()
        # Inject a scheduler that denies the first try_reserve and admits
        # the second.  Simpler: use real RuntimeScheduler with a snapshot
        # that leaves no headroom and override one of the adapters'
        # requirements to need VRAM.
        a1._reply = "from-big"
        a2._reply = "from-small"

        class _BigReqAdapter(_FakeAdapter):
            def requirements(self, fingerprint=None):
                return ModelRequirements(vram_mb=10000, prefers_gpu=True,
                                         cost_class="free")

        a1.__class__ = _BigReqAdapter

        sched = RuntimeScheduler(
            snapshot_provider=lambda: HardwareSnapshot(
                vram_total_mb=8000, vram_used_mb=7800, pressure="normal",
            ),
            gpu_pool_cap=4, vram_safety_mb=256,
        )
        router_with_sched = ModelRouter(scheduler=sched)
        reg.register_chat_routes(router=router_with_sched, task_type="reason_chat")
        # Primary requires 10 GB but only ~200 MB free -> deny -> fall back.
        out = router_with_sched.call("reason_chat",
                                     [{"role": "user", "content": "hi"}])
        self.assertEqual(out, "from-small")

    def test_to_dict_shape(self):
        a1 = _FakeAdapter(name="A1", available=True)
        a2 = _PaidFakeAdapter(name="A2", available=False)
        reg = ProviderRegistry(adapters=[a1, a2])
        reg.discover()
        d = reg.to_dict()
        self.assertEqual(d["total_count"], 2)
        self.assertEqual(d["available_count"], 1)
        names = {p["name"] for p in d["providers"]}
        self.assertEqual(names, {"A1", "A2"})


if __name__ == "__main__":
    unittest.main()
