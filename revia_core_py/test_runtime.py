"""Unit tests for the V2.1 runtime layer.

Covers:
    * HardwareProfiler classification + persistence
    * HardwareAgent snapshot + pressure classification
    * RuntimeScheduler concurrency caps + VRAM budget + priority
    * ModelRouter scheduler-aware fallback + admission

Run from inside ``revia_core_py``::

    python -m unittest test_runtime -v
"""
from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from agents.agent_base import AgentContext, CancellationToken
from agents.model_router import ModelRouter, NoRouteAdmittedError

from runtime import (
    HardwareAgent,
    HardwareFingerprint,
    HardwareProfiler,
    HardwareSnapshot,
    ModelRequirements,
    RuntimeScheduler,
)
from runtime.hardware_profiler import GpuInfo


# ---------------------------------------------------------------------------
# HardwareProfiler
# ---------------------------------------------------------------------------

class TestHardwareProfiler(unittest.TestCase):

    def test_classify_cpu_only_when_no_gpu(self):
        fp = HardwareFingerprint()
        self.assertEqual(HardwareProfiler._classify(fp), "cpu_only")

    def test_classify_low_8gb(self):
        fp = HardwareFingerprint(
            has_cuda=True,
            cuda_devices=[GpuInfo(index=0, name="RTX 4060", vram_total_mb=8192)],
        )
        self.assertEqual(HardwareProfiler._classify(fp), "low_8gb")

    def test_classify_mid_12gb(self):
        fp = HardwareFingerprint(
            has_cuda=True,
            cuda_devices=[GpuInfo(index=0, name="RTX 4070 Ti", vram_total_mb=12 * 1024)],
        )
        self.assertEqual(HardwareProfiler._classify(fp), "mid_12gb")

    def test_classify_high_24gb(self):
        fp = HardwareFingerprint(
            has_cuda=True,
            cuda_devices=[GpuInfo(index=0, name="RTX 4090", vram_total_mb=24 * 1024)],
        )
        self.assertEqual(HardwareProfiler._classify(fp), "high_24gb")

    def test_defaults_for_low_disable_critic_llm(self):
        d = HardwareProfiler._defaults_for("low_8gb")
        self.assertFalse(d["allow_critic_llm"])
        self.assertEqual(d["max_concurrent_llms"], 1)
        self.assertEqual(d["tts_model_size"], "0.6B")

    def test_defaults_for_high_allow_two_llms(self):
        d = HardwareProfiler._defaults_for("high_24gb")
        self.assertTrue(d["allow_critic_llm"])
        self.assertEqual(d["max_concurrent_llms"], 2)
        self.assertTrue(d["allow_vision"])

    def test_persist_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fp.json"
            profiler = HardwareProfiler(fingerprint_path=path)
            fp = HardwareFingerprint(
                cpu_brand="x86_64",
                cpu_cores_logical=12,
                ram_total_mb=32 * 1024,
                has_cuda=True,
                cuda_devices=[GpuInfo(index=0, name="RTX 4070", vram_total_mb=12288)],
                suggested_profile="mid_12gb",
            )
            profiler.save(fp)
            self.assertTrue(path.exists())
            loaded = profiler.load()
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.suggested_profile, "mid_12gb")
            self.assertEqual(len(loaded.cuda_devices), 1)
            self.assertEqual(loaded.cuda_devices[0].vram_total_mb, 12288)
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn("cuda_devices", data)


# ---------------------------------------------------------------------------
# HardwareAgent
# ---------------------------------------------------------------------------

class TestHardwareAgent(unittest.TestCase):

    def _agent(self, gpu_stats: dict | None, fingerprint=None) -> HardwareAgent:
        return HardwareAgent(
            fingerprint=fingerprint,
            gpu_stats_provider=lambda: gpu_stats or {},
            ttl_seconds=0.001,  # effectively no cache for tests
        )

    def test_take_snapshot_returns_struct(self):
        agent = self._agent({"gpu_percent": 30.0, "vram_used_mb": 2048,
                             "vram_total_mb": 8192})
        snap = agent.take_snapshot(force=True)
        self.assertIsInstance(snap, HardwareSnapshot)
        self.assertEqual(snap.vram_total_mb, 8192)
        self.assertEqual(snap.vram_used_mb, 2048)

    def test_pressure_critical_when_vram_full(self):
        agent = self._agent({"gpu_percent": 99.0, "vram_used_mb": 7900,
                             "vram_total_mb": 8000})
        snap = agent.take_snapshot(force=True)
        self.assertEqual(snap.pressure, "critical")

    def test_pressure_tight_when_85_percent(self):
        agent = self._agent({"gpu_percent": 80.0, "vram_used_mb": 7000,
                             "vram_total_mb": 8000})
        snap = agent.take_snapshot(force=True)
        self.assertEqual(snap.pressure, "tight")

    def test_recommendation_blocks_critic_under_critical(self):
        fp = HardwareFingerprint(
            recommended_defaults={"max_concurrent_llms": 2,
                                  "allow_critic_llm": True,
                                  "allow_vision": True,
                                  "vram_safety_mb": 512},
        )
        agent = self._agent({"gpu_percent": 99.0, "vram_used_mb": 23900,
                             "vram_total_mb": 24000}, fingerprint=fp)
        snap = agent.take_snapshot(force=True)
        self.assertEqual(snap.pressure, "critical")
        self.assertEqual(snap.recommendation["max_concurrent_llms"], 0)
        self.assertFalse(snap.recommendation["allow_critic_llm"])
        self.assertFalse(snap.recommendation["allow_vision"])

    def test_runs_as_agent(self):
        agent = self._agent({"gpu_percent": 10.0, "vram_used_mb": 1000,
                             "vram_total_mb": 8000})
        ctx = AgentContext(user_text="ignored",
                           cancel_token=CancellationToken("t"))
        result = agent.execute(ctx)
        self.assertTrue(result.success)
        self.assertIn("pressure", result.result)
        self.assertIn("recommendation", result.result)


# ---------------------------------------------------------------------------
# RuntimeScheduler
# ---------------------------------------------------------------------------

class TestRuntimeScheduler(unittest.TestCase):

    def _snap(self, vram_used=0, vram_total=8000, pressure="normal"):
        return HardwareSnapshot(
            vram_total_mb=vram_total, vram_used_mb=vram_used, pressure=pressure,
        )

    def test_concurrency_cap_blocks_second_admission(self):
        sched = RuntimeScheduler(
            snapshot_provider=lambda: self._snap(),
            gpu_pool_cap=1,
        )
        req = ModelRequirements(vram_mb=1000, prefers_gpu=True)
        first = sched.try_reserve(req, priority="high")
        self.assertIsNotNone(first)
        second = sched.try_reserve(req, priority="high")
        self.assertIsNone(second)
        self.assertIn("gpu_pool_full", sched.status().last_denied_reason)
        sched.release(first)
        third = sched.try_reserve(req, priority="high")
        self.assertIsNotNone(third)
        sched.release(third)

    def test_critical_priority_bypasses_cap(self):
        sched = RuntimeScheduler(
            snapshot_provider=lambda: self._snap(),
            gpu_pool_cap=0,  # everyone except critical is denied
        )
        req = ModelRequirements(vram_mb=1000, prefers_gpu=True)
        denied = sched.try_reserve(req, priority="high")
        self.assertIsNone(denied)
        admitted = sched.try_reserve(req, priority="critical")
        self.assertIsNotNone(admitted)
        sched.release(admitted)

    def test_vram_budget_denies_when_no_room(self):
        sched = RuntimeScheduler(
            snapshot_provider=lambda: self._snap(vram_used=7500, vram_total=8000),
            gpu_pool_cap=4,
            vram_safety_mb=256,
        )
        req = ModelRequirements(vram_mb=1000)  # 7500 + 1000 + 256 > 8000
        self.assertIsNone(sched.try_reserve(req, priority="high"))
        self.assertIn("vram_budget", sched.status().last_denied_reason)

    def test_vram_budget_admits_when_fits(self):
        sched = RuntimeScheduler(
            snapshot_provider=lambda: self._snap(vram_used=2000, vram_total=8000),
            gpu_pool_cap=4,
            vram_safety_mb=256,
        )
        req = ModelRequirements(vram_mb=4000)
        res = sched.try_reserve(req, priority="high")
        self.assertIsNotNone(res)
        sched.release(res)

    def test_pressure_critical_blocks_non_critical(self):
        sched = RuntimeScheduler(
            snapshot_provider=lambda: self._snap(pressure="critical"),
            gpu_pool_cap=4,
        )
        req = ModelRequirements(vram_mb=500)
        self.assertIsNone(sched.try_reserve(req, priority="high"))

    def test_release_is_idempotent(self):
        sched = RuntimeScheduler(snapshot_provider=lambda: self._snap(),
                                  gpu_pool_cap=2)
        req = ModelRequirements(vram_mb=100)
        res = sched.try_reserve(req)
        sched.release(res)
        sched.release(res)  # second call must not double-decrement
        self.assertEqual(sched.status().pool_in_use["gpu"], 0)
        self.assertEqual(sched.status().vram_in_use_mb, 0)

    def test_blocking_reserve_waits_then_returns_none_on_timeout(self):
        sched = RuntimeScheduler(
            snapshot_provider=lambda: self._snap(),
            gpu_pool_cap=1,
        )
        req = ModelRequirements(vram_mb=100)
        first = sched.try_reserve(req)
        self.assertIsNotNone(first)
        t0 = time.monotonic()
        second = sched.reserve(req, priority="high", timeout_ms=80)
        elapsed = (time.monotonic() - t0) * 1000.0
        self.assertIsNone(second)
        self.assertGreaterEqual(elapsed, 70)
        sched.release(first)

    def test_blocking_reserve_unblocks_on_release(self):
        sched = RuntimeScheduler(
            snapshot_provider=lambda: self._snap(),
            gpu_pool_cap=1,
        )
        req = ModelRequirements(vram_mb=100)
        first = sched.try_reserve(req)
        self.assertIsNotNone(first)
        result = {"res": None}

        def waiter():
            result["res"] = sched.reserve(req, priority="high", timeout_ms=2000)

        t = threading.Thread(target=waiter, daemon=True)
        t.start()
        time.sleep(0.05)
        sched.release(first)
        t.join(timeout=2.0)
        self.assertIsNotNone(result["res"])
        sched.release(result["res"])

    def test_configure_from_fingerprint_high_profile(self):
        fp = HardwareFingerprint(
            cpu_cores_logical=12,
            recommended_defaults={"max_concurrent_llms": 2,
                                  "vram_safety_mb": 1024},
        )
        sched = RuntimeScheduler(snapshot_provider=lambda: self._snap())
        sched.configure_from_fingerprint(fp)
        st = sched.status()
        self.assertEqual(st.pool_caps["gpu"], 2)
        self.assertEqual(st.pool_caps["cpu"], 10)


# ---------------------------------------------------------------------------
# ModelRouter (scheduler-aware)
# ---------------------------------------------------------------------------

class TestModelRouterScheduler(unittest.TestCase):

    def _snap(self, vram_used=0, vram_total=8000, pressure="normal"):
        return HardwareSnapshot(
            vram_total_mb=vram_total, vram_used_mb=vram_used, pressure=pressure,
        )

    def test_register_multiple_routes_and_pick_primary(self):
        router = ModelRouter()
        router.register("reason_chat", "primary",
                        handler=lambda *a, **k: "primary",
                        rank=10)
        router.register("reason_chat", "fallback",
                        handler=lambda *a, **k: "fallback",
                        rank=50)
        self.assertEqual(router.get("reason_chat").backend_name, "primary")
        self.assertEqual(router.call("reason_chat"), "primary")

    def test_scheduler_falls_back_when_primary_denied(self):
        sched = RuntimeScheduler(
            snapshot_provider=lambda: self._snap(vram_used=7900, vram_total=8000),
            gpu_pool_cap=4, vram_safety_mb=256,
        )
        router = ModelRouter(scheduler=sched)
        router.register(
            "reason_chat", "big_local",
            handler=lambda *a, **k: "big_local",
            requirements=ModelRequirements(vram_mb=4000, prefers_gpu=True),
            priority_class="high", rank=10,
        )
        router.register(
            "reason_chat", "openai_cloud",
            handler=lambda *a, **k: "openai_cloud",
            requirements=ModelRequirements(vram_mb=0, prefers_gpu=False,
                                           cost_class="paid"),
            priority_class="high", rank=50,
        )
        # Big local requires 4000 MB but only ~100 MB free above safety -> deny.
        # Cloud route has 0 vram cost -> admitted.
        self.assertEqual(router.call("reason_chat"), "openai_cloud")

    def test_router_raises_when_all_routes_denied(self):
        sched = RuntimeScheduler(
            snapshot_provider=lambda: self._snap(pressure="critical"),
            gpu_pool_cap=4,
        )
        router = ModelRouter(scheduler=sched)
        router.register(
            "reason_chat", "only_gpu",
            handler=lambda *a, **k: "x",
            requirements=ModelRequirements(vram_mb=2000, prefers_gpu=True),
            priority_class="high",
        )
        with self.assertRaises(NoRouteAdmittedError):
            router.call("reason_chat")

    def test_no_scheduler_means_v1_behaviour(self):
        router = ModelRouter()  # no scheduler
        router.register("reason_chat", "primary",
                        handler=lambda *a, **k: "primary")
        self.assertEqual(router.call("reason_chat"), "primary")

    def test_select_route_dry_runs(self):
        sched = RuntimeScheduler(
            snapshot_provider=lambda: self._snap(),
            gpu_pool_cap=1,
        )
        router = ModelRouter(scheduler=sched)
        router.register(
            "reason_chat", "primary",
            handler=lambda *a, **k: "x",
            requirements=ModelRequirements(vram_mb=100),
        )
        route = router.select_route("reason_chat")
        self.assertIsNotNone(route)
        # Dry run should NOT consume the slot.
        self.assertEqual(sched.status().pool_in_use["gpu"], 0)


if __name__ == "__main__":
    unittest.main()
