"""Live hardware monitor agent.

Plugs into the existing :class:`Agent` base class so it can be added to
the orchestrator like any other agent (default priority: low / background).

Each ``run`` call produces a fresh :class:`HardwareSnapshot` that includes
a *pressure* label and a *recommendation* dict.  Both the
:class:`ModelRouter` and the :class:`RuntimeScheduler` consult the most
recent snapshot before admitting new work.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

try:  # pragma: no cover - optional dep
    import psutil as _psutil
except ImportError:  # pragma: no cover
    _psutil = None  # type: ignore[assignment]

try:  # pragma: no cover - both package and direct import
    from ..agents.agent_base import Agent, AgentContext
except ImportError:  # pragma: no cover
    from agents.agent_base import Agent, AgentContext  # type: ignore[no-redef]

from .hardware_profiler import HardwareFingerprint

_log = logging.getLogger(__name__)


@dataclass
class HardwareSnapshot:
    """A single live reading, derived from the existing nvidia-smi cache."""

    timestamp: float = 0.0
    cpu_percent: float = 0.0
    ram_used_mb: int = 0
    ram_total_mb: int = 0
    gpu_percent: float = 0.0
    vram_used_mb: int = 0
    vram_total_mb: int = 0
    in_flight_llms: int = 0
    last_llm_latency_ms: float = 0.0
    pressure: str = "normal"        # "spacious" | "normal" | "tight" | "critical"
    recommendation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": round(self.timestamp, 3),
            "cpu_percent": round(self.cpu_percent, 1),
            "ram_used_mb": int(self.ram_used_mb),
            "ram_total_mb": int(self.ram_total_mb),
            "gpu_percent": round(self.gpu_percent, 1),
            "vram_used_mb": int(self.vram_used_mb),
            "vram_total_mb": int(self.vram_total_mb),
            "in_flight_llms": int(self.in_flight_llms),
            "last_llm_latency_ms": round(self.last_llm_latency_ms, 1),
            "pressure": self.pressure,
            "recommendation": dict(self.recommendation),
        }

    @property
    def vram_free_mb(self) -> int:
        return max(0, int(self.vram_total_mb) - int(self.vram_used_mb))

    @property
    def ram_free_mb(self) -> int:
        return max(0, int(self.ram_total_mb) - int(self.ram_used_mb))


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class HardwareAgent(Agent):
    """Periodic live snapshot.  Caller can also use it stand-alone via
    :meth:`take_snapshot`."""

    name = "HardwareAgent"
    default_timeout_ms = 600

    def __init__(
        self,
        fingerprint: HardwareFingerprint | None = None,
        gpu_stats_provider: Callable[[], dict[str, Any]] | None = None,
        in_flight_provider: Callable[[], int] | None = None,
        latency_provider: Callable[[], float] | None = None,
        ttl_seconds: float = 2.0,
    ):
        self._fingerprint = fingerprint
        self._gpu_provider = gpu_stats_provider
        self._in_flight_provider = in_flight_provider
        self._latency_provider = latency_provider
        self._ttl = max(0.1, float(ttl_seconds))

        self._last_snapshot: HardwareSnapshot | None = None
        self._last_at: float = 0.0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def latest(self) -> HardwareSnapshot | None:
        with self._lock:
            return self._last_snapshot

    def take_snapshot(self, force: bool = False) -> HardwareSnapshot:
        now = time.monotonic()
        with self._lock:
            if (
                not force
                and self._last_snapshot is not None
                and (now - self._last_at) < self._ttl
            ):
                return self._last_snapshot

        snap = self._build_snapshot()

        with self._lock:
            self._last_snapshot = snap
            self._last_at = now
        return snap

    # ------------------------------------------------------------------
    # Agent protocol
    # ------------------------------------------------------------------

    def run(self, context: AgentContext) -> dict[str, Any]:
        context.cancel_token.raise_if_cancelled()
        snap = self.take_snapshot(force=False)
        return {
            "_confidence": 1.0 if snap.vram_total_mb or snap.ram_total_mb else 0.5,
            **snap.to_dict(),
        }

    # ------------------------------------------------------------------
    # Snapshot builders
    # ------------------------------------------------------------------

    def _build_snapshot(self) -> HardwareSnapshot:
        cpu_percent = self._cpu_percent()
        ram_used, ram_total = self._ram()
        gpu = self._gpu_stats()
        gpu_percent = float(gpu.get("gpu_percent", 0.0) or 0.0)
        vram_used = int(gpu.get("vram_used_mb", 0) or 0)
        vram_total = int(gpu.get("vram_total_mb", 0) or 0)

        in_flight = 0
        if self._in_flight_provider is not None:
            try:
                in_flight = int(self._in_flight_provider() or 0)
            except Exception:
                in_flight = 0

        latency = 0.0
        if self._latency_provider is not None:
            try:
                latency = float(self._latency_provider() or 0.0)
            except Exception:
                latency = 0.0

        pressure = self._classify_pressure(
            ram_used, ram_total, vram_used, vram_total, gpu_percent, cpu_percent,
        )
        recommendation = self._build_recommendation(pressure, vram_used, vram_total)

        return HardwareSnapshot(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            ram_used_mb=ram_used,
            ram_total_mb=ram_total,
            gpu_percent=gpu_percent,
            vram_used_mb=vram_used,
            vram_total_mb=vram_total,
            in_flight_llms=in_flight,
            last_llm_latency_ms=latency,
            pressure=pressure,
            recommendation=recommendation,
        )

    @staticmethod
    def _cpu_percent() -> float:
        if _psutil is None:
            return 0.0
        try:
            # interval=None uses the rolling reading (non-blocking).
            return float(_psutil.cpu_percent(interval=None) or 0.0)
        except Exception:
            return 0.0

    @staticmethod
    def _ram() -> tuple[int, int]:
        if _psutil is None:
            return 0, 0
        try:
            mem = _psutil.virtual_memory()
            return int(mem.used // (1024 * 1024)), int(mem.total // (1024 * 1024))
        except Exception:
            return 0, 0

    def _gpu_stats(self) -> dict[str, Any]:
        if self._gpu_provider is not None:
            try:
                stats = self._gpu_provider() or {}
                if isinstance(stats, dict):
                    return stats
            except Exception:
                pass
        # Fallback when no provider is wired (e.g. unit tests).
        return {"gpu_percent": 0.0, "vram_used_mb": 0, "vram_total_mb": 0}

    @staticmethod
    def _classify_pressure(
        ram_used: int,
        ram_total: int,
        vram_used: int,
        vram_total: int,
        gpu_pct: float,
        cpu_pct: float,
    ) -> str:
        """Coarse buckets used by the scheduler."""
        # If we have no hardware data at all, default to "normal".
        ram_ratio = (ram_used / ram_total) if ram_total else 0.0
        vram_ratio = (vram_used / vram_total) if vram_total else 0.0

        # CRITICAL: VRAM nearly full or sustained 100% GPU pressure.
        if vram_ratio >= 0.95 or (vram_total and vram_used >= vram_total - 256):
            return "critical"
        if cpu_pct >= 97 and ram_ratio >= 0.95:
            return "critical"

        # TIGHT: little headroom for new model launches.
        if vram_ratio >= 0.85 or ram_ratio >= 0.90:
            return "tight"
        if gpu_pct >= 92 and vram_total > 0:
            return "tight"

        # SPACIOUS: lots of headroom (used by Low priority work).
        if vram_total > 0 and vram_ratio < 0.40 and gpu_pct < 30 and cpu_pct < 60:
            return "spacious"
        if vram_total == 0 and ram_ratio < 0.50 and cpu_pct < 50:
            return "spacious"

        return "normal"

    def _build_recommendation(
        self, pressure: str, vram_used: int, vram_total: int,
    ) -> dict[str, Any]:
        defaults = {}
        if self._fingerprint is not None:
            defaults = dict(self._fingerprint.recommended_defaults or {})
        rec = {
            "max_concurrent_llms": int(defaults.get("max_concurrent_llms", 1)),
            "allow_critic_llm": bool(defaults.get("allow_critic_llm", False)),
            "allow_vision": bool(defaults.get("allow_vision", False)),
            "vram_safety_mb": int(defaults.get("vram_safety_mb", 256)),
            "defer_background": False,
        }
        if pressure == "tight":
            rec["max_concurrent_llms"] = max(1, rec["max_concurrent_llms"] - 1)
            rec["allow_critic_llm"] = False
            rec["defer_background"] = True
        elif pressure == "critical":
            rec["max_concurrent_llms"] = 0   # no new LLM admissions
            rec["allow_critic_llm"] = False
            rec["allow_vision"] = False
            rec["defer_background"] = True
        elif pressure == "spacious":
            # Only widen when the hardware profile actually allows it.
            if defaults.get("max_concurrent_llms", 1) >= 2:
                rec["max_concurrent_llms"] = int(defaults["max_concurrent_llms"])
        rec["vram_free_mb"] = max(0, int(vram_total) - int(vram_used))
        return rec
