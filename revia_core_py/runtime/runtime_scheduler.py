"""Hardware-aware runtime scheduler for the agent layer.

The scheduler is the gate between the Model Router and the LLM/vision/TTS
providers.  Every model call asks for a :class:`Reservation` first; if
admitting the call would exceed the active VRAM/CPU/concurrency budget,
the reservation is denied and the router can fall back to a smaller
route.  Already-running tasks are never preempted; the queue only
controls *new* admissions.

Design notes:
    * Three soft pools: ``gpu`` (LLM/vision), ``cpu`` (rule-based agents,
      embeddings on CPU), ``io`` (HTTP, embeddings via Ollama, TTS HTTP).
    * Each pool has a max-concurrency cap derived from the
      :class:`HardwareFingerprint`.
    * Priority classes: ``critical`` > ``high`` > ``normal`` > ``low``.
    * VRAM accounting is *advisory*.  When the latest snapshot's
      ``vram_free_mb`` drops below ``req.vram_mb + safety``, admission
      is denied unless priority is ``critical``.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

_log = logging.getLogger(__name__)


PRIORITY_CLASSES: tuple[str, ...] = ("critical", "high", "normal", "low")
_PRIORITY_RANK = {p: i for i, p in enumerate(PRIORITY_CLASSES)}


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ModelRequirements:
    """Resource requirements declared by a :class:`ModelRoute`."""

    vram_mb: int = 0
    cpu_bound: bool = False
    prefers_gpu: bool = True
    supports_streaming: bool = False
    latency_budget_ms: int = 8000
    cost_class: str = "free"   # "free" | "metered" | "paid"

    def to_dict(self) -> dict[str, Any]:
        return {
            "vram_mb": int(self.vram_mb),
            "cpu_bound": bool(self.cpu_bound),
            "prefers_gpu": bool(self.prefers_gpu),
            "supports_streaming": bool(self.supports_streaming),
            "latency_budget_ms": int(self.latency_budget_ms),
            "cost_class": self.cost_class,
        }


@dataclass
class Reservation:
    """Handle returned by :meth:`RuntimeScheduler.reserve`."""

    pool: str           # "gpu" | "cpu" | "io"
    vram_mb: int
    priority: str
    granted_at: float
    task_id: str
    _released: bool = field(default=False, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pool": self.pool,
            "vram_mb": int(self.vram_mb),
            "priority": self.priority,
            "granted_at": round(self.granted_at, 3),
            "task_id": self.task_id,
            "released": bool(self._released),
        }


@dataclass
class SchedulerStatus:
    """Read-only snapshot of scheduler state for debugging / `/api/agents/hardware`."""

    pool_caps: dict[str, int]
    pool_in_use: dict[str, int]
    vram_in_use_mb: int
    pending_by_priority: dict[str, int]
    last_denied_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "pool_caps": dict(self.pool_caps),
            "pool_in_use": dict(self.pool_in_use),
            "vram_in_use_mb": int(self.vram_in_use_mb),
            "pending_by_priority": dict(self.pending_by_priority),
            "last_denied_reason": self.last_denied_reason,
        }


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class RuntimeScheduler:
    """Cooperative, in-process priority scheduler with VRAM/concurrency caps.

    The scheduler is intentionally simple: it does not own thread pools.
    The agent runtime already has its own pools (see
    ``ParallelPipeline`` / ``AgentOrchestrator``).  The scheduler's job
    is to decide *whether* a new model-bearing task may run *right now*
    based on the latest :class:`HardwareSnapshot`.
    """

    def __init__(
        self,
        snapshot_provider: Callable[[], Any] | None = None,
        gpu_pool_cap: int = 1,
        cpu_pool_cap: int | None = None,
        io_pool_cap: int = 8,
        vram_safety_mb: int = 256,
        log_fn=None,
    ):
        self._snapshot_provider = snapshot_provider
        self._gpu_cap = max(0, int(gpu_pool_cap))
        self._cpu_cap = (
            max(1, int(cpu_pool_cap))
            if cpu_pool_cap is not None
            else 4
        )
        self._io_cap = max(1, int(io_pool_cap))
        self._vram_safety_mb = max(0, int(vram_safety_mb))
        self._log = log_fn or _log.info

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._in_use: dict[str, int] = {"gpu": 0, "cpu": 0, "io": 0}
        self._vram_in_use: int = 0
        self._pending: dict[str, int] = {p: 0 for p in PRIORITY_CLASSES}
        self._last_denied_reason: str = ""
        self._task_counter: int = 0

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def configure_from_fingerprint(self, fingerprint) -> None:
        """Apply caps derived from a :class:`HardwareFingerprint`."""
        if fingerprint is None:
            return
        defaults = dict(getattr(fingerprint, "recommended_defaults", {}) or {})
        max_llms = int(defaults.get("max_concurrent_llms", 1))
        cpu_cores = int(getattr(fingerprint, "cpu_cores_logical", 0) or 0)
        with self._lock:
            self._gpu_cap = max(0, max_llms)
            if cpu_cores:
                self._cpu_cap = max(1, cpu_cores - 2)
            self._vram_safety_mb = int(defaults.get("vram_safety_mb", self._vram_safety_mb))

    def set_caps(
        self,
        gpu: int | None = None,
        cpu: int | None = None,
        io: int | None = None,
        vram_safety_mb: int | None = None,
    ) -> None:
        with self._lock:
            if gpu is not None:
                self._gpu_cap = max(0, int(gpu))
            if cpu is not None:
                self._cpu_cap = max(1, int(cpu))
            if io is not None:
                self._io_cap = max(1, int(io))
            if vram_safety_mb is not None:
                self._vram_safety_mb = max(0, int(vram_safety_mb))

    # ------------------------------------------------------------------
    # Core admission API
    # ------------------------------------------------------------------

    def try_reserve(
        self,
        requirements: ModelRequirements,
        priority: str = "normal",
        task_id: str = "",
    ) -> Reservation | None:
        """Non-blocking reservation attempt.  Returns ``None`` on denial."""
        priority = priority if priority in _PRIORITY_RANK else "normal"
        pool = self._select_pool(requirements)
        with self._lock:
            ok, reason = self._can_admit_locked(pool, requirements, priority)
            if not ok:
                self._last_denied_reason = reason
                return None
            self._task_counter += 1
            tid = task_id or f"task-{self._task_counter}"
            self._in_use[pool] += 1
            self._vram_in_use += int(requirements.vram_mb)
            res = Reservation(
                pool=pool,
                vram_mb=int(requirements.vram_mb),
                priority=priority,
                granted_at=time.monotonic(),
                task_id=tid,
            )
        return res

    def reserve(
        self,
        requirements: ModelRequirements,
        priority: str = "normal",
        timeout_ms: int = 0,
        task_id: str = "",
    ) -> Reservation | None:
        """Blocking variant.  Waits up to ``timeout_ms`` for a slot.

        ``timeout_ms <= 0`` is equivalent to :meth:`try_reserve`.
        """
        if timeout_ms <= 0:
            return self.try_reserve(requirements, priority=priority, task_id=task_id)

        priority = priority if priority in _PRIORITY_RANK else "normal"
        deadline = time.monotonic() + timeout_ms / 1000.0
        pool = self._select_pool(requirements)
        with self._cond:
            self._pending[priority] += 1
            try:
                while True:
                    ok, reason = self._can_admit_locked(pool, requirements, priority)
                    if ok:
                        self._task_counter += 1
                        tid = task_id or f"task-{self._task_counter}"
                        self._in_use[pool] += 1
                        self._vram_in_use += int(requirements.vram_mb)
                        return Reservation(
                            pool=pool,
                            vram_mb=int(requirements.vram_mb),
                            priority=priority,
                            granted_at=time.monotonic(),
                            task_id=tid,
                        )
                    self._last_denied_reason = reason
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                    self._cond.wait(timeout=remaining)
            finally:
                self._pending[priority] = max(0, self._pending[priority] - 1)

    def release(self, reservation: Reservation | None) -> None:
        """Return reservation resources to the pool.  Safe to call twice."""
        if reservation is None:
            return
        with self._cond:
            if reservation._released:
                return
            reservation._released = True
            self._in_use[reservation.pool] = max(
                0, self._in_use.get(reservation.pool, 0) - 1
            )
            self._vram_in_use = max(0, self._vram_in_use - int(reservation.vram_mb))
            self._cond.notify_all()

    def reservation_context(
        self,
        requirements: ModelRequirements,
        priority: str = "normal",
        timeout_ms: int = 0,
        task_id: str = "",
    ):
        """Convenience context manager around reserve/release."""
        return _ReservationCtx(self, requirements, priority, timeout_ms, task_id)

    # ------------------------------------------------------------------
    # Status / introspection
    # ------------------------------------------------------------------

    def status(self) -> SchedulerStatus:
        with self._lock:
            return SchedulerStatus(
                pool_caps={
                    "gpu": self._gpu_cap,
                    "cpu": self._cpu_cap,
                    "io": self._io_cap,
                },
                pool_in_use=dict(self._in_use),
                vram_in_use_mb=int(self._vram_in_use),
                pending_by_priority=dict(self._pending),
                last_denied_reason=self._last_denied_reason,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _select_pool(req: ModelRequirements) -> str:
        if req.cpu_bound:
            return "cpu"
        if req.prefers_gpu and req.vram_mb > 0:
            return "gpu"
        # Network-bound stuff (TTS HTTP, Ollama embeddings, OpenAI):
        # if vram_mb is zero and not explicitly cpu_bound, treat as IO.
        if req.vram_mb == 0 and not req.prefers_gpu:
            return "io"
        # GPU-preferred but with vram_mb=0 (e.g. tiny inference) -> gpu pool.
        return "gpu" if req.prefers_gpu else "io"

    def _can_admit_locked(
        self,
        pool: str,
        req: ModelRequirements,
        priority: str,
    ) -> tuple[bool, str]:
        # Concurrency cap check
        cap = {"gpu": self._gpu_cap, "cpu": self._cpu_cap, "io": self._io_cap}.get(pool, 0)
        if cap <= 0 and priority != "critical":
            return False, f"{pool}_pool_disabled"
        if cap > 0 and self._in_use.get(pool, 0) >= cap and priority != "critical":
            return False, f"{pool}_pool_full ({self._in_use[pool]}/{cap})"

        # VRAM check (only meaningful when the snapshot has total > 0)
        if req.vram_mb > 0:
            snap = self._snapshot()
            vram_total = int(getattr(snap, "vram_total_mb", 0) or 0)
            vram_used = int(getattr(snap, "vram_used_mb", 0) or 0)
            if vram_total > 0:
                projected_used = vram_used + self._vram_in_use + int(req.vram_mb)
                if (vram_total - projected_used) < self._vram_safety_mb and priority != "critical":
                    return False, (
                        f"vram_budget (need {req.vram_mb}MB + buffer {self._vram_safety_mb}MB; "
                        f"used {vram_used}+{self._vram_in_use}MB / {vram_total}MB)"
                    )

            pressure = str(getattr(snap, "pressure", "normal") or "normal")
            if pressure == "critical" and priority != "critical":
                return False, "hardware_pressure_critical"
        return True, ""

    def _snapshot(self):
        if self._snapshot_provider is None:
            return None
        try:
            return self._snapshot_provider()
        except Exception:
            return None


class _ReservationCtx:
    def __init__(self, scheduler, req, priority, timeout_ms, task_id):
        self._scheduler = scheduler
        self._req = req
        self._priority = priority
        self._timeout_ms = timeout_ms
        self._task_id = task_id
        self._reservation: Reservation | None = None

    def __enter__(self) -> Reservation | None:
        self._reservation = self._scheduler.reserve(
            self._req, self._priority, self._timeout_ms, self._task_id,
        )
        return self._reservation

    def __exit__(self, exc_type, exc, tb):
        self._scheduler.release(self._reservation)
        return False
