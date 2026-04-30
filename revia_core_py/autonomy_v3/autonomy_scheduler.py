"""V3.4 AutonomyScheduler — background self-improvement loop.

The scheduler runs ONE thread that wakes on a fixed interval (default
30s) and dispatches registered :class:`AutonomyTask` callables in
priority order.  Tasks are short, idempotent, and respect the runtime
:class:`HardwareSnapshot`: when pressure is critical the loop sleeps a
full cycle to leave compute headroom for the chat path.

Default tasks (registered by ``register_default_tasks``):

    persist_memory      -> EpisodicMemoryStore.save() if dirty
    expire_goals        -> GoalTracker.expire_overdue()
    persist_goals       -> GoalTracker.save() if changed
    summarize_recent    -> compress oldest episode block (placeholder hook)

The scheduler never blocks the caller.  Start with ``start()``, stop
with ``stop()``; tasks fire from the dedicated thread.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

_log = logging.getLogger(__name__)


@dataclass
class AutonomyTask:
    name: str
    callable: Callable[[], Any]
    interval_s: float = 30.0
    priority: int = 50         # smaller = sooner
    last_run_at: float = 0.0
    last_status: str = ""
    runs: int = 0
    failures: int = 0
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "interval_s": float(self.interval_s),
            "priority": int(self.priority),
            "last_run_at": round(self.last_run_at, 3),
            "last_status": self.last_status,
            "runs": int(self.runs),
            "failures": int(self.failures),
            "enabled": bool(self.enabled),
        }


class AutonomyScheduler:
    """Single-thread background scheduler with pressure awareness."""

    DEFAULT_TICK_S = 5.0

    def __init__(
        self,
        snapshot_provider: Callable[[], Any] | None = None,
        tick_s: float = DEFAULT_TICK_S,
        log_fn: Callable[[str], None] | None = None,
    ):
        self._snapshot_provider = snapshot_provider
        self._tick_s = max(0.5, float(tick_s))
        self._log = log_fn or _log.info
        self._tasks: dict[str, AutonomyTask] = {}
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._last_pressure: str = ""

    # ------------------------------------------------------------------
    # Task registration
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        fn: Callable[[], Any],
        interval_s: float = 30.0,
        priority: int = 50,
    ) -> AutonomyTask:
        if not callable(fn):
            raise TypeError("AutonomyTask requires a callable")
        task = AutonomyTask(
            name=name, callable=fn,
            interval_s=max(1.0, float(interval_s)),
            priority=int(priority),
        )
        with self._lock:
            self._tasks[name] = task
        return task

    def unregister(self, name: str) -> bool:
        with self._lock:
            return self._tasks.pop(name, None) is not None

    def set_enabled(self, name: str, on: bool) -> bool:
        with self._lock:
            t = self._tasks.get(name)
            if t is None:
                return False
            t.enabled = bool(on)
            return True

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="revia-autonomy", daemon=True,
        )
        self._thread.start()

    def stop(self, timeout_s: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=float(timeout_s))
            self._thread = None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        with self._lock:
            tasks = [t.to_dict() for t in self._tasks.values()]
        return {
            "running": self.is_running(),
            "tick_s": self._tick_s,
            "last_pressure": self._last_pressure,
            "tasks": tasks,
        }

    def run_once(self) -> dict[str, str]:
        """Execute due tasks once (testable; no thread)."""
        return self._tick(now=time.monotonic(), force_all=False)

    def run_all(self) -> dict[str, str]:
        """Force every enabled task to run regardless of schedule (testable)."""
        return self._tick(now=time.monotonic(), force_all=True)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        self._log("[Autonomy] scheduler started")
        while not self._stop.is_set():
            try:
                self._tick(now=time.monotonic(), force_all=False)
            except Exception as exc:  # pragma: no cover - defensive
                self._log(f"[Autonomy] tick error: {exc}")
            self._stop.wait(timeout=self._tick_s)
        self._log("[Autonomy] scheduler stopped")

    def _tick(self, now: float, force_all: bool) -> dict[str, str]:
        # Pressure check: skip the whole tick when hardware is critical
        # (unless explicitly forced).
        pressure = self._pressure()
        self._last_pressure = pressure
        if pressure == "critical" and not force_all:
            return {"_skipped": "hardware_pressure_critical"}

        with self._lock:
            tasks = sorted(self._tasks.values(), key=lambda t: t.priority)

        results: dict[str, str] = {}
        for t in tasks:
            if not t.enabled:
                continue
            elapsed = now - t.last_run_at
            if not force_all and elapsed < t.interval_s:
                continue
            try:
                value = t.callable()
                t.runs += 1
                t.last_status = "ok"
                t.last_run_at = now
                results[t.name] = f"ok:{_short_repr(value)}"
            except Exception as exc:
                t.failures += 1
                t.last_status = f"err: {type(exc).__name__}: {exc}"
                t.last_run_at = now
                self._log(f"[Autonomy] task {t.name} failed: {exc}")
                results[t.name] = f"err:{type(exc).__name__}"
        return results

    def _pressure(self) -> str:
        if self._snapshot_provider is None:
            return "normal"
        try:
            snap = self._snapshot_provider()
        except Exception:
            return "normal"
        return str(getattr(snap, "pressure", "normal") or "normal")


def _short_repr(value: Any) -> str:
    if value is None:
        return ""
    s = str(value)
    return s[:40]


# ---------------------------------------------------------------------------
# Default task registration helper
# ---------------------------------------------------------------------------

def register_default_tasks(
    scheduler: AutonomyScheduler,
    *,
    episode_store=None,
    goal_tracker=None,
    summarize_fn: Callable[[], Any] | None = None,
) -> None:
    """Wire the standard self-improvement tasks onto a scheduler.

    Pass-through so callers can add their own tasks via
    :meth:`AutonomyScheduler.register` afterwards.
    """
    if episode_store is not None:
        scheduler.register(
            "persist_memory",
            fn=lambda: episode_store.save() if getattr(episode_store, "_dirty", False) else 0,
            interval_s=60.0,
            priority=20,
        )

    if goal_tracker is not None:
        scheduler.register(
            "expire_goals",
            fn=lambda: goal_tracker.expire_overdue(),
            interval_s=120.0,
            priority=40,
        )
        scheduler.register(
            "persist_goals",
            fn=lambda: goal_tracker.save(),
            interval_s=180.0,
            priority=50,
        )

    if summarize_fn is not None:
        scheduler.register(
            "summarize_recent",
            fn=summarize_fn,
            interval_s=600.0,
            priority=80,
        )
