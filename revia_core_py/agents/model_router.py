"""ModelRouter — task_type to provider mapping with hardware awareness.

V2 changes vs. V1:
    * A route now declares :class:`ModelRequirements` so the
      :class:`RuntimeScheduler` can decide whether the route is
      admissible right now (VRAM budget, pool concurrency, pressure).
    * A task_type may have multiple registered routes (primary +
      fallbacks).  ``call(task_type, ...)`` walks the list in priority
      order, asks the scheduler for a :class:`Reservation`, and falls
      back automatically when the primary cannot run.
    * If no scheduler is wired (e.g. unit tests, V1 callers), the router
      degrades gracefully to first-route-wins behaviour, identical to V1.

Public surface stays backward compatible:
    * ``register(task_type, backend_name, handler)`` still works (with
      defaulted requirements).
    * ``has(task_type)``, ``get(task_type)``, ``call(task_type, *a, **kw)``
      are unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

try:  # pragma: no cover - both package and direct contexts
    from ..runtime.runtime_scheduler import (
        ModelRequirements,
        Reservation,
        RuntimeScheduler,
    )
except ImportError:  # pragma: no cover
    from runtime.runtime_scheduler import (  # type: ignore[no-redef]
        ModelRequirements,
        Reservation,
        RuntimeScheduler,
    )


@dataclass
class ModelRoute:
    """A single task_type binding (one of N candidates per task_type)."""

    task_type: str
    backend_name: str
    handler: Callable[..., Any]
    requirements: ModelRequirements = field(default_factory=ModelRequirements)
    priority_class: str = "normal"
    description: str = ""
    rank: int = 100   # smaller = preferred; ties broken by registration order

    def __call__(self, *args, **kwargs) -> Any:
        return self.handler(*args, **kwargs)


class NoRouteAdmittedError(RuntimeError):
    """Raised when every candidate route was denied by the scheduler."""


class ModelRouter:
    """Resolve a logical task_type to one of its registered backends.

    Usage::

        router = ModelRouter(scheduler=runtime_scheduler)
        router.register(
            "reason_chat", "llm_backend",
            llm_backend.generate_response,
            requirements=ModelRequirements(vram_mb=5000),
            priority_class="high",
            rank=10,
        )
        # optional fallback:
        router.register(
            "reason_chat", "openai", openai_handler,
            requirements=ModelRequirements(vram_mb=0, prefers_gpu=False, cost_class="paid"),
            priority_class="high",
            rank=50,
        )

        text = router.call("reason_chat", user_text)
    """

    def __init__(self, scheduler: RuntimeScheduler | None = None):
        self._scheduler = scheduler
        # task_type -> ordered list of routes (sorted by rank)
        self._routes: dict[str, list[ModelRoute]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        task_type: str,
        backend_name: str,
        handler: Callable[..., Any],
        *,
        requirements: ModelRequirements | None = None,
        priority_class: str = "normal",
        rank: int = 100,
        description: str = "",
        replace: bool = False,
    ) -> ModelRoute:
        if not callable(handler):
            raise TypeError(f"handler for {task_type!r} must be callable")
        route = ModelRoute(
            task_type=task_type,
            backend_name=backend_name,
            handler=handler,
            requirements=requirements or ModelRequirements(),
            priority_class=priority_class,
            description=description or task_type,
            rank=int(rank),
        )
        candidates = self._routes.setdefault(task_type, [])
        if replace:
            candidates.clear()
        candidates.append(route)
        candidates.sort(key=lambda r: (r.rank, r.backend_name))
        return route

    def set_scheduler(self, scheduler: RuntimeScheduler | None) -> None:
        self._scheduler = scheduler

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def has(self, task_type: str) -> bool:
        return bool(self._routes.get(task_type))

    def get(self, task_type: str) -> ModelRoute:
        """Return the *primary* (highest priority) route for the task.

        Kept for V1 callers that just want the handler back.  Note this
        does NOT consult the scheduler; use :meth:`call` for that.
        """
        candidates = self._routes.get(task_type) or []
        if not candidates:
            raise KeyError(
                f"No route registered for task_type {task_type!r}. "
                f"Known: {sorted(self._routes)}"
            )
        return candidates[0]

    def candidates(self, task_type: str) -> list[ModelRoute]:
        return list(self._routes.get(task_type) or [])

    def routes(self) -> dict[str, list[str]]:
        """Return ``{task_type: [backend_name, ...]}`` debug map."""
        return {k: [r.backend_name for r in v] for k, v in self._routes.items()}

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def call(self, task_type: str, *args, **kwargs) -> Any:
        """Dispatch with scheduler-aware admission and automatic fallback.

        Walks the registered routes for ``task_type`` in rank order.  For
        each candidate:

        1. If a scheduler is wired, ask for a :class:`Reservation` based on
           the route's ``ModelRequirements`` and ``priority_class``.
        2. If the reservation is denied, try the next candidate.
        3. If admitted, call the handler under the reservation; release it
           on completion.

        When no scheduler is wired, the router behaves like V1: first
        candidate wins.

        Raises :class:`NoRouteAdmittedError` only when *every* candidate
        was denied.  Provider-side exceptions still propagate normally.
        """
        candidates = self.candidates(task_type)
        if not candidates:
            raise KeyError(
                f"No route registered for task_type {task_type!r}. "
                f"Known: {sorted(self._routes)}"
            )

        # Fast path: no scheduler -> V1 behaviour.
        if self._scheduler is None:
            return candidates[0](*args, **kwargs)

        last_reason = ""
        for route in candidates:
            reservation = self._scheduler.try_reserve(
                route.requirements, priority=route.priority_class,
                task_id=f"{task_type}:{route.backend_name}",
            )
            if reservation is None:
                last_reason = self._scheduler.status().last_denied_reason
                continue
            try:
                return route(*args, **kwargs)
            finally:
                self._scheduler.release(reservation)

        raise NoRouteAdmittedError(
            f"No route admitted for {task_type!r}. "
            f"Last denial: {last_reason or 'unknown'}. "
            f"Tried: {[r.backend_name for r in candidates]}"
        )

    def select_route(
        self,
        task_type: str,
        *,
        priority: str | None = None,
    ) -> ModelRoute | None:
        """Return the first admissible route without dispatching.

        Useful for inspecting *which* backend would run before paying the
        latency of a real call.  ``None`` if every route is denied (and a
        scheduler is wired).
        """
        candidates = self.candidates(task_type)
        if not candidates:
            return None
        if self._scheduler is None:
            return candidates[0]
        for route in candidates:
            pri = priority or route.priority_class
            reservation = self._scheduler.try_reserve(
                route.requirements, priority=pri,
                task_id=f"select:{task_type}:{route.backend_name}",
            )
            if reservation is not None:
                # Release immediately; this was a dry-run.
                self._scheduler.release(reservation)
                return route
        return None
