"""InterfaceRouter — fan one canonical answer out to every active channel.

The router is the *only* layer above the AgentOrchestrator that touches
external IO (chat panel, TTS, avatar, OS notifications).  Agents and the
orchestrator stay pure: they reason and produce text + style metadata.

Design choices:
    * Channels run in parallel (thread pool) by default — voice TTS
      should not block the chat bubble from rendering.
    * One channel failing never aborts the others; the failure is
      captured in :class:`InterfaceDecision` and surfaced through
      :class:`DispatchOutput`.
    * The router is *additive* — REVIA's existing chat path keeps
      working unchanged; the router is opt-in.
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from typing import Any, Iterable

from .base import Interface, InterfaceContext, InterfaceDecision

_log = logging.getLogger(__name__)


@dataclass
class DispatchOutput:
    """Result of one router dispatch."""

    decisions: list[InterfaceDecision] = field(default_factory=list)
    elapsed_ms: float = 0.0

    def delivered_count(self) -> int:
        return sum(1 for d in self.decisions if d.delivered)

    def by_kind(self) -> dict[str, list[InterfaceDecision]]:
        out: dict[str, list[InterfaceDecision]] = {}
        for d in self.decisions:
            out.setdefault(d.kind, []).append(d)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "decisions": [d.to_dict() for d in self.decisions],
            "delivered_count": self.delivered_count(),
            "total": len(self.decisions),
            "elapsed_ms": round(float(self.elapsed_ms), 2),
        }


class InterfaceRouter:
    """Registry + parallel dispatcher for output channels.

    Usage::

        router = InterfaceRouter([
            TextChatInterface(broadcast_fn=ws_broadcast),
            VoiceInterface(speak_fn=tts.speak, enabled=False),
            LogInterface(log_fn=server_log),
        ])
        decisions = router.dispatch(InterfaceContext(final=output.final, ...))
    """

    DEFAULT_PER_INTERFACE_TIMEOUT_S: float = 5.0

    def __init__(
        self,
        interfaces: Iterable[Interface] = (),
        executor: ThreadPoolExecutor | None = None,
        max_workers: int = 4,
        parallel: bool = True,
        per_interface_timeout_s: float | None = None,
        log_fn=None,
    ):
        self._lock = threading.Lock()
        self._interfaces: dict[str, Interface] = {}
        for iface in interfaces:
            self._interfaces[iface.name] = iface
        self._owns_executor = executor is None
        self._executor = executor or ThreadPoolExecutor(
            max_workers=max(2, int(max_workers)),
            thread_name_prefix="revia-iface",
        )
        self._parallel = bool(parallel)
        self._timeout_s = float(
            per_interface_timeout_s
            if per_interface_timeout_s is not None
            else self.DEFAULT_PER_INTERFACE_TIMEOUT_S
        )
        self._log = log_fn or _log.info

    # ------------------------------------------------------------------
    # Registry
    # ------------------------------------------------------------------

    def add(self, interface: Interface) -> None:
        with self._lock:
            self._interfaces[interface.name] = interface

    def remove(self, name: str) -> Interface | None:
        with self._lock:
            return self._interfaces.pop(name, None)

    def get(self, name: str) -> Interface | None:
        with self._lock:
            return self._interfaces.get(name)

    def names(self) -> list[str]:
        with self._lock:
            return list(self._interfaces.keys())

    def set_enabled(self, name: str, on: bool) -> bool:
        """Toggle a single interface.  Returns True if the channel exists."""
        with self._lock:
            iface = self._interfaces.get(name)
            if iface is None:
                return False
            iface.set_enabled(on)
            return True

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "interfaces": [
                    {
                        "name": i.name,
                        "kind": i.kind,
                        "enabled": bool(i.enabled),
                    }
                    for i in self._interfaces.values()
                ],
                "parallel": self._parallel,
                "per_interface_timeout_s": self._timeout_s,
                "count": len(self._interfaces),
            }

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def dispatch(self, ctx: InterfaceContext) -> DispatchOutput:
        t0 = time.monotonic()
        with self._lock:
            ifaces = list(self._interfaces.values())

        if not ifaces:
            return DispatchOutput(decisions=[], elapsed_ms=0.0)

        if not self._parallel:
            decisions = [self._safe_run(i, ctx) for i in ifaces]
            return DispatchOutput(
                decisions=decisions,
                elapsed_ms=(time.monotonic() - t0) * 1000.0,
            )

        # Parallel path.
        futures: dict[str, Future] = {
            i.name: self._executor.submit(self._safe_run, i, ctx)
            for i in ifaces
        }
        decisions: list[InterfaceDecision] = []
        for name, fut in futures.items():
            try:
                decisions.append(fut.result(timeout=self._timeout_s))
            except FuturesTimeoutError:
                decisions.append(InterfaceDecision(
                    interface=name,
                    kind=self._kind_of(name),
                    delivered=False,
                    elapsed_ms=self._timeout_s * 1000.0,
                    error=f"timeout after {int(self._timeout_s * 1000)}ms",
                ))
                fut.cancel()
            except Exception as exc:
                decisions.append(InterfaceDecision(
                    interface=name,
                    kind=self._kind_of(name),
                    delivered=False,
                    elapsed_ms=0.0,
                    error=f"{type(exc).__name__}: {exc}",
                ))
        return DispatchOutput(
            decisions=decisions,
            elapsed_ms=(time.monotonic() - t0) * 1000.0,
        )

    def shutdown(self) -> None:
        if self._owns_executor:
            try:
                self._executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:  # pragma: no cover - py < 3.9
                self._executor.shutdown(wait=False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_run(iface: Interface, ctx: InterfaceContext) -> InterfaceDecision:
        try:
            return iface.execute(ctx)
        except Exception as exc:  # pragma: no cover - last-line defence
            return InterfaceDecision(
                interface=getattr(iface, "name", iface.__class__.__name__),
                kind=getattr(iface, "kind", "text"),
                delivered=False,
                elapsed_ms=0.0,
                error=f"{type(exc).__name__}: {exc}",
            )

    def _kind_of(self, name: str) -> str:
        with self._lock:
            iface = self._interfaces.get(name)
            return getattr(iface, "kind", "text") if iface else "text"
