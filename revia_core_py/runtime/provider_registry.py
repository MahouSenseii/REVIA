"""ProviderRegistry — discover available LLM backends and register them.

Boot flow::

    fingerprint = HardwareProfiler().detect_and_persist(...)
    registry = ProviderRegistry()
    registry.discover()                       # HEAD-probe every adapter
    registry.register_chat_routes(            # add to ModelRouter as fallbacks
        router, task_type="reason_chat",
        fingerprint=fingerprint,
    )

Ranking rules (lower rank wins inside a task type):
    * Primary handler bound by core_server keeps rank 10
      (so user's chosen ``LLMBackend`` is always tried first).
    * Local adapters are added at rank 30..40 (free, prefer first one
      already running).
    * Cloud adapters land at rank 80 (paid, last-resort fallback).

The registry is intentionally additive: it never removes the existing
LLMBackend route — it only adds *fallbacks* the router can use when the
:class:`RuntimeScheduler` denies the primary route under VRAM pressure.
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .providers.base import ProviderAdapter, ProviderInfo
from .providers import (
    KoboldCppAdapter,
    LlamaCppAdapter,
    LmStudioAdapter,
    OllamaAdapter,
    OpenAIAdapter,
    TabbyApiAdapter,
    VllmAdapter,
)

_log = logging.getLogger(__name__)


@dataclass
class ProviderEntry:
    adapter: ProviderAdapter
    rank: int
    priority_class: str = "high"
    info: ProviderInfo = field(default_factory=lambda: ProviderInfo(name="", base_url=""))

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": int(self.rank),
            "priority_class": self.priority_class,
            **self.info.to_dict(),
        }


class ProviderRegistry:
    """Discovers + ranks all known LLM provider adapters.

    The registry is *not* a router itself — it produces a sorted list of
    :class:`ProviderEntry` and registers each one as a fallback handler
    on a caller-provided :class:`ModelRouter`.
    """

    # Default base ranks.  Lower wins.
    DEFAULT_RANK_LOCAL_PRIMARY = 30
    DEFAULT_RANK_LOCAL_FALLBACK = 40
    DEFAULT_RANK_CLOUD = 80

    def __init__(
        self,
        adapters: Iterable[ProviderAdapter] | None = None,
        log_fn: Callable[[str], None] | None = None,
    ):
        self._lock = threading.Lock()
        self._adapters: list[ProviderAdapter] = list(adapters) if adapters else self._default_adapters()
        self._entries: list[ProviderEntry] = []
        self._log = log_fn or _log.info

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def discover(self) -> list[ProviderEntry]:
        """Probe every adapter and rank the survivors.  Idempotent."""
        entries: list[ProviderEntry] = []
        with self._lock:
            for adapter in self._adapters:
                try:
                    available = bool(adapter.is_available())
                except Exception as exc:
                    self._log(f"[ProviderRegistry] {adapter.name} probe error: {exc}")
                    available = False
                rank = self._rank_for(adapter, first_local=not entries)
                entries.append(ProviderEntry(
                    adapter=adapter,
                    rank=rank,
                    priority_class=getattr(adapter, "default_priority_class", "high"),
                    info=adapter.info,
                ))
            entries.sort(key=lambda e: (
                # available first, then rank, then name (stable)
                0 if e.info.available else 1,
                e.rank,
                e.info.name,
            ))
            self._entries = entries
        avail = [e.info.name for e in entries if e.info.available]
        self._log(
            f"[ProviderRegistry] discovered {len(avail)}/{len(entries)} "
            f"providers: {avail}"
        )
        return list(entries)

    def entries(self) -> list[ProviderEntry]:
        with self._lock:
            return list(self._entries)

    def available_entries(self) -> list[ProviderEntry]:
        with self._lock:
            return [e for e in self._entries if e.info.available]

    def adapter(self, name: str) -> ProviderAdapter | None:
        with self._lock:
            for e in self._entries:
                if e.adapter.name.lower() == name.lower():
                    return e.adapter
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "providers": [e.to_dict() for e in self.entries()],
            "available_count": sum(1 for e in self.entries() if e.info.available),
            "total_count": len(self.entries()),
        }

    # ------------------------------------------------------------------
    # Router registration
    # ------------------------------------------------------------------

    def register_chat_routes(
        self,
        router,
        task_type: str = "reason_chat",
        fingerprint=None,
        system_prompt_provider: Callable[[], str] | None = None,
        rank_offset: int = 0,
        only_available: bool = True,
    ) -> int:
        """Register every (available) provider as a route on ``router``.

        Returns the number of routes added.  Each provider is wrapped in
        a ``ProviderAdapter.make_handler()`` so the resulting callable
        matches ``LLMBackend.generate_response``'s ``(text, broadcast_fn,
        **kw)`` signature.

        ``rank_offset`` lets the caller leave room for a primary route
        (e.g. the user's chosen LLMBackend) at rank 10 by passing 20.
        """
        added = 0
        if not self._entries:
            self.discover()

        for entry in self._entries:
            if only_available and not entry.info.available:
                continue
            adapter = entry.adapter
            handler = adapter.make_handler(system_prompt_provider=system_prompt_provider)
            requirements = adapter.requirements(fingerprint=fingerprint)
            try:
                router.register(
                    task_type,
                    backend_name=f"provider:{adapter.name}",
                    handler=handler,
                    requirements=requirements,
                    priority_class=entry.priority_class,
                    rank=entry.rank + rank_offset,
                    description=f"{adapter.name} @ {adapter.base_url}",
                )
                added += 1
            except Exception as exc:  # pragma: no cover - defensive
                self._log(
                    f"[ProviderRegistry] could not register {adapter.name}: {exc}"
                )
        return added

    # ------------------------------------------------------------------
    # Internal: defaults + ranking
    # ------------------------------------------------------------------

    @staticmethod
    def _default_adapters() -> list[ProviderAdapter]:
        """Default adapter set — overridden by callers in tests."""
        adapters: list[ProviderAdapter] = [
            LlamaCppAdapter(),
            OllamaAdapter(),
            LmStudioAdapter(),
            VllmAdapter(),
            KoboldCppAdapter(),
            TabbyApiAdapter(),
        ]
        if os.environ.get("OPENAI_API_KEY"):
            adapters.append(OpenAIAdapter())
        return adapters

    def _rank_for(self, adapter: ProviderAdapter, first_local: bool) -> int:
        if adapter.cost_class == "paid":
            return self.DEFAULT_RANK_CLOUD
        if adapter.cost_class == "metered":
            return self.DEFAULT_RANK_CLOUD - 10
        # free / local
        return (
            self.DEFAULT_RANK_LOCAL_PRIMARY
            if first_local else self.DEFAULT_RANK_LOCAL_FALLBACK
        )
