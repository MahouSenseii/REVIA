"""Phase 6 adapter boundary for web/search tools.

Issue #5 (REVIA_DEEP_DIVE) -- the bare class fails at construction so any
agent that wires the stub directly raises a clear, attributable error
during startup rather than failing silently mid-turn.
"""
from __future__ import annotations

from typing import Any


_NOT_MIGRATED_MSG = (
    "WebSearchAdapter is a Phase 6 boundary stub and cannot be instantiated "
    "directly. Subclass WebSearchAdapter and override search()."
)


class WebSearchAdapter:
    """Phase 6 abstract boundary for web search.

    Subclasses MUST override __init__ and search.
    """

    def __init__(self) -> None:
        if type(self) is WebSearchAdapter:
            raise NotImplementedError(_NOT_MIGRATED_MSG)

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        raise NotImplementedError(
            "WebSearchAdapter.search is not migrated yet -- override in a subclass."
        )
