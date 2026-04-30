"""Phase 6 adapter boundary for memory access.

Issue #5 (REVIA_DEEP_DIVE) -- the bare class fails at construction so any
consumer that wires the stub directly raises a clear, attributable error
during startup rather than failing silently mid-turn inside Agent.execute.
"""
from __future__ import annotations

from typing import Any


_NOT_MIGRATED_MSG = (
    "MemoryAdapter is a Phase 6 boundary stub and cannot be instantiated "
    "directly. Wire the concrete MemoryStore via core_server, or subclass "
    "MemoryAdapter and provide search()/write_turn() implementations."
)


class MemoryAdapter:
    """Phase 6 abstract boundary for memory.

    Subclasses MUST override __init__, search and write_turn.
    The bare class deliberately fails at construction.
    """

    def __init__(self) -> None:
        if type(self) is MemoryAdapter:
            raise NotImplementedError(_NOT_MIGRATED_MSG)

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        raise NotImplementedError(
            "MemoryAdapter.search is not migrated yet -- override in a subclass."
        )

    def write_turn(self, user_text: str, assistant_text: str) -> None:
        raise NotImplementedError(
            "MemoryAdapter.write_turn is not migrated yet -- override in a subclass."
        )
