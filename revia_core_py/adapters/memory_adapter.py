"""Phase 6 adapter boundary for memory access."""
from __future__ import annotations

from typing import Any


class MemoryAdapter:
    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        raise NotImplementedError("memory adapter implementation is not migrated yet")

    def write_turn(self, user_text: str, assistant_text: str) -> None:
        raise NotImplementedError("memory adapter implementation is not migrated yet")
