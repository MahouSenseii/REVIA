"""Phase 6 adapter boundary for web/search tools."""
from __future__ import annotations

from typing import Any


class WebSearchAdapter:
    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        raise NotImplementedError("web search adapter implementation is not migrated yet")
