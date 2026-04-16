"""Phase 6 adapter boundary for LLM access.

Legacy code still owns the concrete implementation in core_server.py. New Core
work should depend on this adapter boundary instead of importing the god module.
"""
from __future__ import annotations

from typing import Any, Mapping


class LlmAdapter:
    def complete(self, messages: list[Mapping[str, Any]], **options) -> str:
        raise NotImplementedError("LLM adapter implementation is not migrated yet")
