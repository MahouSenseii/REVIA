"""Phase 6 adapter boundary for LLM access.

Issue #5 (REVIA_DEEP_DIVE) -- the bare class fails at construction so any
new agent or module that wires it by mistake gets a clear startup error
instead of a silently swallowed exception inside Agent.execute.
"""
from __future__ import annotations

from typing import Any, Mapping


_NOT_MIGRATED_MSG = (
    "LlmAdapter is a Phase 6 boundary stub and cannot be instantiated "
    "directly. Use the concrete LLMBackend wired through core_server, or "
    "subclass LlmAdapter and override complete()."
)


class LlmAdapter:
    """Phase 6 abstract boundary for the LLM backend.

    Subclasses MUST override __init__ and complete. The bare class
    deliberately fails at construction.
    """

    def __init__(self) -> None:
        if type(self) is LlmAdapter:
            raise NotImplementedError(_NOT_MIGRATED_MSG)

    def complete(self, messages: list[Mapping[str, Any]], **options) -> str:
        raise NotImplementedError(
            "LlmAdapter.complete is not migrated yet -- override in a subclass."
        )
