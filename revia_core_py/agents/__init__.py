"""agents/ — RETIRED (WS-2).

The entire agents/ package was wired exclusively through AgentOrchestrator
and /api/agents/chat, both of which have been removed from the active turn
path.  The production path is /api/chat → parallel_pipeline only.

Individual files are kept as inert placeholders (deletion blocked by mount
permissions).  Delete the whole agents/ directory from your IDE.

DO NOT import from this package.
"""
from __future__ import annotations

__all__: list[str] = []
