"""Read-only assistant status view model.

Phase 6 cleanup boundary:
    This module is the replacement direction for the legacy
    assistant_status_manager.py god object. It is intentionally read-only and
    accepts Core snapshots instead of reaching into UI tabs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass
class AssistantStatusSnapshot:
    state: str = "Idle"
    mode: str = "Normal"
    health: str = "Unknown"
    version: int = 0
    details: dict[str, Any] = field(default_factory=dict)


class AssistantStatusView:
    def __init__(self) -> None:
        self._snapshot = AssistantStatusSnapshot()

    def update_from_core(self, raw: Mapping[str, Any]) -> AssistantStatusSnapshot:
        state = raw.get("current_state", raw.get("state", self._snapshot.state))
        mode = raw.get("current_mode", raw.get("mode", self._snapshot.mode))
        version = int(raw.get("version", self._snapshot.version) or 0)
        self._snapshot = AssistantStatusSnapshot(
            state=str(state),
            mode=str(mode),
            health=str(raw.get("health", self._snapshot.health)),
            version=version,
            details=dict(raw),
        )
        return self._snapshot

    def snapshot(self) -> AssistantStatusSnapshot:
        return self._snapshot
