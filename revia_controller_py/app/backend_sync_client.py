"""Controller-side backend sync client for Core config changes."""
from __future__ import annotations

from typing import Any, Mapping

try:
    from revia_core_py.ipc import get_event_publisher
except Exception:
    get_event_publisher = None


class BackendSyncClient:
    """Thin bridge for validated UI config changes into the Core event bus."""

    def __init__(self, event_bus=None):
        self.event_bus = event_bus
        self._publisher = get_event_publisher() if get_event_publisher else None

    def publish_config_change(self, payload: Mapping[str, Any]) -> bool:
        if not self._publisher:
            return False
        return self._publisher.publish_config_change(payload=dict(payload))
