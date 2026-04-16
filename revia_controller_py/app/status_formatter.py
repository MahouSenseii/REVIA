"""Pure formatting helpers for assistant status snapshots."""
from __future__ import annotations

from typing import Mapping


def format_state_label(snapshot: Mapping) -> str:
    state = snapshot.get("current_state", snapshot.get("state", "Idle"))
    mode = snapshot.get("current_mode", snapshot.get("mode", "Normal"))
    return f"{state} / {mode}"


def format_health_label(snapshot: Mapping) -> str:
    health = snapshot.get("health", "Unknown")
    version = snapshot.get("version", "")
    suffix = f" v{version}" if version not in ("", None) else ""
    return f"{health}{suffix}"
