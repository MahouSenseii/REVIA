"""Profile IO boundary for Phase 6 cleanup."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ProfileIo:
    def __init__(self, base_dir: str | Path = "profiles") -> None:
        self.base_dir = Path(base_dir)

    def load(self, profile_id: str = "default_profile") -> dict[str, Any]:
        path = self.base_dir / f"{profile_id}.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def save(self, profile_id: str, profile: dict[str, Any]) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        path = self.base_dir / f"{profile_id}.json"
        path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
