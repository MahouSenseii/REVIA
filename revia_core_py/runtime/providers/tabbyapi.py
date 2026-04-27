"""TabbyAPI adapter (port 5000 by default)."""
from __future__ import annotations

from .base import ModelRequirements
from .openai_compat import OpenAICompatAdapter


class TabbyApiAdapter(OpenAICompatAdapter):
    name = "TabbyAPI"
    cost_class = "free"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:5000/v1",
        default_model: str = "",
        **kw,
    ):
        super().__init__(base_url=base_url, default_model=default_model, **kw)

    def requirements(self, fingerprint=None) -> ModelRequirements:
        vram = 0
        if fingerprint is not None:
            vram = {
                "high_24gb": 14000,
                "mid_12gb": 7000,
                "low_8gb": 4500,
                "cpu_only": 0,
            }.get(getattr(fingerprint, "suggested_profile", "cpu_only"), 4500)
        return ModelRequirements(
            vram_mb=int(vram),
            prefers_gpu=vram > 0,
            cpu_bound=vram == 0,
            supports_streaming=True,
            latency_budget_ms=18000,
            cost_class=self.cost_class,
        )
