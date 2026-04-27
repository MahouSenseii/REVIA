"""LM Studio adapter (port 1234 by default)."""
from __future__ import annotations

from .base import ModelRequirements
from .openai_compat import OpenAICompatAdapter


class LmStudioAdapter(OpenAICompatAdapter):
    name = "LM Studio"
    cost_class = "free"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:1234/v1",
        default_model: str = "",
        **kw,
    ):
        super().__init__(base_url=base_url, default_model=default_model, **kw)

    def requirements(self, fingerprint=None) -> ModelRequirements:
        vram = 0
        if fingerprint is not None:
            vram = {
                "high_24gb": 12000,
                "mid_12gb": 6000,
                "low_8gb": 4000,
                "cpu_only": 0,
            }.get(getattr(fingerprint, "suggested_profile", "cpu_only"), 4000)
        return ModelRequirements(
            vram_mb=int(vram),
            prefers_gpu=vram > 0,
            cpu_bound=vram == 0,
            supports_streaming=True,
            latency_budget_ms=20000,
            cost_class=self.cost_class,
        )
