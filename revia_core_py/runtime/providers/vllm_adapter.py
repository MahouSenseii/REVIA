"""vLLM adapter (port 8000 by default).

vLLM serves the OpenAI-compatible chat completions endpoint at /v1/...
when launched with ``--api-server``.  Higher VRAM cost than llama.cpp
since vLLM keeps a large KV cache.
"""
from __future__ import annotations

from .base import ModelRequirements
from .openai_compat import OpenAICompatAdapter


class VllmAdapter(OpenAICompatAdapter):
    name = "vLLM"
    cost_class = "free"
    default_priority_class = "high"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000/v1",
        default_model: str = "",
        **kw,
    ):
        super().__init__(base_url=base_url, default_model=default_model, **kw)

    def requirements(self, fingerprint=None) -> ModelRequirements:
        vram = 0
        if fingerprint is not None:
            # vLLM is paged-attention heavy; budget more than llama.cpp.
            vram = {
                "high_24gb": 16000,
                "mid_12gb": 8000,
                "low_8gb": 5000,
                "cpu_only": 0,
            }.get(getattr(fingerprint, "suggested_profile", "cpu_only"), 5000)
        return ModelRequirements(
            vram_mb=int(vram),
            prefers_gpu=vram > 0,
            cpu_bound=vram == 0,
            supports_streaming=True,
            latency_budget_ms=15000,
            cost_class=self.cost_class,
        )
