"""OpenAI cloud adapter (api.openai.com / Azure / generic OpenAI-compatible cloud).

Only registered when ``OPENAI_API_KEY`` (or the explicit ``api_key`` arg)
is set.  The base URL defaults to the public OpenAI endpoint but can
point to Azure OpenAI or any OpenAI-compatible cloud (Together, Groq,
Anyscale, ...) by overriding ``base_url`` via ``OPENAI_BASE_URL``.

Cost class is ``paid`` so the registry parks it as a fallback behind any
working local provider.
"""
from __future__ import annotations

import os

from .base import ModelRequirements
from .openai_compat import OpenAICompatAdapter


class OpenAIAdapter(OpenAICompatAdapter):
    name = "OpenAI"
    cost_class = "paid"
    default_priority_class = "high"

    def __init__(
        self,
        base_url: str | None = None,
        default_model: str = "",
        api_key: str | None = None,
        **kw,
    ):
        url = (base_url or os.environ.get("OPENAI_BASE_URL")
               or "https://api.openai.com/v1")
        key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        super().__init__(
            base_url=url,
            default_model=default_model or "gpt-4o-mini",
            api_key=key,
            **kw,
        )

    def is_available(self) -> bool:
        # Cloud is "available" when we have a key.  We avoid pre-flight
        # network probes here because those would stall startup if the
        # user's machine is offline; we let the real call fail loudly
        # if needed.
        ok = bool(self._api_key)
        self._info.available = ok
        self._info.last_error = "" if ok else "missing_api_key"
        return ok

    def requirements(self, fingerprint=None) -> ModelRequirements:
        # Cloud = no local VRAM, prefers IO pool, cost class = paid so
        # the registry / router know to use it as a fallback only.
        return ModelRequirements(
            vram_mb=0,
            prefers_gpu=False,
            cpu_bound=False,
            supports_streaming=True,
            latency_budget_ms=12000,
            cost_class=self.cost_class,
        )
