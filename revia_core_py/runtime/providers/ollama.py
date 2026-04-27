"""Ollama adapter — uses Ollama's native ``/api/chat`` (not OpenAI-compat).

Native API gives us a richer health check (``/api/tags`` lists models)
and more reliable streaming semantics than Ollama's optional OpenAI shim.

Endpoint: ``http://127.0.0.1:11434`` (no /v1 suffix).
"""
from __future__ import annotations

from typing import Any, Callable

from .base import ModelRequirements, ProviderAdapter, ProviderError


class OllamaAdapter(ProviderAdapter):
    name = "Ollama"
    cost_class = "free"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        default_model: str = "llama3",
        **kw,
    ):
        # Strip /v1 suffix if user accidentally passed the OpenAI-compat URL.
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        super().__init__(base_url=base_url, default_model=default_model, **kw)

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str = "",
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: list[str] | None = None,
        broadcast_fn: Callable[..., None] | None = None,
        **kwargs: Any,
    ) -> str:
        body: dict[str, Any] = {
            "model": model or self.default_model or "llama3",
            "messages": list(messages or []),
            "stream": False,
            "options": {
                "temperature": float(temperature),
                "num_predict": int(max_tokens),
            },
        }
        if stop:
            body["options"]["stop"] = list(stop)
        for k, v in kwargs.items():
            if k in ("top_p", "top_k", "repeat_penalty", "seed", "num_ctx") and v is not None:
                body["options"][k] = v

        data = self._http_post_json("/api/chat", body)
        if not isinstance(data, dict):
            raise ProviderError("Ollama: response was not a JSON object")
        msg = data.get("message")
        if isinstance(msg, dict) and "content" in msg:
            return str(msg.get("content") or "")
        # Fallback for /api/generate-style replies.
        if "response" in data:
            return str(data.get("response") or "")
        raise ProviderError(f"Ollama: unexpected response shape: {list(data)[:6]}")

    def list_models(self) -> list[str]:
        try:
            data = self._http_get_json("/api/tags", timeout_s=2.0)
        except ProviderError:
            return [self.default_model] if self.default_model else []
        models = data.get("models") if isinstance(data, dict) else None
        if not isinstance(models, list):
            return [self.default_model] if self.default_model else []
        names: list[str] = []
        for m in models:
            if isinstance(m, dict) and "name" in m:
                names.append(str(m["name"]))
        return names or ([self.default_model] if self.default_model else [])

    def requirements(self, fingerprint=None) -> ModelRequirements:
        # Ollama runs locally — same VRAM hints as llama.cpp.
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

    def _probe(self) -> tuple[bool, str]:
        try:
            self._http_get_json("/api/tags", timeout_s=self._probe_timeout_s)
            return True, ""
        except ProviderError as e:
            return False, str(e)[:80]
