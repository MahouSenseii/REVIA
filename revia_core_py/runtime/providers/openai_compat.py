"""Shared OpenAI-compatible adapter base.

Used by every local server that exposes the standard
``POST /v1/chat/completions`` endpoint: llama.cpp, LM Studio, koboldcpp,
TabbyAPI, vLLM (when launched in OpenAI mode).
"""
from __future__ import annotations

from typing import Any, Callable

from .base import ProviderAdapter, ProviderError


class OpenAICompatAdapter(ProviderAdapter):
    """Generic adapter for OpenAI-compatible HTTP servers.

    Does NOT auto-discover models on construct — call :meth:`list_models`
    explicitly when needed (some local servers stall this endpoint).
    """

    name = "OpenAICompat"
    cost_class = "free"

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
            "model": model or self.default_model or "default",
            "messages": list(messages or []),
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "stream": False,
        }
        if stop:
            body["stop"] = list(stop)
        # Pass-through any extra OpenAI-compatible kwargs (top_p, presence_penalty, ...).
        for k in ("top_p", "presence_penalty", "frequency_penalty",
                  "repetition_penalty", "logprobs", "n"):
            if k in kwargs and kwargs[k] is not None:
                body[k] = kwargs[k]

        data = self._http_post_json("/chat/completions", body)
        return _extract_chat_text(data, provider_name=self.name)

    def list_models(self) -> list[str]:
        try:
            data = self._http_get_json("/models", timeout_s=2.0)
        except ProviderError:
            return [self.default_model] if self.default_model else []
        items = data.get("data") if isinstance(data, dict) else None
        if not isinstance(items, list):
            return [self.default_model] if self.default_model else []
        ids: list[str] = []
        for it in items:
            if isinstance(it, dict) and "id" in it:
                ids.append(str(it["id"]))
        return ids or ([self.default_model] if self.default_model else [])

    def _probe(self) -> tuple[bool, str]:
        # /v1/models is the standard cheap health check; fall back to bare URL.
        try:
            self._http_get_json("/models", timeout_s=self._probe_timeout_s)
            return True, ""
        except ProviderError as e:
            # Some servers (e.g. llama.cpp without --api) return 404 on /models
            # but are still alive; check the bare URL.
            return super()._probe()


def _extract_chat_text(data: dict[str, Any], provider_name: str) -> str:
    """Pull the assistant text from a non-streaming OpenAI chat response."""
    if not isinstance(data, dict):
        raise ProviderError(f"{provider_name}: response was not a JSON object")
    if "error" in data:
        err = data["error"]
        msg = err.get("message") if isinstance(err, dict) else str(err)
        raise ProviderError(f"{provider_name}: {msg}")
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ProviderError(f"{provider_name}: empty choices in response")
    first = choices[0]
    if not isinstance(first, dict):
        raise ProviderError(f"{provider_name}: bad first choice")
    msg = first.get("message")
    if isinstance(msg, dict) and "content" in msg:
        return str(msg.get("content") or "")
    # Some servers return only "text" (non-chat completion shape).
    if "text" in first:
        return str(first.get("text") or "")
    return ""
