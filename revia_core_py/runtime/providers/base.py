"""ProviderAdapter ABC.

A provider adapter is a thin, transport-only wrapper around an LLM
backend.  It does NOT do prompt engineering, retry, or AVS scoring —
those concerns live in the agent layer.  Its job is just:

    chat(messages, **opts) -> str

plus enough metadata for the ProviderRegistry to rank it (cost, default
priority, declared :class:`ModelRequirements`).
"""
from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

try:
    from ..runtime_scheduler import ModelRequirements
except ImportError:  # pragma: no cover
    from runtime.runtime_scheduler import ModelRequirements  # type: ignore[no-redef]


@dataclass
class ProviderInfo:
    """Static info about a provider — what the registry shows in a list."""

    name: str
    base_url: str
    cost_class: str = "free"          # "free" | "metered" | "paid"
    default_model: str = ""
    available: bool = False
    last_probed_at: float = 0.0
    last_error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "base_url": self.base_url,
            "cost_class": self.cost_class,
            "default_model": self.default_model,
            "available": bool(self.available),
            "last_probed_at": round(self.last_probed_at, 3),
            "last_error": self.last_error,
        }


class ProviderAdapter(ABC):
    """Abstract base for all LLM provider adapters.

    Subclasses must set ``name`` and implement :meth:`chat`.  The default
    :meth:`is_available` does an HTTP HEAD/GET against ``base_url``;
    most subclasses can keep it.
    """

    name: str = "ProviderAdapter"
    cost_class: str = "free"   # "free" | "metered" | "paid"
    default_priority_class: str = "high"

    def __init__(
        self,
        base_url: str,
        default_model: str = "",
        api_key: str = "",
        timeout_s: float = 30.0,
        probe_timeout_s: float = 0.6,
    ):
        self.base_url = (base_url or "").rstrip("/")
        self.default_model = default_model or ""
        self._api_key = api_key or ""
        self._timeout_s = float(timeout_s)
        self._probe_timeout_s = float(probe_timeout_s)
        self._info = ProviderInfo(
            name=self.name,
            base_url=self.base_url,
            cost_class=self.cost_class,
            default_model=self.default_model,
        )

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    @property
    def info(self) -> ProviderInfo:
        return self._info

    def is_available(self) -> bool:
        ok, err = self._probe()
        self._info.available = ok
        self._info.last_probed_at = time.time()
        self._info.last_error = err
        return ok

    @abstractmethod
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
        """Send chat messages and return the assistant text reply."""
        raise NotImplementedError

    def list_models(self) -> list[str]:  # pragma: no cover - default no-op
        """Return available model ids on this provider, if discoverable."""
        return [self.default_model] if self.default_model else []

    def requirements(self, fingerprint=None) -> ModelRequirements:
        """Default :class:`ModelRequirements` for this adapter.

        Subclasses override to reflect local-vs-cloud cost, VRAM hints, etc.
        """
        return ModelRequirements(
            vram_mb=0,
            prefers_gpu=False,
            cpu_bound=False,
            supports_streaming=True,
            latency_budget_ms=15000,
            cost_class=self.cost_class,
        )

    def make_handler(
        self,
        system_prompt_provider: Callable[[], str] | None = None,
    ) -> Callable[..., str]:
        """Return a callable suitable for :meth:`ModelRouter.register`.

        The handler accepts the legacy positional signature used by
        ``LLMBackend.generate_response`` so existing callers keep working::

            handler(user_text, broadcast_fn=None, **kw) -> str
        """

        def _handler(user_text: str, broadcast_fn=None, **kw: Any) -> str:
            sys_prompt = ""
            if system_prompt_provider is not None:
                try:
                    sys_prompt = str(system_prompt_provider() or "")
                except Exception:
                    sys_prompt = ""
            messages: list[dict[str, str]] = []
            if sys_prompt:
                messages.append({"role": "system", "content": sys_prompt})
            messages.append({"role": "user", "content": str(user_text or "")})
            return self.chat(
                messages,
                broadcast_fn=broadcast_fn,
                model=str(kw.pop("model", "") or self.default_model),
                max_tokens=int(kw.pop("max_tokens", 512)),
                temperature=float(kw.pop("temperature", 0.7)),
                **kw,
            )

        return _handler

    # ------------------------------------------------------------------
    # Internal probes / HTTP
    # ------------------------------------------------------------------

    def _probe(self) -> tuple[bool, str]:
        """Default probe: HEAD/GET on the base URL.  Subclasses can override
        with a more meaningful health check (e.g. /api/tags for Ollama)."""
        if not self.base_url:
            return False, "missing_base_url"
        return _http_alive(self.base_url, timeout_s=self._probe_timeout_s)

    def _http_post_json(
        self,
        path: str,
        body: dict[str, Any],
        timeout_s: float | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        import urllib.error
        import urllib.request

        url = self.base_url + path
        data = json.dumps(body).encode("utf-8")
        req_headers = {"Content-Type": "application/json"}
        if self._api_key:
            req_headers["Authorization"] = f"Bearer {self._api_key}"
        if headers:
            req_headers.update(headers)

        req = urllib.request.Request(url, data=data, headers=req_headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout_s or self._timeout_s) as resp:
                raw = resp.read().decode("utf-8", "ignore")
        except urllib.error.HTTPError as e:
            try:
                detail = e.read().decode("utf-8", "ignore")
            except Exception:
                detail = ""
            raise ProviderError(
                f"{self.name} HTTP {e.code}: {detail[:300]}"
            ) from None
        except urllib.error.URLError as e:
            raise ProviderError(f"{self.name} network error: {e.reason}") from None
        except Exception as e:  # pragma: no cover - defensive
            raise ProviderError(f"{self.name} unexpected error: {e}") from None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            raise ProviderError(
                f"{self.name} returned non-JSON: {raw[:200]}"
            ) from None

    def _http_get_json(
        self,
        path: str,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        import urllib.error
        import urllib.request

        url = self.base_url + path
        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        req = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout_s or self._timeout_s) as resp:
                raw = resp.read().decode("utf-8", "ignore")
        except Exception as e:
            raise ProviderError(f"{self.name} list error: {e}") from None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}


class ProviderError(RuntimeError):
    """Raised when an adapter call fails (network / 4xx / 5xx / parse)."""


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _http_alive(base_url: str, timeout_s: float = 0.6) -> tuple[bool, str]:
    """Generic reachability check.  Returns (ok, error_message)."""
    import urllib.error
    import urllib.request

    url = (base_url or "").rstrip("/")
    if not url:
        return False, "missing_base_url"
    for method in ("HEAD", "GET"):
        try:
            req = urllib.request.Request(url, method=method)
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                status = int(getattr(resp, "status", 200) or 200)
                if 200 <= status < 500:
                    return True, ""
        except urllib.error.HTTPError as e:
            # Many local servers return 404 on the bare base url but 200
            # for /v1/models; treat 4xx as alive (server is up, path is wrong).
            if 200 <= e.code < 500:
                return True, ""
            return False, f"http_{e.code}"
        except urllib.error.URLError as e:
            return False, str(e.reason)[:80]
        except Exception as e:  # pragma: no cover - defensive
            return False, f"{type(e).__name__}: {e}"
    return False, "no_response"
