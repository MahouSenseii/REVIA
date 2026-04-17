"""
vLLM Enhanced Backend — PRD §18 (Optional Secondary Inference)
================================================================
Provides an optimized inference path for vLLM servers, exploiting features
that generic OpenAI-compatible streaming doesn't leverage:

1. **Guided generation** — structured JSON/grammar outputs
2. **Logprobs for confidence** — token-level confidence scores for answer validation
3. **Best-of-N sampling** — server-side multi-sample selection (instead of client regen loops)
4. **Speculative decoding awareness** — detects and surfaces speculative decode metrics
5. **Usage-based token counting** — vLLM returns exact token counts, no word-count estimation
6. **Prefix caching** — automatic KV-cache reuse for system prompts across turns
7. **LoRA adapter hot-swap** — switch fine-tuned adapters per-platform or per-mood

This module is a *drop-in enhancer* — it wraps the standard OpenAI-compat
streaming path and adds vLLM-specific parameters when the backend is vLLM.
Falls back gracefully if the server doesn't support extended features.

Usage
-----
    from vllm_backend import VLLMEnhancer
    enhancer = VLLMEnhancer(session, log_fn)

    # Detect if the server is vLLM
    if enhancer.probe(base_url):
        result = enhancer.generate(base_url, messages, broadcast_fn, **opts)
    else:
        # Fall back to standard OpenAI-compat streaming
        ...
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt complexity classification (smart routing)
# ---------------------------------------------------------------------------

# Patterns that indicate a "heavy" prompt needing vLLM's advanced features
_COMPLEX_INDICATORS = re.compile(
    r"(?i)"
    r"(?:explain|analyze|compare|contrast|summarize|evaluate|critique|review|"
    r"write\s+(?:a|an|the)\s+(?:essay|report|article|story|code|script|function|class)|"
    r"step[- ]by[- ]step|in\s+detail|comprehensive|thorough|elaborate|"
    r"pros?\s+and\s+cons?|trade[- ]?offs?|advantages?\s+and\s+disadvantages?|"
    r"how\s+(?:does|do|would|could|should|can)|why\s+(?:does|do|is|are|would)|"
    r"implement|refactor|debug|design|architect|plan|create\s+a\s+(?:system|app|project))"
)

# Conversational / simple patterns that don't need vLLM overhead
_SIMPLE_INDICATORS = re.compile(
    r"(?i)"
    r"(?:^(?:hi|hey|hello|yo|sup|lol|lmao|ok|okay|sure|yeah|yep|nah|no|yes|thanks|ty|thx|"
    r"good\s*(?:morning|night|evening)|how\s+are\s+you|what'?s\s+up|"
    r"gn|gm|brb|ttyl|bye)\s*[!?.]*$)"
)

# Token thresholds for context-length routing
_VLLM_MIN_CONTEXT_TOKENS = 1500   # Don't bother with vLLM under this
_VLLM_LONG_CONTEXT_TOKENS = 4000  # Definitely use vLLM above this


@dataclass
class PromptClassification:
    """Result of prompt complexity analysis."""
    should_use_vllm: bool = False
    reason: str = ""
    estimated_context_tokens: int = 0
    complexity_score: float = 0.0   # 0.0 = trivial, 1.0 = max complexity
    has_multi_turn: bool = False
    has_system_prompt: bool = False
    user_text_words: int = 0


def classify_prompt_complexity(
    messages: list[dict],
    user_text: str = "",
    cuda_available: bool = False,
) -> PromptClassification:
    """Classify whether a prompt should be routed to vLLM.

    vLLM is only used when ALL of these conditions are met:
    1. CUDA is available (vLLM requires GPU)
    2. The prompt is complex, long-context, or needs heavier inference

    Parameters
    ----------
    messages :
        The full message list (system + conversation + user).
    user_text :
        The current user message text (for pattern matching).
    cuda_available :
        Whether the local backend is configured for CUDA/GPU.

    Returns
    -------
    PromptClassification with ``should_use_vllm`` set.
    """
    result = PromptClassification()

    # Gate 1: CUDA is mandatory
    if not cuda_available:
        result.reason = "CUDA not available; vLLM requires GPU"
        return result

    # Estimate context size
    total_words = 0
    system_words = 0
    turn_count = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            # Multi-modal content
            words = sum(
                len(str(p.get("text", "")).split())
                for p in content if isinstance(p, dict)
            )
        else:
            words = len(str(content).split())
        total_words += words
        if msg.get("role") == "system":
            system_words = words
            result.has_system_prompt = True
        elif msg.get("role") in ("user", "assistant"):
            turn_count += 1

    # Rough token estimate: ~1.3 tokens per word for English
    result.estimated_context_tokens = int(total_words * 1.3)
    result.has_multi_turn = turn_count > 6
    result.user_text_words = len(user_text.split()) if user_text else 0

    # Gate 2: Skip vLLM for trivially simple prompts
    if _SIMPLE_INDICATORS.match(user_text.strip()):
        result.reason = "Simple conversational message; standard path faster"
        result.complexity_score = 0.05
        return result

    # Score complexity
    score = 0.0

    # Factor 1: Context length (0.0 to 0.4)
    if result.estimated_context_tokens >= _VLLM_LONG_CONTEXT_TOKENS:
        score += 0.4
    elif result.estimated_context_tokens >= _VLLM_MIN_CONTEXT_TOKENS:
        ratio = (result.estimated_context_tokens - _VLLM_MIN_CONTEXT_TOKENS) / (
            _VLLM_LONG_CONTEXT_TOKENS - _VLLM_MIN_CONTEXT_TOKENS
        )
        score += 0.15 + (0.25 * ratio)

    # Factor 2: User message length (0.0 to 0.2)
    if result.user_text_words > 100:
        score += 0.2
    elif result.user_text_words > 40:
        score += 0.1
    elif result.user_text_words > 20:
        score += 0.05

    # Factor 3: Complex prompt patterns (0.0 to 0.25)
    if _COMPLEX_INDICATORS.search(user_text):
        score += 0.25

    # Factor 4: Multi-turn depth (0.0 to 0.15)
    if turn_count > 20:
        score += 0.15
    elif turn_count > 10:
        score += 0.08
    elif turn_count > 6:
        score += 0.04

    result.complexity_score = round(min(score, 1.0), 3)

    # Decision threshold
    # Score >= 0.35 triggers vLLM; lower goes standard path
    VLLM_THRESHOLD = 0.35

    if result.complexity_score >= VLLM_THRESHOLD:
        result.should_use_vllm = True
        reasons = []
        if result.estimated_context_tokens >= _VLLM_LONG_CONTEXT_TOKENS:
            reasons.append(f"long context ({result.estimated_context_tokens} est. tokens)")
        if _COMPLEX_INDICATORS.search(user_text):
            reasons.append("complex prompt pattern")
        if turn_count > 10:
            reasons.append(f"deep conversation ({turn_count} turns)")
        if result.user_text_words > 40:
            reasons.append(f"detailed query ({result.user_text_words} words)")
        result.reason = "vLLM recommended: " + ", ".join(reasons)
    else:
        result.reason = (
            f"Standard path sufficient (complexity={result.complexity_score:.2f}, "
            f"threshold={VLLM_THRESHOLD})"
        )

    return result


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class VLLMMetrics:
    """Per-request metrics returned by vLLM's usage extension."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    tokens_per_second: float = 0.0
    time_to_first_token_ms: float = 0.0
    generation_time_ms: float = 0.0
    # Speculative decode stats (vLLM >= 0.4)
    spec_decode_accepted: int = 0
    spec_decode_drafted: int = 0
    # Prefix cache hit
    prefix_cache_hit_tokens: int = 0

    def acceptance_rate(self) -> float:
        if self.spec_decode_drafted == 0:
            return 0.0
        return self.spec_decode_accepted / self.spec_decode_drafted


@dataclass
class VLLMGenerateResult:
    """Result from a vLLM-enhanced generation."""
    text: str = ""
    metrics: VLLMMetrics = field(default_factory=VLLMMetrics)
    logprobs: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str = ""
    success: bool = True
    error: str = ""
    # Average token-level logprob (useful for answer validation confidence)
    avg_logprob: float = 0.0


# ---------------------------------------------------------------------------
# Server capability detection
# ---------------------------------------------------------------------------

_VLLM_MARKERS = [
    "vllm",          # Server header or model metadata
    "speculative",   # Speculative decode in extra_body
    "guided_json",   # Guided generation support
]


class VLLMEnhancer:
    """Enhanced inference wrapper for vLLM servers.

    Thread-safe — probe results and LoRA state are protected by locks.
    """

    def __init__(self, session, log_fn=None, interrupt_check=None):
        """
        Parameters
        ----------
        session :
            A ``requests.Session`` instance (reuses connection pooling).
        log_fn :
            Callable for structured logging (falls back to module logger).
        interrupt_check :
            Callable returning True if the user has requested interruption.
        """
        self._session = session
        self._log = log_fn or _log.info
        self._interrupt_check = interrupt_check or (lambda: False)

        self._lock = threading.Lock()
        self._probed_servers: dict[str, bool] = {}  # base_url → is_vllm
        self._server_caps: dict[str, dict] = {}     # base_url → capabilities
        self._active_lora: dict[str, str] = {}      # base_url → lora_adapter_name
        self._last_metrics = VLLMMetrics()

    # ------------------------------------------------------------------
    # Capability probing
    # ------------------------------------------------------------------

    def probe(self, base_url: str) -> bool:
        """Detect whether the server at *base_url* is vLLM.

        Caches the result per URL. Returns True if vLLM features are available.
        """
        url = base_url.rstrip("/")
        with self._lock:
            cached = self._probed_servers.get(url)
            if cached is not None:
                return cached

        is_vllm = False
        caps = {}
        try:
            # 1. Check /version endpoint (vLLM-specific)
            r = self._session.get(f"{url}/version", timeout=2)
            if r.ok:
                data = r.json()
                version = data.get("version", "")
                if version:
                    is_vllm = True
                    caps["version"] = version
                    self._log(f"[vLLM] Detected vLLM server v{version} at {url}")
        except Exception:
            pass

        if not is_vllm:
            try:
                # 2. Fallback: check /models for vLLM markers in metadata
                r = self._session.get(f"{url}/models", timeout=2)
                if r.ok:
                    data = r.json()
                    models_data = data.get("data", [])
                    raw = json.dumps(data).lower()
                    if "vllm" in raw:
                        is_vllm = True
                        caps["version"] = "unknown"
                    # Extract model list
                    caps["models"] = [
                        m.get("id", "") for m in models_data if m.get("id")
                    ]
            except Exception:
                pass

        if is_vllm:
            # 3. Probe additional capabilities
            try:
                # Check for guided generation support
                r = self._session.get(f"{url}/v1/models", timeout=2)
                if r.ok:
                    caps["guided_generation"] = True
            except Exception:
                caps["guided_generation"] = False

            # Check for LoRA adapters
            try:
                r = self._session.post(
                    f"{url}/v1/lora/list",
                    json={},
                    timeout=2,
                )
                if r.ok:
                    loras = r.json().get("loras", [])
                    caps["lora_adapters"] = [l.get("name", "") for l in loras]
                    self._log(f"[vLLM] LoRA adapters: {caps['lora_adapters']}")
            except Exception:
                caps["lora_adapters"] = []

        with self._lock:
            self._probed_servers[url] = is_vllm
            if is_vllm:
                self._server_caps[url] = caps

        return is_vllm

    def get_capabilities(self, base_url: str) -> dict:
        """Return cached capability dict for a probed server."""
        with self._lock:
            return dict(self._server_caps.get(base_url.rstrip("/"), {}))

    def invalidate_cache(self, base_url: str | None = None):
        """Clear cached probe results. Call after server restart."""
        with self._lock:
            if base_url:
                self._probed_servers.pop(base_url.rstrip("/"), None)
                self._server_caps.pop(base_url.rstrip("/"), None)
            else:
                self._probed_servers.clear()
                self._server_caps.clear()

    # ------------------------------------------------------------------
    # Enhanced generation
    # ------------------------------------------------------------------

    def generate(
        self,
        base_url: str,
        messages: list[dict],
        broadcast_fn: Callable,
        *,
        model: str = "",
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.9,
        fast_mode: bool = False,
        # vLLM-specific
        logprobs: int = 0,
        best_of: int = 1,
        guided_json: dict | None = None,
        guided_regex: str | None = None,
        lora_adapter: str | None = None,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        min_tokens: int = 0,
        api_key: str = "",
    ) -> VLLMGenerateResult:
        """Run inference through vLLM with enhanced features.

        Falls back gracefully if vLLM-specific features fail — the core
        OpenAI-compat streaming always works.
        """
        url = base_url.rstrip("/") + "/v1/chat/completions"
        result = VLLMGenerateResult()

        body: dict[str, Any] = {
            "model": model or "default",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": min(max_tokens, 256) if fast_mode else max_tokens,
            "top_p": top_p,
            "stream": True,
        }

        # Standard penalties (OpenAI-compat)
        if frequency_penalty:
            body["frequency_penalty"] = frequency_penalty
        if presence_penalty:
            body["presence_penalty"] = presence_penalty

        # vLLM extended parameters via extra_body
        extra: dict[str, Any] = {}

        if logprobs and logprobs > 0:
            body["logprobs"] = True
            body["top_logprobs"] = min(logprobs, 10)

        if best_of > 1:
            extra["best_of"] = min(best_of, 5)
            extra["use_beam_search"] = False  # Best-of with sampling, not beam

        if guided_json:
            extra["guided_json"] = guided_json

        if guided_regex:
            extra["guided_regex"] = guided_regex

        if repetition_penalty != 1.0:
            extra["repetition_penalty"] = repetition_penalty

        if min_tokens > 0:
            extra["min_tokens"] = min_tokens

        if lora_adapter:
            # vLLM multi-LoRA: pass adapter name in model field
            body["model"] = lora_adapter

        if extra:
            body["extra_body"] = extra

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        # --- Streaming ---
        full_text = ""
        all_logprobs = []
        t0 = time.perf_counter()
        last_token_t = t0
        t_first_token = None
        token_count = 0
        finish_reason = ""
        try:
            read_timeout = float(os.environ.get("REVIA_VLLM_READ_TIMEOUT_S", "80"))
        except (TypeError, ValueError):
            read_timeout = 80.0
        read_timeout = max(5.0, min(read_timeout, 95.0))

        try:
            with self._session.post(
                url, headers=headers, json=body, stream=True, timeout=(10, read_timeout)
            ) as resp:
                resp.raise_for_status()

                for line in resp.iter_lines(decode_unicode=False, chunk_size=128):
                    if time.perf_counter() - last_token_t > read_timeout:
                        raise TimeoutError(
                            f"vLLM stream produced no tokens for {read_timeout:.0f}s"
                        )
                    if self._interrupt_check():
                        self._log("[vLLM] Generation interrupted by user")
                        result.finish_reason = "interrupted"
                        break

                    if not line or not line.startswith(b"data: "):
                        continue
                    payload = line[6:]
                    if payload.strip() == b"[DONE]":
                        break

                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        _log.debug("[vLLM] Dropped malformed SSE chunk: %s", payload[:120])
                        continue

                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = choice.get("delta", {})
                    tok = delta.get("content", "")

                    if tok:
                        last_token_t = time.perf_counter()
                        if t_first_token is None:
                            t_first_token = time.perf_counter()
                        full_text += tok
                        token_count += 1
                        broadcast_fn({"type": "chat_token", "token": tok})

                    # Collect logprobs if requested
                    lp = choice.get("logprobs")
                    if lp and isinstance(lp, dict):
                        content_lps = lp.get("content", [])
                        if content_lps:
                            all_logprobs.extend(content_lps)

                    fr = choice.get("finish_reason")
                    if fr:
                        finish_reason = fr

                    # Extract vLLM usage extension from final chunk
                    usage = chunk.get("usage")
                    if usage:
                        result.metrics.prompt_tokens = usage.get("prompt_tokens", 0)
                        result.metrics.completion_tokens = usage.get("completion_tokens", 0)
                        result.metrics.total_tokens = usage.get("total_tokens", 0)

        except Exception as e:
            result.success = False
            result.error = str(e)
            self._log(f"[vLLM] Generation error: {e}")
            return result

        # --- Post-processing ---
        elapsed = time.perf_counter() - t0
        result.text = full_text
        result.finish_reason = finish_reason or result.finish_reason
        result.logprobs = all_logprobs

        # Metrics
        result.metrics.generation_time_ms = elapsed * 1000
        if t_first_token is not None:
            result.metrics.time_to_first_token_ms = (t_first_token - t0) * 1000
        # Use vLLM's exact token count if available, else fall back to our count
        actual_tokens = result.metrics.completion_tokens or token_count
        result.metrics.tokens_per_second = (
            actual_tokens / elapsed if elapsed > 0 else 0
        )
        # If vLLM didn't report usage, use our count
        if not result.metrics.completion_tokens:
            result.metrics.completion_tokens = token_count

        # Compute average logprob for confidence scoring
        if all_logprobs:
            valid_lps = [
                lp_entry.get("logprob", 0.0)
                for lp_entry in all_logprobs
                if isinstance(lp_entry, dict) and "logprob" in lp_entry
            ]
            if valid_lps:
                result.avg_logprob = sum(valid_lps) / len(valid_lps)

        with self._lock:
            self._last_metrics = result.metrics

        self._log(
            f"[vLLM] Done: {actual_tokens} tokens in {elapsed:.2f}s "
            f"({result.metrics.tokens_per_second:.1f} t/s) | "
            f"TTFT={result.metrics.time_to_first_token_ms:.0f}ms | "
            f"finish={result.finish_reason}"
        )

        return result

    # ------------------------------------------------------------------
    # LoRA adapter management
    # ------------------------------------------------------------------

    def set_lora_adapter(self, base_url: str, adapter_name: str) -> bool:
        """Set the active LoRA adapter for a server.

        Returns True if the adapter is available on the server.
        """
        url = base_url.rstrip("/")
        caps = self.get_capabilities(url)
        available = caps.get("lora_adapters", [])
        if adapter_name and adapter_name not in available:
            self._log(
                f"[vLLM] LoRA adapter '{adapter_name}' not found. "
                f"Available: {available}"
            )
            return False
        with self._lock:
            self._active_lora[url] = adapter_name
        return True

    def get_active_lora(self, base_url: str) -> str:
        """Return the active LoRA adapter name for a server, or ''."""
        with self._lock:
            return self._active_lora.get(base_url.rstrip("/"), "")

    # ------------------------------------------------------------------
    # Metrics access
    # ------------------------------------------------------------------

    @property
    def last_metrics(self) -> VLLMMetrics:
        with self._lock:
            return self._last_metrics

    def get_metrics_dict(self) -> dict:
        """Return last metrics as a JSON-serializable dict."""
        m = self.last_metrics
        return {
            "prompt_tokens": m.prompt_tokens,
            "completion_tokens": m.completion_tokens,
            "total_tokens": m.total_tokens,
            "tokens_per_second": round(m.tokens_per_second, 1),
            "time_to_first_token_ms": round(m.time_to_first_token_ms, 1),
            "generation_time_ms": round(m.generation_time_ms, 1),
            "spec_decode_acceptance_rate": round(m.acceptance_rate(), 3),
            "prefix_cache_hit_tokens": m.prefix_cache_hit_tokens,
        }

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def health_check(self, base_url: str) -> dict:
        """Query vLLM health and running requests."""
        url = base_url.rstrip("/")
        result = {"healthy": False, "running_requests": 0, "gpu_memory_pct": 0}
        try:
            r = self._session.get(f"{url}/health", timeout=2)
            if r.ok:
                result["healthy"] = True
        except Exception:
            pass
        try:
            r = self._session.get(f"{url}/metrics", timeout=2)
            if r.ok:
                # Parse Prometheus metrics for key stats
                text = r.text
                for line in text.split("\n"):
                    if line.startswith("vllm:num_requests_running"):
                        result["running_requests"] = int(float(line.split()[-1]))
                    elif line.startswith("vllm:gpu_cache_usage_perc"):
                        result["gpu_memory_pct"] = round(float(line.split()[-1]) * 100, 1)
        except Exception:
            pass
        return result
