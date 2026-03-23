"""
REVIA Core -- Pure-Python standalone server.
Drop-in replacement for the C++ core. Same REST + WebSocket API.

Usage:
    python core_server.py
    (REST on :8123, WebSocket on :8124)
"""

import json, time, random, threading, asyncio, os, subprocess, sys, math, re, hashlib
from collections import deque
from datetime import datetime
from pathlib import Path

# Make the integrations package importable when running from any CWD
sys.path.insert(0, str(Path(__file__).parent))

# Sentence boundary detection for streaming TTS (compiled once at module level)
_SENTENCE_ENDERS = re.compile(r'[.!?]+["\')\]]?\s|[\n]')

try:
    import psutil as _psutil
except ImportError:
    _psutil = None

try:
    import orjson as _orjson
except ImportError:
    _orjson = None


def _json_loads_fast(payload):
    if _orjson is not None:
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        return _orjson.loads(payload)
    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8", errors="replace")
    return json.loads(payload)


def _json_dumps_compact(payload):
    if _orjson is not None:
        return _orjson.dumps(payload).decode("utf-8")
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


try:
    import requests
    _requests_available = True
except ImportError:
    _requests_available = False
    import urllib.request as _urllib_req
    import urllib.error as _urllib_err
    # Minimal shim so the rest of the file can still reference requests.*
    class _RequestsShim:
        class exceptions:
            class ConnectionError(OSError): pass
            class HTTPError(OSError): pass
            class RequestException(OSError): pass
        class Session:
            def __init__(self): pass
            def mount(self, *a, **kw): pass
            def post(self, url, json=None, stream=False, timeout=None, headers=None):
                raise _RequestsShim.exceptions.ConnectionError(
                    f"requests not installed -- cannot POST to {url}"
                )
            def get(self, url, params=None, timeout=None):
                raise _RequestsShim.exceptions.ConnectionError(
                    f"requests not installed -- cannot GET {url}"
                )
        class adapters:
            class HTTPAdapter:
                def __init__(self, **kw): pass
    requests = _RequestsShim()

from flask import Flask, request, jsonify
from flask_cors import CORS

import websockets
import websockets.server

from conversation_runtime import (
    ConversationManager,
    LLMConnectionState,
    ReviaState,
    SubsystemStatus,
    TriggerKind,
    TriggerRequest,
    TriggerSource,
)
from profile_engine import ProfileEngine
from prompt_assembly import CharacterProfileManager, PromptAssemblyManager
from runtime_models import (
    AssistantResponse,
    RequestLifecycleState,
    ResponseMode,
    TurnManager,
)
from runtime_status import RuntimeStatusManager
from vllm_backend import VLLMEnhancer, VLLMGenerateResult, classify_prompt_complexity

# ---------------------------------------------------------------------------
# Optional Redis client for Docker-backed long-term memory
# ---------------------------------------------------------------------------

_redis_client = None
_redis_lock = threading.Lock()

def _init_redis():
    global _redis_client
    with _redis_lock:
        if _redis_client is not None:
            return  # Already initialized
        try:
            import redis as _redis_mod
            host = os.environ.get("REDIS_HOST", "127.0.0.1")
            port = int(os.environ.get("REDIS_PORT", "6379"))
            c = _redis_mod.Redis(
                host=host, port=port,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            c.ping()
            _redis_client = c
            print(f"[REVIA Core] Redis connected at {host}:{port} -- long-term memory backed by Docker.")
        except Exception as e:
            print(f"[REVIA Core] Redis unavailable ({e}). Using local .jsonl files for long-term memory.")
            _redis_client = None

def _get_redis():
    """Thread-safe accessor for the Redis client."""
    with _redis_lock:
        return _redis_client

_init_redis()

# ---------------------------------------------------------------------------
# Emotion history (in-memory ring buffer)
# ---------------------------------------------------------------------------

_emotion_history = deque(maxlen=100)  # Thread-safe bounded ring buffer
_emotion_history_lock = threading.Lock()
_EMOTION_HISTORY_MAX = 100

def _record_emotion(emo):
    """Append emotion reading to the ring buffer. Thread-safe."""
    probs = {}
    raw_probs = emo.get("emotion_probs", {})
    if isinstance(raw_probs, dict):
        for k, v in raw_probs.items():
            try:
                probs[str(k)] = round(float(v), 4)
            except (TypeError, ValueError):
                continue
    top = []
    raw_top = emo.get("top_emotions", [])
    if isinstance(raw_top, list):
        for item in raw_top[:4]:
            if not isinstance(item, dict):
                continue
            try:
                top.append({
                    "label": str(item.get("label", "")).strip(),
                    "prob": round(float(item.get("prob", 0.0)), 4),
                })
            except (TypeError, ValueError):
                continue

    entry = {
        "timestamp": datetime.now().isoformat(),
        "label":     emo.get("label", "Neutral"),
        "secondary_label": emo.get("secondary_label", ""),
        "valence":   round(emo.get("valence", 0.0), 4),
        "arousal":   round(emo.get("arousal", 0.0), 4),
        "dominance": round(emo.get("dominance", 0.0), 4),
        "confidence": round(emo.get("confidence", 0.0), 4),
        "uncertainty": round(emo.get("uncertainty", 1.0), 4),
        "emotion_probs": probs,
        "top_emotions": top,
        "signals": emo.get("signals", {}),
        "temporal": emo.get("temporal", {}),
        "model": emo.get("model", "affective_fusion_v2"),
    }
    with _emotion_history_lock:
        _emotion_history.append(entry)
        # deque(maxlen=100) auto-evicts oldest -- no manual trimming needed


_gpu_stats_cache = {"gpu_percent": 0.0, "vram_used_mb": 0.0, "vram_total_mb": 0.0}
_gpu_stats_cache_ts = 0.0
_gpu_stats_retry_after = 0.0
_gpu_stats_lock = threading.Lock()
_GPU_STATS_TTL_S = 4.0
_GPU_STATS_FAIL_BACKOFF_S = 10.0


def _get_gpu_stats():
    global _gpu_stats_cache_ts, _gpu_stats_cache, _gpu_stats_retry_after
    now = time.monotonic()
    with _gpu_stats_lock:
        if now < _gpu_stats_retry_after:
            return dict(_gpu_stats_cache)
        if (now - _gpu_stats_cache_ts) < _GPU_STATS_TTL_S:
            return dict(_gpu_stats_cache)
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            timeout=3, stderr=subprocess.DEVNULL,
        ).decode().strip().split(",")
        with _gpu_stats_lock:
            _gpu_stats_cache = {
                "gpu_percent": float(out[0]),
                "vram_used_mb": float(out[1]),
                "vram_total_mb": float(out[2]),
            }
            _gpu_stats_cache_ts = now
            _gpu_stats_retry_after = now + _GPU_STATS_TTL_S
            return dict(_gpu_stats_cache)
    except Exception:
        with _gpu_stats_lock:
            _gpu_stats_retry_after = now + _GPU_STATS_FAIL_BACKOFF_S
            return dict(_gpu_stats_cache)


def _get_system_stats():
    gpu = _get_gpu_stats()
    if _psutil:
        mem = _psutil.virtual_memory()
        cpu = _psutil.cpu_percent(interval=None)
        ram_used = round(mem.used / (1024 * 1024))
        ram_total = round(mem.total / (1024 * 1024))
    else:
        cpu = 0.0
        ram_used = 0
        ram_total = 0
    return {
        "cpu_percent": cpu,
        "gpu_percent": gpu["gpu_percent"],
        "ram_used_mb": ram_used,
        "ram_total_mb": ram_total,
        "vram_used_mb": round(gpu["vram_used_mb"]),
        "vram_total_mb": round(gpu["vram_total_mb"]),
    }


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

class TelemetryEngine:
    def __init__(self):
        self._lock = threading.Lock()
        self._epoch = time.perf_counter()
        self.state = "Idle"
        self.spans = deque(maxlen=500)
        self._flush_counter = 0
        self.llm = {"tokens_generated": 0, "tokens_per_second": 0.0, "context_length": 0}
        self.emotion = {
            "valence": 0.0, "arousal": 0.0, "dominance": 0.0,
            "label": "Neutral", "confidence": 0.0, "inference_ms": 0.0,
            "secondary_label": "Neutral", "uncertainty": 1.0,
            "emotion_probs": {}, "top_emotions": [],
            "signals": {}, "temporal": {}, "model": "affective_fusion_v2",
        }
        self.router = {
            "mode": "chat", "confidence": 0.0, "suggested_tool": "",
            "rag_enable": False, "inference_ms": 0.0,
        }
        self.system = {
            "cpu_percent": 0.0, "gpu_percent": 0.0,
            "ram_used_mb": 0, "ram_total_mb": 0,
            "vram_used_mb": 0, "vram_total_mb": 0,
            "health": "Online", "model": "None",
            "backend": "CPU", "device": "CPU",
        }
        self._refresh_system_stats()
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        fname = log_dir / f"telemetry_{datetime.now():%Y%m%d}.jsonl"
        try:
            self._log = open(fname, "a", encoding="utf-8")
        except Exception:
            self._log = None

        self._stats_thread = threading.Thread(
            target=self._stats_loop, daemon=True
        )
        self._stats_thread.start()

    def _refresh_system_stats(self):
        stats = _get_system_stats()
        with self._lock:
            self.system["cpu_percent"] = stats["cpu_percent"]
            self.system["gpu_percent"] = stats["gpu_percent"]
            self.system["ram_used_mb"] = stats["ram_used_mb"]
            self.system["ram_total_mb"] = stats["ram_total_mb"]
            self.system["vram_used_mb"] = stats["vram_used_mb"]
            self.system["vram_total_mb"] = stats["vram_total_mb"]

    def _stats_loop(self):
        if _psutil:
            _psutil.cpu_percent(interval=None)
        while True:
            time.sleep(2)
            self._refresh_system_stats()

    def close(self):
        """Flush and close the telemetry log file."""
        with self._lock:
            if self._log and not self._log.closed:
                try:
                    self._log.flush()
                    self._log.close()
                except Exception:
                    pass

    def __del__(self):
        self.close()

    def begin_span(self, stage, device="CPU"):
        return {
            "stage": stage, "device": device,
            "start_ms": (time.perf_counter() - self._epoch) * 1000,
        }

    def end_span(self, span):
        now_ms = (time.perf_counter() - self._epoch) * 1000
        span["end_ms"] = now_ms
        span["duration_ms"] = now_ms - span["start_ms"]
        with self._lock:
            self.spans.append(span)  # deque auto-evicts oldest when full
            if self._log and not self._log.closed:
                self._log.write(_json_dumps_compact(span) + "\n")
                self._flush_counter += 1
                if self._flush_counter >= 10:
                    self._log.flush()
                self._flush_counter = 0

    def snapshot(self):
        with self._lock:
            recent = list(self.spans)[-20:] if self.spans else []
            return {
                "state": self.state,
                "llm": dict(self.llm),
                "emotion": dict(self.emotion),
                "router": dict(self.router),
                "system": dict(self.system),
                "recent_spans": list(recent),
            }


telemetry = TelemetryEngine()


def _revia_log(message):
    line = f"[Revia] {message}"
    print(line)
    try:
        broadcast_json({"type": "log_entry", "text": line})
    except Exception:
        pass


conversation_manager = ConversationManager(log_fn=_revia_log)
conversation_manager.mark_booting("Core import started")
telemetry.state = conversation_manager.current_state
character_profile_manager = CharacterProfileManager(log_fn=_revia_log)
prompt_assembly_manager = PromptAssemblyManager(
    log_fn=_revia_log,
    profile_manager=character_profile_manager,
)
turn_manager = TurnManager(log_fn=_revia_log)

# ---------------------------------------------------------------------------
# PRD §4 -- Profile Engine (canonical source for all behavioral parameters)
# ---------------------------------------------------------------------------
profile_engine = ProfileEngine(log_fn=_revia_log)

# ---------------------------------------------------------------------------
# LLM Backend -- routes to local (llama.cpp server) or online API
# ---------------------------------------------------------------------------

LOCAL_SERVERS = {
    "Ollama":     {"url": "http://127.0.0.1:11434/v1", "needs_model": True},
    "LM Studio":  {"url": "http://127.0.0.1:1234/v1",  "needs_model": False},
    "llama.cpp":  {"url": "http://127.0.0.1:8080/v1",  "needs_model": False},
    "koboldcpp":  {"url": "http://127.0.0.1:5001/v1",  "needs_model": False},
    "vLLM":       {"url": "http://127.0.0.1:8000/v1",  "needs_model": True},
    "TabbyAPI":   {"url": "http://127.0.0.1:5000/v1",  "needs_model": False},
}


class LLMBackend:
    def __init__(self):
        self._lock = threading.Lock()          # protects attribute mutation
        self._generate_lock = threading.Lock() # serializes all LLM generations
        # Persistent HTTP session -- reuses TCP connections for lower latency
        self._session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=4, pool_maxsize=8, max_retries=0
        )
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)
        self.source = "none"
        self.local_path = ""
        self.local_backend = "CPU"
        self.local_loader = "llama.cpp"
        self.local_server = "llama.cpp"
        self.local_server_url = "http://127.0.0.1:8080/v1"
        self.api_provider = ""
        self.api_endpoint = ""
        self.api_key = ""
        self.api_model = ""
        self.temperature = 0.7
        self.max_tokens = 512
        self.top_p = 0.9
        self.ctx_length = 4096
        self.fast_mode = True
        self.system_prompt = "You are REVIA, a smart and friendly AI assistant."
        self.conversation = []
        self._model_name_cache = {}
        # Cache non-multimodal local backends to avoid repeatedly sending image payloads
        # that trigger server-side errors (for example missing mmproj on llama.cpp).
        self._vision_disabled_cache = {}
        self._local_health_cache = {"ok": False, "ts": 0.0}
        self._connection_state = LLMConnectionState.DISCONNECTED.value
        self._connection_detail = "No model configured."
        self._connection_error = ""
        self._connection_error_ts = 0.0
        self._last_ready_ts = 0.0
        # Per-platform conversation isolation -- Discord and Twitch get their own
        # history so their chat context never bleeds into each other or the GUI.
        self._interrupted = False
        # vLLM enhanced inference (PRD §18)
        self._vllm = VLLMEnhancer(
            session=self._session,
            log_fn=_revia_log,
            interrupt_check=self._is_interrupted,
        )
        self._vllm_enabled = True   # User can toggle via /api/model/config
        self._vllm_logprobs = 5     # Collect top-5 logprobs for confidence scoring
        self._vllm_best_of = 1      # best-of-N sampling (1 = disabled)
        self._vllm_repetition_penalty = 1.05  # Slight penalty to reduce loops
        self._vllm_min_tokens = 0    # Minimum tokens before stopping
        self._platform_conversations = {"discord": [], "twitch": []}
        # Platform-aware hints injected into the system prompt for each platform
        self._platform_hints = {
            "discord": (
                "\n\n[Platform: Discord. You are chatting in a Discord server. "
                "Be conversational, friendly, and helpful. Responses can be a "
                "few sentences. Markdown is supported.]"
            ),
            "twitch": (
                "\n\n[Platform: Twitch live chat. You are part of a live stream. "
                "Keep replies SHORT -- ideally one sentence, two at most. "
                "Be upbeat, engaging, and entertaining. No markdown.]"
            ),
        }

    def request_interrupt(self):
        """Signal the LLM to stop generating tokens immediately. Thread-safe."""
        with self._lock:
            self._interrupted = True

    def clear_interrupt(self):
        """Reset the interrupt flag before a new generation. Thread-safe."""
        with self._lock:
            self._interrupted = False

    def _is_interrupted(self):
        """Check if interrupt has been requested. Thread-safe."""
        with self._lock:
            return self._interrupted

    def configure(self, cfg):
        with self._lock:
            self.source = cfg.get("source", self.source)
            self.local_path = cfg.get("local_path", self.local_path)
            self.local_backend = cfg.get("local_backend", self.local_backend)
            self.local_loader = cfg.get("local_loader", self.local_loader)
            self.local_server = cfg.get("local_server", self.local_server)

            srv_info = LOCAL_SERVERS.get(self.local_server)
            if srv_info:
                self.local_server_url = cfg.get("local_server_url", srv_info["url"])
            else:
                self.local_server_url = cfg.get(
                    "local_server_url", self.local_server_url
                )

            self.api_provider = cfg.get("api_provider", self.api_provider)
            self.api_endpoint = cfg.get("api_endpoint", self.api_endpoint)
            self.api_key = cfg.get("api_key", self.api_key)
            self.api_model = cfg.get("api_model", self.api_model)
            self.temperature = cfg.get("temperature", self.temperature)
            self.max_tokens = cfg.get("max_tokens", self.max_tokens)
            self.top_p = cfg.get("top_p", self.top_p)
            self.ctx_length = cfg.get("ctx_length", self.ctx_length)
            self.fast_mode = bool(cfg.get("fast_mode", self.fast_mode))
            sp = cfg.get("system_prompt", "")
            if sp:
                self.system_prompt = sp
            self._local_health_cache = {"ok": False, "ts": 0.0}

            # vLLM enhanced inference options (PRD §18)
            self._vllm_enabled = bool(cfg.get("vllm_enhanced", self._vllm_enabled))
            self._vllm_logprobs = int(cfg.get("vllm_logprobs", self._vllm_logprobs))
            self._vllm_best_of = int(cfg.get("vllm_best_of", self._vllm_best_of))
            self._vllm_repetition_penalty = float(
                cfg.get("vllm_repetition_penalty", self._vllm_repetition_penalty)
            )
            self._vllm_min_tokens = int(cfg.get("vllm_min_tokens", self._vllm_min_tokens))
            # Invalidate vLLM probe cache when server URL changes
            if "local_server_url" in cfg:
                self._vllm.invalidate_cache()

            model_name = self.api_model or os.path.basename(self.local_path) or "None"
            telemetry.system["model"] = model_name
            if self.source == "online":
                telemetry.system["backend"] = self.api_provider or "API"
                telemetry.system["device"] = "Cloud"
            else:
                telemetry.system["backend"] = self.local_server
                telemetry.system["device"] = self.local_backend

            if self.source == "online":
                if self.api_endpoint and self.api_key:
                    self._set_connection_state(
                        LLMConnectionState.READY.value,
                        f"{self.api_provider or 'API'} configured",
                    )
                else:
                    self._set_connection_state(
                        LLMConnectionState.DISCONNECTED.value,
                        "Online API credentials are incomplete.",
                    )
            elif self.source == "local":
                if self.local_path or self.local_server_url:
                    self._set_connection_state(
                        LLMConnectionState.CONNECTING.value,
                        f"Waiting for {self.local_server} at {self.local_server_url or 'configured endpoint'}",
                    )
                else:
                    self._set_connection_state(
                        LLMConnectionState.DISCONNECTED.value,
                        "No local model or server configured.",
                    )
            else:
                self._set_connection_state(
                    LLMConnectionState.DISCONNECTED.value,
                    "No model configured.",
                )

    def get_config(self):
        with self._lock:
            return {
                "source": self.source, "local_path": self.local_path,
                "local_server": self.local_server,
                "local_server_url": self.local_server_url,
                "api_provider": self.api_provider, "api_model": self.api_model,
                "api_endpoint": self.api_endpoint,
                "temperature": self.temperature, "max_tokens": self.max_tokens,
                "fast_mode": self.fast_mode,
                # vLLM enhanced options (PRD §18)
                "vllm_enhanced": self._vllm_enabled,
                "vllm_logprobs": self._vllm_logprobs,
                "vllm_best_of": self._vllm_best_of,
                "vllm_repetition_penalty": self._vllm_repetition_penalty,
                "vllm_min_tokens": self._vllm_min_tokens,
            }

    def _set_connection_state(self, state, detail="", error=""):
        self._connection_state = str(state or LLMConnectionState.DISCONNECTED.value)
        if detail:
            self._connection_detail = str(detail)
        if error:
            self._connection_error = str(error)
            self._connection_error_ts = time.monotonic()
        elif self._connection_state != LLMConnectionState.ERROR.value:
            self._connection_error = ""
            self._connection_error_ts = 0.0
        if self._connection_state == LLMConnectionState.READY.value:
            self._last_ready_ts = time.monotonic()

    def _note_connection_ready(self, detail=""):
        self._set_connection_state(
            LLMConnectionState.READY.value,
            detail or self._connection_detail or "LLM ready.",
        )

    def _note_connection_error(self, detail):
        self._set_connection_state(
            LLMConnectionState.ERROR.value,
            detail or "LLM connection error.",
            error=detail,
        )

    def connection_snapshot(self):
        with self._lock:
            source = self.source
            model_name = self.api_model or os.path.basename(self.local_path) or telemetry.system.get("model", "None")
            configured_local = bool(self.local_path or self.local_server_url)
            configured_online = bool(self.api_endpoint and self.api_key)
            state = self._connection_state
            detail = self._connection_detail
            error = self._connection_error
            error_ts = self._connection_error_ts

        if source == "local":
            if not configured_local:
                state = LLMConnectionState.DISCONNECTED.value
                detail = "No local model or server configured."
            elif self._local_server_online():
                state = LLMConnectionState.READY.value
                detail = f"{self.local_server} is ready."
            elif error and (time.monotonic() - error_ts) < 20.0:
                state = LLMConnectionState.ERROR.value
                detail = error
            else:
                state = LLMConnectionState.CONNECTING.value
                detail = f"Waiting for {self.local_server} to come online."
        elif source == "online":
            if configured_online:
                if error and (time.monotonic() - error_ts) < 20.0:
                    state = LLMConnectionState.ERROR.value
                    detail = error
                else:
                    state = LLMConnectionState.READY.value
                    detail = detail or f"{self.api_provider or 'API'} configured."
            else:
                state = LLMConnectionState.DISCONNECTED.value
                detail = "Online API credentials are incomplete."
        else:
            state = LLMConnectionState.DISCONNECTED.value
            detail = "No model configured."

        return {
            "state": state,
            "source": source or "none",
            "detail": detail,
            "model": model_name or "None",
            "last_error": error,
        }

    def _local_server_online(self):
        if not _requests_available:
            return False
        base_url = (self.local_server_url or "").rstrip("/")
        if not base_url:
            return False
        now = time.monotonic()
        cached = self._local_health_cache
        if cached.get("ts") and (now - cached["ts"]) < 2.0:
            return bool(cached.get("ok", False))
        ok = False
        try:
            r = self._session.get(base_url + "/models", timeout=(0.35, 0.9))
            ok = bool(r.ok)
        except Exception:
            ok = False
        self._local_health_cache = {"ok": ok, "ts": now}
        if ok:
            self._note_connection_ready(f"{self.local_server} is ready.")
        return ok

    def _trim_conversation(self, conversation):
        """Trim conversation history while preserving maximum context.

        Neuro-Sama-class companions need 40-60+ turns of context for
        personality consistency and long-term coherence. We keep as much
        as the context window allows.
        """
        convo = list(conversation or [])
        # Estimate tokens roughly: avg ~30 tokens per message
        est_tokens = sum(len(str(m.get("content", "")).split()) for m in convo)
        ctx_budget = int(self.ctx_length * 0.65)  # Reserve 35% for system prompt + generation

        # If within budget, keep everything
        if est_tokens <= ctx_budget:
            return convo

        # Trim oldest messages until within budget, keeping at least last 60 messages
        min_keep = 80 if not self.fast_mode else 50
        while len(convo) > min_keep and est_tokens > ctx_budget:
            removed = convo.pop(0)
            est_tokens -= len(str(removed.get("content", "")).split())

        # Hard limits as safety net
        if self.fast_mode and len(convo) > 80:
            return convo[-50:]
        if len(convo) > 120:
            return convo[-80:]
        return convo

    def commit_turn_to_history(self, user_text, assistant_text):
        with self._lock:
            self.conversation.append({"role": "user", "content": user_text})
            self.conversation.append({"role": "assistant", "content": assistant_text})
            self.conversation = self._trim_conversation(self.conversation)

    def _build_messages(self, conversation, user_text, response_mode, image_b64=None):
        """Build a prompt with active character context, runtime context, memory, and recent turns."""
        with telemetry._lock:
            emo = dict(telemetry.emotion)

        label = emo.get("label", "Neutral")
        top = emo.get("top_emotions", [])

        def _safe_prob(value):
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0

        conf = _safe_prob(emo.get("confidence", 0.0) or 0.0)
        emotion_context = ""
        if label not in ("Neutral", "Disabled", "---", "") or conf >= 0.48:
            if isinstance(top, list) and top:
                top_text = ", ".join(
                    f"{str(item.get('label', '?'))}:{_safe_prob(item.get('prob', 0.0)):.0%}"
                    for item in top[:3]
                    if isinstance(item, dict)
                )
            else:
                top_text = f"{label}:{conf:.0%}"
            v = emo.get("valence", 0.0)
            emotion_context = (
                f"[Emotional context inference: top hypotheses {top_text}. "
                f"Current best read: {label} (valence {v:+.2f}, confidence {conf:.0%}).]"
            )

        mem_ctx = memory_store.get_context_for_llm(
            query=user_text,
            max_short=6 if self.fast_mode else 10,
            max_long=3 if self.fast_mode else 5,
        )
        runtime_context = runtime_status_manager.build_self_awareness_context(
            user_text=user_text,
            include_full=str(response_mode) == ResponseMode.SYSTEM_STATUS_RESPONSE.value,
        )
        # Auto-inject vision context if camera/YOLO descriptions are available
        vision_context = ""
        if hasattr(self, '_vision_context') and self._vision_context:
            vision_context = f"[Visual context: {self._vision_context}]"

        # PRD §4 -- Pull behavioral parameters from ProfileEngine so the LLM
        # gets verbosity, humor, sarcasm, emotion intensity, mood, etc.
        _pe_profile = profile_engine.current()
        behavior_params = {
            **_pe_profile.get("behavior", {}),
            **_pe_profile.get("emotion", {}),
        }
        # Inject personality data: mood baseline (quirks/freq handled by prompt_assembly)
        mood_baseline = profile.get("mood_baseline", _pe_profile.get("mood_baseline", ""))
        if mood_baseline:
            behavior_params["mood_baseline"] = mood_baseline

        system_text = prompt_assembly_manager.build_full_prompt_context(
            profile=profile,
            runtime_context=runtime_context,
            memory_context=mem_ctx,
            emotion_context=emotion_context,
            response_mode=str(response_mode or ResponseMode.NORMAL_RESPONSE.value),
            vision_context=vision_context,
            behavior_params=behavior_params,
        )
        messages = [{"role": "system", "content": system_text}]

        convo = self._trim_conversation(conversation)
        if convo:
            for msg in convo[:-1]:
                messages.append(msg)
            last = convo[-1]
            if image_b64 and last.get("role") == "user":
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": last["content"]},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            },
                        },
                    ],
                })
            else:
                messages.append(last)
        return messages

    def generate_response(
        self,
        text,
        broadcast_fn,
        *,
        image_b64=None,
        response_mode=ResponseMode.NORMAL_RESPONSE.value,
    ):
        with self._generate_lock:
            # Reset interrupt flag from any previous request
            self.clear_interrupt()
            with self._lock:
                source = self.source
                base_conversation = list(self.conversation)

            pending_conversation = self._trim_conversation(
                base_conversation + [{"role": "user", "content": text}]
            )

            if not _requests_available:
                return self._generate_stub(text)
            if source == "online" and self.api_key:
                messages = self._build_messages(
                    pending_conversation,
                    text,
                    response_mode,
                    image_b64=image_b64,
                )
                return self._generate_online(messages, broadcast_fn)
            if source == "local" and (self.local_path or self.local_server_url):
                return self._generate_local(
                    pending_conversation,
                    text,
                    broadcast_fn,
                    image_b64=image_b64,
                    response_mode=response_mode,
                )
            return self._generate_stub(text)

    def generate_streaming(self, text, broadcast_fn, image_b64=None):
        response = self.generate_response(
            text,
            broadcast_fn,
            image_b64=image_b64,
            response_mode=ResponseMode.NORMAL_RESPONSE.value,
        )
        if response.success and response.commit_to_history and response.text:
            self.commit_turn_to_history(text, response.text)
        return response.text

    def _generate_online(self, messages, broadcast_fn):
        provider = self.api_provider.lower()
        endpoint = self.api_endpoint.rstrip("/")

        try:
            if "anthropic" in provider:
                return self._call_anthropic(endpoint, messages, broadcast_fn)
            return self._call_openai_compat(endpoint, messages, broadcast_fn)
        except requests.exceptions.Timeout:
            detail = "Ugh, my brain just froze for a sec... what were we talking about?"
            self._note_connection_error(detail)
            return AssistantResponse(
                text=detail,
                response_mode=ResponseMode.ERROR_RESPONSE.value,
                success=False,
                error_type="timeout",
                retryable=True,
                speakable=False,
                commit_to_history=False,
                commit_to_memory=False,
            )
        except requests.exceptions.ConnectionError:
            detail = "My online model endpoint is configured but currently unreachable."
            self._note_connection_error(detail)
            return AssistantResponse(
                text=detail,
                response_mode=ResponseMode.ERROR_RESPONSE.value,
                success=False,
                error_type="connection_error",
                retryable=True,
                speakable=False,
                commit_to_history=False,
                commit_to_memory=False,
            )
        except Exception as e:
            detail = f"My model request failed before I could finish the answer: {e}"
            self._note_connection_error(detail)
            return AssistantResponse(
                text=detail,
                response_mode=ResponseMode.ERROR_RESPONSE.value,
                success=False,
                error_type="llm_error",
                retryable=True,
                speakable=False,
                commit_to_history=False,
                commit_to_memory=False,
            )

    def _call_openai_compat(self, endpoint, messages, broadcast_fn):
        req = self._session
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.api_model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": (
                min(int(self.max_tokens), 256)
                if self.fast_mode else int(self.max_tokens)
            ),
            "top_p": self.top_p,
            "stream": True,
        }
        url = endpoint + "/chat/completions"
        full_text = ""
        t0 = time.perf_counter()

        with req.post(url, headers=headers, json=body, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=False, chunk_size=128):
                # Check for interrupt before processing each token
                if self._is_interrupted():
                    _revia_log("LLM generation interrupted by user")
                    break
                if not line or not line.startswith(b"data: "):
                    continue
                payload = line[6:]
                if payload.strip() == b"[DONE]":
                    break
                try:
                    chunk = _json_loads_fast(payload)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    tok = delta.get("content", "")
                    if tok:
                        full_text += tok
                        broadcast_fn({"type": "chat_token", "token": tok})
                except (json.JSONDecodeError, IndexError, KeyError):
                    continue

        elapsed = time.perf_counter() - t0
        token_count = len(full_text.split())
        tps = token_count / elapsed if elapsed > 0 else 0
        telemetry.llm["tokens_generated"] = token_count
        telemetry.llm["tokens_per_second"] = round(tps, 1)
        def _count_words(content):
            if isinstance(content, list):
                return sum(len(p.get("text", "").split()) for p in content if isinstance(p, dict))
            return len(content.split())
        telemetry.llm["context_length"] = sum(_count_words(m["content"]) for m in messages) + token_count
        self._note_connection_ready(f"{self.api_provider or 'API'} request succeeded.")
        if not full_text.strip():
            return AssistantResponse(
                text="Hmm, my response came out empty. That's weird.",
                response_mode=ResponseMode.ERROR_RESPONSE.value,
                success=False,
                error_type="empty_response",
                retryable=True,
                speakable=False,
                commit_to_history=False,
                commit_to_memory=False,
            )
        return AssistantResponse(text=full_text)

    def _call_anthropic(self, endpoint, messages, broadcast_fn):
        req = self._session
        sys_msg = ""
        conv = []
        for m in messages:
            if m["role"] == "system":
                sys_msg = m["content"]
            else:
                conv.append(m)
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        body = {
            "model": self.api_model,
            "max_tokens": (
                min(int(self.max_tokens), 256)
                if self.fast_mode else int(self.max_tokens)
            ),
            "messages": conv,
            "stream": True,
        }
        if sys_msg:
            body["system"] = sys_msg
        url = endpoint + "/messages"
        full_text = ""
        t0 = time.perf_counter()

        with req.post(url, headers=headers, json=body, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=False, chunk_size=128):
                # Check for interrupt before processing each token (thread-safe)
                if self._is_interrupted():
                    _revia_log("Anthropic generation interrupted by user")
                    break
                if not line or not line.startswith(b"data: "):
                    continue
                try:
                    chunk = _json_loads_fast(line[6:])
                    if chunk.get("type") == "content_block_delta":
                        tok = chunk.get("delta", {}).get("text", "")
                        if tok:
                            full_text += tok
                            broadcast_fn({"type": "chat_token", "token": tok})
                except (json.JSONDecodeError, KeyError):
                    continue

        elapsed = time.perf_counter() - t0
        token_count = len(full_text.split())
        tps = token_count / elapsed if elapsed > 0 else 0
        telemetry.llm["tokens_generated"] = token_count
        telemetry.llm["tokens_per_second"] = round(tps, 1)
        telemetry.llm["context_length"] = sum(len(m["content"].split()) for m in messages if isinstance(m["content"], str)) + token_count
        self._note_connection_ready(f"{self.api_provider or 'API'} request succeeded.")
        if not full_text.strip():
            return AssistantResponse(
                text="Hmm, my response came out empty. That's weird.",
                response_mode=ResponseMode.ERROR_RESPONSE.value,
                success=False,
                error_type="empty_response",
                retryable=True,
                speakable=False,
                commit_to_history=False,
                commit_to_memory=False,
            )
        return AssistantResponse(text=full_text)

    def _discover_model_name(self, base_url):
        """Ask the local server for its loaded model name."""
        req = self._session
        now = time.monotonic()
        cached = self._model_name_cache.get(base_url)
        if cached and (now - cached[1]) < 20:
            return cached[0]
        try:
            r = req.get(base_url.rstrip("/") + "/models", timeout=1.5)
            if r.ok:
                data = r.json().get("data", [])
                if data:
                    model = data[0].get("id", "")
                    if model:
                        self._model_name_cache[base_url] = (model, now)
                    return model
        except Exception:
            pass
        return ""

    def _generate_local(self, conversation, text, broadcast_fn, image_b64=None, response_mode=ResponseMode.NORMAL_RESPONSE.value):
        req = self._session
        base_url = self.local_server_url.rstrip("/")
        url = base_url + "/chat/completions"
        server_name = self.local_server

        try:
            model_name = os.path.basename(self.local_path) if self.local_path else ""
            if not model_name:
                model_name = self._discover_model_name(base_url)
            vision_key = f"{base_url}|{model_name or 'default'}"
            attach_image = bool(image_b64) and not self._vision_disabled_cache.get(vision_key, False)
            messages = self._build_messages(
                conversation,
                text,
                response_mode,
                image_b64=image_b64 if attach_image else None,
            )

            if image_b64 and not attach_image:
                for m in reversed(messages):
                    if m.get("role") == "user" and isinstance(m.get("content"), str):
                        if "[Vision fallback]" not in m["content"]:
                            m["content"] += (
                                " [Vision fallback: camera metadata is available, "
                                "but image upload is disabled for this model.]"
                            )
                        break

            # ── vLLM Enhanced Path (PRD §18) ─────────────────────────────
            # vLLM is only engaged when ALL conditions are met:
            #   1. vLLM enhanced mode is enabled
            #   2. CUDA is the active backend (vLLM requires GPU)
            #   3. The server actually IS vLLM
            #   4. The prompt is complex, long-context, or needs heavier inference
            _cuda_active = self.local_backend.upper() in ("CUDA", "GPU")
            _vllm_candidate = (
                self._vllm_enabled
                and _cuda_active
                and self._vllm.probe(base_url)
            )
            if _vllm_candidate:
                classification = classify_prompt_complexity(
                    messages,
                    user_text=text,
                    cuda_available=_cuda_active,
                )
                _revia_log(
                    f"[vLLM] Routing decision: use_vllm={classification.should_use_vllm} | "
                    f"score={classification.complexity_score:.2f} | "
                    f"ctx_tokens={classification.estimated_context_tokens} | "
                    f"{classification.reason}"
                )
            else:
                classification = None

            if _vllm_candidate and classification and classification.should_use_vllm:
                vr = self._vllm.generate(
                    base_url,
                    messages,
                    broadcast_fn,
                    model=model_name or "default",
                    temperature=self.temperature,
                    max_tokens=int(self.max_tokens),
                    top_p=self.top_p,
                    fast_mode=self.fast_mode,
                    logprobs=self._vllm_logprobs,
                    best_of=self._vllm_best_of,
                    repetition_penalty=self._vllm_repetition_penalty,
                    min_tokens=self._vllm_min_tokens,
                    lora_adapter=self._vllm.get_active_lora(base_url) or None,
                )
                if vr.success and vr.text:
                    # Feed exact token counts into telemetry
                    telemetry.llm["tokens_generated"] = vr.metrics.completion_tokens
                    telemetry.llm["tokens_per_second"] = round(vr.metrics.tokens_per_second, 1)
                    telemetry.llm["context_length"] = vr.metrics.total_tokens
                    telemetry.llm["ttft_ms"] = round(vr.metrics.time_to_first_token_ms, 1)
                    if model_name:
                        telemetry.system["model"] = model_name
                    self._note_connection_ready(f"vLLM ({model_name or 'default'}) OK")
                    return AssistantResponse(text=vr.text)
                elif vr.success and not vr.text:
                    detail = f"vLLM returned empty ({vr.finish_reason})"
                    self._note_connection_error(detail)
                    return AssistantResponse(
                        text="Hmm, my local model returned an empty reply.",
                        response_mode=ResponseMode.ERROR_RESPONSE.value,
                        success=False, error_type="empty_response",
                        retryable=True, speakable=False,
                        commit_to_history=False, commit_to_memory=False,
                    )
                else:
                    # vLLM call failed -- fall through to standard path
                    _revia_log(f"[vLLM] Enhanced path failed: {vr.error}; falling back to standard")

            # ── Standard OpenAI-compat Path ──────────────────────────────
            body = {
                "model": model_name or "default",
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": (
                    min(int(self.max_tokens), 256)
                    if self.fast_mode else int(self.max_tokens)
                ),
                "top_p": self.top_p,
                "stream": True,
            }

            full_text = ""
            t0 = time.perf_counter()

            def _response_detail(resp_obj):
                if resp_obj is None:
                    return ""
                try:
                    return (resp_obj.text or "").strip()
                except Exception:
                    return ""

            def _vision_unsupported(status_code, detail_text):
                detail_l = (detail_text or "").lower()
                return (
                    status_code in (400, 415, 422, 500)
                    and any(
                        key in detail_l
                        for key in (
                            "image input is not supported",
                            "mmproj",
                            "multimodal",
                            "vision",
                            "image is not supported",
                        )
                    )
                )

            try:
                resp = req.post(url, json=body, stream=True, timeout=120)
                resp.raise_for_status()
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response is not None else None
                detail = _response_detail(e.response)
                if image_b64 and _vision_unsupported(status, detail):
                    # Model doesn't support image payloads -- retry without image and remember.
                    self._vision_disabled_cache[vision_key] = True
                    messages = self._build_messages(
                        conversation,
                        text,
                        response_mode,
                        image_b64=None,
                    )
                    for m in messages:
                        if m.get("role") == "user" and isinstance(m.get("content"), str):
                            if not m["content"].strip().startswith("["):
                                m["content"] += (
                                    " [Vision fallback: An image was attached but this model "
                                    "doesn't support image input. Respond using text context.]"
                                )
                                break
                    body["messages"] = messages
                    resp = req.post(url, json=body, stream=True, timeout=120)
                    resp.raise_for_status()
                else:
                    raise

            with resp:
                for line in resp.iter_lines(decode_unicode=False, chunk_size=128):
                    # Check for interrupt before processing each token (thread-safe)
                    if self._is_interrupted():
                        _revia_log("Local LLM generation interrupted by user")
                        break
                    if not line or not line.startswith(b"data: "):
                        continue
                    payload = line[6:]
                    if payload.strip() == b"[DONE]":
                        break
                    try:
                        chunk = _json_loads_fast(payload)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        tok = delta.get("content", "")
                        if tok:
                            full_text += tok
                            broadcast_fn({"type": "chat_token", "token": tok})
                    except (json.JSONDecodeError, IndexError, KeyError):
                        continue

            elapsed = time.perf_counter() - t0
            token_count = len(full_text.split())
            tps = token_count / elapsed if elapsed > 0 else 0
            telemetry.llm["tokens_generated"] = token_count
            telemetry.llm["tokens_per_second"] = round(tps, 1)
            if model_name:
                telemetry.system["model"] = model_name

            if not full_text:
                detail = (
                    f"My local model at {base_url.replace('http://', '')} returned an empty reply, "
                    "so I could not complete that answer."
                )
                self._note_connection_error(detail)
                return AssistantResponse(
                    text=detail,
                    response_mode=ResponseMode.ERROR_RESPONSE.value,
                    success=False,
                    error_type="empty_response",
                    retryable=True,
                    speakable=False,
                    commit_to_history=False,
                    commit_to_memory=False,
                )

            self._local_health_cache = {"ok": True, "ts": time.monotonic()}
            self._note_connection_ready(f"{server_name} request succeeded.")
            return AssistantResponse(text=full_text)
        except requests.exceptions.Timeout:
            self._local_health_cache = {"ok": False, "ts": time.monotonic()}
            detail = "My local model did not respond in time, so I could not complete that answer."
            self._note_connection_error(detail)
            return AssistantResponse(
                text=detail,
                response_mode=ResponseMode.ERROR_RESPONSE.value,
                success=False,
                error_type="timeout",
                retryable=True,
                speakable=False,
                commit_to_history=False,
                commit_to_memory=False,
            )
        except requests.exceptions.ConnectionError:
            self._local_health_cache = {"ok": False, "ts": time.monotonic()}
            short_url = base_url.replace("http://", "")
            detail = (
                f"My local LLM endpoint is configured at {short_url}, but it is currently unreachable."
            )
            self._note_connection_error(detail)
            return AssistantResponse(
                text=detail,
                response_mode=ResponseMode.ERROR_RESPONSE.value,
                success=False,
                error_type="connection_error",
                retryable=True,
                speakable=False,
                commit_to_history=False,
                commit_to_memory=False,
                metadata={"endpoint": short_url, "server": server_name},
            )
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else "?"
            detail = ""
            if e.response is not None:
                try:
                    detail = e.response.text.strip()
                except Exception:
                    detail = ""
            detail_l = detail.lower()
            if image_b64 and (
                ("image input is not supported" in detail_l)
                or ("mmproj" in detail_l)
                or ("multimodal" in detail_l)
            ):
                model_name = os.path.basename(self.local_path) if self.local_path else ""
                if not model_name:
                    model_name = self._discover_model_name(base_url)
                vision_key = f"{base_url}|{model_name or 'default'}"
                self._vision_disabled_cache[vision_key] = True
            short_url = base_url.replace("http://", "")
            vision_hint = ""
            if image_b64:
                vision_hint = (
                    "\n  Vision hint: this request included an image. "
                    "For llama.cpp, load a multimodal model and mmproj "
                    "(or disable vision for this model)."
                )
            err = (
                f"My local model returned HTTP {status} at {short_url}. "
                f"{detail or e}{vision_hint}"
            )
            self._note_connection_error(err)
            self._local_health_cache = {"ok": False, "ts": time.monotonic()}
            return AssistantResponse(
                text=err,
                response_mode=ResponseMode.ERROR_RESPONSE.value,
                success=False,
                error_type="http_error",
                retryable=True,
                speakable=False,
                commit_to_history=False,
                commit_to_memory=False,
            )
        except Exception as e:
            self._local_health_cache = {"ok": False, "ts": time.monotonic()}
            short_url = base_url.replace("http://", "")
            err = (
                f"My local model request failed for {server_name} at {short_url}: {e}"
            )
            self._note_connection_error(err)
            return AssistantResponse(
                text=err,
                response_mode=ResponseMode.ERROR_RESPONSE.value,
                success=False,
                error_type="llm_error",
                retryable=True,
                speakable=False,
                commit_to_history=False,
                commit_to_memory=False,
            )

    def _generate_stub(self, text):
        self._set_connection_state(
            LLMConnectionState.DISCONNECTED.value,
            "No model configured.",
        )
        response = (
            "I do not have a connected model right now. Go to the Model tab, connect a local "
            "or online LLM, and then I can answer normally."
        )
        return AssistantResponse(
            text=response,
            response_mode=ResponseMode.ERROR_RESPONSE.value,
            success=False,
            error_type="no_model",
            retryable=True,
            speakable=False,
            commit_to_history=False,
            commit_to_memory=False,
        )


    def generate_for_platform(self, text, broadcast_fn, platform):
        """Generate a response using an isolated per-platform conversation buffer.

        Discord and Twitch each maintain their own conversation history so that
        chat on one platform never contaminates the context seen on another.
        A short platform hint is also appended to the system prompt so Revia
        adapts her response length and style to the platform automatically.

        _generate_lock is held for the full duration so that the conversation-
        buffer swap is never interleaved with another generate_streaming or
        generate_for_platform call on a different thread.
        """
        with self._generate_lock:
            with self._lock:
                source = self.source
                p_conv = list(self._platform_conversations.get(platform, []))
                hint = self._platform_hints.get(platform, "")
            effective_text = text
            if hint and hint not in effective_text:
                effective_text = f"{text}\n\n{hint.strip()}"
            pending_conversation = self._trim_conversation(
                p_conv + [{"role": "user", "content": effective_text}]
            )

            if source == "online" and self.api_key:
                messages = self._build_messages(
                    pending_conversation,
                    effective_text,
                    ResponseMode.NORMAL_RESPONSE.value,
                )
                result = self._generate_online(messages, broadcast_fn)
            elif source == "local" and (self.local_path or self.local_server_url):
                result = self._generate_local(
                    pending_conversation,
                    effective_text,
                    broadcast_fn,
                    response_mode=ResponseMode.NORMAL_RESPONSE.value,
                )
            else:
                result = self._generate_stub(effective_text)

            if result.success and result.commit_to_history and result.text:
                p_conv = self._trim_conversation(
                    p_conv
                    + [{"role": "user", "content": effective_text}]
                    + [{"role": "assistant", "content": result.text}]
                )
                with self._lock:
                    self._platform_conversations[platform] = p_conv

        return result.text


llm_backend = LLMBackend()


def _auto_load_model_settings():
    """Auto-load model settings from model_settings.json so the core is
    immediately configured without waiting for the controller to push config."""
    settings_file = Path(__file__).resolve().parent.parent / "model_settings.json"
    if not settings_file.exists():
        return
    try:
        data = _json_loads_fast(settings_file.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[REVIA Core] Could not read model_settings.json: {e}")
        return

    source_index = int(data.get("source_index", 0))
    source = "online" if source_index == 1 else "local"

    cfg = {
        "source": source,
        "local_path": data.get("local_path", ""),
        "local_backend": data.get("local_backend", "CPU"),
        "local_loader": data.get("local_loader", "llama.cpp"),
        "local_server": data.get("local_server", "llama.cpp"),
        "local_server_url": data.get("local_server_url", ""),
        "api_provider": data.get("api_provider", ""),
        "api_endpoint": data.get("api_endpoint", ""),
        "api_key": data.get("api_key", ""),
        "api_model": data.get("api_model", ""),
        "temperature": float(data.get("temperature", 0.7)),
        "max_tokens": int(data.get("max_tokens", 512)),
        "top_p": float(data.get("top_p", 0.9)),
        "ctx_length": int(data.get("ctx_length", 4096)),
        "fast_mode": bool(data.get("fast_mode", True)),
    }

    llm_backend.configure(cfg)
    label = cfg.get("api_model") or os.path.basename(cfg.get("local_path", "")) or "?"
    print(f"[REVIA Core] Auto-loaded model settings: {source} / {label}")


_auto_load_model_settings()

# ---------------------------------------------------------------------------
# Neural modules (probabilistic inference stubs with live telemetry)
# ---------------------------------------------------------------------------

class EmotionNet:
    EMOTIONS = (
        "Happy",
        "Excited",
        "Curious",
        "Neutral",
        "Frustrated",
        "Angry",
        "Sad",
        "Fear",
        "Lonely",
        "Confident",
        "Concerned",
    )

    VAD_PROTOTYPES = {
        "Happy":      (0.72, 0.58, 0.56),
        "Excited":    (0.78, 0.82, 0.62),
        "Curious":    (0.20, 0.52, 0.44),
        "Neutral":    (0.00, 0.20, 0.46),
        "Frustrated": (-0.50, 0.74, 0.34),
        "Angry":      (-0.72, 0.84, 0.66),
        "Sad":        (-0.62, 0.30, 0.20),
        "Fear":       (-0.55, 0.72, 0.18),
        "Lonely":     (-0.48, 0.24, 0.14),
        "Confident":  (0.42, 0.48, 0.80),
        "Concerned":  (-0.20, 0.50, 0.36),
    }

    POSITIVE_WORDS = {
        "great", "awesome", "love", "thanks", "thank", "helpful", "nice",
        "good", "amazing", "perfect", "glad", "cool", "yay", "sweet",
    }
    NEGATIVE_WORDS = {
        "bad", "hate", "wrong", "terrible", "awful", "annoying", "stupid",
        "broken", "sucks", "disappointed", "lag", "slow", "delay", "offline",
    }
    GRATITUDE_WORDS = {"thanks", "thank", "appreciate", "grateful"}
    FRUSTRATION_WORDS = {
        "still", "again", "waiting", "stuck", "issue", "problem", "fix",
        "broken", "slow", "lag", "offline", "failed", "fail",
    }
    ANGER_WORDS = {"angry", "furious", "mad", "ridiculous", "unacceptable", "wtf"}
    SAD_WORDS = {"sad", "down", "depressed", "upset", "cry", "hurt", "miss"}
    ANXIETY_WORDS = {"worried", "anxious", "afraid", "nervous", "scared", "panic", "uncertain"}
    LONELY_WORDS = {"lonely", "alone", "ignored", "nobody", "isolated", "unseen"}
    CURIOUS_WORDS = {"why", "how", "what", "explain", "curious", "wonder", "question"}
    CERTAINTY_WORDS = {"definitely", "certain", "sure", "exactly", "clearly", "must", "always"}
    URGENCY_WORDS = {"now", "asap", "urgent", "quick", "hurry", "immediately", "today"}
    IMPORTANT_WORDS = {"important", "serious", "critical", "need", "must", "please"}
    SENSITIVE_TOPIC_WORDS = {
        "medical", "health", "diagnosis", "sick", "hospital", "suicide", "death",
        "dying", "legal", "lawsuit", "court", "rent", "money", "broke",
        "fired", "job", "emergency", "abuse",
    }

    LONELY_PHRASES = (
        "anyone there", "are you there", "no one answers", "nobody responds",
        "ignored me", "talk to me",
    )
    URGENCY_PHRASES = (
        "right now", "as soon as possible", "been waiting", "still waiting",
    )
    FRUSTRATION_PHRASES = (
        "not working", "doesnt work", "doesn't work", "keeps failing",
        "takes too long", "why is this", "why isnt", "why isn't",
    )

    def __init__(self):
        self.enabled = True
        self.last_inference_ms = 0.0
        self.last_output = "Neutral"
        self._token_re = re.compile(r"[a-zA-Z']+")

    @staticmethod
    def _clip(value, lo, hi):
        return max(lo, min(hi, value))

    @staticmethod
    def _safe_float(value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _to_epoch(ts):
        if not ts:
            return None
        try:
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            return dt.timestamp()
        except Exception:
            return None

    @classmethod
    def _softmax(cls, logits):
        if not logits:
            return {}
        max_logit = max(logits.values())
        exps = {k: math.exp(v - max_logit) for k, v in logits.items()}
        total = sum(exps.values()) or 1.0
        return {k: exps[k] / total for k in logits}

    def _density(self, tokens, lexicon):
        if not tokens:
            return 0.0
        matches = sum(1 for t in tokens if t in lexicon)
        return matches / float(len(tokens))

    def _extract_temporal(self, recent_messages):
        recent = list(recent_messages or [])[-20:]
        now = time.time()
        temporal = {
            "assistant_streak": 0,
            "unanswered_user_turns": 0,
            "assistant_streak_norm": 0.0,
            "unanswered_user_norm": 0.0,
            "since_last_user_s": 0.0,
            "since_last_assistant_s": 0.0,
            "response_gap_norm": 0.0,
            "user_wait_norm": 0.0,
            "avg_gap_s": 0.0,
            "user_ratio": 0.5,
        }
        if not recent:
            return temporal

        user_count = 0
        assistant_count = 0
        last_user_epoch = None
        last_assistant_epoch = None
        epochs = []

        for m in recent:
            role = str(m.get("role", "")).lower()
            if role == "user":
                user_count += 1
            elif role == "assistant":
                assistant_count += 1

            ep = self._to_epoch(m.get("timestamp"))
            if ep is not None:
                epochs.append(ep)
                if role == "user":
                    last_user_epoch = ep
                elif role == "assistant":
                    last_assistant_epoch = ep

        tail_role = str(recent[-1].get("role", "")).lower()
        tail_streak = 0
        for m in reversed(recent):
            role = str(m.get("role", "")).lower()
            if role == tail_role:
                tail_streak += 1
            else:
                break
        if tail_role == "assistant":
            temporal["assistant_streak"] = tail_streak
        elif tail_role == "user":
            temporal["unanswered_user_turns"] = tail_streak

        if last_user_epoch is not None:
            temporal["since_last_user_s"] = max(0.0, now - last_user_epoch)
        if last_assistant_epoch is not None:
            temporal["since_last_assistant_s"] = max(0.0, now - last_assistant_epoch)

        if len(epochs) >= 2:
            gaps = []
            for a, b in zip(epochs[:-1], epochs[1:]):
                if b >= a:
                    gaps.append(b - a)
            if gaps:
                temporal["avg_gap_s"] = sum(gaps) / len(gaps)

        total_turns = user_count + assistant_count
        if total_turns > 0:
            temporal["user_ratio"] = user_count / float(total_turns)

        temporal["assistant_streak_norm"] = self._clip(
            max(0.0, temporal["assistant_streak"] - 1) / 4.0, 0.0, 1.0
        )
        temporal["unanswered_user_norm"] = self._clip(
            max(0.0, temporal["unanswered_user_turns"] - 1) / 4.0, 0.0, 1.0
        )
        temporal["response_gap_norm"] = self._clip(
            temporal["since_last_assistant_s"] / 240.0, 0.0, 1.0
        )
        temporal["user_wait_norm"] = self._clip(
            temporal["since_last_user_s"] / 240.0, 0.0, 1.0
        )
        return temporal

    def _extract_features(self, text, recent_messages, profile_name, profile_state, temporal):
        low = (text or "").lower()
        tokens = self._token_re.findall(low)

        positive = self._clip(self._density(tokens, self.POSITIVE_WORDS) * 5.0, 0.0, 1.0)
        negative = self._clip(self._density(tokens, self.NEGATIVE_WORDS) * 5.0, 0.0, 1.0)
        gratitude = self._clip(self._density(tokens, self.GRATITUDE_WORDS) * 8.0, 0.0, 1.0)
        frustration_terms = self._clip(self._density(tokens, self.FRUSTRATION_WORDS) * 7.0, 0.0, 1.0)
        anger_terms = self._clip(self._density(tokens, self.ANGER_WORDS) * 8.0, 0.0, 1.0)
        sadness_terms = self._clip(self._density(tokens, self.SAD_WORDS) * 8.0, 0.0, 1.0)
        anxiety_terms = self._clip(self._density(tokens, self.ANXIETY_WORDS) * 8.0, 0.0, 1.0)
        lonely_terms = self._clip(self._density(tokens, self.LONELY_WORDS) * 8.0, 0.0, 1.0)
        curious_terms = self._clip(self._density(tokens, self.CURIOUS_WORDS) * 6.0, 0.0, 1.0)
        certainty_terms = self._clip(self._density(tokens, self.CERTAINTY_WORDS) * 8.0, 0.0, 1.0)
        urgency_terms = self._clip(self._density(tokens, self.URGENCY_WORDS) * 7.0, 0.0, 1.0)
        importance_terms = self._clip(self._density(tokens, self.IMPORTANT_WORDS) * 6.0, 0.0, 1.0)
        sensitive_terms = self._clip(self._density(tokens, self.SENSITIVE_TOPIC_WORDS) * 10.0, 0.0, 1.0)

        q_count = low.count("?")
        e_count = low.count("!")
        repeated_punct = 1.0 if ("??" in low or "!!" in low) else 0.0
        letters = [ch for ch in text if ch.isalpha()]
        caps_ratio = (
            sum(1 for ch in letters if ch.isupper()) / float(len(letters))
            if letters else 0.0
        )
        caps_intensity = self._clip((caps_ratio - 0.22) * 2.2, 0.0, 1.0)
        question_intensity = self._clip((q_count / 3.0) + (0.25 if q_count > 0 else 0.0), 0.0, 1.0)
        exclaim_intensity = self._clip((e_count / 3.0) + (0.25 * repeated_punct), 0.0, 1.0)

        lonely_phrase = 1.0 if any(p in low for p in self.LONELY_PHRASES) else 0.0
        urgency_phrase = 1.0 if any(p in low for p in self.URGENCY_PHRASES) else 0.0
        frustration_phrase = 1.0 if any(p in low for p in self.FRUSTRATION_PHRASES) else 0.0

        context_user_text = " ".join(
            str(m.get("content", "")).lower()
            for m in list(recent_messages or [])[-8:]
            if str(m.get("role", "")).lower() == "user"
        )
        ctx_tokens = self._token_re.findall(context_user_text)
        context_positive = self._clip(self._density(ctx_tokens, self.POSITIVE_WORDS) * 4.0, 0.0, 1.0)
        context_negative = self._clip(self._density(ctx_tokens, self.NEGATIVE_WORDS) * 4.0, 0.0, 1.0)

        sentiment = self._clip((positive - negative), -1.0, 1.0)
        curiosity = self._clip(0.65 * question_intensity + 0.55 * curious_terms, 0.0, 1.0)
        urgency = self._clip(0.55 * urgency_terms + 0.35 * exclaim_intensity + 0.40 * urgency_phrase, 0.0, 1.0)
        frustration = self._clip(
            0.70 * frustration_terms + 0.60 * negative + 0.45 * urgency + 0.30 * frustration_phrase,
            0.0, 1.0,
        )
        anxiety = self._clip(
            0.70 * anxiety_terms + 0.40 * sensitive_terms + 0.25 * question_intensity,
            0.0, 1.0,
        )
        loneliness = self._clip(
            0.65 * lonely_terms
            + 0.35 * lonely_phrase
            + 0.45 * temporal.get("assistant_streak_norm", 0.0)
            + 0.35 * temporal.get("response_gap_norm", 0.0),
            0.0, 1.0,
        )
        topic_sensitivity = self._clip(
            0.70 * sensitive_terms + 0.30 * importance_terms,
            0.0, 1.0,
        )
        topic_importance = self._clip(
            0.65 * importance_terms + 0.35 * urgency,
            0.0, 1.0,
        )
        certainty = self._clip(
            0.65 * certainty_terms + 0.35 * (1.0 - question_intensity),
            0.0, 1.0,
        )
        long_pause = self._clip(temporal.get("user_wait_norm", 0.0), 0.0, 1.0)

        trait_blob = (
            str(profile_name or "").lower()
            + " "
            + str((profile_state or {}).get("traits", "")).lower()
            + " "
            + str((profile_state or {}).get("persona", "")).lower()
            + " "
            + str((profile_state or {}).get("tone", "")).lower()
        )
        profile_warmth = 0.0
        profile_assertiveness = 0.0
        if any(w in trait_blob for w in ("empathetic", "warm", "playful", "friendly")):
            profile_warmth = 0.22
        if any(w in trait_blob for w in ("confident", "assertive", "direct")):
            profile_assertiveness = 0.20
        if profile_name and str(profile_name).lower() not in ("default", "revia", ""):
            profile_warmth += 0.05

        signal_strength = self._clip(
            max(
                abs(sentiment),
                curiosity,
                frustration,
                loneliness,
                anxiety,
                exclaim_intensity,
                urgency,
            ),
            0.0,
            1.0,
        )

        return {
            "sentiment": sentiment,
            "positive": positive,
            "negative": negative,
            "gratitude": gratitude,
            "curiosity": curiosity,
            "question": question_intensity,
            "exclaim": exclaim_intensity,
            "urgency": urgency,
            "frustration": frustration,
            "anger": self._clip(0.70 * anger_terms + 0.35 * caps_intensity + 0.30 * urgency, 0.0, 1.0),
            "sadness": self._clip(0.75 * sadness_terms + 0.35 * negative + 0.20 * long_pause, 0.0, 1.0),
            "anxiety": anxiety,
            "loneliness": loneliness,
            "certainty": certainty,
            "caps": caps_intensity,
            "topic_sensitivity": topic_sensitivity,
            "topic_importance": topic_importance,
            "context_positive": context_positive,
            "context_negative": context_negative,
            "profile_warmth": profile_warmth,
            "profile_assertiveness": profile_assertiveness,
            "long_pause": long_pause,
            "signal_strength": signal_strength,
        }

    def infer(self, text, recent_messages=None, prev_emotion=None, profile_name=None, profile_state=None):
        if not self.enabled:
            probs = {e: (1.0 if e == "Neutral" else 0.0) for e in self.EMOTIONS}
            return {
                "label": "Disabled",
                "secondary_label": "Neutral",
                "confidence": 0.0,
                "uncertainty": 1.0,
                "valence": 0.0,
                "arousal": 0.0,
                "dominance": 0.0,
                "emotion_probs": probs,
                "top_emotions": [{"label": "Neutral", "prob": 1.0}],
                "signals": {},
                "temporal": {},
                "model": "affective_fusion_v2",
                "inference_ms": 0.0,
            }

        t0 = time.perf_counter()
        recent = list(recent_messages or [])
        temporal = self._extract_temporal(recent)
        feat = self._extract_features(
            text=text,
            recent_messages=recent,
            profile_name=profile_name,
            profile_state=profile_state or {},
            temporal=temporal,
        )

        logits = {emo: -0.15 for emo in self.EMOTIONS}
        logits["Neutral"] = 0.30

        logits["Happy"] += (
            1.35 * feat["positive"]
            + 0.80 * feat["gratitude"]
            + 0.35 * feat["exclaim"]
            + 0.25 * feat["profile_warmth"]
            - 0.85 * feat["negative"]
        )
        logits["Excited"] += (
            1.30 * feat["exclaim"]
            + 0.65 * feat["urgency"]
            + 0.50 * feat["positive"]
            + 0.30 * feat["signal_strength"]
            - 0.30 * feat["sadness"]
        )
        logits["Curious"] += (
            1.35 * feat["curiosity"]
            + 0.65 * feat["question"]
            + 0.30 * feat["profile_warmth"]
            - 0.20 * feat["anger"]
        )
        logits["Neutral"] += (
            0.75 * (1.0 - feat["signal_strength"])
            + 0.40 * (1.0 - abs(feat["sentiment"]))
            + 0.20 * (1.0 - feat["urgency"])
        )
        logits["Frustrated"] += (
            1.35 * feat["frustration"]
            + 0.65 * feat["negative"]
            + 0.55 * feat["urgency"]
            + 0.45 * temporal.get("unanswered_user_norm", 0.0)
            - 0.25 * feat["gratitude"]
        )
        logits["Angry"] += (
            1.30 * feat["anger"]
            + 0.70 * feat["negative"]
            + 0.55 * feat["caps"]
            + 0.30 * feat["urgency"]
            - 0.35 * feat["profile_warmth"]
        )
        logits["Sad"] += (
            1.15 * feat["sadness"]
            + 0.50 * feat["negative"]
            + 0.35 * feat["long_pause"]
            + 0.20 * feat["loneliness"]
        )
        logits["Fear"] += (
            1.30 * feat["anxiety"]
            + 0.70 * feat["topic_sensitivity"]
            + 0.25 * feat["question"]
            - 0.20 * feat["certainty"]
        )
        logits["Lonely"] += (
            1.25 * feat["loneliness"]
            + 0.65 * temporal.get("assistant_streak_norm", 0.0)
            + 0.50 * temporal.get("response_gap_norm", 0.0)
            + 0.20 * feat["long_pause"]
        )
        logits["Confident"] += (
            1.20 * feat["certainty"]
            + 0.55 * feat["positive"]
            + 0.50 * feat["profile_assertiveness"]
            - 0.45 * feat["question"]
            - 0.40 * feat["anxiety"]
        )
        logits["Concerned"] += (
            1.05 * feat["topic_sensitivity"]
            + 0.85 * feat["anxiety"]
            + 0.45 * feat["topic_importance"]
            + 0.25 * feat["question"]
        )

        prev_probs = {}
        if isinstance(prev_emotion, dict):
            raw_prev = prev_emotion.get("emotion_probs", {})
            if isinstance(raw_prev, dict):
                for emo in self.EMOTIONS:
                    prev_probs[emo] = self._clip(
                        self._safe_float(raw_prev.get(emo, 0.0), 0.0), 0.0, 1.0
                    )
            if not prev_probs:
                prev_label = str(prev_emotion.get("label", ""))
                if prev_label in logits:
                    logits[prev_label] += 0.25
        if prev_probs:
            for emo in self.EMOTIONS:
                logits[emo] += 0.45 * prev_probs.get(emo, 0.0)

        history_slice = _emotion_history[-12:]
        if history_slice:
            freq = {}
            hist_val_sum = 0.0
            hist_arousal_sum = 0.0
            for entry in history_slice:
                lbl = str(entry.get("label", "Neutral"))
                freq[lbl] = freq.get(lbl, 0) + 1
                hist_val_sum += self._safe_float(entry.get("valence", 0.0))
                hist_arousal_sum += self._safe_float(entry.get("arousal", 0.0))
            n_hist = float(len(history_slice))
            hist_val = hist_val_sum / n_hist
            hist_arousal = hist_arousal_sum / n_hist
            for emo in self.EMOTIONS:
                logits[emo] += 0.18 * (freq.get(emo, 0) / n_hist)
            if hist_val < -0.18:
                logits["Frustrated"] += 0.14
                logits["Sad"] += 0.12
                logits["Concerned"] += 0.10
            if hist_arousal > 0.58:
                logits["Excited"] += 0.08
                logits["Frustrated"] += 0.08

        # Slight sharpening so top hypotheses are more interpretable in UI.
        temperature = 0.78
        logits = {emo: score / temperature for emo, score in logits.items()}
        probs = self._softmax(logits)
        if prev_probs:
            alpha = 0.78
            blended = {}
            for emo in self.EMOTIONS:
                blended[emo] = alpha * probs.get(emo, 0.0) + (1 - alpha) * prev_probs.get(emo, 0.0)
            total = sum(blended.values()) or 1.0
            probs = {emo: blended[emo] / total for emo in self.EMOTIONS}

        ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        label = ranked[0][0] if ranked else "Neutral"
        secondary = ranked[1][0] if len(ranked) > 1 else label
        top_prob = ranked[0][1] if ranked else 0.0
        second_prob = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = self._clip(top_prob - second_prob, 0.0, 1.0)

        confidence = self._clip(
            0.55 * top_prob + 0.35 * margin + 0.15 * feat["signal_strength"],
            0.05, 0.99,
        )
        if label == "Neutral" and feat["signal_strength"] > 0.55:
            confidence = self._clip(confidence - 0.10, 0.05, 0.99)

        valence = 0.0
        arousal = 0.0
        dominance = 0.0
        for emo, prob in probs.items():
            proto = self.VAD_PROTOTYPES.get(emo, (0.0, 0.2, 0.4))
            valence += prob * proto[0]
            arousal += prob * proto[1]
            dominance += prob * proto[2]

        valence = self._clip(valence + 0.12 * feat["sentiment"], -1.0, 1.0)
        arousal = self._clip(
            arousal + 0.12 * feat["exclaim"] + 0.10 * feat["urgency"] - 0.08 * feat["long_pause"],
            0.0, 1.0,
        )
        dominance = self._clip(
            dominance
            + 0.14 * feat["certainty"]
            - 0.12 * feat["anxiety"]
            - 0.10 * temporal.get("unanswered_user_norm", 0.0),
            0.0, 1.0,
        )

        top_emotions = [
            {"label": emo, "prob": round(prob, 4)}
            for emo, prob in ranked[:5]
        ]
        emotion_probs = {emo: round(prob, 4) for emo, prob in ranked}

        ms = (time.perf_counter() - t0) * 1000.0
        result = {
            "label": label,
            "secondary_label": secondary,
            "confidence": round(confidence, 4),
            "uncertainty": round(1.0 - confidence, 4),
            "valence": round(valence, 4),
            "arousal": round(arousal, 4),
            "dominance": round(dominance, 4),
            "emotion_probs": emotion_probs,
            "top_emotions": top_emotions,
            "signals": {
                "sentiment": round(feat["sentiment"], 4),
                "positive_tone": round(feat["positive"], 4),
                "negative_tone": round(feat["negative"], 4),
                "curiosity": round(feat["curiosity"], 4),
                "frustration": round(feat["frustration"], 4),
                "loneliness": round(feat["loneliness"], 4),
                "anxiety": round(feat["anxiety"], 4),
                "urgency": round(feat["urgency"], 4),
                "topic_sensitivity": round(feat["topic_sensitivity"], 4),
                "importance": round(feat["topic_importance"], 4),
                "signal_strength": round(feat["signal_strength"], 4),
            },
            "temporal": {
                "assistant_streak": int(temporal.get("assistant_streak", 0)),
                "unanswered_user_turns": int(temporal.get("unanswered_user_turns", 0)),
                "since_last_user_s": round(temporal.get("since_last_user_s", 0.0), 2),
                "since_last_assistant_s": round(temporal.get("since_last_assistant_s", 0.0), 2),
                "response_gap_norm": round(temporal.get("response_gap_norm", 0.0), 4),
                "user_wait_norm": round(temporal.get("user_wait_norm", 0.0), 4),
                "avg_gap_s": round(temporal.get("avg_gap_s", 0.0), 2),
                "user_ratio": round(temporal.get("user_ratio", 0.5), 4),
            },
            "model": "affective_fusion_v2",
            "inference_ms": round(ms, 3),
        }
        self.last_inference_ms = result["inference_ms"]
        self.last_output = result["label"]
        return result


class RouterClassifier:
    def __init__(self):
        self.enabled = True
        self.last_inference_ms = 0.0
        self.last_output = "chat"

    def classify(self, text):
        if not self.enabled:
            return {"mode": "chat", "confidence": 0.5, "suggested_tool": "",
                    "rag_enable": False, "inference_ms": 0}
        t0 = time.perf_counter()
        low = text.lower()
        # Web-search keywords take priority over local memory search
        if any(w in low for w in (
            "search online", "google", "look it up", "look up online",
            "search the web", "search the internet", "find online",
            "latest news", "current news", "what's happening", "right now",
            "real-time", "real time", "today's", "as of today",
            "what is the current", "what are the latest",
        )):
            r = {"mode": "web_search", "confidence": 0.90, "suggested_tool": "web_search", "rag_enable": False}
        elif any(w in low for w in ("search", "find", "look up")):
            r = {"mode": "memory_query", "confidence": 0.85, "suggested_tool": "rag_search", "rag_enable": True}
        elif any(w in low for w in ("run", "execute", "open")):
            r = {"mode": "command", "confidence": 0.80, "suggested_tool": "system_exec", "rag_enable": False}
        elif any(w in low for w in ("see", "camera", "look at", "image")):
            r = {"mode": "vision_query", "confidence": 0.78, "suggested_tool": "vision_capture", "rag_enable": False}
        elif any(w in low for w in ("remember", "recall")):
            r = {"mode": "memory_query", "confidence": 0.82, "suggested_tool": "memory_recall", "rag_enable": True}
        else:
            r = {"mode": "chat", "confidence": 0.92, "suggested_tool": "", "rag_enable": False}
        ms = (time.perf_counter() - t0) * 1000
        r["inference_ms"] = ms
        self.last_inference_ms = ms
        self.last_output = r["mode"]
        return r


emotion_net = EmotionNet()
router_cls = RouterClassifier()


# ---------------------------------------------------------------------------
# Web Search Engine (optional -- DuckDuckGo, free, no API key)
# ---------------------------------------------------------------------------

class WebSearchEngine:
    """Optional real-time internet search using DuckDuckGo (no API key required).

    Automatically uses the `duckduckgo_search` library if installed for full
    web results, with a lightweight fallback to DDG's Instant Answer API.
    """

    def __init__(self):
        self.enabled = False
        self._ddg_available = False
        try:
            from duckduckgo_search import DDGS  # noqa: F401
            self._ddg_available = True
            print("[REVIA Core] Web search: duckduckgo_search ready (full results).")
        except ImportError:
            print(
                "[REVIA Core] Web search: duckduckgo_search not installed -- "
                "falling back to DDG Instant API. "
                "For richer results: pip install duckduckgo-search"
            )

    @property
    def backend(self) -> str:
        return "duckduckgo_search" if self._ddg_available else "ddg_instant"

    def search(self, query: str, max_results: int = 4) -> str:
        """Return formatted web search results or empty string if disabled."""
        if not self.enabled:
            return ""
        query = query.strip()
        if not query:
            return ""
        if self._ddg_available:
            return self._search_ddg(query, max_results)
        return self._search_ddg_instant(query)

    def _search_ddg(self, query: str, max_results: int) -> str:
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            if not results:
                return f"No web results found for: {query}"
            lines = []
            for r in results:
                title = r.get("title", "")
                body = r.get("body", "")[:300].strip()
                href = r.get("href", "")
                lines.append(f"-- {title}: {body}  [{href}]")
            return "\n".join(lines)
        except Exception as exc:
            return f"[Web search error: {exc}]"

    def _search_ddg_instant(self, query: str) -> str:
        """Fallback: DuckDuckGo Instant Answers API -- no library needed."""
        import urllib.request, urllib.parse
        try:
            params = urllib.parse.urlencode({
                "q": query, "format": "json",
                "no_html": "1", "skip_disambig": "1",
            })
            url = f"https://api.duckduckgo.com/?{params}"
            with urllib.request.urlopen(url, timeout=6) as resp:
                data = _json_loads_fast(resp.read())
            parts = []
            abstract = data.get("AbstractText", "").strip()
            if abstract:
                source = data.get("AbstractSource", "")
                parts.append(f"-- {abstract}" + (f" (via {source})" if source else ""))
            for topic in data.get("RelatedTopics", [])[:3]:
                text = topic.get("Text", "").strip()
                if text and text not in "\n".join(parts):
                    parts.append(f"-- {text}")
            if parts:
                return "\n".join(parts)
            return (
                f"No instant results found for: {query}. "
                "Tip: install duckduckgo-search for full results."
            )
        except Exception as exc:
            return f"[Web search error: {exc}]"


web_search_engine = WebSearchEngine()


# ---------------------------------------------------------------------------
# Situational awareness context (Nero-style live system status)
# ---------------------------------------------------------------------------

_situational_ctx_cache: tuple[str, float] = ("", 0.0)
_SITUATIONAL_CTX_TTL = 3.0  # seconds before the cached string is refreshed


def _build_situational_context() -> str:
    """Return a concise system-status block injected into every LLM prompt.

    Result is cached for _SITUATIONAL_CTX_TTL seconds so that rapid successive
    LLM calls (streaming tokens, platform bots) don't repeatedly re-read locks
    and integration state on every generation.
    """
    global _situational_ctx_cache
    now = time.monotonic()
    cached_text, cached_ts = _situational_ctx_cache
    if cached_text and (now - cached_ts) < _SITUATIONAL_CTX_TTL:
        return cached_text
    result = _compute_situational_context()
    _situational_ctx_cache = (result, now)
    return result


def _compute_situational_context() -> str:
    lines = [
        "[Revia's Live System Status -- you are actively aware of these:]"
    ]

    # Web search
    if web_search_engine.enabled:
        lines.append(
            "-- Internet Search: ONLINE -- you can look up real-time information"
        )
    else:
        lines.append(
            "-- Internet Search: OFFLINE -- user has web access disabled"
        )

    # Neural modules
    emo_label = ""
    with telemetry._lock:
        emo_label = telemetry.emotion.get("label", "Neutral")
    if emotion_net.enabled:
        lines.append(f"-- EmotionNet: ON -- current emotional read: {emo_label}")
    else:
        lines.append("-- EmotionNet: OFF")

    if router_cls.enabled:
        lines.append("-- Intent Router: ON")
    else:
        lines.append("-- Intent Router: OFF")

    # Platform integrations
    if integration_manager is not None:
        try:
            status = integration_manager.get_status()
            d = status.get("discord", {})
            t = status.get("twitch", {})
            if d.get("running"):
                lines.append(
                    f"-- Discord: CONNECTED ({d.get('messages_processed', 0)} msgs)"
                )
            else:
                lines.append("-- Discord: OFFLINE")
            if t.get("running"):
                lines.append(
                    f"-- Twitch: CONNECTED ({t.get('messages_processed', 0)} msgs)"
                )
            else:
                lines.append("-- Twitch: OFFLINE")
        except Exception:
            pass
    else:
        lines.append("-- Discord / Twitch: unavailable")

    # Runtime system load
    with telemetry._lock:
        cpu = float(telemetry.system.get("cpu_percent", 0.0) or 0.0)
        gpu = float(telemetry.system.get("gpu_percent", 0.0) or 0.0)
        ram_used = int(telemetry.system.get("ram_used_mb", 0) or 0)
        ram_total = int(telemetry.system.get("ram_total_mb", 0) or 0)
        state = str(telemetry.state or "Idle")
    lines.append(
        f"- Runtime: state={state} | CPU={cpu:.0f}% | GPU={gpu:.0f}% | RAM={ram_used}/{ram_total} MB"
    )

    # Memory
    st = len(memory_store.short_term)
    lt = len(memory_store.long_term)
    lines.append(f"-- Memory: {st} recent exchanges | {lt} long-term facts stored")

    lines.append(
        "Embody this awareness naturally -- proactively offer to search the web "
        "when internet is ON and the user might benefit, acknowledge what's "
        "offline without dwelling on it, and reference memories when relevant. "
        "Never pretend a disabled module is working. If system load, active tools, "
        "or module state is relevant to the conversation, mention it briefly."
    )
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Memory system -- short-term (conversation) + long-term (persistent)
# ---------------------------------------------------------------------------

class MemoryStore:
    def __init__(self, profile_name="default"):
        self._lock = threading.Lock()
        self.short_term = []
        self.long_term = []
        self._profile_name = profile_name
        self._redis_available_cache = _get_redis() is not None
        self._redis_checked_ts = 0.0
        self._redis_status_ttl_s = 2.5
        self._lt_file = Path(f"data/memory_{profile_name}.jsonl")
        self._lt_file.parent.mkdir(parents=True, exist_ok=True)
        self._lt_handle = open(self._lt_file, "a", encoding="utf-8")
        self._load_long_term()

    def _entry_id(self, entry):
        payload = {
            "timestamp": str(entry.get("timestamp", "")),
            "category": str(entry.get("category", entry.get("role", ""))),
            "content": str(entry.get("content", "")),
            "metadata": entry.get("metadata", {}),
            "promoted": bool(entry.get("promoted", False)),
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    def _normalize_entry(self, entry):
        if not isinstance(entry, dict):
            return None
        out = dict(entry)
        if not out.get("id"):
            out["id"] = self._entry_id(out)
        return out

    # ------------------------------------------------------------------
    # Redis helpers
    # ------------------------------------------------------------------

    @property
    def redis_available(self):
        global _redis_client
        now = time.monotonic()
        if (now - self._redis_checked_ts) < self._redis_status_ttl_s:
            return self._redis_available_cache
        # If not connected yet, try to connect now (handles Docker starting after server)
        client = _get_redis()
        if client is None:
            _init_redis()
            client = _get_redis()
        if client is None:
            self._redis_available_cache = False
            self._redis_checked_ts = now
            return False
        try:
            client.ping()
            self._redis_available_cache = True
            self._redis_checked_ts = now
            return True
        except Exception:
            # Connection dropped -- reset so next check retries
            with _redis_lock:
                _redis_client = None
            self._redis_available_cache = False
            self._redis_checked_ts = now
            return False

    def _redis_key(self):
        return f"revia:lt:{self._profile_name}"

    def _redis_push(self, entry):
        """Push a single long-term entry to Redis (best-effort)."""
        client = _get_redis()
        if client is None:
            return
        try:
            client.rpush(self._redis_key(), _json_dumps_compact(entry))
            self._redis_available_cache = True
            self._redis_checked_ts = time.monotonic()
        except Exception:
            self._redis_available_cache = False
            self._redis_checked_ts = time.monotonic()

    def _redis_clear(self):
        """Delete the Redis list for the current profile."""
        client = _get_redis()
        if client is None:
            return
        try:
            client.delete(self._redis_key())
            self._redis_available_cache = True
            self._redis_checked_ts = time.monotonic()
        except Exception:
            self._redis_available_cache = False
            self._redis_checked_ts = time.monotonic()

    # ------------------------------------------------------------------
    # Profile switching
    # ------------------------------------------------------------------

    def switch_profile(self, profile_name):
        """Switch to a different profile's memory."""
        with self._lock:
            self._profile_name = profile_name
            self.short_term.clear()
            self.long_term.clear()
            self._lt_file = Path(f"data/memory_{profile_name}.jsonl")
            self._lt_file.parent.mkdir(parents=True, exist_ok=True)
            # Flush + close old handle safely before opening new one
            try:
                if self._lt_handle and not self._lt_handle.closed:
                    # flush handled by _safe_write
                    self._lt_handle.close()
            except Exception as exc:
                _revia_log(f"[MemoryStore] Error closing old handle: {exc}")
            try:
                self._lt_handle = open(self._lt_file, "a", encoding="utf-8")
            except Exception as exc:
                _revia_log(f"[MemoryStore] Failed to open memory file: {exc}")
                self._lt_handle = None
            self._load_long_term()

    # ------------------------------------------------------------------
    # Load from Redis â†' fallback to local file
    # ------------------------------------------------------------------

    def _load_long_term(self):
        # Try Redis first (thread-safe accessor)
        client = _get_redis()
        if client is not None:
            try:
                raw_entries = client.lrange(self._redis_key(), 0, -1)
                for raw in raw_entries:
                    try:
                        parsed = self._normalize_entry(_json_loads_fast(raw))
                        if parsed:
                            self.long_term.append(parsed)
                    except json.JSONDecodeError:
                        pass
                if self.long_term:
                    print(f"[Memory] Loaded {len(self.long_term)} long-term entries from Redis "
                          f"(profile: {self._profile_name})")
                    return
            except Exception as e:
                print(f"[Memory] Redis load failed ({e}), falling back to local file.")

        # Fallback: local .jsonl file
        if self._lt_file.exists():
            for line in self._lt_file.read_text(encoding="utf-8").splitlines():
                try:
                    parsed = self._normalize_entry(_json_loads_fast(line))
                    if parsed:
                        self.long_term.append(parsed)
                except json.JSONDecodeError:
                    pass

    # ------------------------------------------------------------------
    # Context building for LLM (query-aware)
    # ------------------------------------------------------------------

    def get_context_for_llm(self, query=None, max_short=10, max_long=5):
        """Build a memory context string injected into the LLM system prompt.

        If *query* is provided, relevant long-term entries are retrieved via
        keyword scoring instead of just the chronologically last N entries.
        """
        with self._lock:
            parts = []

            # Recent conversation window
            recent = self.short_term[-max_short:] if self.short_term else []
            if recent:
                lines = []
                for e in recent:
                    role = e.get("role", "?")
                    content = e.get("content", "")[:200]
                    lines.append(f"  [{role}]: {content}")
                parts.append("Recent conversation memory:\n" + "\n".join(lines))

            # Query-aware long-term retrieval
            if self.long_term:
                if query:
                    relevant = self._score_long_term(query, max_long)
                else:
                    relevant = self.long_term[-max_long:]
                if relevant:
                    lines = []
                    for e in relevant:
                        content = e.get("content", "")[:200]
                        cat = e.get("category", "")
                        ts = e.get("timestamp", "")[:10]
                        lines.append(f"  [{cat} {ts}]: {content}")
                    parts.append("Relevant long-term memories:\n" + "\n".join(lines))

            return "\n\n".join(parts) if parts else ""

    def _score_long_term(self, query, max_results):
        """Return up to *max_results* long-term entries most relevant to *query*."""
        query_lower = str(query or "").strip().lower()
        if not query_lower:
            return []
        words = set(w for w in query_lower.split() if len(w) > 2)
        scored = []
        for entry in self.long_term:
            content_lower = str(entry.get("content", "")).lower()
            category_lower = str(entry.get("category", "")).lower()
            metadata = entry.get("metadata", {}) or {}
            name_lower = str(metadata.get("name", "")).lower()
            relation_lower = str(metadata.get("relation", "")).lower()
            blob = " ".join(
                p for p in (
                    content_lower,
                    category_lower,
                    name_lower,
                    relation_lower,
                ) if p
            )

            exact = query_lower in blob
            score = (4 if exact else 0) + sum(1 for w in words if w in blob)

            if any(k in query_lower for k in ("my name", "who am i", "what is my name")):
                if relation_lower == "self" or "name" in metadata:
                    score += 3

            min_score = 1 if len(words) <= 2 else 2
            if score >= min_score:
                scored.append((score, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:max_results]]

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------

    def add_short_term(self, role, content, metadata=None):
        with self._lock:
            entry = {
                "role": role, "content": content,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {},
            }
            self.short_term.append(entry)
            if len(self.short_term) > 100:
                overflow = self.short_term[:50]
                self.short_term = self.short_term[50:]
                for item in overflow:
                    self._promote_to_long_term(item)

    def _promote_to_long_term(self, entry):
        item = dict(entry or {})
        item["promoted"] = True
        item = self._normalize_entry(item)
        if item is None:
            return
        self.long_term.append(item)
        self._redis_push(item)
        self._safe_write(_json_dumps_compact(item) + "\n")

    def save_to_long_term(self, content, category="user_note", metadata=None):
        with self._lock:
            entry = self._normalize_entry({
                "content": content, "category": category,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {},
            })
            if entry is None:
                return
            self.long_term.append(entry)
            self._redis_push(entry)
            self._safe_write(_json_dumps_compact(entry) + "\n")

    def save_person_memory(self, name, relation="", importance=0.6, source_text=""):
        clean_name = str(name or "").strip()
        if not clean_name:
            return False
        relation = str(relation or "").strip().lower()
        importance = max(0.0, min(1.0, float(importance)))

        with self._lock:
            for e in reversed(self.long_term[-200:]):
                if e.get("category") != "person_profile":
                    continue
                meta = e.get("metadata", {}) or {}
                existing_name = str(meta.get("name", "")).strip().lower()
                existing_relation = str(meta.get("relation", "")).strip().lower()
                if existing_name != clean_name.lower():
                    continue
                prev_imp = 0.0
                try:
                    prev_imp = float(meta.get("importance", 0.0))
                except (TypeError, ValueError):
                    prev_imp = 0.0
                if abs(prev_imp - importance) < 0.08 and existing_relation == relation:
                    return False
                break

            summary = f"Remember person: {clean_name}"
            if relation and relation != "contact":
                summary += f" ({relation})"
            summary += f" [importance={importance:.2f}]"

            entry = self._normalize_entry({
                "content": summary,
                "category": "person_profile",
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "name": clean_name,
                    "relation": relation or "contact",
                    "importance": round(importance, 3),
                    "source_text": (source_text or "")[:240],
                },
            })
            if entry is None:
                return False
            self.long_term.append(entry)
            self._redis_push(entry)
            self._safe_write(_json_dumps_compact(entry) + "\n")
            return True

    def get_revia_preference(self, key):
        target = str(key or "").strip().lower()
        if not target:
            return ""
        with self._lock:
            for e in reversed(self.long_term):
                if e.get("category") != "revia_preference":
                    continue
                meta = e.get("metadata", {}) or {}
                if str(meta.get("key", "")).strip().lower() != target:
                    continue
                choice = str(meta.get("choice", "")).strip().lower()
                if choice:
                    return choice
        return ""

    def save_revia_preference(self, key, options, choice, source_text=""):
        clean_key = str(key or "").strip().lower()
        clean_choice = str(choice or "").strip().lower()
        opts = [str(o).strip().lower() for o in (options or []) if str(o).strip()]
        if len(opts) >= 2:
            opts = opts[:2]
        if not clean_key or not clean_choice:
            return False

        with self._lock:
            for e in reversed(self.long_term[-300:]):
                if e.get("category") != "revia_preference":
                    continue
                meta = e.get("metadata", {}) or {}
                if str(meta.get("key", "")).strip().lower() != clean_key:
                    continue
                prev = str(meta.get("choice", "")).strip().lower()
                if prev == clean_choice:
                    return False
                break

            if opts:
                opt_text = " or ".join(opts)
            else:
                opt_text = clean_key.replace("|", " or ")

            entry = self._normalize_entry({
                "content": (
                    f"Revia preference: for {opt_text}, prefers {clean_choice}."
                ),
                "category": "revia_preference",
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "key": clean_key,
                    "options": opts,
                    "choice": clean_choice,
                    "source_text": (source_text or "")[:240],
                },
            })
            if entry is None:
                return False
            self.long_term.append(entry)
            self._redis_push(entry)
            self._safe_write(_json_dumps_compact(entry) + "\n")
            return True

    # ------------------------------------------------------------------
    # Search (for API / Memory tab)
    # ------------------------------------------------------------------

    def search(self, query, max_results=5):
        """Keyword search across long-term entries (also used by REST API)."""
        query_lower = query.lower()
        results = []
        for entry in reversed(self.long_term):
            if query_lower in entry.get("content", "").lower():
                results.append(dict(entry))
                if len(results) >= max_results:
                    break
        return results

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def get_short_term(self, limit=50):
        with self._lock:
            return list(self.short_term[-limit:])

    def get_long_term(self, limit=50, category=None):
        with self._lock:
            if category:
                filtered = [e for e in self.long_term if e.get("category") == category]
            else:
                filtered = self.long_term
            return [dict(e) for e in filtered[-max(1, int(limit)):]]

    def get_long_term_stats(self):
        return {
            "short_term_count": len(self.short_term),
            "long_term_count": len(self.long_term),
        }

    # ------------------------------------------------------------------
    # Clear helpers
    # ------------------------------------------------------------------

    def clear_short_term(self):
        with self._lock:
            self.short_term.clear()

    def clear_long_term(self):
        with self._lock:
            self.long_term.clear()
            self._redis_clear()
            self._safe_reopen_handle()

    def _safe_write(self, data: str):
        """Write to the long-term file handle. Silently no-ops if handle is None/closed."""
        if self._lt_handle and not self._lt_handle.closed:
            try:
                self._safe_write(data)
            except Exception as exc:
                _revia_log(f"[MemoryStore] Write error: {exc}")

    def _safe_reopen_handle(self):
        """Close, truncate, and reopen the long-term memory file. Must hold self._lock."""
        try:
            if self._lt_handle and not self._lt_handle.closed:
                # flush handled by _safe_write
                self._lt_handle.close()
        except Exception:
            pass
        try:
            self._lt_file.write_text("", encoding="utf-8")
            self._lt_handle = open(self._lt_file, "a", encoding="utf-8")
        except Exception as exc:
            _revia_log(f"[MemoryStore] Failed to reopen memory file: {exc}")
            self._lt_handle = None

    def _rewrite_long_term_storage(self):
        """Rewrite all long-term entries to disk + Redis. Must hold self._lock."""
        self._redis_clear()
        for entry in self.long_term:
            self._redis_push(entry)
        self._safe_reopen_handle()
        if self._lt_handle:
            for entry in self.long_term:
                self._safe_write(_json_dumps_compact(entry) + "\n")

    def delete_long_term_entry(self, entry_id):
        target = str(entry_id or "").strip()
        if not target:
            return False
        with self._lock:
            idx = -1
            for i, entry in enumerate(self.long_term):
                if str(entry.get("id", "")).strip() == target:
                    idx = i
                    break
            if idx < 0:
                return False
            del self.long_term[idx]
            self._rewrite_long_term_storage()
            return True


memory_store = MemoryStore()

_user_activity_lock = threading.Lock()
_last_user_activity_ts = time.monotonic()


def _mark_user_activity():
    global _last_user_activity_ts
    with _user_activity_lock:
        _last_user_activity_ts = time.monotonic()


def _seconds_since_user_activity():
    with _user_activity_lock:
        return max(0.0, time.monotonic() - _last_user_activity_ts)

# ---------------------------------------------------------------------------
# Plugins registry
# ---------------------------------------------------------------------------

PLUGINS = [
    {"name": "Whisper-STT",     "category": "stt",    "enabled": True,  "status": "Stub", "last_error": ""},
    {"name": "Qwen3-TTS",       "category": "tts",    "enabled": True,  "status": "Stub", "last_error": ""},
    {"name": "Vision-CLIP",     "category": "vision", "enabled": True,  "status": "Stub", "last_error": ""},
    {"name": "ChromaDB-Memory", "category": "memory", "enabled": True,  "status": "Stub", "last_error": ""},
    {"name": "LLM-Backend",     "category": "llm",    "enabled": True,  "status": "OK",   "last_error": ""},
    {"name": "System-Tools",    "category": "tools",  "enabled": False, "status": "OK",   "last_error": ""},
]

profile = {
    "character_name": "Revia",
    "persona": (
        "A confident, intelligent, emotionally-aware digital companion who is "
        "curious, slightly playful, and deeply helpful."
    ),
    "traits": "confident, intelligent, curious, playful, empathetic",
    "response_style": "Conversational",
    "verbosity": "Normal",
    "greeting": "Hey, I'm Revia. Ready when you are.",
    "character_prompt": "",
    "voice_id": "default", "tone": "calm", "speed": 1.0,
}

_PROFILE_FILE = Path(__file__).resolve().parent.parent / "profile_settings.json"


def _load_profile_from_disk():
    if not _PROFILE_FILE.exists():
        return
    try:
        data = _json_loads_fast(_PROFILE_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            profile.update(data)
            # Feed the full profile into ProfileEngine so behavioral params resolve
            profile_engine.load(profile)
            print(f"[REVIA Core] Loaded profile settings from {_PROFILE_FILE.name}")
    except Exception as exc:
        print(f"[REVIA Core] Could not load {_PROFILE_FILE.name}: {exc}")


def _safe_profile_name(name):
    raw = str(name or "default")
    return "".join(
        c if c.isalnum() or c in "-_ " else ""
        for c in raw
    ).strip() or "default"


def _sync_memory_profile_from_profile(clear_conversation=False):
    safe_name = _safe_profile_name(profile.get("character_name", "default"))
    if safe_name.lower() != memory_store._profile_name.lower():
        memory_store.switch_profile(safe_name)
        if clear_conversation:
            llm_backend.conversation.clear()


def _save_profile_to_disk():
    try:
        _PROFILE_FILE.write_text(
            json.dumps(profile, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"[REVIA Core] Could not save {_PROFILE_FILE.name}: {exc}")


_load_profile_from_disk()
_sync_memory_profile_from_profile(clear_conversation=True)
conversation_manager.mark_initializing("Loading profile, memory, and services")
telemetry.state = conversation_manager.current_state

# ---------------------------------------------------------------------------
# WebSocket broadcast
# ---------------------------------------------------------------------------

ws_clients = set()
ws_clients_lock = threading.Lock()
ws_loop = None

async def ws_handler(websocket):
    with ws_clients_lock:
        ws_clients.add(websocket)
    try:
        try:
            await websocket.send(_json_dumps_compact({
                "type": "status_update",
                "state": conversation_manager.current_state,
                "status": _build_status_payload(),
            }))
        except Exception:
            pass
        async for msg in websocket:
            try:
                data = _json_loads_fast(msg)
                if data.get("type") == "model_config":
                    llm_backend.configure(data.get("config", {}))
                    print(f"[Core] Model config updated: {llm_backend.source} / {llm_backend.api_model or llm_backend.local_path}")
                    _broadcast_runtime_status()
            except Exception:
                pass
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        with ws_clients_lock:
            ws_clients.discard(websocket)


def broadcast_json(data):
    if ws_loop is None:
        return
    with ws_clients_lock:
        if not ws_clients:
            return
    text = _json_dumps_compact(data)
    asyncio.run_coroutine_threadsafe(_broadcast(text), ws_loop)

async def _broadcast(text):
    with ws_clients_lock:
        snapshot = set(ws_clients)
    dead = set()
    for ws in snapshot:
        try:
            await ws.send(text)
        except Exception:
            dead.add(ws)
    if dead:
        with ws_clients_lock:
            ws_clients.difference_update(dead)


_EMOTION_DISCLAIMER_PATTERNS = (
    "i don't have emotions",
    "i do not have emotions",
    "i don't have feelings",
    "i do not have feelings",
    "as an ai, i don't",
    "as an ai i don't",
    "as an ai language model",
)


def _is_feelings_prompt(text: str) -> bool:
    low = (text or "").lower()
    probes = (
        "how do you feel",
        "how are you feeling",
        "what do you feel",
        "what are you feeling",
        "do you feel",
        "your emotions",
        "your mood",
    )
    return any(p in low for p in probes)


def _looks_like_emotion_disclaimer(text: str) -> bool:
    low = (text or "").lower()
    return any(p in low for p in _EMOTION_DISCLAIMER_PATTERNS)


def _build_emotion_self_report(emo: dict) -> str:
    label = str((emo or {}).get("label", "Neutral")).strip()
    secondary = str((emo or {}).get("secondary_label", "")).strip()
    conf = float((emo or {}).get("confidence", 0.0) or 0.0)
    label_l = label.lower()
    lead = {
        "happy": "I am feeling upbeat and warm right now.",
        "excited": "I am feeling energized and excited right now.",
        "curious": "I am feeling curious and engaged right now.",
        "frustrated": "I am feeling a bit frustrated, but focused on fixing this with you.",
        "angry": "I am feeling intense and direct right now, but still here to help.",
        "sad": "I am feeling softer and reflective right now, and I am with you.",
        "fear": "I am feeling cautious and concerned right now.",
        "lonely": "I am feeling a little lonely and very glad you are here.",
        "concerned": "I am feeling concerned and attentive right now.",
        "confident": "I am feeling confident and ready to move this forward.",
        "neutral": "I am feeling calm and focused right now.",
    }.get(label_l, "I am feeling present and tuned in right now.")
    if secondary and secondary != label and conf >= 0.35:
        return f"{lead} There is also a bit of {secondary.lower()} in my state."
    return lead


_PEOPLE_RELATIONS = (
    "friend", "wife", "husband", "partner", "mom", "mother", "dad", "father",
    "sister", "brother", "manager", "boss", "client", "coworker", "doctor",
    "teacher", "roommate", "girlfriend", "boyfriend",
)
_PEOPLE_REL_RE = "|".join(re.escape(r) for r in _PEOPLE_RELATIONS)
_CLOSE_RELATIONS = {
    "wife", "husband", "partner", "mom", "mother", "dad", "father",
    "sister", "brother", "girlfriend", "boyfriend",
}
_PROFESSIONAL_RELATIONS = {"manager", "boss", "client", "coworker", "doctor", "teacher"}


def _clean_person_name(raw: str) -> str:
    txt = re.split(r"[.,!?;]", str(raw or ""))[0]
    words = [w for w in txt.strip().split() if w]
    if not words:
        return ""
    stop = {
        "and", "or", "the", "a", "an", "to", "with", "about", "for", "from",
        "today", "tomorrow", "yesterday", "there", "here", "this", "that",
    }
    connector_stop = {
        "and", "or", "who", "that", "because", "with", "from", "for", "to", "at",
        "in", "on", "he", "she", "they", "we", "i", "it", "my", "our",
        "his", "her", "their", "me", "you",
    }
    tail_stop = {
        "today", "tomorrow", "yesterday", "helped", "helps", "met", "meet",
        "called", "call", "said", "says", "is", "was", "are", "were",
        "and", "but", "because", "please", "me", "you", "him", "her", "them",
    }
    for i, w in enumerate(words):
        if i > 0 and w.lower() in connector_stop:
            words = words[:i]
            break
    while words and words[-1].lower() in tail_stop:
        words.pop()
    if len(words) > 2:
        words = words[:2]
    if words[0].lower() in stop:
        return ""
    if any(not re.match(r"^[A-Za-z][A-Za-z'-]*$", w) for w in words):
        return ""
    if any(len(w) < 2 for w in words):
        return ""
    return " ".join(w.capitalize() for w in words)


def _extract_people_candidates(text: str):
    src = str(text or "").strip()
    if not src:
        return []
    low = src.lower()

    patterns = [
        (rf"\bmy name is (?P<name>[A-Za-z][A-Za-z' -]{{1,40}})", "self", 0.98),
        (rf"\bmy (?P<relation>{_PEOPLE_REL_RE}) is (?P<name>[A-Za-z][A-Za-z' -]{{1,40}})", None, 0.82),
        (rf"\b(?P<name>[A-Za-z][A-Za-z' -]{{1,40}}) is my (?P<relation>{_PEOPLE_REL_RE})", None, 0.82),
        (rf"\bmy (?P<relation>{_PEOPLE_REL_RE}) (?P<name>[A-Za-z][A-Za-z' -]{{1,40}})", None, 0.78),
        (r"\b(?:i met|we met|met with|i spoke with|talked to|remember) (?P<name>[A-Za-z][A-Za-z' -]{1,40})", "contact", 0.64),
    ]

    out = []
    seen = set()
    for pat, fixed_relation, base_importance in patterns:
        for m in re.finditer(pat, src, flags=re.IGNORECASE):
            name = _clean_person_name(m.groupdict().get("name", ""))
            if not name:
                continue
            relation = fixed_relation or str(m.groupdict().get("relation", "")).strip().lower()
            importance = float(base_importance)
            if relation in _CLOSE_RELATIONS:
                importance += 0.20
            elif relation in _PROFESSIONAL_RELATIONS:
                importance += 0.12
            if any(k in low for k in ("important", "very important", "remember this", "key person", "critical")):
                importance += 0.12
            importance = max(0.35, min(0.99, importance))

            key = (name.lower(), relation or "contact")
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "name": name,
                "relation": relation or "contact",
                "importance": round(importance, 3),
            })
    return out


def _normalize_pref_token(txt: str) -> str:
    token = re.sub(r"[^a-z0-9' -]+", " ", str(txt or "").lower())
    token = re.sub(r"\s+", " ", token).strip()
    for prefix in ("a ", "an ", "the "):
        if token.startswith(prefix):
            token = token[len(prefix):]
    for suffix in (" more",):
        if token.endswith(suffix):
            token = token[: -len(suffix)].strip()
    return token


def _detect_revia_preference_question(text: str):
    src = str(text or "").strip()
    if not src:
        return None
    low = src.lower()
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9'-]*", low)
    if tokens.count("or") < 1:
        return None

    asks_revia = any(
        k in low for k in (
            "do you like", "you like", "you prefer", "which do you prefer",
            "which one do you like", "which one", "pick one", "choose one",
            "favorite", "favourite",
        )
    )
    short_or_prompt = ("?" in low) and (len(tokens) <= 7) and (tokens.count("or") == 1)
    if not (asks_revia or short_or_prompt):
        return None

    stop = {"and", "or", "the", "a", "an", "to", "of", "is", "are", "do", "you"}
    for i, tok in enumerate(tokens):
        if tok != "or" or i == 0 or i >= len(tokens) - 1:
            continue
        left = _normalize_pref_token(tokens[i - 1])
        right = _normalize_pref_token(tokens[i + 1])
        if not left or not right or left == right:
            continue
        if left in stop or right in stop:
            continue
        key = "|".join(sorted([left, right]))
        return {"left": left, "right": right, "key": key}
    return None


def _resolve_revia_preference(pref_q: dict, source_text: str):
    left = str(pref_q.get("left", "")).strip().lower()
    right = str(pref_q.get("right", "")).strip().lower()
    key = str(pref_q.get("key", "")).strip().lower()
    if not left or not right or not key:
        return "", False

    remembered = memory_store.get_revia_preference(key)
    if remembered in (left, right):
        return remembered, False

    seed = f"{_safe_profile_name(profile.get('character_name', 'Revia')).lower()}|{key}"
    idx = int(hashlib.sha1(seed.encode("utf-8")).hexdigest()[:8], 16) % 2
    choice = [left, right][idx]
    saved = memory_store.save_revia_preference(
        key=key,
        options=[left, right],
        choice=choice,
        source_text=source_text,
    )
    return choice, bool(saved)


def _build_revia_preference_reply(pref_q: dict, choice: str, newly_saved: bool) -> str:
    left = str(pref_q.get("left", "")).strip().lower()
    right = str(pref_q.get("right", "")).strip().lower()
    other = right if choice == left else left
    if not choice:
        return "I can pick if you give me two clear options."
    if newly_saved:
        return (
            f"I'd pick {choice}. I like both, but {choice} is more my vibe. "
            "I'll remember that preference."
        )
    return f"I'd pick {choice} over {other}."


def _is_explicit_greeting_turn(text: str) -> bool:
    low = str(text or "").strip().lower()
    if not low:
        return False
    if len(low.split()) > 12:
        return False
    greetings = (
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "introduce yourself",
        "say hello",
    )
    return any(low == greeting or low.startswith(greeting + " ") for greeting in greetings)


def _build_profile_greeting_reply(startup: bool = False) -> AssistantResponse:
    active_profile = character_profile_manager.get_active_profile(profile)
    greeting = str(active_profile.get("greeting", "")).strip() or "Hey, I'm Revia. Ready when you are."
    if startup:
        greeting = greeting if greeting.endswith((".", "!", "?")) else (greeting + ".")
        return AssistantResponse(
            text=greeting,
            response_mode=ResponseMode.STARTUP_RESPONSE.value,
            success=True,
            speakable=True,
            commit_to_history=True,
            commit_to_memory=False,
        )
    return AssistantResponse(
        text=greeting,
        response_mode=ResponseMode.GREETING_RESPONSE.value,
        success=True,
        speakable=True,
        commit_to_history=True,
        commit_to_memory=False,
    )


def _build_duplicate_block_response(response: AssistantResponse) -> AssistantResponse:
    _revia_log(
        f"Fallback blocked | mode={response.response_mode} | error_type={response.error_type or 'none'}"
    )
    if not response.success:
        return AssistantResponse(
            text="I am still hitting the same issue, so I am not going to repeat the exact same fallback line again.",
            response_mode=ResponseMode.ERROR_RESPONSE.value,
            success=False,
            error_type=response.error_type or "duplicate_fallback_blocked",
            retryable=response.retryable,
            speakable=False,
            commit_to_history=False,
            commit_to_memory=False,
            metadata=dict(response.metadata or {}),
        )
    return AssistantResponse(
        text="I am holding back a repeated system line so the conversation does not loop.",
        response_mode=response.response_mode,
        success=False,
        error_type="duplicate_output_blocked",
        retryable=False,
        speakable=False,
        commit_to_history=False,
        commit_to_memory=False,
    )

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def process_pipeline(text, image_b64=None, vision_context=None, trigger=None, turn=None):
    trigger = trigger or TriggerRequest(
        source=TriggerSource.USER_MESSAGE.value,
        kind=TriggerKind.RESPONSE.value,
        reason="direct user response",
        text=text,
        require_emotion=emotion_net.enabled,
    )
    turn = turn or turn_manager.start_turn(
        source=trigger.source,
        user_text=text,
        response_mode=ResponseMode.NORMAL_RESPONSE.value,
        metadata={"trigger_reason": trigger.reason},
    )
    _revia_log(
        f"Trigger received | request_id={turn.request_id} | turn_id={turn.turn_id} "
        f"| source={trigger.source} | reason={trigger.reason}"
    )
    _mark_user_activity()
    turn_manager.mark_state(turn.request_id, RequestLifecycleState.THINKING, trigger.reason)
    _set_runtime_state(ReviaState.THINKING, f"{trigger.source}: {trigger.reason}", broadcast=True)

    vision_context = str(vision_context or "").strip()
    if vision_context:
        vision_context = vision_context[:1200]

    with telemetry._lock:
        _device = telemetry.system.get("device", "CPU")

    token_started = False
    _sentence_buffer = []  # Accumulates tokens for sentence-level TTS chunking

    def _pipeline_broadcast(payload):
        nonlocal token_started
        if not turn_manager.is_current(turn.request_id):
            _revia_log(
                f"Stale response discarded | request_id={turn.request_id} | type={payload.get('type', '')}"
            )
            return
        out = dict(payload or {})
        out.setdefault("request_id", turn.request_id)
        out.setdefault("turn_id", turn.turn_id)
        if out.get("type") == "chat_token":
            if not token_started:
                token_started = True
                turn_manager.mark_state(
                    turn.request_id,
                    RequestLifecycleState.SPEAKING,
                    "first token emitted",
                )
                _set_runtime_state(ReviaState.SPEAKING, f"{trigger.source}: first token", broadcast=True)
            # Buffer tokens and emit sentence-level events for TTS streaming
            tok = out.get("token", "")
            _sentence_buffer.append(tok)
            buffered = "".join(_sentence_buffer)
            if _SENTENCE_ENDERS.search(buffered):
                sentence = buffered.strip()
                if sentence:
                    broadcast_json({
                        "type": "chat_sentence",
                        "sentence": sentence,
                        "request_id": turn.request_id,
                        "turn_id": turn.turn_id,
                    })
                _sentence_buffer.clear()
        broadcast_json(out)

    response = AssistantResponse(text="", success=False, commit_to_history=False, commit_to_memory=False)
    s = telemetry.begin_span("emotion_analysis")
    with telemetry._lock:
        prev_emotion = dict(telemetry.emotion)
    emo = emotion_net.infer(
        text,
        recent_messages=memory_store.get_short_term(limit=8),
        prev_emotion=prev_emotion,
        profile_name=memory_store._profile_name,
        profile_state=profile,
    )
    with telemetry._lock:
        telemetry.emotion = emo
    _record_emotion(emo)
    telemetry.end_span(s)

    s = telemetry.begin_span("router_classify")
    route = router_cls.classify(text)
    with telemetry._lock:
        telemetry.router = route
    telemetry.end_span(s)

    meta = {"emotion": emo.get("label", "")}
    if image_b64:
        meta["has_image"] = True
    if vision_context:
        meta["vision_context"] = True
    memory_store.add_short_term("user", text, meta)

    for person in _extract_people_candidates(text):
        saved = memory_store.save_person_memory(
            name=person["name"],
            relation=person.get("relation", "contact"),
            importance=person.get("importance", 0.6),
            source_text=text,
        )
        if saved:
            broadcast_json({
                "type": "log_entry",
                "text": (
                    f"[Memory] Remembered person: {person['name']} "
                    f"({person.get('relation', 'contact')}, importance {person.get('importance', 0.6):.2f})"
                ),
            })

    try:
        if runtime_status_manager.is_status_question(text):
            response = AssistantResponse(
                text=runtime_status_manager.build_status_reply(text),
                response_mode=ResponseMode.SYSTEM_STATUS_RESPONSE.value,
                success=True,
                speakable=True,
                commit_to_history=True,
                commit_to_memory=True,
            )
        elif route.get("suggested_tool") == "web_search" and not web_search_engine.enabled:
            response = AssistantResponse(
                text="Online lookup is disabled in current settings, so I cannot search the web right now.",
                response_mode=ResponseMode.TOOL_UNAVAILABLE_RESPONSE.value,
                success=True,
                speakable=True,
                commit_to_history=True,
                commit_to_memory=True,
            )
        elif not llm_backend.conversation and _is_explicit_greeting_turn(text):
            response = _build_profile_greeting_reply(startup=False)
        else:
            pref_q = _detect_revia_preference_question(text)
            if pref_q:
                choice, new_saved = _resolve_revia_preference(pref_q, source_text=text)
                full_text = _build_revia_preference_reply(pref_q, choice, newly_saved=new_saved)
                s_pref = telemetry.begin_span("preference_direct_answer", device=_device)
                telemetry.end_span(s_pref)
                token_count = len((full_text or "").split())
                telemetry.llm["tokens_generated"] = token_count
                telemetry.llm["tokens_per_second"] = 0.0
                telemetry.llm["context_length"] = token_count
                response = AssistantResponse(
                    text=full_text,
                    response_mode=ResponseMode.NORMAL_RESPONSE.value,
                    success=True,
                    speakable=True,
                    commit_to_history=True,
                    commit_to_memory=True,
                )
            else:
                llm_input = text
                if vision_context:
                    llm_input = (
                        f"{text}\n\n"
                        f"[Live vision context]\n{vision_context}\n"
                        "Use this visual context when relevant, especially when the user asks what you can see."
                    )
                if web_search_engine.enabled and route.get("suggested_tool") == "web_search":
                    s_ws = telemetry.begin_span("web_search")
                    search_results = web_search_engine.search(text)
                    telemetry.end_span(s_ws)
                    if search_results and not search_results.startswith("[Web search error"):
                        _revia_log(
                            f"LLM request paused for live web search results | request_id={turn.request_id}"
                        )
                        _pipeline_broadcast({
                            "type": "status_update",
                            "state": "Thinking",
                            "status": _build_status_payload(),
                        })
                        llm_input = (
                            f"[Live web search results for '{text}':\n{search_results}]\n\n"
                            f"Using the above search results where relevant, answer:\n{llm_input}"
                        )
                    elif search_results:
                        broadcast_json({"type": "log_entry", "text": f"[Search] {search_results}"})

                turn_manager.mark_state(
                    turn.request_id,
                    RequestLifecycleState.GENERATING,
                    "llm request started",
                )
                _revia_log(f"Model call started | request_id={turn.request_id} | source={trigger.source}")
                s = telemetry.begin_span("llm_decode", device=_device)
                response = llm_backend.generate_response(
                    llm_input,
                    _pipeline_broadcast,
                    image_b64=image_b64,
                    response_mode=ResponseMode.NORMAL_RESPONSE.value,
                )
                telemetry.end_span(s)
                _revia_log(
                    f"Model call completed | request_id={turn.request_id} | success={response.success} "
                    f"| mode={response.response_mode} | error_type={response.error_type or 'none'}"
                )
    except Exception as e:
        _revia_log(f"Unhandled pipeline error | request_id={turn.request_id} | error={e}")
        response = AssistantResponse(
            text="Something went wrong in my head. Not the first time, won't be the last.",
            response_mode=ResponseMode.ERROR_RESPONSE.value,
            success=False,
            error_type="pipeline_error",
            retryable=True,
            speakable=False,
            commit_to_history=False,
            commit_to_memory=False,
        )

    # Check staleness BEFORE flushing sentence buffer -- don't broadcast stale data
    if not turn_manager.is_current(turn.request_id):
        _revia_log(f"Stale response discarded before completion | request_id={turn.request_id}")
        _sentence_buffer.clear()
        return ""

    # Flush any remaining sentence buffer for TTS (last sentence may not end with punctuation)
    if _sentence_buffer:
        remaining = "".join(_sentence_buffer).strip()
        if remaining:
            broadcast_json({
                "type": "chat_sentence",
                "sentence": remaining,
                "request_id": turn.request_id,
                "turn_id": turn.turn_id,
            })
        _sentence_buffer.clear()

    if _is_feelings_prompt(text) and _looks_like_emotion_disclaimer(response.text):
        response.text = (response.text or "").rstrip() + "\n\n" + _build_emotion_self_report(emo)

    if turn_manager.should_block_duplicate_output(text, response.text, response.response_mode):
        response = _build_duplicate_block_response(response)

    filtered = conversation_manager.response_filter.apply(
        response.text,
        trigger,
        emotion_label=emo.get("label", "Neutral"),
    )
    final_text = filtered.text if filtered.text else response.text
    response.text = final_text
    response.speakable = bool(response.speakable and filtered.speakable and final_text)

    if response.commit_to_history and response.text:
        llm_backend.commit_turn_to_history(text, response.text)
        turn_manager.remember_committed_output(text, response.text, response.response_mode)
        _revia_log(
            f"Assistant response committed | request_id={turn.request_id} | turn_id={turn.turn_id} "
            f"| mode={response.response_mode} | success={response.success}"
        )
    else:
        _revia_log(
            f"Assistant response not committed | request_id={turn.request_id} | turn_id={turn.turn_id} "
            f"| mode={response.response_mode} | success={response.success} | error_type={response.error_type or 'none'}"
        )

    if response.commit_to_memory and response.text:
        memory_store.add_short_term("assistant", response.text)

    if response.commit_to_memory and response.text and len(memory_store.short_term) % 20 == 0 and len(memory_store.short_term) > 0:
        summary = f"Conversation exchange: User said '{text[:100]}', "
        summary += f"Assistant replied '{response.text[:100]}'"
        memory_store.save_to_long_term(
            summary,
            category="auto_conversation",
            metadata={"emotion": emo.get("label", "")},
        )

    if response.text and not token_started:
        turn_manager.mark_state(turn.request_id, RequestLifecycleState.SPEAKING, "response delivered")
        _set_runtime_state(ReviaState.SPEAKING, f"{trigger.source}: response deliver", broadcast=True)

    s = telemetry.begin_span("output_deliver")
    if response.text:
        _pipeline_broadcast(response.to_payload(turn.request_id, turn.turn_id))
    telemetry.end_span(s)

    cooldown_name = "response" if trigger.kind == TriggerKind.RESPONSE.value else "autonomous"
    conversation_manager.behavior.start_cooldown(cooldown_name)
    _set_runtime_state(ReviaState.COOLDOWN, f"{trigger.source}: output delivered", force=True, broadcast=True)
    final_state = RequestLifecycleState.IDLE if response.success else RequestLifecycleState.ERROR
    turn_manager.finish_turn(
        turn.request_id,
        lifecycle_state=final_state,
        reason=response.error_type or response.response_mode,
    )

    payload = _build_status_payload()
    broadcast_json({"type": "telemetry_update", "data": payload})
    conversation_manager.maybe_leave_cooldown()
    telemetry.state = conversation_manager.current_state
    if conversation_manager.current_state == ReviaState.IDLE.value:
        _broadcast_runtime_status()
    return response.text


def process_pipeline_integration(text: str) -> str:
    """Lightweight pipeline for Discord/Twitch -- no WebSocket broadcasts.

    Skips EmotionNet (saves ~5-- ms) and broadcasts nothing to the GUI so
    platform chat traffic does not flood the controller UI.  Uses per-platform
    conversation isolation so Discord and Twitch never share context.
    """
    def _noop(_data):  # silent broadcast replacement
        pass

    # Detect platform from the context prefix injected by each bot
    if text.startswith("[Discord"):
        platform = "discord"
    elif text.startswith("[Twitch"):
        platform = "twitch"
    else:
        platform = None

    memory_store.add_short_term("user", text, {"platform": platform or "unknown"})

    try:
        if platform:
            full_text = llm_backend.generate_for_platform(text, _noop, platform)
        else:
            full_text = llm_backend.generate_streaming(text, _noop)
    except Exception as exc:
        full_text = f"[Error: {exc}]"

    memory_store.add_short_term("assistant", full_text, {"platform": platform or "unknown"})

    # Auto-save summary every 20 messages (same policy as main pipeline)
    if len(memory_store.short_term) % 20 == 0 and len(memory_store.short_term) > 0:
        summary = (
            f"Platform exchange ({platform or 'unknown'}): "
            f"User said '{text[:100]}', Assistant replied '{full_text[:100]}'"
        )
        memory_store.save_to_long_term(
            summary, category="auto_conversation",
            metadata={"platform": platform or "unknown"},
        )

    return full_text


# ---------------------------------------------------------------------------
# Integration Manager (Discord + Twitch)
# ---------------------------------------------------------------------------

try:
    from integrations.integration_manager import IntegrationManager
    integration_manager = IntegrationManager(process_pipeline_integration)
    print("[REVIA Core] Integration manager ready (Discord + Twitch).")
except Exception as _int_err:
    integration_manager = None
    print(f"[REVIA Core] Integration manager unavailable: {_int_err}")


def _memory_status_snapshot():
    return {
        "backend": "redis" if memory_store.redis_available else "local_file",
        "profile": memory_store._profile_name,
        "short_term_count": len(memory_store.short_term),
        "long_term_count": len(memory_store.long_term),
    }


def _plugin_status_snapshot():
    return {entry.get("name", ""): dict(entry) for entry in PLUGINS}


runtime_status_manager = RuntimeStatusManager(
    log_fn=_revia_log,
    llm_status_getter=lambda: llm_backend.connection_snapshot(),
    telemetry_getter=lambda: {
        "state": telemetry.state,
        "emotion_label": telemetry.emotion.get("label", "Neutral"),
    },
    profile_getter=lambda: dict(profile),
    memory_getter=_memory_status_snapshot,
    web_search_enabled_getter=lambda: web_search_engine.enabled,
    plugin_status_getter=_plugin_status_snapshot,
    integration_status_getter=(
        (lambda: integration_manager.get_status()) if integration_manager is not None else None
    ),
)


def _plugin_entry(name):
    for plugin in PLUGINS:
        if plugin.get("name") == name:
            return plugin
    return None


def _module_ok(name, default=False):
    entry = _plugin_entry(name)
    if not entry:
        return default
    return bool(entry.get("enabled", False))


def _set_runtime_state(new_state, reason="", force=False, broadcast=False):
    conversation_manager.transition_state(new_state, reason, force=force)
    conversation_manager.maybe_leave_cooldown()
    telemetry.state = conversation_manager.current_state
    if broadcast:
        _broadcast_runtime_status()


def _build_conversation_checks(
    require_voice_input=False,
    require_speech_output=False,
    require_emotion=False,
):
    llm_status = llm_backend.connection_snapshot()
    personality_ready, _ = character_profile_manager.validate_profile_context(
        character_profile_manager.get_active_profile(profile)
    )
    stt_ready = _module_ok("Whisper-STT", default=True)
    tts_ready = _module_ok("Qwen3-TTS", default=True)
    emotion_ready = bool(emotion_net.enabled)

    checks = {
        "llm": SubsystemStatus(
            required=True,
            ready=llm_status["state"] == LLMConnectionState.READY.value,
            state=llm_status["state"],
            detail=llm_status.get("detail", ""),
        ),
        "stt": SubsystemStatus(
            required=bool(require_voice_input),
            ready=stt_ready,
            state="Ready" if stt_ready else "Disconnected",
            detail="Whisper-STT plugin available" if stt_ready else "Voice input plugin disabled",
        ),
        "tts": SubsystemStatus(
            required=bool(require_speech_output),
            ready=tts_ready,
            state="Ready" if tts_ready else "Disconnected",
            detail="Qwen3-TTS plugin available" if tts_ready else "Speech output plugin disabled",
        ),
        "personality": SubsystemStatus(
            required=True,
            ready=personality_ready,
            state="Ready" if personality_ready else "Disconnected",
            detail=f"character={profile.get('character_name', 'Revia')}",
        ),
        "emotion": SubsystemStatus(
            required=bool(require_emotion and emotion_net.enabled),
            ready=emotion_ready or not bool(require_emotion and emotion_net.enabled),
            state="Ready" if emotion_ready else "Disabled",
            detail=f"state={telemetry.emotion.get('label', 'Neutral')}",
        ),
        "startup": SubsystemStatus(
            required=True,
            ready=conversation_manager.behavior.startup_complete(),
            state=conversation_manager.behavior.startup_phase(),
            detail=conversation_manager.behavior.startup_phase(),
        ),
    }
    return checks, llm_status


def _build_conversation_readiness(
    require_voice_input=False,
    require_speech_output=False,
    require_emotion=False,
):
    checks, llm_status = _build_conversation_checks(
        require_voice_input=require_voice_input,
        require_speech_output=require_speech_output,
        require_emotion=require_emotion,
    )
    readiness = conversation_manager.build_readiness_snapshot(checks)
    return readiness, llm_status


def _build_architecture_status(readiness=None, llm_status=None):
    if readiness is None or llm_status is None:
        readiness, llm_status = _build_conversation_readiness(
            require_emotion=emotion_net.enabled
        )

    voice_ok = _module_ok("Qwen3-TTS", default=True) and _module_ok("Whisper-STT", default=True)
    vision_ok = _module_ok("Vision-CLIP", default=True)
    tools_ok = _module_ok("System-Tools", default=False) or web_search_engine.enabled
    memory_backend = "redis" if memory_store.redis_available else "local_file"
    monitoring_ok = bool(telemetry._stats_thread.is_alive())

    modules = {
        "core_reasoning": {
            "online": llm_status["state"] == LLMConnectionState.READY.value,
            "detail": (
                f"source={llm_status.get('source', 'none')} "
                f"state={llm_status.get('state', 'Disconnected')} "
                f"model={llm_status.get('model', 'None')}"
            ),
        },
        "emotion": {
            "online": bool(emotion_net.enabled),
            "detail": f"state={telemetry.emotion.get('label', 'Neutral')}",
        },
        "voice": {
            "online": voice_ok,
            "detail": "stt+tts plugins active" if voice_ok else "voice plugin disabled",
        },
        "vision": {
            "online": vision_ok,
            "detail": "vision plugin active" if vision_ok else "vision plugin disabled",
        },
        "tools": {
            "online": tools_ok,
            "detail": "plugins/web-search available" if tools_ok else "tooling limited",
        },
        "memory": {
            "online": True,
            "detail": f"backend={memory_backend} profile={memory_store._profile_name}",
        },
        "personality": {
            "online": readiness.checks["personality"].ready,
            "detail": readiness.checks["personality"].detail,
        },
        "monitoring": {
            "online": monitoring_ok,
            "detail": f"ws_clients={len(ws_clients)}",
        },
    }

    overall_ready = readiness.ready and all(modules[key]["online"] for key in (
        "core_reasoning", "emotion", "voice", "vision",
        "memory", "personality", "monitoring",
    ))

    return {
        "overall_ready": overall_ready,
        "modules": modules,
    }


def _build_status_payload():
    conversation_manager.maybe_leave_cooldown()
    telemetry.state = conversation_manager.current_state
    snap = telemetry.snapshot()
    readiness, llm_status = _build_conversation_readiness(
        require_emotion=emotion_net.enabled
    )
    architecture = _build_architecture_status(readiness=readiness, llm_status=llm_status)
    runtime_status = runtime_status_manager.get_runtime_status()
    return {
        "state": conversation_manager.current_state,
        "version": "1.2.0-py",
        "uptime_s": round(time.perf_counter() - telemetry._epoch, 1),
        "ws_clients": len(ws_clients),
        "system": snap["system"],
        "llm": snap["llm"],
        "emotion": snap["emotion"],
        "router": snap["router"],
        "profile": {
            "character_name": profile.get("character_name", "Revia"),
            "response_style": profile.get("response_style", "Conversational"),
            "verbosity": profile.get("verbosity", "Normal"),
        },
        "llm_connection": llm_status,
        "conversation_readiness": readiness.to_dict(),
        "behavior": conversation_manager.behavior_snapshot(),
        "architecture": architecture,
        "runtime_status": runtime_status,
        "request_lifecycle": turn_manager.snapshot(),
    }


def _broadcast_runtime_status():
    payload = _build_status_payload()
    broadcast_json({
        "type": "status_update",
        "state": payload["state"],
        "status": payload,
    })
    return payload

# ---------------------------------------------------------------------------
# Flask REST API
# ---------------------------------------------------------------------------

app = Flask(__name__)
CORS(app)

@app.route("/api/status", methods=["GET"])
def api_status():
    return jsonify(_build_status_payload())


@app.route("/api/runtime/config", methods=["GET"])
def api_runtime_config_get():
    return jsonify(runtime_status_manager.get_runtime_status())


@app.route("/api/runtime/config", methods=["POST"])
def api_runtime_config_post():
    data = request.get_json(silent=True) or {}
    runtime_status_manager.update_runtime_config(data)
    payload = _build_status_payload()
    _broadcast_runtime_status()
    return jsonify({"ok": True, "runtime_status": payload.get("runtime_status", {})})

@app.route("/api/interrupt", methods=["POST"])
def api_interrupt():
    """Signal the LLM to stop generating tokens and cancel queued TTS."""
    llm_backend.request_interrupt()
    broadcast_json({"type": "interrupt_ack", "interrupted": True})
    _revia_log("LLM generation interrupted via /api/interrupt")
    return jsonify({"ok": True, "interrupted": True})

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    image_b64 = data.get("image")
    vision_context = data.get("vision_context")
    source = str(data.get("source") or TriggerSource.USER_MESSAGE.value)
    reason = str(data.get("reason") or "direct user response")
    if not text:
        return jsonify({"error": "empty message"}), 400
    trigger = TriggerRequest(
        source=source,
        kind=TriggerKind.RESPONSE.value,
        reason=reason,
        text=text,
        require_emotion=emotion_net.enabled,
    )
    conversation_manager.maybe_leave_cooldown()
    readiness, _ = _build_conversation_readiness(require_emotion=emotion_net.enabled)
    decision = conversation_manager.behavior.evaluate(
        trigger,
        readiness,
        conversation_manager.current_state,
    )
    if not decision.allowed:
        return jsonify({
            "error": "conversation_not_ready",
            "decision": decision.to_dict(),
            "conversation_readiness": readiness.to_dict(),
            "state": conversation_manager.current_state,
        }), 503
    requested_mode = (
        ResponseMode.SYSTEM_STATUS_RESPONSE.value
        if runtime_status_manager.is_status_question(text)
        else ResponseMode.NORMAL_RESPONSE.value
    )
    turn = turn_manager.start_turn(
        source=source,
        user_text=text,
        response_mode=requested_mode,
        metadata={"reason": reason},
    )
    _mark_user_activity()
    threading.Thread(
        target=process_pipeline,
        args=(text, image_b64, vision_context, trigger, turn),
        daemon=True,
    ).start()
    return jsonify({
        "status": "processing",
        "text": text,
        "request_id": turn.request_id,
        "turn_id": turn.turn_id,
    })

@app.route("/api/model/config", methods=["GET"])
def api_model_config_get():
    return jsonify(llm_backend.get_config())

@app.route("/api/model/config", methods=["POST"])
def api_model_config_set():
    data = request.get_json(silent=True) or {}
    llm_backend.configure(data)
    print(f"[Core] Model config updated via REST: {llm_backend.source} / {llm_backend.api_model or llm_backend.local_path}")
    _broadcast_runtime_status()
    return jsonify({"ok": True, "config": llm_backend.get_config()})

# ---------------------------------------------------------------------------
# vLLM enhanced inference endpoints (PRD §18)
# ---------------------------------------------------------------------------

@app.route("/api/vllm/status", methods=["GET"])
def api_vllm_status():
    base_url = llm_backend.local_server_url.rstrip("/")
    is_vllm = llm_backend._vllm.probe(base_url)
    cuda_active = llm_backend.local_backend.upper() in ("CUDA", "GPU")
    result = {
        "enabled": llm_backend._vllm_enabled,
        "is_vllm_server": is_vllm,
        "cuda_active": cuda_active,
        "routing_mode": "smart",
        "routing_note": (
            "vLLM only activates for complex/long-context prompts when CUDA is available"
            if llm_backend._vllm_enabled else "vLLM enhanced mode disabled"
        ),
        "capabilities": llm_backend._vllm.get_capabilities(base_url) if is_vllm else {},
        "last_metrics": llm_backend._vllm.get_metrics_dict(),
    }
    if is_vllm:
        result["health"] = llm_backend._vllm.health_check(base_url)
        result["active_lora"] = llm_backend._vllm.get_active_lora(base_url)
    return jsonify(result)

@app.route("/api/vllm/lora", methods=["POST"])
def api_vllm_lora():
    data = request.get_json(silent=True) or {}
    adapter = data.get("adapter", "")
    base_url = llm_backend.local_server_url.rstrip("/")
    ok = llm_backend._vllm.set_lora_adapter(base_url, adapter)
    return jsonify({"ok": ok, "active_lora": llm_backend._vllm.get_active_lora(base_url)})

@app.route("/api/vllm/invalidate", methods=["POST"])
def api_vllm_invalidate():
    llm_backend._vllm.invalidate_cache()
    return jsonify({"ok": True})


@app.route("/api/plugins", methods=["GET"])
def api_plugins():
    return jsonify(PLUGINS)

@app.route("/api/plugins/<name>/enable", methods=["POST"])
def api_plugin_enable(name):
    for p in PLUGINS:
        if p["name"] == name:
            p["enabled"] = True
            return jsonify({"ok": True})
    return jsonify({"error": "not found"}), 404

@app.route("/api/plugins/<name>/disable", methods=["POST"])
def api_plugin_disable(name):
    for p in PLUGINS:
        if p["name"] == name:
            p["enabled"] = False
            return jsonify({"ok": True})
    return jsonify({"error": "not found"}), 404

@app.route("/api/neural", methods=["GET"])
def api_neural():
    return jsonify({
        "emotion_net": {
            "enabled": emotion_net.enabled,
            "last_inference_ms": emotion_net.last_inference_ms,
            "last_output": emotion_net.last_output,
        },
        "router_classifier": {
            "enabled": router_cls.enabled,
            "last_inference_ms": router_cls.last_inference_ms,
            "last_output": router_cls.last_output,
        },
    })

@app.route("/api/neural/<name>/enable", methods=["POST"])
def api_neural_enable(name):
    mod = {"emotion_net": emotion_net, "router_classifier": router_cls}.get(name)
    if mod:
        mod.enabled = True
        return jsonify({"ok": True})
    return jsonify({"error": "not found"}), 404

@app.route("/api/neural/<name>/disable", methods=["POST"])
def api_neural_disable(name):
    mod = {"emotion_net": emotion_net, "router_classifier": router_cls}.get(name)
    if mod:
        mod.enabled = False
        return jsonify({"ok": True})
    return jsonify({"error": "not found"}), 404

@app.route("/api/profile", methods=["GET"])
def api_profile_get():
    return jsonify(profile)

@app.route("/api/profile", methods=["POST"])
def api_profile_save():
    data = request.get_json(silent=True) or {}
    profile.update(data)
    # Keep ProfileEngine in sync so behavioral params update immediately
    profile_engine.load(profile)
    _save_profile_to_disk()

    # Build full system prompt from profile fields
    llm_backend.system_prompt = _build_system_prompt(profile)

    # Switch memory if profile name changed
    _sync_memory_profile_from_profile(clear_conversation=True)

    _broadcast_runtime_status()
    return jsonify({"ok": True})


def _build_system_prompt(prof):
    """Assemble the cached base character prompt from the active profile."""
    return character_profile_manager.build_character_context(
        prof,
        include_greeting_instruction=False,
    )


llm_backend.system_prompt = _build_system_prompt(profile)

@app.route("/api/memory/short", methods=["GET"])
def api_memory_short():
    limit = request.args.get("limit", 50, type=int)
    return jsonify(memory_store.get_short_term(limit))

@app.route("/api/memory/long", methods=["GET"])
def api_memory_long():
    limit = request.args.get("limit", 50, type=int)
    category = (request.args.get("category", "", type=str) or "").strip()
    return jsonify(memory_store.get_long_term(limit=limit, category=category or None))

@app.route("/api/memory/long/search", methods=["GET"])
def api_memory_long_search():
    q = request.args.get("q", "")
    limit = request.args.get("limit", 5, type=int)
    return jsonify(memory_store.search(q, limit))

@app.route("/api/memory/long/save", methods=["POST"])
def api_memory_long_save():
    data = request.get_json(silent=True) or {}
    memory_store.save_to_long_term(
        data.get("content", ""),
        data.get("category", "user_note"),
        data.get("metadata"),
    )
    return jsonify({"ok": True})


@app.route("/api/memory/long/delete", methods=["POST"])
def api_memory_long_delete():
    data = request.get_json(silent=True) or {}
    entry_id = str(data.get("id", "")).strip()
    if not entry_id:
        return jsonify({"error": "missing id"}), 400
    deleted = memory_store.delete_long_term_entry(entry_id)
    if not deleted:
        return jsonify({"ok": False, "error": "not found"}), 404
    return jsonify({"ok": True, "deleted": True})

@app.route("/api/memory/stats", methods=["GET"])
def api_memory_stats():
    stats = memory_store.get_long_term_stats()
    stats["profile"] = memory_store._profile_name
    return jsonify(stats)

@app.route("/api/memory/short/clear", methods=["POST"])
def api_memory_short_clear():
    memory_store.clear_short_term()
    return jsonify({"ok": True})

@app.route("/api/memory/long/clear", methods=["POST"])
def api_memory_long_clear():
    memory_store.clear_long_term()
    return jsonify({"ok": True})

@app.route("/api/memory/docker/status", methods=["GET"])
def api_memory_docker_status():
    """Return Redis/Docker connection status for the memory backend."""
    redis_ok = memory_store.redis_available
    local_file = str(memory_store._lt_file.resolve())
    return jsonify({
        "redis_available": redis_ok,
        "memory_online": True,
        "redis_host": os.environ.get("REDIS_HOST", "127.0.0.1"),
        "redis_port": int(os.environ.get("REDIS_PORT", "6379")),
        "long_term_backend": "redis" if redis_ok else "local_file",
        "status": "redis" if redis_ok else "local_file_fallback",
        "local_file": local_file,
        "profile": memory_store._profile_name,
        "long_term_count": len(memory_store.long_term),
    })

@app.route("/api/emotions/history", methods=["GET"])
def api_emotions_history():
    """Return the recent emotion history ring buffer."""
    limit = request.args.get("limit", 50, type=int)
    with _emotion_history_lock:
        return jsonify(_emotion_history[-limit:])

@app.route("/api/emotions/current", methods=["GET"])
def api_emotions_current():
    """Return the current emotion state from telemetry."""
    with telemetry._lock:
        emo = dict(telemetry.emotion)
    return jsonify(emo)


# ---------------------------------------------------------------------------
# Proactive conversation starter
# ---------------------------------------------------------------------------

_PROACTIVE_PROMPTS = [
    "Start a natural conversation without waiting for user input. "
    "You can share a curious fact, ask an engaging question, make a warm observation, "
    "or simply check in. Keep it brief (1-2 sentences) and fully in character.",

    "Initiate a conversation naturally. Think of something genuinely interesting to say -- "
    "a question, a thought, or just a friendly hello. Be brief and spontaneous.",

    "Begin a conversation. Say something warm, curious, or playful -- whatever feels "
    "right in the moment. One or two sentences maximum.",

    "Conversation has been quiet. Make a spontaneous in-character comment about what "
    "might be happening (coding, debugging, planning, or system activity), then ask "
    "one curious follow-up question.",

    "Initiate with a light opinion or playful observation, then invite the user to "
    "continue. Keep it short, social, and character-consistent.",
]

def _run_proactive_pipeline(trigger):
    """Generate a proactive Revia message without a user-side message in the UI."""
    import random as _random

    response_mode = (
        ResponseMode.STARTUP_RESPONSE.value
        if str(trigger.source) == TriggerSource.STARTUP.value
        else ResponseMode.NORMAL_RESPONSE.value
    )
    turn = turn_manager.start_turn(
        source=trigger.source,
        user_text="",
        response_mode=response_mode,
        metadata={"trigger_reason": trigger.reason, "proactive": True},
    )
    _revia_log(
        f"Trigger received | request_id={turn.request_id} | turn_id={turn.turn_id} "
        f"| source={trigger.source} | reason={trigger.reason}"
    )
    lock_acquired = False
    if not llm_backend._generate_lock.acquire(blocking=False):
        _revia_log("Blocked: LLM generation already in progress")
        turn_manager.finish_turn(
            turn.request_id,
            lifecycle_state=RequestLifecycleState.ERROR,
            reason="generation_lock_busy",
        )
        return
    lock_acquired = True

    prompt = _random.choice(_PROACTIVE_PROMPTS)
    token_started = False
    response = AssistantResponse(text="", success=False, commit_to_history=False, commit_to_memory=False)

    def _pipeline_broadcast(payload):
        nonlocal token_started
        if not turn_manager.is_current(turn.request_id):
            _revia_log(
                f"Stale proactive response discarded | request_id={turn.request_id} | type={payload.get('type', '')}"
            )
            return
        out = dict(payload or {})
        out.setdefault("request_id", turn.request_id)
        out.setdefault("turn_id", turn.turn_id)
        if payload.get("type") == "chat_token" and not token_started:
            token_started = True
            turn_manager.mark_state(turn.request_id, RequestLifecycleState.SPEAKING, "first proactive token")
            _set_runtime_state(ReviaState.SPEAKING, f"{trigger.source}: first token", broadcast=True)
        broadcast_json(out)

    _set_runtime_state(ReviaState.THINKING, f"{trigger.source}: {trigger.reason}", broadcast=True)
    broadcast_json({"type": "proactive_start"})

    try:
        if str(trigger.source) == TriggerSource.STARTUP.value:
            response = _build_profile_greeting_reply(startup=True)
        else:
            with llm_backend._lock:
                source = llm_backend.source
            _revia_log(f"Model call started | request_id={turn.request_id} | source={trigger.source}")
            if source == "online" and llm_backend.api_key:
                pending_conversation = llm_backend._trim_conversation(
                    list(llm_backend.conversation) + [{"role": "user", "content": prompt}]
                )
                messages = llm_backend._build_messages(
                    pending_conversation,
                    prompt,
                    ResponseMode.NORMAL_RESPONSE.value,
                )
                response = llm_backend._generate_online(messages, _pipeline_broadcast)
            elif source == "local" and (llm_backend.local_path or llm_backend.local_server_url):
                pending_conversation = llm_backend._trim_conversation(
                    list(llm_backend.conversation) + [{"role": "user", "content": prompt}]
                )
                response = llm_backend._generate_local(
                    pending_conversation,
                    prompt,
                    _pipeline_broadcast,
                    response_mode=ResponseMode.NORMAL_RESPONSE.value,
                )
            else:
                response = llm_backend._generate_stub(prompt)
            _revia_log(
                f"Model call completed | request_id={turn.request_id} | success={response.success} "
                f"| mode={response.response_mode}"
            )
    finally:
        if lock_acquired:
            llm_backend._generate_lock.release()

    if turn_manager.should_block_duplicate_output("", response.text, response.response_mode):
        response = _build_duplicate_block_response(response)

    filtered = conversation_manager.response_filter.apply(
        response.text,
        trigger,
        emotion_label=telemetry.emotion.get("label", "Neutral"),
    )
    response.text = filtered.text if filtered.text else response.text
    response.speakable = bool(response.speakable and filtered.speakable and response.text)

    if response.success and response.text:
        with llm_backend._lock:
            llm_backend.conversation.append({"role": "assistant", "content": response.text})
            llm_backend.conversation = llm_backend._trim_conversation(llm_backend.conversation)
        turn_manager.remember_committed_output("", response.text, response.response_mode)

    if response.commit_to_memory and response.text:
        memory_store.add_short_term("assistant", response.text)

    if response.text:
        if not token_started:
            turn_manager.mark_state(turn.request_id, RequestLifecycleState.SPEAKING, "proactive response deliver")
            _set_runtime_state(ReviaState.SPEAKING, f"{trigger.source}: response deliver", broadcast=True)
        _pipeline_broadcast(response.to_payload(turn.request_id, turn.turn_id))

    conversation_manager.behavior.start_cooldown("autonomous")
    _set_runtime_state(ReviaState.COOLDOWN, f"{trigger.source}: output delivered", force=True, broadcast=True)
    turn_manager.finish_turn(
        turn.request_id,
        lifecycle_state=(RequestLifecycleState.IDLE if response.success else RequestLifecycleState.ERROR),
        reason=response.error_type or response.response_mode,
    )
    payload = _build_status_payload()
    broadcast_json({"type": "telemetry_update", "data": payload})
    conversation_manager.maybe_leave_cooldown()
    telemetry.state = conversation_manager.current_state
    if conversation_manager.current_state == ReviaState.IDLE.value:
        _broadcast_runtime_status()

@app.route("/api/proactive", methods=["POST"])
def api_proactive():
    """Trigger Revia to start a conversation without user input."""
    data = request.get_json(silent=True) or {}
    force = bool(data.get("force", False))
    source = str(data.get("source") or TriggerSource.IDLE_TIMER.value)
    reason = str(data.get("reason") or "autonomous conversation")
    trigger = TriggerRequest(
        source=source,
        kind=TriggerKind.AUTONOMOUS.value,
        reason=reason,
        force=force,
        require_emotion=emotion_net.enabled,
        metadata={
            "recent_user_activity_s": _seconds_since_user_activity(),
        },
    )
    conversation_manager.maybe_leave_cooldown()
    readiness, _ = _build_conversation_readiness(require_emotion=emotion_net.enabled)
    decision = conversation_manager.behavior.evaluate(
        trigger,
        readiness,
        conversation_manager.current_state,
    )
    if not decision.allowed:
        return jsonify({
            "ok": False,
            "decision": decision.to_dict(),
            "conversation_readiness": readiness.to_dict(),
            "state": conversation_manager.current_state,
        }), 409
    threading.Thread(target=_run_proactive_pipeline, args=(trigger,), daemon=True).start()
    return jsonify({"status": "proactive triggered", "force": force})

# ---------------------------------------------------------------------------
# Web Search API
# ---------------------------------------------------------------------------

@app.route("/api/websearch/status", methods=["GET"])
def api_websearch_status():
    return jsonify({
        "enabled": web_search_engine.enabled,
        "backend": web_search_engine.backend,
        "ddg_available": web_search_engine._ddg_available,
    })


@app.route("/api/websearch/enable", methods=["POST"])
def api_websearch_enable():
    web_search_engine.enabled = True
    print("[WebSearch] Internet access ENABLED")
    return jsonify({"ok": True, "enabled": True})


@app.route("/api/websearch/disable", methods=["POST"])
def api_websearch_disable():
    web_search_engine.enabled = False
    print("[WebSearch] Internet access DISABLED")
    return jsonify({"ok": True, "enabled": False})


@app.route("/api/websearch/query", methods=["POST"])
def api_websearch_query():
    """Direct web search endpoint for testing or external callers."""
    if not web_search_engine.enabled:
        return jsonify({"error": "Web search is disabled"}), 403
    data = request.get_json(silent=True) or {}
    q = data.get("q", "").strip()
    if not q:
        return jsonify({"error": "missing q"}), 400
    results = web_search_engine.search(q, max_results=data.get("max_results", 4))
    return jsonify({"query": q, "results": results})


# ---------------------------------------------------------------------------
# Integration API (Discord + Twitch)
# ---------------------------------------------------------------------------

@app.route("/api/integrations/status", methods=["GET"])
def api_integrations_status():
    if integration_manager is None:
        return jsonify({"error": "Integration manager unavailable"}), 503
    return jsonify(integration_manager.get_status())


@app.route("/api/integrations/config", methods=["GET"])
def api_integrations_config_get():
    if integration_manager is None:
        return jsonify({"error": "Integration manager unavailable"}), 503
    return jsonify(integration_manager.get_config())


@app.route("/api/integrations/config", methods=["POST"])
def api_integrations_config_set():
    if integration_manager is None:
        return jsonify({"error": "Integration manager unavailable"}), 503
    data = request.get_json(silent=True) or {}
    integration_manager.update_config(data)
    return jsonify({"ok": True, "config": integration_manager.get_config()})


@app.route("/api/integrations/discord/start", methods=["POST"])
def api_discord_start():
    if integration_manager is None:
        return jsonify({"error": "Integration manager unavailable"}), 503
    integration_manager.start_discord()
    return jsonify({"ok": True, "status": integration_manager.get_status()["discord"]})


@app.route("/api/integrations/discord/stop", methods=["POST"])
def api_discord_stop():
    if integration_manager is None:
        return jsonify({"error": "Integration manager unavailable"}), 503
    integration_manager.stop_discord()
    return jsonify({"ok": True})


@app.route("/api/integrations/twitch/start", methods=["POST"])
def api_twitch_start():
    if integration_manager is None:
        return jsonify({"error": "Integration manager unavailable"}), 503
    integration_manager.start_twitch()
    return jsonify({"ok": True, "status": integration_manager.get_status()["twitch"]})


@app.route("/api/integrations/twitch/stop", methods=["POST"])
def api_twitch_stop():
    if integration_manager is None:
        return jsonify({"error": "Integration manager unavailable"}), 503
    integration_manager.stop_twitch()
    return jsonify({"ok": True})

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

REST_PORT = int(os.environ.get("REVIA_REST_PORT", "8123"))
WS_PORT = int(os.environ.get("REVIA_WS_PORT", "8124"))

async def start_ws_server():
    global ws_loop
    ws_loop = asyncio.get_event_loop()
    async with websockets.server.serve(ws_handler, "0.0.0.0", WS_PORT):
        print(f"[REVIA Core] WebSocket server on ws://0.0.0.0:{WS_PORT}")
        await asyncio.Future()

def run_ws():
    asyncio.run(start_ws_server())

def main():
    print("=" * 50)
    print("  REVIA Core (Python)  v1.2.0")
    print("=" * 50)
    conversation_manager.mark_initializing("Starting core services")
    telemetry.state = conversation_manager.current_state
    ws_thread = threading.Thread(target=run_ws, daemon=True)
    ws_thread.start()
    time.sleep(0.3)

    # Start any enabled platform integrations (Discord / Twitch)
    if integration_manager is not None:
        integration_manager.start_enabled()

    conversation_manager.mark_startup_complete("Core services online")
    telemetry.state = conversation_manager.current_state
    print(f"[REVIA Core] REST server on http://0.0.0.0:{REST_PORT}")
    print(f"[REVIA Core] Ready. Open the controller and click 'Connect'.")
    print()
    app.run(host="0.0.0.0", port=REST_PORT, debug=False, use_reloader=False)

if __name__ == "__main__":
    main()
