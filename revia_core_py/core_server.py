"""
REVIA Core -- Pure-Python standalone server.
Drop-in replacement for the C++ core. Same REST + WebSocket API.

Usage:
    python core_server.py
    (REST on :8123, WebSocket on :8124)
"""

import json, time, random, threading, asyncio, os, subprocess
from datetime import datetime
from pathlib import Path

import psutil
from flask import Flask, request, jsonify
from flask_cors import CORS

import websockets
import websockets.server


def _get_gpu_stats():
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            timeout=3, stderr=subprocess.DEVNULL,
        ).decode().strip().split(",")
        return {
            "gpu_percent": float(out[0]),
            "vram_used_mb": float(out[1]),
            "vram_total_mb": float(out[2]),
        }
    except Exception:
        return {"gpu_percent": 0.0, "vram_used_mb": 0.0, "vram_total_mb": 0.0}


def _get_system_stats():
    mem = psutil.virtual_memory()
    gpu = _get_gpu_stats()
    return {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "gpu_percent": gpu["gpu_percent"],
        "ram_used_mb": round(mem.used / (1024 * 1024)),
        "ram_total_mb": round(mem.total / (1024 * 1024)),
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
        self.spans = []
        self.llm = {"tokens_generated": 0, "tokens_per_second": 0.0, "context_length": 0}
        self.emotion = {
            "valence": 0.0, "arousal": 0.0, "dominance": 0.0,
            "label": "Neutral", "confidence": 0.0, "inference_ms": 0.0,
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
        self._log = open(fname, "a", encoding="utf-8")

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
        psutil.cpu_percent(interval=None)
        while True:
            time.sleep(2)
            self._refresh_system_stats()

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
            self.spans.append(span)
            if len(self.spans) > 500:
                self.spans = self.spans[-400:]
            self._log.write(json.dumps(span) + "\n")
            self._log.flush()

    def snapshot(self):
        with self._lock:
            recent = self.spans[-20:] if self.spans else []
            return {
                "state": self.state,
                "llm": dict(self.llm),
                "emotion": dict(self.emotion),
                "router": dict(self.router),
                "system": dict(self.system),
                "recent_spans": list(recent),
            }


telemetry = TelemetryEngine()

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
        self._lock = threading.Lock()
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
        self.system_prompt = "You are REVIA, a smart and friendly AI assistant."
        self.conversation = []

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
            sp = cfg.get("system_prompt", "")
            if sp:
                self.system_prompt = sp

            model_name = self.api_model or os.path.basename(self.local_path) or "None"
            telemetry.system["model"] = model_name
            if self.source == "online":
                telemetry.system["backend"] = self.api_provider or "API"
                telemetry.system["device"] = "Cloud"
            else:
                telemetry.system["backend"] = self.local_server
                telemetry.system["device"] = self.local_backend

    def get_config(self):
        with self._lock:
            return {
                "source": self.source, "local_path": self.local_path,
                "local_server": self.local_server,
                "local_server_url": self.local_server_url,
                "api_provider": self.api_provider, "api_model": self.api_model,
                "api_endpoint": self.api_endpoint,
                "temperature": self.temperature, "max_tokens": self.max_tokens,
            }

    def _build_messages(self, image_b64=None):
        """Build the full message list with system prompt + memory + conversation."""
        sys_content = self.system_prompt

        mem_ctx = memory_store.get_context_for_llm()
        if mem_ctx:
            sys_content += "\n\n--- Memory Context ---\n" + mem_ctx

        messages = [{"role": "system", "content": sys_content}]

        # Add conversation history (all but last user msg)
        for msg in self.conversation[:-1]:
            messages.append(msg)

        # Last user message -- attach image if provided
        if self.conversation:
            last = self.conversation[-1]
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

    def generate_streaming(self, text, broadcast_fn, image_b64=None):
        with self._lock:
            source = self.source

        self.conversation.append({"role": "user", "content": text})
        if len(self.conversation) > 40:
            self.conversation = self.conversation[-30:]

        if source == "online" and self.api_key:
            return self._generate_online(broadcast_fn, image_b64=image_b64)
        elif source == "local" and (self.local_path or self.local_server_url):
            return self._generate_local(text, broadcast_fn, image_b64=image_b64)
        else:
            return self._generate_stub(text, broadcast_fn)

    def _generate_online(self, broadcast_fn, image_b64=None):
        import requests as req
        provider = self.api_provider.lower()
        endpoint = self.api_endpoint.rstrip("/")
        messages = self._build_messages(image_b64=image_b64)

        try:
            if "anthropic" in provider:
                return self._call_anthropic(endpoint, messages, broadcast_fn)
            else:
                return self._call_openai_compat(endpoint, messages, broadcast_fn)
        except Exception as e:
            err = f"[LLM Error: {e}]"
            broadcast_fn({"type": "chat_token", "token": err})
            self.conversation.append({"role": "assistant", "content": err})
            return err

    def _call_openai_compat(self, endpoint, messages, broadcast_fn):
        import requests as req
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.api_model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stream": True,
        }
        url = endpoint + "/chat/completions"
        full_text = ""
        token_count = 0
        t0 = time.perf_counter()

        with req.post(url, headers=headers, json=body, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    tok = delta.get("content", "")
                    if tok:
                        full_text += tok
                        token_count += 1
                        broadcast_fn({"type": "chat_token", "token": tok})
                except (json.JSONDecodeError, IndexError, KeyError):
                    continue

        elapsed = time.perf_counter() - t0
        tps = token_count / elapsed if elapsed > 0 else 0
        telemetry.llm["tokens_generated"] = token_count
        telemetry.llm["tokens_per_second"] = round(tps, 1)
        telemetry.llm["context_length"] = sum(len(m["content"].split()) for m in messages) + token_count
        self.conversation.append({"role": "assistant", "content": full_text})
        return full_text

    def _call_anthropic(self, endpoint, messages, broadcast_fn):
        import requests as req
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
            "max_tokens": self.max_tokens,
            "messages": conv,
            "stream": True,
        }
        if sys_msg:
            body["system"] = sys_msg
        url = endpoint + "/messages"
        full_text = ""
        token_count = 0
        t0 = time.perf_counter()

        with req.post(url, headers=headers, json=body, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                try:
                    chunk = json.loads(line[6:])
                    if chunk.get("type") == "content_block_delta":
                        tok = chunk.get("delta", {}).get("text", "")
                        if tok:
                            full_text += tok
                            token_count += 1
                            broadcast_fn({"type": "chat_token", "token": tok})
                except (json.JSONDecodeError, KeyError):
                    continue

        elapsed = time.perf_counter() - t0
        tps = token_count / elapsed if elapsed > 0 else 0
        telemetry.llm["tokens_generated"] = token_count
        telemetry.llm["tokens_per_second"] = round(tps, 1)
        self.conversation.append({"role": "assistant", "content": full_text})
        return full_text

    def _discover_model_name(self, base_url):
        """Ask the local server for its loaded model name."""
        import requests as req
        try:
            r = req.get(base_url.rstrip("/") + "/models", timeout=3)
            if r.ok:
                data = r.json().get("data", [])
                if data:
                    return data[0].get("id", "")
        except Exception:
            pass
        return ""

    def _generate_local(self, text, broadcast_fn, image_b64=None):
        import requests as req
        base_url = self.local_server_url.rstrip("/")
        url = base_url + "/chat/completions"
        server_name = self.local_server

        try:
            messages = self._build_messages(image_b64=image_b64)

            model_name = os.path.basename(self.local_path) if self.local_path else ""
            if not model_name:
                model_name = self._discover_model_name(base_url)

            body = {
                "model": model_name or "default",
                "messages": messages,
                "temperature": self.temperature,
                "stream": True,
            }

            full_text = ""
            token_count = 0
            t0 = time.perf_counter()

            try:
                resp = req.post(url, json=body, stream=True, timeout=120)
                resp.raise_for_status()
            except req.exceptions.HTTPError as e:
                if e.response and e.response.status_code == 500 and image_b64:
                    # Model doesn't support vision -- retry without image
                    messages = self._build_messages(image_b64=None)
                    for m in messages:
                        if m.get("role") == "user" and isinstance(m.get("content"), str):
                            if not m["content"].strip().startswith("["):
                                m["content"] += " [Note: An image was attached but this model doesn't support vision.]"
                                break
                    body["messages"] = messages
                    resp = req.post(url, json=body, stream=True, timeout=120)
                    resp.raise_for_status()
                else:
                    raise

            with resp:
                for line in resp.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        tok = delta.get("content", "")
                        if tok:
                            full_text += tok
                            token_count += 1
                            broadcast_fn({"type": "chat_token", "token": tok})
                    except (json.JSONDecodeError, IndexError, KeyError):
                        continue

            elapsed = time.perf_counter() - t0
            tps = token_count / elapsed if elapsed > 0 else 0
            telemetry.llm["tokens_generated"] = token_count
            telemetry.llm["tokens_per_second"] = round(tps, 1)
            if model_name:
                telemetry.system["model"] = model_name
            self.conversation.append({"role": "assistant", "content": full_text})
            return full_text
        except Exception as e:
            short_url = base_url.replace("http://", "")
            err = (
                f"[Local LLM Error] Cannot reach {server_name} at {short_url}\n"
                f"  Error: {e}\n"
                f"  Make sure {server_name} is running with a model loaded."
            )
            broadcast_fn({"type": "chat_token", "token": err})
            self.conversation.append({"role": "assistant", "content": err})
            return err

    def _generate_stub(self, text, broadcast_fn):
        response = (
            "No LLM connected. Go to the Model tab, select a local model file "
            "or an online API provider, enter your settings, and click Connect. "
            "Then your messages will be routed to the real LLM."
        )
        tokens = response.split()
        full = ""
        t0 = time.perf_counter()
        for tok in tokens:
            time.sleep(random.uniform(0.02, 0.05))
            word = tok + " "
            full += word
            broadcast_fn({"type": "chat_token", "token": word})
        elapsed = time.perf_counter() - t0
        tps = len(tokens) / elapsed if elapsed > 0 else 0
        telemetry.llm["tokens_generated"] = len(tokens)
        telemetry.llm["tokens_per_second"] = round(tps, 1)
        self.conversation.append({"role": "assistant", "content": full.strip()})
        return full.strip()


llm_backend = LLMBackend()

# ---------------------------------------------------------------------------
# Neural modules (keyword stubs with realistic timing)
# ---------------------------------------------------------------------------

class EmotionNet:
    def __init__(self):
        self.enabled = True
        self.last_inference_ms = 0.0
        self.last_output = "Neutral"

    def infer(self, text):
        if not self.enabled:
            return {"label": "Disabled", "confidence": 0, "valence": 0,
                    "arousal": 0, "dominance": 0, "inference_ms": 0}
        t0 = time.perf_counter()
        low = text.lower()
        if any(w in low for w in ("happy", "great", "awesome", "love", "thank")):
            r = {"valence": 0.8, "arousal": 0.6, "dominance": 0.5, "label": "Happy", "confidence": 0.82}
        elif any(w in low for w in ("angry", "hate", "furious")):
            r = {"valence": -0.7, "arousal": 0.8, "dominance": 0.7, "label": "Angry", "confidence": 0.78}
        elif any(w in low for w in ("sad", "depressed", "sorry")):
            r = {"valence": -0.6, "arousal": 0.3, "dominance": 0.2, "label": "Sad", "confidence": 0.75}
        elif "?" in low:
            r = {"valence": 0.1, "arousal": 0.4, "dominance": 0.3, "label": "Curious", "confidence": 0.68}
        else:
            r = {"valence": 0.0, "arousal": 0.2, "dominance": 0.4, "label": "Neutral", "confidence": 0.90}
        time.sleep(random.uniform(0.004, 0.012))
        ms = (time.perf_counter() - t0) * 1000
        r["inference_ms"] = ms
        self.last_inference_ms = ms
        self.last_output = r["label"]
        return r


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
        if any(w in low for w in ("search", "find", "look up")):
            r = {"mode": "memory_query", "confidence": 0.85, "suggested_tool": "rag_search", "rag_enable": True}
        elif any(w in low for w in ("run", "execute", "open")):
            r = {"mode": "command", "confidence": 0.80, "suggested_tool": "system_exec", "rag_enable": False}
        elif any(w in low for w in ("see", "camera", "look at", "image")):
            r = {"mode": "vision_query", "confidence": 0.78, "suggested_tool": "vision_capture", "rag_enable": False}
        elif any(w in low for w in ("remember", "recall")):
            r = {"mode": "memory_query", "confidence": 0.82, "suggested_tool": "memory_recall", "rag_enable": True}
        else:
            r = {"mode": "chat", "confidence": 0.92, "suggested_tool": "", "rag_enable": False}
        time.sleep(random.uniform(0.003, 0.008))
        ms = (time.perf_counter() - t0) * 1000
        r["inference_ms"] = ms
        self.last_inference_ms = ms
        self.last_output = r["mode"]
        return r


emotion_net = EmotionNet()
router_cls = RouterClassifier()

# ---------------------------------------------------------------------------
# Memory system -- short-term (conversation) + long-term (persistent)
# ---------------------------------------------------------------------------

class MemoryStore:
    def __init__(self, profile_name="default"):
        self._lock = threading.Lock()
        self.short_term = []
        self.long_term = []
        self._profile_name = profile_name
        self._lt_file = Path(f"data/memory_{profile_name}.jsonl")
        self._lt_file.parent.mkdir(parents=True, exist_ok=True)
        self._load_long_term()

    def switch_profile(self, profile_name):
        """Switch to a different profile's memory."""
        with self._lock:
            self._profile_name = profile_name
            self.short_term.clear()
            self.long_term.clear()
            self._lt_file = Path(f"data/memory_{profile_name}.jsonl")
            self._lt_file.parent.mkdir(parents=True, exist_ok=True)
            self._load_long_term()

    def _load_long_term(self):
        if self._lt_file.exists():
            for line in self._lt_file.read_text(encoding="utf-8").splitlines():
                try:
                    self.long_term.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    def get_context_for_llm(self, max_short=10, max_long=5):
        """Build a memory context string to inject into the LLM prompt."""
        with self._lock:
            parts = []
            recent = self.short_term[-max_short:] if self.short_term else []
            if recent:
                lines = []
                for e in recent:
                    role = e.get("role", "?")
                    content = e.get("content", "")[:200]
                    lines.append(f"  [{role}]: {content}")
                parts.append("Recent conversation memory:\n" + "\n".join(lines))

            if self.long_term:
                notes = self.long_term[-max_long:]
                lines = []
                for e in notes:
                    content = e.get("content", "")[:200]
                    cat = e.get("category", "")
                    ts = e.get("timestamp", "")[:10]
                    lines.append(f"  [{cat} {ts}]: {content}")
                parts.append("Long-term memories:\n" + "\n".join(lines))

            return "\n\n".join(parts) if parts else ""

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
        entry["promoted"] = True
        self.long_term.append(entry)
        with open(self._lt_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def save_to_long_term(self, content, category="user_note", metadata=None):
        with self._lock:
            entry = {
                "content": content, "category": category,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {},
            }
            self.long_term.append(entry)
            with open(self._lt_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

    def search(self, query, max_results=5):
        query_lower = query.lower()
        results = []
        for entry in reversed(self.long_term):
            if query_lower in entry.get("content", "").lower():
                results.append(entry)
                if len(results) >= max_results:
                    break
        return results

    def get_short_term(self, limit=50):
        with self._lock:
            return list(self.short_term[-limit:])

    def get_long_term_stats(self):
        return {
            "short_term_count": len(self.short_term),
            "long_term_count": len(self.long_term),
        }

    def clear_short_term(self):
        with self._lock:
            self.short_term.clear()

    def clear_long_term(self):
        with self._lock:
            self.long_term.clear()
            if self._lt_file.exists():
                self._lt_file.write_text("", encoding="utf-8")


memory_store = MemoryStore()

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
    "persona": "A helpful neural assistant.",
    "traits": "friendly, curious, witty",
    "response_style": "Conversational",
    "verbosity": "Normal",
    "greeting": "Hello! I'm Revia, your neural assistant.",
    "character_prompt": "",
    "voice_id": "default", "tone": "calm", "speed": 1.0,
}

# ---------------------------------------------------------------------------
# WebSocket broadcast
# ---------------------------------------------------------------------------

ws_clients = set()
ws_loop = None

async def ws_handler(websocket):
    ws_clients.add(websocket)
    try:
        async for msg in websocket:
            try:
                data = json.loads(msg)
                if data.get("type") == "model_config":
                    llm_backend.configure(data.get("config", {}))
                    print(f"[Core] Model config updated: {llm_backend.source} / {llm_backend.api_model or llm_backend.local_path}")
            except Exception:
                pass
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        ws_clients.discard(websocket)


def broadcast_json(data):
    if not ws_clients or ws_loop is None:
        return
    text = json.dumps(data)
    asyncio.run_coroutine_threadsafe(_broadcast(text), ws_loop)

async def _broadcast(text):
    dead = set()
    for ws in ws_clients:
        try:
            await ws.send(text)
        except Exception:
            dead.add(ws)
    ws_clients.difference_update(dead)

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def process_pipeline(text, image_b64=None):
    telemetry.state = "Processing"
    broadcast_json({"type": "status_update", "state": "Processing"})

    s = telemetry.begin_span("input_capture")
    time.sleep(random.uniform(0.002, 0.005))
    telemetry.end_span(s)

    s = telemetry.begin_span("stt_decode")
    time.sleep(random.uniform(0.005, 0.010))
    telemetry.end_span(s)

    s = telemetry.begin_span("emotion_analysis")
    emo = emotion_net.infer(text)
    telemetry.emotion = emo
    telemetry.end_span(s)

    s = telemetry.begin_span("router_classify")
    route = router_cls.classify(text)
    telemetry.router = route
    telemetry.end_span(s)

    s = telemetry.begin_span("context_gather")
    time.sleep(random.uniform(0.008, 0.020))
    telemetry.end_span(s)

    s = telemetry.begin_span("llm_prefill")
    time.sleep(random.uniform(0.010, 0.030))
    telemetry.end_span(s)

    # Store user message in short-term memory
    meta = {"emotion": emo.get("label", "")}
    if image_b64:
        meta["has_image"] = True
    memory_store.add_short_term("user", text, meta)

    # LLM decode -- stream tokens via the configured backend
    s = telemetry.begin_span("llm_decode")
    full_text = llm_backend.generate_streaming(
        text, broadcast_json, image_b64=image_b64
    )
    telemetry.end_span(s)

    # Store assistant response in short-term memory
    memory_store.add_short_term("assistant", full_text)

    # Auto-save conversation summary to long-term every 20 messages
    if len(memory_store.short_term) % 20 == 0 and len(memory_store.short_term) > 0:
        summary = f"Conversation exchange: User said '{text[:100]}', "
        summary += f"Assistant replied '{full_text[:100]}'"
        memory_store.save_to_long_term(
            summary, category="auto_conversation",
            metadata={"emotion": emo.get("label", "")},
        )

    s = telemetry.begin_span("tts_encode")
    time.sleep(random.uniform(0.008, 0.020))
    telemetry.end_span(s)

    s = telemetry.begin_span("memory_write")
    time.sleep(random.uniform(0.003, 0.010))
    telemetry.end_span(s)

    s = telemetry.begin_span("output_deliver")
    broadcast_json({"type": "chat_complete", "text": full_text})
    telemetry.end_span(s)

    snap = telemetry.snapshot()
    broadcast_json({"type": "telemetry_update", "data": snap})

    telemetry.state = "Idle"
    broadcast_json({"type": "status_update", "state": "Idle"})
    return full_text

# ---------------------------------------------------------------------------
# Flask REST API
# ---------------------------------------------------------------------------

app = Flask(__name__)
CORS(app)

@app.route("/api/status", methods=["GET"])
def api_status():
    snap = telemetry.snapshot()
    return jsonify({
        "state": snap["state"],
        "version": "1.1.0-py",
        "uptime_s": round(time.perf_counter() - telemetry._epoch, 1),
        "ws_clients": len(ws_clients),
        "system": snap["system"],
        "llm": snap["llm"],
    })

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    image_b64 = data.get("image")
    if not text:
        return jsonify({"error": "empty message"}), 400
    threading.Thread(
        target=process_pipeline, args=(text, image_b64), daemon=True
    ).start()
    return jsonify({"status": "processing", "text": text})

@app.route("/api/model/config", methods=["GET"])
def api_model_config_get():
    return jsonify(llm_backend.get_config())

@app.route("/api/model/config", methods=["POST"])
def api_model_config_set():
    data = request.get_json(silent=True) or {}
    llm_backend.configure(data)
    print(f"[Core] Model config updated via REST: {llm_backend.source} / {llm_backend.api_model or llm_backend.local_path}")
    return jsonify({"ok": True, "config": llm_backend.get_config()})

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
    old_name = profile.get("character_name", "default")
    profile.update(data)

    # Build full system prompt from profile fields
    llm_backend.system_prompt = _build_system_prompt(profile)

    # Switch memory if profile name changed
    new_name = profile.get("character_name", "default")
    safe_name = "".join(c if c.isalnum() or c in "-_ " else "" for c in new_name).strip() or "default"
    if safe_name.lower() != memory_store._profile_name.lower():
        memory_store.switch_profile(safe_name)
        llm_backend.conversation.clear()

    return jsonify({"ok": True})


def _build_system_prompt(prof):
    """Assemble a system prompt from all profile fields."""
    name = prof.get("character_name", "REVIA")
    persona = prof.get("persona", "")
    traits = prof.get("traits", "")
    style = prof.get("response_style", "Conversational")
    verbosity = prof.get("verbosity", "Normal")
    greeting = prof.get("greeting", "")
    char_prompt = prof.get("character_prompt", "")

    parts = []

    if char_prompt:
        parts.append(char_prompt)
    else:
        parts.append(f"You are {name}, an AI assistant.")

    if persona:
        parts.append(f"Persona: {persona}")
    if traits:
        parts.append(f"Personality traits: {traits}")

    parts.append(f"Response style: {style}. Verbosity: {verbosity}.")

    if greeting:
        parts.append(
            f"When starting a new conversation, greet the user with "
            f"something like: \"{greeting}\""
        )

    parts.append(
        "Use your memory context (provided below the conversation) to "
        "recall past interactions and user preferences. Reference them "
        "naturally when relevant."
    )

    return "\n".join(parts)

@app.route("/api/memory/short", methods=["GET"])
def api_memory_short():
    limit = request.args.get("limit", 50, type=int)
    return jsonify(memory_store.get_short_term(limit))

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
    print("  REVIA Core (Python)  v1.1.0")
    print("=" * 50)
    ws_thread = threading.Thread(target=run_ws, daemon=True)
    ws_thread.start()
    time.sleep(0.3)
    print(f"[REVIA Core] REST server on http://0.0.0.0:{REST_PORT}")
    print(f"[REVIA Core] Ready. Open the controller and click 'Connect'.")
    print()
    app.run(host="0.0.0.0", port=REST_PORT, debug=False, use_reloader=False)

if __name__ == "__main__":
    main()
