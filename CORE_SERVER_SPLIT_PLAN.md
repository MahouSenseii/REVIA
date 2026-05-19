# core_server.py Split — Migration Plan

**Status:** Draft  
**Author:** Architecture Review  
**File audited:** `revia_core_py/core_server.py` — 7,517 lines, 80+ routes, 7 classes  
**Goal:** Decompose into `revia_core_py/server/` sub-package with focused modules, eliminating the `globals().get(...)` anti-pattern and unblocking every downstream refactor.

---

## 1. Why This Has to Come First

Every other fix on the audit list — memory store hot path, thread pool consolidation, lazy agent loading, unbounded lists — requires touching code that is **entangled in one file**. You cannot confidently refactor `MemoryStore`, `EmotionNet`, or the pipeline without first knowing which routes own them. The split creates stable seams.

---

## 2. Proposed Module Boundaries

```
revia_core_py/
├── server/
│   ├── __init__.py          # re-exports app, run_rest, run_ws — drop-in shim
│   ├── _container.py        # AppContainer: single source of all shared singletons
│   ├── tts_utils.py         # TTS sentence-chunking helpers (pure functions, no I/O)
│   ├── telemetry.py         # TelemetryEngine + GPU/system stats helpers
│   ├── llm_backend.py       # LLMBackend class + LOCAL_SERVERS + EmotionNet (for now)
│   ├── memory_store.py      # MemoryStore class + Redis init/reconnect threads
│   ├── emotion.py           # EmotionNet, NeuralRefiner wiring, emotion history ring
│   ├── pipeline.py          # process_pipeline_safe, process_pipeline, TurnWatchdog
│   ├── websocket.py         # ws_handler, _broadcast, broadcast_json, ws_clients
│   ├── routes/
│   │   ├── __init__.py      # registers all blueprints on the Flask app
│   │   ├── chat.py          # POST /api/chat, POST /api/interrupt
│   │   ├── model.py         # GET/POST /api/model/config, /api/vllm/*
│   │   ├── profile.py       # GET/POST /api/profile, /api/tts/output
│   │   ├── memory.py        # /api/memory/*, /api/emotions/*
│   │   ├── telemetry.py     # GET /api/status, /api/neural/*, /api/rl/*
│   │   ├── agents.py        # /api/agents/*, /api/hardware/*
│   │   ├── integrations.py  # /api/integrations/*, /api/plugins/*
│   │   ├── autonomy.py      # POST /api/proactive
│   │   └── lifecycle.py     # POST /api/shutdown, /api/runtime/config, /api/websearch/*
└── core_server.py           # KEPT as thin entry point: imports server, calls server.run()
```

### Module responsibilities at a glance

| Module | Owns | Does NOT own |
|--------|------|-------------|
| `_container.py` | All singleton creation + wiring | No Flask routes, no threads |
| `tts_utils.py` | `_extract_complete_tts_sentences`, `_tts_*` helpers | Nothing else |
| `telemetry.py` | `TelemetryEngine`, `_get_gpu_stats`, `_get_system_stats` | LLM backend, emotion |
| `llm_backend.py` | `LLMBackend`, `LOCAL_SERVERS`, `RouterClassifier`, `WebSearchEngine` | Memory, emotion, pipeline |
| `memory_store.py` | `MemoryStore`, Redis init/reconnect threads | LLM, emotion, routes |
| `emotion.py` | `EmotionNet`, `NeuralRefiner` wiring, `_emotion_history` ring, helpers | Pipeline, routes |
| `pipeline.py` | `process_pipeline`, `process_pipeline_safe`, `_run_proactive_pipeline` | Routes, WS |
| `websocket.py` | `ws_handler`, `_broadcast`, `broadcast_json`, `ws_clients` | Flask app |
| `routes/*.py` | Flask blueprints | Business logic (only calls container + pipeline) |

---

## 3. The Critical Fix: AppContainer

The `globals().get(...)` anti-pattern exists because six singletons are instantiated at module scope in sequence and reference each other. The fix is one container object that owns all of them.

```python
# server/_container.py

from __future__ import annotations
from pathlib import Path
from .telemetry import TelemetryEngine
from .llm_backend import LLMBackend
from .memory_store import MemoryStore
from .emotion import EmotionNet, build_neural_refiner
from conversation_runtime import ConversationManager
from profile_engine import ProfileEngine
from prompt_assembly import CharacterProfileManager, PromptAssemblyManager
from runtime_models import TurnManager
from reinforcement_learner import ReinforcementLearner
from human_feel_layer import HumanFeelLayer
from answer_validation import AnswerValidationSystem
import threading

class AppContainer:
    """Single source of truth for all shared singletons.

    Constructed once in server/__init__.py. Passed (or imported from here)
    wherever shared state is needed. Eliminates globals().get() entirely.
    """

    def __init__(self, data_dir: Path):
        self.telemetry           = TelemetryEngine()
        self.conversation        = ConversationManager(log_fn=self._log)
        self.character_profiles  = CharacterProfileManager(log_fn=self._log)
        self.prompt_assembly     = PromptAssemblyManager(
                                       log_fn=self._log,
                                       profile_manager=self.character_profiles)
        self.turn_manager        = TurnManager(log_fn=self._log)
        self.rl_engine           = ReinforcementLearner(data_dir=data_dir,
                                       log_fn=self._log)
        self.profile_engine      = ProfileEngine(log_fn=self._log)
        self.human_feel          = HumanFeelLayer(
                                       profile_engine=self.profile_engine)
        self.avs_engine          = AnswerValidationSystem(
                                       profile_engine=self.profile_engine)
        self.llm_backend         = LLMBackend(telemetry=self.telemetry,
                                       log_fn=self._log)
        self.memory_store        = MemoryStore(log_fn=self._log)
        self.emotion_net         = EmotionNet()
        self.neural_refiner      = build_neural_refiner()   # None if torch absent
        self.profile: dict       = {}                       # loaded in __init__ or route

        # Late-bound (set by server/__init__.py after WS loop is up)
        self.broadcast_json      = None   # callable | None

    def _log(self, message: str) -> None:
        """Centralized logger — no globals().get() needed."""
        line = f"[Revia] {message}"
        print(line)
        if callable(self.broadcast_json):
            try:
                self.broadcast_json({"type": "log_entry", "text": line})
            except Exception as e:
                print(f"[Revia] (log broadcast failed: {type(e).__name__}: {e})")

# Module-level singleton — imported by all other server/* modules
_container: AppContainer | None = None

def get_container() -> AppContainer:
    if _container is None:
        raise RuntimeError("AppContainer not initialized — call init_container() first")
    return _container

def init_container(data_dir: Path) -> AppContainer:
    global _container
    _container = AppContainer(data_dir)
    return _container
```

Every route and pipeline function that previously did `globals().get("profile")` now does:
```python
from server._container import get_container
container = get_container()
profile = container.profile
```

This is explicit, testable, and IDE-navigable.

---

## 4. Route Blueprints Pattern

Each `routes/*.py` file becomes a Flask Blueprint:

```python
# server/routes/memory.py
from flask import Blueprint, request, jsonify
from .._container import get_container

bp = Blueprint("memory", __name__, url_prefix="/api")

@bp.get("/memory/short")
def api_memory_short():
    store = get_container().memory_store
    return jsonify(store.get_short_term())

@bp.get("/memory/long")
def api_memory_long():
    ...
```

`routes/__init__.py` registers them all:
```python
from flask import Flask
from . import chat, model, profile, memory, telemetry_routes, agents, integrations, autonomy, lifecycle

def register_all(app: Flask) -> None:
    for module in [chat, model, profile, memory, telemetry_routes,
                   agents, integrations, autonomy, lifecycle]:
        app.register_blueprint(module.bp)
```

---

## 5. Migration Phases

### Phase 0 — Guard rails (1 day, zero behaviour change)
Before touching anything:
1. `pip install pytest` and write a smoke-test that starts the server in-process and hits `/api/status` — this is your regression fence.
2. Add type stubs / docstrings to the six most-called functions so moves are verifiable.
3. Commit baseline. Tag it `pre-split`.

### Phase 1 — Pure-function extractions (1–2 days, safe)
These have **no shared state** and can move with a simple cut-paste + import update:

| What | Source lines | Destination |
|------|-------------|-------------|
| `_tts_*` helpers, `_extract_complete_tts_sentences` | 37–162 | `server/tts_utils.py` |
| `_json_loads_fast`, `_json_dumps_compact` | 175–188 | `server/json_utils.py` |
| `_revia_log`, `_revia_log_throttled` | 773–820 | Move into AppContainer._log + _log_throttled |
| `RouterClassifier` | 2778–2813 | `server/llm_backend.py` |
| `WebSearchEngine` | 2842–2922 | `server/llm_backend.py` |

**Validation:** run smoke test after each move.

### Phase 2 — Telemetry module (½ day, minimal coupling)
Move `TelemetryEngine`, `_get_gpu_stats`, `_get_system_stats` into `server/telemetry.py`.  
`TelemetryEngine` currently reads `globals().get("profile")` inside `_get_gpu_stats` — inject `profile_getter: Callable[[], dict]` instead.

### Phase 3 — Emotion module (1 day, touch EmotionNet)
Move `EmotionNet`, `_neutral_emotion_state`, `_annotate_emotion_state`, `_infer_expression_emotion`, `_record_emotion`, `_emotion_history_snapshot` into `server/emotion.py`.  
`EmotionNet.infer()` calls `globals().get("neural_refiner")` at lines 2671 + 2768 — inject `neural_refiner` into `EmotionNet.__init__` (or use a setter called by AppContainer post-construction).

### Phase 4 — Memory module (1 day, critical path for hot-fix)
Move `MemoryStore`, `_init_redis`, `_redis_reconnect_loop`, `_get_redis` into `server/memory_store.py`.  
This is also when you fix the hot-path (O(N) scan, JSONL batching, deque caps) since it's now isolated.  
**See Section 7 below for the hot-path fixes to apply simultaneously.**

### Phase 5 — LLM Backend module (2 days, largest class)
Move `LLMBackend` into `server/llm_backend.py`.  
Key wiring changes:
- Constructor receives `telemetry: TelemetryEngine` and `log_fn: Callable` (eliminates two `globals().get()` usages)
- `globals().get("profile")` at line 1360 becomes `self._profile_getter()` — inject via `set_profile_getter(fn)` on the container post-init

### Phase 6 — Pipeline module (3 days, highest risk)
Move `process_pipeline`, `process_pipeline_safe`, `_run_proactive_pipeline`, `TurnWatchdog` into `server/pipeline.py`.  
`process_pipeline` (~400 lines) references every singleton. After the AppContainer exists, each reference becomes `container.X` — a mechanical substitution.

**Split the function while you're in there:**
```
process_pipeline()
  └── _pipeline_perception(container, turn, msg)   # emotion + memory retrieval
  └── _pipeline_cognition(container, turn, prompt)  # routing + LLM call
  └── _pipeline_expression(container, turn, result) # HFL + TTS + RL
```

### Phase 7 — Routes (2–3 days)
Create `server/routes/` and migrate route handlers one blueprint at a time, starting with the smallest (lifecycle, websearch, plugins) and finishing with chat + agents.

### Phase 8 — Cleanup (½ day)
- Delete everything that moved from `core_server.py`
- `core_server.py` becomes 30-line entry point:
  ```python
  from server import init_and_run
  if __name__ == "__main__":
      init_and_run()
  ```
- Remove all remaining `globals().get(...)` calls (should be zero after Phase 3–5)
- Run full smoke test + any integration tests

---

## 6. Shared State Strategy

| Shared resource | Strategy |
|-----------------|----------|
| `profile` dict | Lives on `container.profile`. Routes that mutate it (`/api/profile POST`) call `container.reload_profile()` which fan-outs to `llm_backend.configure()` etc. |
| `broadcast_json` | Set on `container.broadcast_json` by `server/__init__.py` once WS loop signals ready. All modules import from container — no `globals().get()`. |
| `ws_clients` / `ws_loop` | Live in `server/websocket.py` as module-level (not global in the entry point). |
| `neural_refiner` | Injected into `EmotionNet` at container init time via `emotion_net.set_refiner(neural_refiner)`. |
| `PLUGINS` list | Moves to `server/routes/integrations.py` (only two routes touch it). |

---

## 7. Hot-Path Fixes to Apply During Phase 4 (Memory Store)

While `MemoryStore` is isolated, apply these simultaneously — they are low-risk and high-value:

### 7a. Cap all unbounded history lists with `deque`
```python
# Before (in MemoryStore.__init__)
self._short_term = []

# After
from collections import deque
self._short_term = deque(maxlen=500)
```
Same pattern for `AnswerValidationSystem._history`, `AntiLoopEngine._history`, `InterruptionHandler._history` — all one-line fixes.

### 7b. Batch JSONL flushes
```python
# Before: _safe_write flushes per entry
def _safe_write(self, path, entry):
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")
        f.flush()  # ← kills performance

# After: flush every N entries (already done in TelemetryEngine — replicate pattern)
self._flush_counter += 1
if self._flush_counter >= 20:
    f.flush()
    self._flush_counter = 0
```

### 7c. O(N) scan → inverted index
`get_context_for_llm` does a full substring scan of up to 10k entries inside a lock on every chat turn. Minimum viable fix:
```python
# Add to MemoryStore.__init__:
self._keyword_index: dict[str, set[int]] = {}   # word → set of entry indices

# Update _add_long_term_entry to index words:
for word in re.findall(r'\w+', entry["text"].lower()):
    if len(word) > 3:  # skip stop words
        self._keyword_index.setdefault(word, set()).add(entry_id)

# get_context_for_llm: lookup union of word sets, score only those candidates
candidate_ids = set()
for word in query_words:
    candidate_ids |= self._keyword_index.get(word, set())
```

### 7d. Release the lock before scoring
```python
# Before: score inside lock (blocks all readers for the full scan duration)
with self._lock:
    results = [score(e) for e in self._long_term]

# After: snapshot outside, score outside
with self._lock:
    candidates = [self._long_term[i] for i in candidate_ids]
# Lock released — score freely
results = sorted([score(e) for e in candidates], key=..., reverse=True)
```

---

## 8. Rollback Strategy

- Each phase ends with a git commit tagged `split/phase-N`
- `core_server.py` is **not deleted** until Phase 8; earlier phases only add new files and update imports inside the existing file progressively
- The smoke test (`/api/status` + `/api/chat` roundtrip) runs as a pre-commit hook from Phase 0 onward
- If a phase breaks the smoke test, `git revert split/phase-N` restores to a working state

---

## 9. Files to Create (in order)

```
revia_core_py/server/__init__.py
revia_core_py/server/_container.py
revia_core_py/server/json_utils.py
revia_core_py/server/tts_utils.py
revia_core_py/server/telemetry.py
revia_core_py/server/emotion.py
revia_core_py/server/memory_store.py
revia_core_py/server/llm_backend.py
revia_core_py/server/pipeline.py
revia_core_py/server/websocket.py
revia_core_py/server/routes/__init__.py
revia_core_py/server/routes/chat.py
revia_core_py/server/routes/model.py
revia_core_py/server/routes/profile.py
revia_core_py/server/routes/memory.py
revia_core_py/server/routes/telemetry_routes.py
revia_core_py/server/routes/agents.py
revia_core_py/server/routes/integrations.py
revia_core_py/server/routes/autonomy.py
revia_core_py/server/routes/lifecycle.py
```

---

## 10. What Stays in core_server.py (temporarily, until Phase 8)

During the migration, `core_server.py` continues to be the running entry point. Each phase replaces sections of it with `from server.X import Y` shims. The file shrinks phase by phase. At Phase 8 it becomes:

```python
"""REVIA Core entry point."""
from server import init_and_run

if __name__ == "__main__":
    init_and_run()
```

---

## 11. Quick Wins to Ship Before the Split (≤2 hours total)

These don't require restructuring and can be merged to main immediately:

1. **`globals().get("broadcast_json")` → pass `log_fn` into classes** — already done for some classes, three remaining instances
2. **`len(text.split())` → `len(_WORD_RE.findall(text))`** where `_WORD_RE = re.compile(r'\w+')` — one line in the LLM streaming path
3. **`asyncio.gather` in `_broadcast`** — replace sequential `await websocket.send(text)` loop with `asyncio.gather(*[ws.send(text) for ws in clients], return_exceptions=True)`
4. **Cap `_throttle_last_ts` / `_throttle_suppressed`** — add `if len(_throttle_last_ts) > 1000: _throttle_last_ts.clear()` guard

---

## 12. Estimated Timeline

| Phase | Work | Duration |
|-------|------|----------|
| 0 | Smoke test + tagging | 1 day |
| 1 | Pure-function extractions | 1–2 days |
| 2 | Telemetry module | ½ day |
| 3 | Emotion module | 1 day |
| 4 | Memory module + hot-path fixes | 1–2 days |
| 5 | LLM Backend module | 2 days |
| 6 | Pipeline module | 3 days |
| 7 | Routes (all blueprints) | 2–3 days |
| 8 | Cleanup + final smoke | ½ day |
| **Total** | | **~12–14 working days** |

One engineer can do this solo. Two engineers can parallelize Phases 1–3 (no shared state conflicts).

---

*Generated from live audit of `core_server.py` (7,517 lines). Verify line numbers against current HEAD before starting Phase 1.*
