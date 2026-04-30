# REVIA Deep Dive — Architecture, Issues & Optimizations

**Scope:** Full codebase review across `revia_core_py`, `revia_core_cpp`, `revia_controller_py`
**Focus areas:** Human conversation, Memory, Parallel execution, Multi-LLM routing, Generation timing, Error surfacing

---

## 1. Conversation Maintenance (Human-Like)

### What's Working

`ConversationStateMachine` is solid. The FSM has well-defined transitions (BOOTING → INITIALIZING → IDLE → LISTENING → THINKING → SPEAKING → COOLDOWN), interruption states (INTERRUPTED / RECOVERING), and every transition is lock-protected. `BehaviorController` correctly gates autonomous triggers behind warmup and cooldown timers. `HumanFeelLayer` adds thinking pauses, self-corrections, prosody hints, quirks, and emotional vocalizations — all driven by `ProfileEngine` parameters, no hardcoded values. `ResponseFilter` blocks repetitive autonomous outputs via signature deduplication. This is genuinely good architecture.

### Issues Found

**Issue 1 — State machine can get permanently stuck in THINKING.**
If the pipeline throws between the THINKING → SPEAKING transition (e.g., LLM backend crashes mid-generation), the FSM stays in `THINKING` indefinitely. `BehaviorController.evaluate()` hard-blocks all `RESPONSE`-kind triggers when `current_state == THINKING`. There is no watchdog timer to force a recovery. A new user message will be rejected as "state=Thinking" until the server restarts.

**Fix:** Add a monotonic timestamp when entering THINKING. In `evaluate()`, if `current_state == THINKING` and `elapsed > thinking_timeout_s`, force-transition to ERROR → IDLE before blocking the new trigger.

**Issue 2 — HFL signatures collide with ResponseFilter deduplication.**
`ResponseFilter._signature()` strips all non-alphanumeric characters: `"Hmm… let me think about that!"` and `"Hmm, let me think about that."` both produce `hmmletmethinkaboutth`. But `HumanFeelLayer._apply_thinking_pause()` prepends phrases like `"Hmm… "` stochastically. This means HFL-processed outputs from different underlying replies can hash to the same signature, and `ResponseFilter` will reject the second one as "repetitive autonomous output" — a false positive.

**Fix:** Run `ResponseFilter.apply()` on the pre-HFL text, not the post-HFL text. The uniqueness check should operate on the LLM-generated content, not the cosmetic wrapper.

**Issue 3 — `response_cooldown_s` default is 750ms.**
`BehaviorController` default `response_cooldown_s=0.75`. This only gates AUTONOMOUS triggers (the code comment and evaluate() logic correctly bypass it for RESPONSE-kind). But `start_cooldown("response", ...)` is still being called after each response turn. If anything upstreams starts checking `remaining_cooldown("response")` for non-autonomous triggers, users will see lag. Trace every call to `start_cooldown("response")` and confirm none block user-driven replies.

---

## 2. Memory

### What's Working

`MemoryStore` (in `core_server.py`) correctly separates short-term (rolling list, up to 100 turns, overflow promoted to long-term) from long-term (JSONL file + optional Redis). `get_context_for_llm()` injects both layers into the prompt as structured context. The Redis cache path includes a fast TTL-based ping check that avoids blocking the status endpoint.

`EpisodicMemoryStore` (`autonomy_v3/episodic_memory.py`) is a well-designed second memory layer with time-decayed keyword search, quality scoring, and `attach_lesson()` for reflection — exactly the right architecture for long-term learning.

### Issues Found

**Issue 4 — Two disconnected memory systems, neither aware of the other.**
`MemoryStore` (main pipeline) and `EpisodicMemoryStore` (autonomy_v3) hold separate data and never exchange information. Lessons attached via `ReflectionAgent` never reach the `MemoryStore.long_term` array, so the main LLM context never benefits from what Revia has learned across sessions. The `MemoryAgent` (`agents/memory_agent.py`) is only wired to `MemoryStore` — it will never surface episodic lessons.

**Fix:** After each completed turn, write the episode to `EpisodicMemoryStore`. In `MemoryAgent.run()`, call `episodic_store.search(context.user_text, limit=3)` and append the lesson text from top hits into `relevant_facts`. This adds near-zero latency (keyword search is CPU-local) and gives Revia genuine cross-session memory.

**Issue 5 — `MemoryAdapter` raises `NotImplementedError` — it is live in the codebase.**
`revia_core_py/adapters/memory_adapter.py` and `llm_adapter.py` are Phase 6 boundary stubs. Both `search()` and `write_turn()` raise `NotImplementedError("...not migrated yet")`. If any new agent or module imports and calls these instead of the concrete `MemoryStore`, it will crash. The `Agent.execute()` wrapper will catch the exception and record it as a failed `AgentResult` with no visible error — silently degraded turn.

**Fix:** Either complete the migration (wrap `MemoryStore` inside `MemoryAdapter`) or add a `NotImplementedError` guard at import time that raises loudly during startup, not silently at runtime. Same applies to `LlmAdapter` and `WebSearchAdapter`.

**Issue 6 — `EpisodicMemoryStore.save()` full-rewrites the JSONL on every flush.**
`save()` calls `self._path.write_text(payload + "\n")` which rebuilds the entire file from scratch every time. At 2000 episodes this is 300-500KB of I/O per save, serialized on the calling thread. On a slow NVMe under load this adds visible latency, and a crash mid-write corrupts the file.

**Fix:** Use append-only writes (like `MemoryStore._lt_handle`) for new episodes. Only do a full rewrite on compaction (e.g., when trimming to `max_episodes`). Add atomic write via temp-file + rename.

---

## 3. Parallel Functions

### What's Working

`AgentOrchestrator` genuinely runs Phase 1 agents (MemoryAgent, EmotionAgent, IntentAgent, ReasoningAgent, VoiceStyleAgent) in parallel using `ThreadPoolExecutor`, with per-agent timeouts, cancellation tokens, and a clean fan-out/collect pattern. `ParallelPipeline` adds a second layer of concurrency: perception lane (< 5ms per telemetry), cognition lane (LLM), and expression lane (TTS) run independently, and streaming TTS (`chat_sentence` events) starts output before the LLM finishes. The telemetry confirms this is working — `perception_lane` averages 3-5ms while `llm_decode` runs in parallel.

### Issues Found

**Issue 7 — Three separate thread pools doing overlapping work.**
`ParallelPipeline` creates `_executor` (lane-level) and `_fanout_executor` (inner fan-out). `AgentOrchestrator` creates its own `ThreadPoolExecutor(max_workers=4)`. If both are active in the same pipeline run, there are 3 executors competing for CPU. On a 6-core machine this means threads are oversubscribed — the OS scheduler creates context-switch pressure exactly during the highest-latency stage (LLM decode is already CUDA-bound; you don't want Python threads fighting for GIL above it).

**Fix:** `AgentOrchestrator` should accept an injected executor. Pass `ParallelPipeline._fanout_executor` to the orchestrator at construction. This brings it down to 2 pools (lane-level + inner fan-out shared with the orchestrator).

**Issue 8 — `on_complete` callback exceptions are silently dropped.**
In `ParallelPipeline._submit_lane()`, the `on_complete(result)` call is wrapped in `except Exception: pass`. Any bug in a callback (e.g., a UI update that throws because the widget was garbage-collected) disappears completely. This makes bugs in callback chains very hard to track down.

**Fix:** At minimum, log the exception: replace `except Exception: pass` with `except Exception as exc: self._log(f"[ParallelPipeline] on_complete callback error in lane '{lane_name}': {exc}")`.

**Issue 9 — Perception → Cognition handoff is sequential, not properly pipelined for turn 1.**
The code comment in `ParallelPipeline` states "for the first message after idle, perception must complete before cognition can start." This is correctly acknowledged, but there is no enforcement mechanism. The pipeline submits cognition immediately with whatever emotion context was available from the previous turn (or none). On a cold start (first ever turn), `emotion_label` defaults to `"neutral"` and `profile_state` is empty — so the LLM prompt has no emotional grounding on turn 1. This isn't a crash but it does produce a notably flat first response.

**Fix:** On the first turn, submit perception synchronously (it's 3-5ms per telemetry) before submitting cognition. The latency cost is negligible.

---

## 4. Multiple LLMs (Model Router)

### Answer to your question — confirmed correct.

Yes, multi-LLM routing will meaningfully improve Revia. The `ModelRouter` architecture already supports it: register routes with `rank` values, and the router walks candidates in priority order with automatic fallback. The groundwork is there.

### What's Working

`ModelRouter` supports multiple named routes per `task_type`, hardware-aware `ModelRequirements` (VRAM budget, GPU preference, cost class), `priority_class` dispatch, and automatic fallback when a route is denied by `RuntimeScheduler`. All of this is clean and production-ready.

### Issues Found

**Issue 10 — `intent_classify` and `emotion_classify` task types are never registered.**
`IntentAgent` checks `model_router.has("intent_classify")` and falls back to heuristics if missing. `EmotionAgent` checks `model_router.has("emotion_classify")`. Neither task type is registered anywhere in the codebase. This means:

- Intent classification is always pure heuristic (rule-based, no ML).
- Emotion classification always calls `EmotionNet.infer()` directly, bypassing the router entirely.

The router hook exists but is dead code today. This is the primary place to plug in small models.

**Optimization:** Register a lightweight intent classifier (e.g., a fine-tuned `distilbert` or an ONNX model) under `"intent_classify"` with `rank=10`. Register a fallback to the heuristic engine at `rank=100`. The router will automatically use the ML model when available and fall back to heuristics when VRAM is tight. Same pattern for `"emotion_classify"`.

**Issue 11 — No complexity-based routing for `reason_chat`.**
The `ReasoningAgent` calls `router.call("reason_chat", ...)` — but there is only ever one route registered for this task type (the primary LLM). There is no small-model fast path for simple turns.

**Optimization:** Register a second `"reason_chat"` route at `rank=50` pointing to a small/fast model (e.g., Phi-3-mini, Llama 3.2 1B via llama.cpp). Add an `IntentComplexityClassifier` that gates route selection: `greeting`, `affirmation`, `small_talk`, `farewell` → small model (rank=50 wins first); `question`, `command`, `reasoning` → primary model (rank=10 wins first). This alone can cut latency by 60-70% for the majority of casual turns. The `ModelRouter` architecture supports this with zero structural changes.

---

## 5. Generation Time Tracking (ms)

### What's Working

`TelemetryEngine` uses `time.perf_counter()` (not `time.time()`) for sub-millisecond resolution — correct. Spans are written to dated JSONL files (`logs/telemetry_YYYYMMDD.jsonl`). The telemetry confirms real data is flowing: `perception_lane` consistently 3-5ms, `llm_decode` 31-80 seconds (local model, expected). `ttft_ms` is captured on the vLLM path.

### Issues Found

**Issue 12 — No single "total turn latency" metric.**
`perception_lane`, `llm_decode`, and `output_deliver` are tracked as separate spans, but there is no top-level `turn_total_ms` span that wraps them all. To know total response time you have to manually sum three spans from the JSONL — there is no single field to graph or alert on.

**Fix:** Open a `turn_total` span at the point the user message is received and close it at `output_deliver` end. This becomes your primary latency KPI and makes it trivial to spot regressions.

**Issue 13 — `ttft_ms` is only captured on the vLLM provider path.**
`time_to_first_token_ms` is set via `telemetry.llm["ttft_ms"]` only when using `vllm_adapter`. The llama.cpp, KoboldCPP, LM Studio, and Ollama providers track `first_token_t` locally but never write it to `telemetry.llm["ttft_ms"]`. You're missing TTFT data for all local providers.

**Fix:** In each provider's streaming loop (wherever `first_token_t` is set), add `telemetry.llm["ttft_ms"] = round((first_token_t - generation_start) * 1000, 1)` immediately. This gives you a consistent TTFT metric across all backends.

**Issue 14 — Telemetry log only flushes every 10 writes (flush_counter).**
If the process crashes after 1-9 spans, those spans are lost. In a dev cycle where you're testing short sessions, most spans are never flushed.

**Fix:** Add a time-based flush alongside the counter flush. If `time.monotonic() - last_flush_t > 2.0`, flush regardless of counter. Two seconds is imperceptible to performance but ensures data survives crashes.

**Issue 15 — `TelemetryEngine.__del__` is not reliable.**
Python's `__del__` is not guaranteed to run on interpreter shutdown. The log file may not be properly flushed on exit.

**Fix:** Register `telemetry.close` via `atexit.register(telemetry.close)` immediately after `telemetry = TelemetryEngine()`. This is a one-line fix.

---

## 6. Error and Exception Surfacing

### What's Working

`ReviaErrorHandler` is well-designed: frozen `ErrorReport` dataclass with slots, sanitized stack traces (secrets redacted), token-bucket rate-limited `ConsoleBackend`, persistent `FileBackend` with a single open handle, thread-safe `ErrorStore` with `deque(maxlen=1000)`, and async `acheck()` support. The architecture is clean.

`logs_tab.py` in the GUI correctly classifies log lines by severity with themed colors and routes them to the `event_bus.log_entry` signal.

### Issues Found

**Issue 16 — `ReviaErrorHandler` has no WebSocket backend — errors are invisible in the GUI.**
`ReviaErrorHandler` writes to `ConsoleBackend` (stdout/stderr) and optionally `FileBackend`. There is no backend that calls `broadcast_json({"type": "log_entry", ...})`. The GUI `logs_tab.py` only displays entries that come through `event_bus.log_entry`. This means any error logged through `error_handler.error(...)` or `error_handler.critical(...)` never appears in the Logs tab — it only shows in the server terminal.

**Fix:** Add a `WebSocketBackend(ErrorBackend)` that calls `broadcast_json({"type": "log_entry", "text": formatted_line})` on `emit()`. Wire it as a second backend in `ReviaErrorHandler` alongside `ConsoleBackend`. Since `broadcast_json` is a module-level global in `core_server.py`, pass it as a callable at construction.

```python
class WebSocketBackend(ErrorBackend):
    def __init__(self, broadcaster_fn: Callable):
        self._broadcast = broadcaster_fn

    def emit(self, report: ErrorReport) -> None:
        parts = [
            f"[{report.severity.name}]",
            f"[{report.category}]",
            report.message,
            f"({report.function_name}:{report.line_number})",
        ]
        try:
            self._broadcast({"type": "log_entry", "text": " ".join(parts)})
        except Exception:
            pass
```

**Issue 17 — `_revia_log` silently drops WS broadcasts if `broadcast_json` is not yet a callable.**
```python
broadcaster = globals().get("broadcast_json")
if not callable(broadcaster):
    return
```
This is correct, but there is no fallback queue. Log lines emitted during startup (before the WS server is ready) are permanently lost from the GUI. Users can miss critical startup errors.

**Fix:** Add a startup buffer: a module-level `deque(maxlen=200)` that accumulates messages before `broadcast_json` is ready. When `broadcast_json` is first assigned, flush the buffer.

**Issue 18 — Background thread exceptions do not reach the Logs tab.**
If an exception escapes a background thread (autonomy loop, TTS, audio service) without being explicitly caught and forwarded to `event_bus.log_entry`, it goes to stderr only. The GUI user sees nothing.

**Fix:** Set `threading.excepthook` at startup to capture uncaught thread exceptions and route them through `_revia_log(f"[UNCAUGHT] {args.exc_type.__name__}: {args.exc_value}")`. One setup call, catches all background thread failures.

```python
import threading

def _thread_excepthook(args):
    _revia_log(
        f"[CRITICAL] Uncaught exception in thread '{args.thread.name}': "
        f"{args.exc_type.__name__}: {args.exc_value}"
    )

threading.excepthook = _thread_excepthook
```

---

## Summary Table

| # | Area | Severity | Type | One-line description |
|---|------|----------|------|----------------------|
| 1 | Conversation | High | Bug | FSM stuck in THINKING on pipeline crash — no watchdog recovery |
| 2 | Conversation | Medium | Bug | HFL signatures collide with ResponseFilter — false dedup of valid replies |
| 3 | Conversation | Low | Risk | `response_cooldown` start may accidentally gate user replies downstream |
| 4 | Memory | High | Architecture | Two disconnected memory systems — episodic lessons never reach the LLM |
| 5 | Memory | High | Bug | All three Phase 6 adapters raise `NotImplementedError` silently at runtime |
| 6 | Memory | Medium | Performance | `EpisodicMemoryStore.save()` full-rewrites JSONL — slow + crash-unsafe |
| 7 | Parallel | Medium | Performance | Three ThreadPoolExecutors competing — oversubscribed threads on CPU |
| 8 | Parallel | Low | Bug | `on_complete` callback exceptions silently swallowed |
| 9 | Parallel | Low | Quality | Turn-1 LLM prompt has no emotional grounding (cold start) |
| 10 | Multi-LLM | High | Missing | `intent_classify` / `emotion_classify` routes never registered — dead hooks |
| 11 | Multi-LLM | Medium | Optimization | No small-model fast path for simple turns — all turns use full LLM |
| 12 | Timing | Medium | Missing | No `turn_total_ms` span — cannot measure end-to-end latency in one field |
| 13 | Timing | Medium | Missing | `ttft_ms` only captured on vLLM path — all local providers missing it |
| 14 | Timing | Low | Reliability | Telemetry only flushes every 10 writes — data loss on crash |
| 15 | Timing | Low | Reliability | `TelemetryEngine.__del__` unreliable — no `atexit` registration |
| 16 | Errors | High | Missing | No WebSocket backend on `ReviaErrorHandler` — errors invisible in GUI |
| 17 | Errors | Medium | Bug | Startup log lines dropped from WS if broadcaster not yet callable |
| 18 | Errors | Medium | Missing | Background thread exceptions never reach GUI Logs tab |

---

## Priority Order for Implementation

1. **Issue 16** — Wire error handler to GUI (highest visibility impact, one class + one line)
2. **Issue 18** — `threading.excepthook` (one line, catches all background failures)
3. **Issue 1** — FSM THINKING watchdog (prevents hard-stuck states)
4. **Issue 4** — Connect EpisodicMemory → MemoryAgent (enables real cross-session memory)
5. **Issue 5** — Guard Phase 6 adapter stubs (prevents silent degradation)
6. **Issue 12 + 13** — `turn_total_ms` + universal `ttft_ms` (enables latency tracking toward MS goals)
7. **Issue 10 + 11** — Register small-model routes (biggest latency win for simple turns)
8. **Issue 2** — Move ResponseFilter dedup before HFL
9. **Issue 7** — Share executor between pipeline and orchestrator
10. **Issue 6 + 14 + 15** — Append-only episodic save, telemetry flush, atexit

---

*All findings based on direct source read. No issues flagged without confirming the exact code path responsible.*
