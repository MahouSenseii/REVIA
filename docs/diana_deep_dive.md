# REVIA — Diana-Pattern Deep Dive

**Date:** 2026-04-26
**Author:** Sr. Developer audit (Cowork-assisted)
**Scope:** Verify REVIA can multi-task and evolve/grow on the path to a Diana-from-Pragmata-style AI.
**Companion docs:** `REVIA_CORE_REVIEW.md` (2026-04-16 architecture review), `docs/phase6_cleanup_boundaries.md`.
**Disposition:** Audit only. No code changes in this pass.

---

## 0. Executive summary

REVIA is much further along than it looks from the README. The strangler-fig migration described in `REVIA_CORE_REVIEW.md` has actually been *built* — the cpp Core (EventBus, StateManager, ContextManager, DecisionEngine, PriorityResolver, TimingEngine, SafetyGateway, ActionDispatcher, FeedbackManager, BackendSyncService) all exist as real code, gated behind `REVIA_CORE_V2_ENABLED`. There is also a real Diana persona (`persona_manager.py:54-151` `diana_inspired` preset), an online-learning PyTorch refiner with a saved `.pt`, a Thompson-sampling RL engine, a parallel pipeline with three concurrent lanes, multi-platform integrations (Discord, Twitch), and a vLLM backend.

But the system is **not** ready to be Diana yet. Three concrete blockers:

1. **The cpp orchestration loop runs end-to-end but emits nothing**, because every executor registered in `revia_core_cpp/src/main.cpp:97-126` is a `NoopActionExecutor`, and the Phase 6 Python adapters (`memory_adapter.py`, `llm_adapter.py`, `web_search_adapter.py`) all `raise NotImplementedError`. The cpp loop is a beautiful empty pipe.
2. **"Multi-task" today is intra-turn, not inter-task.** `parallel_pipeline.py` parallelizes one conversational turn (Perception / Cognition / Expression). It cannot run goal-driven background skills, and there is no scheduler/supervisor for fire-and-forget jobs that survive across turns. The cpp `EventBus` has a single dispatch worker.
3. **Evolve / grow is partial.** `reinforcement_learner.py` tunes 8 scalar knobs. `neural_refiner.py` does real online learning on emotion logits. But **nothing improves prompts, policies, or skills themselves** based on outcomes, and there is no fine-tune / LoRA lane. The plugin manager (`revia_core_cpp/src/plugins/plugin_manager.cpp`) is metadata-only — it cannot actually load a skill.

The README and the prior review currently disagree about which core is canonical. README (newer) says Python is primary; the 2026-04-16 review says cpp is canonical. **The README is true behaviorally** (because the cpp executors and adapters are stubs); **the review is true architecturally** (the cpp scaffolding is correct and built). This needs a forcing decision before the Diana work goes much further. See §4.

The good news: most of what Diana needs already exists in pieces. This document maps each Diana capability to the concrete file that already implements 30–80% of it, calls out the gap, and proposes additive work that does not require a rewrite.

---

## 1. The Diana pattern (anchor model)

Diana from *Pragmata* is the android partner who pairs with astronaut Hugh Williams on a lunar base. The relationship is the design: Hugh handles the physical domain, Diana handles the digital/cognitive domain. The architectural traits that matter for REVIA are:

- **Pair-bonded operator model.** Diana isn't a chatbot — she has a primary operator and her behavior is shaped by that pair-bond. State is per-relationship, not global.
- **Concurrent agency in the digital domain.** While Hugh is shooting, Diana is hacking, scanning, planning, predicting, and protecting. Multiple goals run at once, not sequentially.
- **Surgical tool use.** Diana doesn't talk to do things; she invokes capabilities. Each capability is a discrete, composable skill.
- **Continuity / memory.** Diana remembers, anticipates, and refers back. Memory is part of the relationship, not just a RAG store.
- **Quiet, precise, protective tone.** This is already encoded in REVIA's `diana_inspired` preset.
- **Evolves with experience.** Diana grows from the partnership — picks up the operator's habits, adjusts to their stress level, learns when to push back.

Below, every architectural move is justified against one of these traits.

---

## 2. Where REVIA is today — ground truth audit

Each row cites the actual file and the actual lines I read. No inferences from filenames.

### 2.1 The two cores and the live contradiction

| Concern | Today | Evidence |
|---|---|---|
| Cpp Core scaffolding | All 6 phases of the strangler-fig review are coded | `revia_core_cpp/src/core/{CoreOrchestrator,EventBus,StateManager,ContextManager,DecisionEngine,PriorityResolver,TimingEngine,SafetyGateway,ActionDispatcher,FeedbackManager}.{h,cpp}` |
| Cpp Core behavior | Does nothing user-visible | `main.cpp:97-126` registers `NoopActionExecutor` for every `ActionType`. `executors/DefaultActionExecutors.{h,cpp}` are deliberately no-ops. |
| Cpp event coverage | Only `EventType::UserText` is subscribed | `CoreOrchestrator.cpp:38-41` subscribes one type even though `IEvent.h` declares 10. |
| Cpp dispatch concurrency | Single worker thread, MPMC queue capacity 1024 | `EventBus.h:38-114`, `worker_loop()` is the only consumer. |
| Cpp plugin loading | Metadata-only, hard-coded stubs | `plugin_manager.cpp:49-59` registers six static `PluginInfo` records. `discover()` reads `*.json` metadata; never loads a binary or spawns a subprocess. |
| Python core_server.py | God module, grew from 4,923 to 6,053 LOC since the prior review | `core_server.py:455 TelemetryEngine`, `:656 LLMBackend`, `:1936 EmotionNet`, `:2518 RouterClassifier`, `:2582 WebSearchEngine`, `:2777 MemoryStore`, `:3922 process_pipeline`, `:4762 _set_runtime_state` |
| Phase 6 adapters | Pure `NotImplementedError` stubs | `revia_core_py/adapters/{llm_adapter,memory_adapter,web_search_adapter}.py` |
| Bridge | Python → cpp event POST works, gated by feature flag | `revia_core_py/ipc/event_publisher.py:50-83`, timeout 0.35 s, fail-open to legacy path |
| README vs prior review | README says Python primary; review says cpp canonical | `README.md:7-12` vs `REVIA_CORE_REVIEW.md:7` |

**Net:** the cpp side is a correct, empty pipe. The Python side is the actual brain. This needs a forcing decision (see §4) before Diana scaffolding is added on top of the contradiction.

### 2.2 Multi-task — what runs concurrently today

| Layer | Concurrency model | Files |
|---|---|---|
| Cpp event dispatch | One worker thread; subscribers serialized | `EventBus.h:114-127`, single `std::thread worker_` |
| Cpp servers | REST + WS each on own thread; telemetry broadcast on its own thread; broadcast flush on its own thread (4 total) | `main.cpp:156-171` |
| Python conversational turn | 3-lane `ThreadPoolExecutor` (max_workers=4) plus a separate fanout executor | `parallel_pipeline.py:91-100`, lanes are `perception/cognition/expression` |
| Python network IO | `asyncio` for WebSocket handler and broadcasts | `core_server.py:3384 ws_handler`, `:3432 _broadcast` |
| Discord bot | Runs in its own daemon thread, calls `pipeline_fn(text) -> str` | `integrations/discord_bot.py`, `integrations/integration_manager.py:120-124` |
| Twitch bot | Same shape as Discord | `integrations/twitch_bot.py`, `integration_manager.py:145-148` |
| Sing mode | Queue + state machine; 837 LOC parallel state engine for music playback | `revia_controller_py/app/sing_mode.py` |
| TTS | 954 LOC backend; sentence-level streaming | `revia_controller_py/app/tts_backend.py`, also see `core_server.py:_extract_complete_tts_sentences` |
| Camera / vision | YOLOv8 on a service thread on the controller side | `revia_controller_py/app/camera_service.py` (806 LOC), root `yolov8n.pt` |
| Anti-loop / cooldowns | Per-call timing logic, no central scheduler | `anti_loop_engine.py` (371 LOC), plus cooldowns in `conversation_runtime.py:264 BehaviorController` |

**What's missing for Diana-style concurrent agency:**

- **No goal-driven background workers.** Every concurrent thread today is reactive: it waits for an event, processes it, returns. Nothing runs a goal loop ("scan recent memory every 10 minutes", "watch this folder", "monitor webcam for face recognition", "tail this log for patterns").
- **No task supervisor.** No scheduler with retry, cancellation, timeouts, and observability. `parallel_pipeline.py` has a `ThreadPoolExecutor`, but its API is per-turn submit/await; there's no concept of a "running skill" that can be inspected, paused, or replaced.
- **No skill registry.** Cpp `PluginManager` is metadata only. Python has no skill registry at all — capabilities are inlined into `core_server.process_pipeline()`.
- **`agent_mesh` was attempted and removed.** `revia_core_py/__pycache__/agent_mesh.cpython-312.pyc` and `test_agent_mesh.cpython-312.pyc` exist on disk; the source files do not. There was a prior attempt at a multi-agent mesh that didn't survive. Worth recovering if the user wants — it may already contain ideas worth reviving.

### 2.3 Evolve / grow — what learns today

| Mechanism | What it tunes | File | Status |
|---|---|---|---|
| Thompson-sampling bandit | 8 scalar knobs: temperature, verbosity, humor, formality, emoji_density, proactivity, interrupt_sensitivity, topic_depth | `reinforcement_learner.py:73-94` | **Real, persists to `data/rl_parameters.json`.** Reward composite from AVS quality, engagement, emotion shift, interruption, loop, correction. |
| Neural emotion refiner | 27→64→32→11 PyTorch network refining emotion logits | `neural_refiner.py:60-86` | **Real, online learning every inference, persists to `data/neural_refiner.pt`.** Trains via KL-div toward the post-blended emotion distribution. |
| Persona presets | Static prompt templates per preset | `persona_manager.py:51-211` | Static. `diana_inspired` is the preset most aligned with the goal; it does not evolve. |
| Memory store | Per-profile JSONL, optional Redis | `memory_<profile>.jsonl`, `core_server.MemoryStore:2777` | Append-only conversational log. **No embeddings, no semantic retrieval visible**, no consolidation/summarization loop. The Phase 6 `MemoryAdapter` is a stub. |
| Profile engine | Validated profile load/save, trait weights | `profile_engine.py` (479 LOC) | Static editor, not a learning loop. |
| Cpp learning sinks | Single `StructuredLogLearningSink` | `feedback/DefaultLearningSinks.{h,cpp}`, registered at `main.cpp:128-129` | Logs feedback events; does not feed back into anything. |

**What's missing for Diana-style growth:**

- **No prompt or policy self-improvement.** RL tunes scalar knobs; nothing rewrites `persona_manager.PERSONA_PRESETS["diana_inspired"]["style_prompt"]` based on consistent failure modes, nothing edits the system prompt. Persona is frozen at import time (it's even pre-serialized for speed, see `persona_manager.py:220`).
- **No fine-tune / LoRA lane.** vLLM backend exists (`vllm_backend.py`, 671 LOC) but there's no scheduled fine-tune or LoRA-adapter swap that consumes accumulated turns from `data/memory_<profile>.jsonl`.
- **No semantic memory.** The memory store appears to be a journal, not an embedding-indexed retrieval surface. The cpp `MemoryContextProvider` would call into the Python `MemoryAdapter` stub, which raises.
- **Plugin/skill system is metadata only.** For Diana to "grow capabilities", REVIA needs a real skill registry where each skill is an executable unit (its own Provider/Rule/Executor or a Python coroutine), discoverable at runtime.
- **The cpp `FeedbackManager` is wired** (`main.cpp:128`, `CoreOrchestrator.cpp:138`), but it only fans out to `StructuredLogLearningSink`. The Python `ReinforcementLearner` and `NeuralRefiner` are not behind `ILearningSink` — they're invoked directly by `core_server.py`'s legacy path.

### 2.4 Persona — Diana already exists as a preset

`persona_manager.py:54-151` defines `diana_inspired` with:

- Identity prompt: *"You are Revia. You feel present, alert, and machine-born … you watch first, think carefully, then say exactly what matters."*
- Style: still, precise, understated, dry wit sparingly.
- Collaboration: *"Act like a close field partner with a cool head. Gather the missing facts, make a plan, and protect the user's time."*
- Traits: observant, precise, calm, quietly warm, protective, dryly witty, curious, self-possessed.
- Speech quirks: "look", "wait", "there it is", "that's the part that matters".
- Modules: voice calibration, technical examples, relational examples, introspection examples, long-form examples (multi-turn training-style demonstrations).

**This is genuinely good prompt design.** The work is to turn it from a static preset into the surface of a real architecture: persona-state that updates with experience, persona-dependent skill set, persona-specific safety tuning, persona-specific TTS voice (the `voices/Revia/voice.json` lives next to `voices/audio.wav`).

---

## 3. The Diana architecture for REVIA

The proposal is **additive** — it does not require demolishing what exists. It introduces three new concepts and reuses the existing pieces underneath them.

### 3.1 Three new concepts

```
┌─────────────────────────────────────────────────────────────────────┐
│  1. SkillRegistry        2. AgentSupervisor       3. EvolutionLoop  │
│     (what Diana          (background task            (how Diana     │
│      can do)              concurrency)                grows)        │
└─────────────────────────────────────────────────────────────────────┘
                  │                    │                    │
                  ▼                    ▼                    ▼
       reuse: cpp PluginManager,   reuse: ParallelPipeline   reuse: ReinforcementLearner,
       DecisionEngine, ActionDispatcher, ThreadPoolExecutor       NeuralRefiner, FeedbackManager
```

#### Concept 1: SkillRegistry — what Diana can do

A skill is a discrete capability bundle. Each skill declares:

- A `name`, `description`, `permissions` (file_read, network, mic, vision, …)
- `triggers`: explicit (`"!summarize"`, `/skills/scan`) and implicit (intent classifier hits "research", "watch")
- `provider`: optional `IContextProvider` it adds when active
- `executor`: an `IActionExecutor` (or a Python coroutine, behind a thin wrapper)
- `feedback`: how outcomes from this skill score on the RL reward
- `state`: per-skill persistent state (e.g., a Watcher skill remembers the last-modified file it saw)

Skills live in `revia_core_py/skills/<skill_name>/` with a `manifest.json` and a `skill.py`. The cpp `PluginManager` is upgraded to actually run them (subprocess for untrusted, in-process for first-party). The hardcoded stubs in `plugin_manager.cpp:49-59` become real skills.

This satisfies the **surgical tool use** trait. Diana's "hacking" in Pragmata maps cleanly to skill invocation: explicit, scoped, observable, composable.

#### Concept 2: AgentSupervisor — background task concurrency

Agents are long-running, goal-driven loops. Each agent:

- Has a name, a goal, an invocation cadence (`event-driven`, `interval=10m`, `cron`, `idle-only`)
- Runs in its own task with cancellation + timeout + retry policy
- Publishes events on the cpp `EventBus` instead of mutating shared state
- Reports health (last_run, success_rate, p95_latency) to the controller UI

Examples for Diana:

- **MemoryConsolidator** — every 15 minutes, take recent turns from `data/memory_<profile>.jsonl`, summarize, embed, write to a semantic store. Frees the conversational log from being the only memory.
- **PatternWatcher** — tail `revia_controller_py/logs/revia_core.jsonl` for repeated user frustration signals; raise an `EmotionShift` event for the orchestrator to consume.
- **FocusGuard** — if the camera service detects the operator hasn't moved for N minutes, raise a soft `IdleTimer` event with a protective-mode hint.
- **ResearchPlanner** — when a `/research` skill is invoked, spin up a goal-driven agent that fans out queries, ranks, and writes a brief.
- **HealthSentinel** — sample CPU/GPU/Redis/vLLM every 30s, raise alerts on degradation.

Implementation: a small supervisor on top of `ParallelPipeline._executor`. Supervisor primitives: `start_agent`, `stop_agent`, `list_agents`, `agent_health`. Persistence: agent declarations live next to skills; agent runtime state in Redis (already present).

This satisfies the **concurrent agency** and **continuity** traits. The supervisor is what makes "while Hugh is shooting, Diana is hacking" possible.

#### Concept 3: EvolutionLoop — how Diana grows

Three lanes, each independently deployable:

- **Lane A — Scalar tuning** (already real). RL bandit + neural refiner. Runs every interaction. Already in `reinforcement_learner.py` and `neural_refiner.py`. **Action:** put both behind `ILearningSink` so the cpp `FeedbackManager` is the only caller. Move them to `revia_core_py/analytics/` per the prior review.
- **Lane B — Policy / prompt evolution** (new). When the FeedbackManager observes a sustained pattern (low reward + low engagement + frequent corrections in a specific topic_category), generate a **prompt patch**: a small overlay applied to the active persona's `style_prompt` or `collaboration_prompt`. Patches are versioned, A/B-able, and reversible. Owner: `revia_core_py/evolution/policy_patcher.py`.
- **Lane C — Local model adaptation** (new). Periodic LoRA fine-tune on accepted turns + reward labels. Runs as a scheduled agent in the AgentSupervisor (e.g., nightly). Owner: `revia_core_py/evolution/lora_trainer.py`. Adapter swap is a `ConfigChange` event so the cpp `BackendSyncService` validates and applies it through the existing pipe.

Each lane closes a feedback loop. The reward signal feeding Lane A already exists and is good; Lane B and Lane C consume the same signal at different cadences.

This satisfies the **evolves with experience** trait without coupling a model rewrite to a prompt edit to a knob change — they're three independent rates.

### 3.2 How the parts compose at runtime

```
                       ┌──────────────────┐
   Mic / Discord ────► │   EventBus (cpp) │
   Twitch / UI         └──────────────────┘
   FileWatcher ─────────┬──┬──┬───────────────┐
   IdleTimer ───────────┘  │  │               │
                           ▼  ▼               ▼
                  CoreOrchestrator    AgentSupervisor (NEW)
                           │            (long-running goal loops,
                           │             can publish back to bus)
                           ▼
                   ContextManager
                   ├─ ConversationProvider
                   ├─ MemoryProvider  (NEW: semantic + episodic)
                   ├─ ProfileProvider (persona-aware)
                   ├─ EmotionProvider
                   ├─ RelationshipProvider (NEW: pair-bond state)
                   ├─ SkillProvider (NEW: which skills are armed)
                   └─ CapabilityProvider
                           ▼
                   DecisionEngine ─── Rules can suggest a SkillInvocation
                   PriorityResolver
                   TimingEngine
                   SafetyGateway
                           ▼
                   ActionDispatcher
                   ├─ SpeakResponseExecutor   (real, via vLLM + Qwen3-TTS)
                   ├─ SendTextExecutor        (real, via Discord/Twitch adapter)
                   ├─ InvokeSkillExecutor     (NEW: routes to SkillRegistry)
                   ├─ SpawnAgentExecutor      (NEW: AgentSupervisor.start)
                   ├─ InterruptOutputExecutor
                   └─ … (rest of existing set)
                           ▼
                   ExecutionResult
                           ▼
                   FeedbackManager  ──►  ILearningSink fanout:
                           │              ├─ ScalarTuningSink (RL bandit)
                           │              ├─ NeuralRefinerSink (online learning)
                           │              ├─ PolicyPatcherSink (Lane B)
                           │              └─ LoraTrainerSink   (Lane C, batched)
                           ▼
                   BackendSyncService publishes state, decision,
                   execution, skill_state, agent_health, learning_progress
                                   ▼
                          Controller UI tabs read all of the above
```

Three things to notice:

- **The skeleton is already there.** Every box without "(NEW)" is a file that exists today. The Diana-specific work is the four new boxes (AgentSupervisor, MemoryProvider real impl, RelationshipProvider, SkillProvider) plus three new executors plus three new sinks.
- **Persona is a runtime input, not a global.** ProfileProvider already snapshots the active persona; RelationshipProvider adds the pair-bond state (operator id, operator habits, operator stress signal, last interaction tone). The prompt is composed per turn from these, not loaded at boot.
- **Multi-task is two layers.** Inside one turn, the existing `ParallelPipeline` keeps doing perception/cognition/expression in parallel. Across turns, the AgentSupervisor runs goal loops. They share the same executor pool but use different submit APIs.

---

## 4. Forcing decisions before more code lands

Three open questions need answers before Diana scaffolding goes much further. Each has a default but the user should pick.

### 4.1 Which core is canonical, today?

**Default I'd recommend:** Cpp Core remains the architectural target, but the next 2 phases of Diana work happen in Python and only call into the cpp Core through the `event_publisher.py` bridge. Reason: the cpp Core is correct but empty (NoopActionExecutors + NotImplementedError adapters). Building Diana skills inside cpp before the Phase 6 adapters are real means writing IPC-marshalling for capabilities that don't exist yet. Python is where the brain lives in 2026-04-26.

**Alternative:** Commit hard to cpp — implement the Phase 6 adapters as the first work, and make the cpp executors real before any Diana skill ships. This is the cleaner long-term path but stalls Diana progress for ~2-3 weeks of plumbing.

### 4.2 Persona composition policy

Once Diana grows (Lane B), the persona prompt isn't a fixed preset anymore. Two options:

- **Layered patch model.** `diana_inspired` preset stays frozen; a `policy_patches` overlay appends/edits at runtime. Easy to roll back; visible in the UI.
- **Branching persona model.** Each accepted patch creates a new persona variant (`diana_inspired_v3.user_q`). User picks which one is active.

I'd recommend layered patches — closer to how RL already works, simpler ops.

### 4.3 Memory shape

Memory today is per-profile JSONL plus Redis. Diana needs at least three memory surfaces:

- **Episodic** — verbatim turns with timestamps. (Have this.)
- **Semantic** — embedding-indexed facts/preferences/observations. (Don't have this.)
- **Relational** — pair-bond state: operator habits, current stress, recurring concerns. (Don't have this; need a small structured store keyed by operator id.)

Decision: implement semantic in Redis (RediSearch / vector type) or stand up Chroma/Qdrant? The hardcoded plugin stub is `ChromaDB-Memory`, which suggests prior intent toward Chroma. Either works; Redis-only is simpler ops because Redis is already the primary memory backend per `README.md:13-19`.

---

## 5. Gap matrix — Diana need → today → action

| Diana trait | What today provides | Gap | First-cut action |
|---|---|---|---|
| Pair-bonded operator | Persona presets exist; no relational state | No `RelationshipManager` writing pair-bond state | Add `revia_core_cpp/src/identity/RelationshipManager.{h,cpp}` (review §6 calls for it; not built yet) + `RelationshipContextProvider` reading it. Add operator-id concept to `runtime_models.py`. |
| Concurrent agency | `ParallelPipeline` (intra-turn) + Discord/Twitch threads | No goal-driven background workers, no supervisor | Build `AgentSupervisor` on top of `ParallelPipeline._executor`. Convert HealthSentinel + MemoryConsolidator first. |
| Surgical tool use | `PluginManager` metadata-only; capabilities inlined | No real skill loader / executor | Promote `revia_core_cpp/src/plugins/plugin_manager.cpp` from metadata to subprocess loader. Add `revia_core_py/skills/<name>/skill.py` convention + `InvokeSkillExecutor`. |
| Continuity / memory | JSONL + Redis | No semantic memory, `MemoryAdapter` is `NotImplementedError` | Implement `MemoryAdapter` against Redis vectors (or Chroma). Stand up `MemoryConsolidator` as the first agent. |
| Quiet, precise tone | `diana_inspired` preset prompts | None (this is good) | Keep frozen; layer patches later (Lane B). |
| Evolves with experience | RL bandit + neural refiner | No prompt evolution, no LoRA lane | Wire RL + refiner behind `ILearningSink`. Build `PolicyPatcher` (Lane B). Build `LoraTrainer` agent (Lane C). |
| Multi-source events | Discord, Twitch, controller UI, mic, vision | All paths join through legacy `process_pipeline` | When core_v2 is on, every external source emits via `event_publisher.py`. Discord already does (`integrations/discord_bot.py`). Twitch + STT + camera need the same forwarding. |
| Background long-running tasks | Sing-mode is the only one (and it's a parallel state machine) | No pattern for new ones | AgentSupervisor + a small declarative manifest format. |
| Self-improving prompts/policies | None | Whole capability missing | `PolicyPatcher` consumes FeedbackManager events; produces versioned, reversible patches; A/B harness for at least 50 turns before adoption. |
| Fine-tune / local model adaptation | None | Whole capability missing | `LoraTrainer` agent. Reads accepted turns + reward labels. Trains adapters offline, swaps via `ConfigChange` event. |

---

## 6. Roadmap (additive, phased, no big-bang)

Phases are sized so each one is independently shippable with `REVIA_CORE_V2_ENABLED=0` still working. Anything that breaks the legacy path is rejected.

### Phase D0 — Decisions and stubs unblocked (1–2 days)

- Resolve §4.1, §4.2, §4.3.
- Implement the three Phase 6 adapters (`memory_adapter`, `llm_adapter`, `web_search_adapter`) so the cpp Core can actually produce output when v2 is on. Use a thin pass-through to the existing `core_server.py` classes for now; refactor later.
- Add an `operator_id` field to `runtime_models.py` and pass it through every event.
- Recover or reconstruct `agent_mesh.py` from the `.pyc` shadow if any value is recoverable, then formally delete the cache. Even if we don't reuse it, we shouldn't ship dead `.pyc` files in source.
- Decide on the canonical core (cpp vs python) and update `README.md` to match the decision.

Exit: cpp v2 path produces a real (text) reply for `EventType::UserText` end to end, even if everything else stays legacy.

### Phase D1 — SkillRegistry minimum viable (3–5 days)

- Add `revia_core_py/skills/__init__.py` with a small `Skill` dataclass, `SkillRegistry`, and a manifest loader that reads `revia_core_py/skills/<name>/manifest.json`.
- Promote `revia_core_cpp/src/plugins/plugin_manager.cpp` to call into the Python skill registry over the existing REST/WS bus. Keep the metadata API for backward compatibility.
- Add `InvokeSkillExecutor` to the cpp `ActionDispatcher` registry. Wire it to publish a `SkillInvocation` REST call into Python.
- Convert exactly one existing capability to a skill — `web_search` is the cheapest because the engine already exists in `core_server.WebSearchEngine`.
- Add a `SkillProvider` to `ContextManager` that surfaces "which skills are armed for this turn".

Exit: `!skills` lists web_search; `process_pipeline` no longer hardcodes its invocation; the controller UI's existing model tab gets a "Skills" sub-section.

### Phase D2 — AgentSupervisor + first two agents (3–5 days)

- Add `revia_core_py/agents/supervisor.py` on top of `ParallelPipeline._executor`. Primitives: declare, start, stop, status, health.
- Implement two agents:
  - `HealthSentinel` (replaces ad-hoc psutil sampling in `core_server.py`).
  - `MemoryConsolidator` (the first user-visible "evolve" win — Diana actually starts having semantic memory).
- Wire agent health into `BackendSyncService.publish_*` so the controller UI's System tab shows agents.
- Add a `SpawnAgentExecutor` so the `DecisionEngine` can decide to start an agent.

Exit: System tab shows two running agents with last_run / success_rate. Memory tab shows growing semantic store.

### Phase D3 — Real memory + RelationshipManager (5–7 days)

- Implement `MemoryAdapter.search` using Redis vectors (recommended) or Chroma.
- Implement `MemoryConsolidator` agent's summarize+embed loop.
- Add `revia_core_cpp/src/identity/RelationshipManager.{h,cpp}` (review §6 already names it). Holds operator_id → habits, stress, last_topic, last_tone.
- Add `RelationshipContextProvider` that contributes pair-bond state to `ContextPackage`.
- Update prompt assembly (`prompt_assembly.py`) to read pair-bond state when persona is `diana_inspired`.

Exit: With persona = diana_inspired, the prompt at turn N+1 reflects that the operator was stressed at turn N. Memory tab can search and surface a fact from a week ago.

### Phase D4 — Evolution Lane A (FeedbackManager wiring, 2–3 days)

- Implement `ScalarTuningSink` and `NeuralRefinerSink`. Each is a thin `ILearningSink` wrapper around the existing classes.
- Move `reinforcement_learner.py` and `neural_refiner.py` to `revia_core_py/analytics/` per the prior review.
- Remove direct calls to RL/refiner from `core_server.py`'s legacy path; route through `FeedbackManager`.
- Surface RL stats + refiner stats in the controller UI's existing System tab.

Exit: `core_server.py` no longer touches RL or refiner directly. Toggling `REVIA_CORE_V2_ENABLED` does not change learning behavior.

### Phase D5 — Evolution Lane B (PolicyPatcher, 5–7 days)

- Add `revia_core_py/evolution/policy_patcher.py`. Consumes FeedbackManager events. Pattern detection: rolling window of (topic_category, persona, reward) tuples. Trigger threshold: ≥ 50 turns and reward delta ≤ −0.15 vs baseline.
- Patch format: small JSON overlay (`{"style_prompt_append": "..."}`, `{"speech_quirks_remove": ["..."]}`). Versioned, signed (so a corrupt patch can't poison the persona), reversible.
- A/B harness: when a candidate patch exists, route 20% of turns through it for at least 50 turns before adoption.
- Controller UI: new "Evolution" sub-tab (or extend Profile tab) showing active patches, candidate patches, and A/B status.

Exit: With sustained user frustration on a topic, Diana proposes a candidate patch and A/B-tests it. Adopted patches survive restarts.

### Phase D6 — Evolution Lane C (LoraTrainer, 7–10 days)

- Add `revia_core_py/evolution/lora_trainer.py` as a scheduled agent (interval=24h, idle-only).
- Trainer reads accepted turns from `data/memory_<profile>.jsonl` filtered by reward ≥ 0.4. Uses LoRA on the local LLM (vLLM-compatible).
- Adapter manager: `vllm_backend.py` gains an `apply_adapter(adapter_path)` API. Adapter swap is a `ConfigChange` event so `BackendSyncService` validates + applies.
- Controller UI: an "Adapters" panel under the Model tab showing available adapters, training history, current active adapter, and a manual rollback button.

Exit: Diana can be re-trained on the operator's accumulated conversation. Rollback works in one click.

### Phase D7 — Cleanup (the original Phase 6 finally finishes)

- With the executors and adapters all real, the cpp Core can be canonical. The legacy path in `core_server.py` retires module by module per `docs/phase6_cleanup_boundaries.md`.
- `process_pipeline()` shrinks to a thin compat shim and is eventually deleted.
- The README is updated to reflect the cpp Core as the runtime brain.

Exit: `REVIA_CORE_V2_ENABLED` is removed from the codebase.

---

## 7. Risks worth flagging

1. **Adapter stub rot.** `memory_adapter.py`, `llm_adapter.py`, `web_search_adapter.py` have been `NotImplementedError` since the original review. They're a structural lie — every consumer that imports them is silently coupled to the legacy path. Phase D0 must fix this or every later phase inherits the lie.
2. **Two state machines diverging.** `conversation_runtime.ConversationStateMachine` (Python, 134-262) and `revia_core_cpp/src/core/StateManager` (cpp) both define allowed-transition tables. They will drift. Adopting cpp as canonical (per §4.1) means `ConversationStateMachine` becomes a thin client of the cpp state via `BackendSyncClient`.
3. **Agent supervisor + Qt event loop.** The controller UI runs Qt. Long-running agents must publish to the cpp `EventBus`, never to the Qt signal bus, or the UI thread will block. Bridge through `backend_sync_client.py` only.
4. **Plugin trust boundary.** The moment skills become real, supply-chain risk arrives. Untrusted skills should run as subprocesses with a JSON-over-stdio contract (already the pattern from the prior review §R9). First-party skills can stay in-process initially; a manifest field gates this.
5. **PolicyPatcher feedback inversion.** A bad patch that silently makes Diana terser could trigger reduced engagement, which the bandit reads as "lower verbosity is bad", which raises verbosity, which fights the patch. Mitigation: the policy patcher reads from FeedbackManager *after* RL has already updated, and patches mark which signals they consume so RL can be told to ignore those slices for K turns post-patch.
6. **LoRA training catastrophic forgetting.** Diana's tone is delicate (`diana_inspired` is specifically *not* bubbly, *not* flirty, *not* theatrical). A naive fine-tune on accumulated turns can erase those guardrails. Mitigation: LoRA-only (don't touch base weights), maintain a held-out style-eval set drawn from `persona_manager.py:121-145` examples, reject adapters whose held-out style score regresses.
7. **`agent_mesh.py` ghost.** The fact that `__pycache__/agent_mesh.cpython-312.pyc` and `test_agent_mesh.cpython-312.pyc` exist without source files means there was earlier multi-agent work. Either recover it (it may have ideas worth using), or delete the caches so they don't shadow imports.

---

## 8. What I did *not* read this pass

For the user's awareness — these would be the next reads if the audit needed to go deeper:

- `revia_core_py/core_server.py` lines 656–4000 (the `LLMBackend` body and most of `process_pipeline`). I read enough to know the shape; I did not audit the request shaping, retry logic, or prompt injection.
- `revia_controller_py/app/assistant_status_manager.py` (1,141 LOC). The prior review already flagged this as a god class; nothing in the Diana plan depends on its internals.
- `revia_controller_py/gui/widgets/chat_panel.py` (1,455 LOC) and tabs/voice_tab.py (1,780 LOC). UI-side; not on the architectural critical path.
- `tts_backend.py` (954 LOC) and `audio_service.py` (342 LOC). Voice is real and works; only matters at Phase D1 when `SpeakResponseExecutor` becomes real.
- The Discord/Twitch bot internals beyond `integration_manager.py`. Both are working integrations; not on the critical path until §D1.

---

## 9. Final answer to "is REVIA able to multi-task and evolve / grow?"

**Today, partially.** Concretely:

- *Multi-task within a turn:* yes, the three-lane `ParallelPipeline` works.
- *Multi-task across turns / goal-driven concurrency:* no, there is no supervisor. Discord and Twitch run as event-driven threads but they are not goal-driven agents.
- *Evolve scalar parameters:* yes, `ReinforcementLearner` is real and persisting.
- *Evolve emotion classifier:* yes, `NeuralRefiner` is real and online-training.
- *Evolve persona, prompts, or skills:* no.
- *Fine-tune the underlying LLM:* no.

**With Phase D0–D6 above:** all six become yes, additively, without a rewrite. The work is real but each phase is small enough to ship and verify independently.

The Diana persona itself (`persona_manager.py:54-151`) is already excellent. The job is to give it an architecture that matches the prompt.

---

*End of audit.*
