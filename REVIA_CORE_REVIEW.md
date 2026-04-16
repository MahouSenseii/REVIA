# REVIA Core — Architecture Review & Update Plan

**Date:** 2026-04-16
**Author:** Sr. Developer review (Claude-assisted)
**Governing workflow:** `MODULAR_CODING_SKILL.md` — Review → Architect → Build/Update → Validate → Refactor → Recheck → Output
**Target architecture:** `REVIA_CORE.md` (REVIA Core Layer spec)
**Canonical runtime (user-confirmed):** `revia_core_cpp` is the real core (event bus, state machine, timing, dispatch, concurrency-sensitive flow, safety-critical execution, lifecycle). `revia_core_py` is the scripting / orchestration / experimentation layer (model adapters, prototyping, analytics, admin tooling, optional isolated decision plugins).
**Migration strategy (user-confirmed):** Strangler-fig, incremental. No big-bang rewrite.
**Theme system (user-confirmed):** Included in this pass.

---

## 0. Executive Summary

The current REVIA codebase is a **feature-rich system without a brain**. The exact failure mode `REVIA_CORE.md` warns against is present:

- There is **no Core Orchestrator**. Behavior is decided by a ~4,900-line god module (`revia_core_py/core_server.py`) and a ~1,070-line Qt god object (`revia_controller_py/app/assistant_status_manager.py`).
- **Runtime state is scattered** across three modules (`revia_states.py` string constants, `runtime_status.py`, `runtime_state_sync.py`, `_set_runtime_state()` function at `core_server.py:3981`). No single owner.
- **No Event Bus in the canonical runtime.** The only bus is `revia_controller_py/app/event_bus.py` — a 37-line PySide6 signal bag on the UI side. The C++ core has no bus at all.
- **No Decision Engine, Priority Resolver, Timing Engine, Safety Gateway, Action Dispatcher, Context Manager, or Feedback Manager** as first-class systems. Their logic is inlined inside `process_pipeline()` (`core_server.py:3415`) and ad-hoc helpers.
- **Ownership is inverted vs. user intent.** The low-latency, concurrency-sensitive runtime loop currently lives in Python; the C++ side (`pipeline.cpp` 273 lines, `main.cpp` 76 lines) is a thin linear stage runner that does not own the runtime.
- **Theme system is a 22-line stub** that reads `.qss` files and toggles two hardcoded names — does not satisfy any of the `REVIA_CORE.md` theme requirements (tokens, validation, custom themes, contrast checks, persistence, sync).

This review proposes a **strangler-fig migration** that introduces the Core layer in C++ alongside the existing code, routes events through it incrementally, and demotes `core_server.py` to a model/experimentation service over time.

---

## 1. Review — Current System vs. Core Spec

### 1.1 Observed current structure

```text
REVIA/
├── revia_core_cpp/                 # Intended canonical core — currently thin (~1,482 LOC)
│   └── src/
│       ├── main.cpp                # 76 LOC — starts REST+WS+telemetry threads
│       ├── pipeline/pipeline.{h,cpp}   # Linear stage runner (STT→router→emotion→RAG→prompt→LLM→TTS→mem)
│       ├── neural/                 # EmotionNet, RouterClassifier
│       ├── api/rest_server, ws_server  # HTTP/WebSocket endpoints
│       ├── plugins/plugin_manager
│       └── telemetry/telemetry
│
├── revia_core_py/                  # Should be scripting/experimentation — currently owns runtime
│   ├── core_server.py              # 4,923 LOC GOD MODULE — owns pipeline, state, memory, LLM, emotion, router, search
│   ├── conversation_runtime.py     # 554 LOC — parallel runtime logic
│   ├── anti_loop_engine.py         # Timing/cooldown logic (owned HERE, not a TimingEngine)
│   ├── interruption_handler.py     # Interrupt logic (owned HERE, not Priority Resolver)
│   ├── human_feel_layer.py         # Pacing/tone logic (owned HERE)
│   ├── reply_planner.py            # Decision-ish logic (owned HERE)
│   ├── answer_validation.py        # Safety-ish logic (owned HERE)
│   ├── reinforcement_learner.py    # Feedback-ish logic (not wired through an orchestrator)
│   ├── error_handler.py, persona_manager.py, profile_engine.py, prompt_assembly.py,
│   ├── runtime_status.py, runtime_models.py, vllm_backend.py
│   └── integrations/               # discord_bot, twitch_bot, integration_manager
│
└── revia_controller_py/            # Qt desktop controller / backend UI
    ├── app/
    │   ├── assistant_status_manager.py  # 1,073 LOC — second god class (state + timing + telemetry + UI signals)
    │   ├── controller_client.py         # 707 LOC — HTTP client to core
    │   ├── event_bus.py                 # 37 LOC — Qt signal bag (UI only)
    │   ├── conversation_policy.py       # 164 LOC — fragment of a decision engine
    │   ├── runtime_state_sync.py        # 109 LOC — debounced snapshot pusher
    │   ├── revia_states.py              # String constants shared by policy + status mgr
    │   ├── theme_manager.py             # 22 LOC STUB — two hardcoded themes, no tokens
    │   ├── sing_mode.py, sing_queue.py, sing_command.py, song_library.py  # Feature silo
    │   ├── audio_service, continuous_audio, tts_backend, voice_*, camera_service
    │   └── …
    └── gui/  (main_window, tabs/, widgets/, qss/)
```

### 1.2 Mapping matrix — Core subsystem → Current location

| Core subsystem (spec §) | Current location(s) | Owner? | Status |
|---|---|---|---|
| **Event System** (§1) | `revia_controller_py/app/event_bus.py` (Qt signals, UI-side only) | Partial | **Missing in core_cpp.** No normalization, timestamps, or source metadata. |
| **State System** (§2) | `revia_states.py` (strings), `runtime_status.py`, `runtime_state_sync.py`, `core_server.py:_set_runtime_state()`, `assistant_status_manager.EAssistantState` | **None** (4+ owners) | Scattered. No transition validation. Invalid transitions silently accepted. |
| **Context System** (§3) | `core_server.py:_build_situational_context()`, `prompt_assembly.py`, inline memory fetches | None | No `IContextProvider` pattern. Context is assembled ad-hoc inside `process_pipeline`. |
| **Decision Engine** (§4) | `reply_planner.py`, fragments in `core_server.py:process_pipeline`, `conversation_policy.py` (controller side), `interruption_handler.py` | Shared | No `IDecisionRule`. Decisions are branches inside a 400-line function. |
| **Priority Resolver** (§5) | Inline `if/else` chains in `core_server.py`, `interruption_handler.py` | None | No explicit conflict resolution. Last-write-wins between emotion/profile/filter. |
| **Timing Engine** (§6) | `anti_loop_engine.py`, `human_feel_layer.py`, timers in `assistant_status_manager.py`, cooldowns in `core_server.py`, `conversation_policy.py` | None (5+ owners) | Classic scattered-timing smell the spec explicitly forbids (Design Warnings §). |
| **Action System** (§7) | Side effects inside `process_pipeline` stages; `controller_client.py` HTTP calls; Qt signal emissions | None | No `IAction` type. No `ActionDispatcher`. No registered executors. |
| **Safety Gateway** (§8) | `answer_validation.py`, filter flags on `FiltersTab`, inline checks in `process_pipeline` | Partial | No single gateway. Filters not composed through one pipeline. |
| **Feedback Loop** (§9) | `reinforcement_learner.py`, scattered emotion writes, `telemetry.py` | Partial | Not triggered by an orchestrator; invoked ad-hoc. |
| **Core Sync Layer** (§10) | `runtime_state_sync.py` (push), `controller_client.py` (pull), WS broadcasts in `ws_server.cpp`/`core_server.py` | Split | Works, but ownership is split between two languages. No rollback path for bad config. |
| **Theme System** | `revia_controller_py/app/theme_manager.py` (22 LOC) + `.qss` files | Stub only | **Nothing in spec is satisfied.** No tokens, no validation, no custom themes, no contrast checks, no per-user persistence, no sync. |

### 1.3 Concrete smells (cross-referenced to file:line)

1. **God module** — `revia_core_py/core_server.py` (4,923 LOC) contains `TelemetryEngine`, `LLMBackend`, `EmotionNet`, `RouterClassifier`, `WebSearchEngine`, `MemoryStore`, pipeline function, WS handler, HTTP endpoints, state setter, profile I/O, greeting logic, and emotion reporting. Spec explicitly warns: *"Avoid giant god classes"*.
2. **God Qt object** — `assistant_status_manager.py` (1,073 LOC) mixes state derivation, request/STT/TTS timing, telemetry subscription, snapshot construction, formatting, and signal wiring. Holds 4+ responsibilities.
3. **Inverted ownership** — The low-latency runtime loop (pipeline stages, STT/TTS coordination, interrupts, timing) runs in Python. User's own constraint says these belong in C++.
4. **Parallel partial implementations** — `conversation_runtime.py` (py, 554 LOC) and `pipeline.cpp` (cpp, 273 LOC) both try to be the runtime loop. Neither is complete, and they disagree.
5. **Event bus is UI-only** — `revia_controller_py/app/event_bus.py` is a `QObject` with `Signal(...)` members. Not usable from the C++ core, not thread-tested for non-Qt producers, not persistent, not typed.
6. **Timing scattered** — `anti_loop_engine` owns cooldowns, `human_feel_layer` owns pacing, `assistant_status_manager._on_live_timer` owns UI timers, `core_server` owns silence timeouts, `conversation_policy` owns its own rules. No TimingEngine.
7. **Hidden coupling via module globals** — `core_server.py` uses module-level functions (`_set_runtime_state`, `_mark_user_activity`, `_seconds_since_user_activity`, `_sync_memory_profile_from_profile`) touching shared global state. This is hidden dependency the Modular Coding Skill forbids.
8. **Theme stub** — `theme_manager.py` has only `apply_theme(theme_name)` reading `.qss` files. Widgets are styled via Qt stylesheets with hardcoded colors embedded in `.qss`, violating the "widgets should read theme tokens" rule.
9. **No `IAction`, `IEvent`, `IContextProvider`, `IDecisionRule`, `IActionExecutor`** — None of the contracts from the spec exist as interfaces in either language. Everything is concrete.
10. **Failure handling is ad-hoc** — `error_handler.py` exists but isn't a Core-owned recovery mode. `_handle_pipeline_crash` (core_server.py:3840) is a module function, not part of a state machine with a `Recovering` state transition.

---

## 2. Architect — Corrected Modular Architecture

### 2.1 Ownership split (per user's canonical-core clarification)

```text
┌────────────────────────────────────────────────────────────────────┐
│                      revia_core_cpp  (CANONICAL)                    │
│  Owns: lifecycle, concurrency, low-latency loop, safety-critical    │
├────────────────────────────────────────────────────────────────────┤
│  Core/                                                              │
│    ├── CoreOrchestrator         ← the brain; single entry per event │
│    ├── EventBus                 ← typed, thread-safe, timestamped   │
│    ├── StateManager             ← authoritative RuntimeState        │
│    ├── TimingEngine             ← ALL delays, cooldowns, bursts     │
│    ├── ActionDispatcher         ← executor registry; IActionExecutor│
│    ├── SafetyGateway            ← chained filter/rule pipeline      │
│    ├── PriorityResolver         ← conflict arbitration              │
│    └── ContextManager           ← providers + ranker                │
│                                                                     │
│  Identity/   EmotionManager, RelationshipManager (runtime state)    │
│  Intelligence/ VoicePipelineCoordinator (STT↔LLM↔TTS handoff)       │
│  Integrations/ PlatformEventAdapter (bridges py/external → EventBus)│
│  UI/         BackendSyncService (publish state, receive config)     │
│  Shared/     Types, Enums, IEvent, IAction, IContextProvider,       │
│              IDecisionRule, IActionExecutor                         │
└────────────────────────────────────────────────────────────────────┘
                               ▲  ▼   (IPC: REST + WS, existing channels)
┌────────────────────────────────────────────────────────────────────┐
│                   revia_core_py  (SCRIPTING LAYER)                  │
│  Owns: model adapters, experimentation, analytics, admin tooling    │
├────────────────────────────────────────────────────────────────────┤
│  adapters/    ModelRouterAdapter (vLLM/OpenAI/etc.),                │
│               MemoryAdapter, ToolAdapter, WebSearchAdapter          │
│  decision_plugins/  (optional, isolated IDecisionRule plugins       │
│                     loaded over IPC by PluginManager in cpp)        │
│  experiments/  prompt_assembly_v2, reply_planner_prototypes         │
│  analytics/    reinforcement_learner, metrics export, eval harness  │
│  admin/        profile_engine, persona_manager, data migrations     │
│  integrations/ discord_bot, twitch_bot  (emit events to cpp bus)    │
└────────────────────────────────────────────────────────────────────┘
                               ▲  ▼
┌────────────────────────────────────────────────────────────────────┐
│                 revia_controller_py  (BACKEND UI)                   │
│  Owns: user-facing config, rendering, theme editor, monitoring      │
├────────────────────────────────────────────────────────────────────┤
│  app/       ControllerClient (thin — just IPC),                     │
│             AssistantStatusView (read-only snapshot consumer),      │
│             ThemeManager (full spec impl), BackendSyncClient        │
│  gui/       tabs/, widgets/ — consume theme tokens only             │
└────────────────────────────────────────────────────────────────────┘
```

### 2.2 Single-ownership table (target state)

| Concern | Single Owner | Lives in |
|---|---|---|
| Runtime state value | `StateManager` | core_cpp/Core |
| Event normalization | `EventBus` | core_cpp/Core |
| Context gathering | `ContextManager` | core_cpp/Core |
| Action selection | `DecisionEngine` | core_cpp/Core (rules may be py plugins) |
| Conflict resolution | `PriorityResolver` | core_cpp/Core |
| Timing & pacing | `TimingEngine` | core_cpp/Core |
| Action execution | `ActionDispatcher` | core_cpp/Core |
| Output moderation | `SafetyGateway` | core_cpp/Core |
| Feedback outcome → learning | `FeedbackManager` | core_cpp/Core (delegates metrics to py) |
| LLM/model routing | `ModelRouterAdapter` | core_py/adapters |
| Memory read/write | `MemoryAdapter` (wraps Redis/store) | core_py/adapters |
| Voice STT/TTS device | `VoicePipelineCoordinator` | core_cpp/Intelligence |
| Theme definition & validation | `ThemeManager` | controller_py/app |
| Theme publish/apply flow | `BackendSyncService` (cpp) ↔ `BackendSyncClient` (py) | both |
| Emotion state | `EmotionManager` | core_cpp/Identity |
| Relationship affinity | `RelationshipManager` | core_cpp/Identity |
| Profile persistence | `ProfileAdmin` | core_py/admin |

### 2.3 Canonical runtime flow (§Core Runtime Flow in spec)

```text
External input ─┐
                ├─▶ EventBus.publish(IEvent)
Integrations ───┘           │
                            ▼
                  CoreOrchestrator.onEvent(IEvent)
                            │
   ┌────────────────────────┼─────────────────────────┐
   ▼                        ▼                         ▼
StateManager.snapshot  ContextManager.build      TimingEngine.pre-check
                            │
                            ▼
                    DecisionEngine.evaluate(ctx, state)
                            │
                            ▼
                    PriorityResolver.resolve(decision, ctx, state)
                            │
                            ▼
                    TimingEngine.applyTiming(decision, ctx, state)
                            │
               [should_act ? yes : publish & return]
                            │
                            ▼
                    SafetyGateway.validate(decision, ctx)
                            │
           [allowed ? dispatch : fallback action]
                            │
                            ▼
                    ActionDispatcher.dispatch(IAction)
                            │
                            ▼
                      ExecutionResult
                            │
                            ▼
                    FeedbackManager.process(result, ctx, state)
                            │
                            ▼
                    BackendSyncService.publish(state, decision, result)
```

Every step emits a structured log line (spec §Logging and Debug Expectations).

### 2.4 Key interfaces (C++ sketches; compile-ready later, design now)

```cpp
// core_cpp/include/revia/core/IEvent.h
namespace revia::core {
  enum class EventType { UserText, UserSpeech, SilenceTimeout, PlatformEvent,
                         ModelCompletion, FilterResult, MemoryUpdate,
                         EmotionShift, Interruption, ConfigChange };
  enum class EventSource { Discord, Twitch, LocalSTT, ControllerUI,
                           InternalTimer, InternalModel };

  struct IEvent {
    std::string id;              // UUID
    EventType type;
    EventSource source;
    std::chrono::system_clock::time_point created_at;
    nlohmann::json payload;      // typed payload per EventType
    virtual ~IEvent() = default;
  };
}
```

```cpp
// core_cpp/include/revia/core/IAction.h
namespace revia::core {
  enum class ActionType { SpeakResponse, SendTextResponse, AskClarifyingQuestion,
                          WaitSilently, FollowUp, IgnoreEvent, TriggerFallback,
                          UpdateInternalStateOnly, InterruptCurrentOutput,
                          QueueProactiveMessage };
  struct IAction {
    std::string id;
    ActionType type;
    int priority;
    ContextPackage context;      // snapshot at decision time
    virtual ~IAction() = default;
  };
}
```

```cpp
// core_cpp/include/revia/core/IContextProvider.h
namespace revia::core {
  class IContextProvider {
  public:
    virtual ContextFragment Collect(const IEvent& e, const RuntimeState& s) = 0;
    virtual std::string Name() const = 0;
    virtual ~IContextProvider() = default;
  };
}
```

```cpp
// core_cpp/include/revia/core/IDecisionRule.h
namespace revia::core {
  class IDecisionRule {
  public:
    virtual DecisionInfluence Evaluate(const ContextPackage& ctx,
                                       const RuntimeState& s) = 0;
    virtual std::string Name() const = 0;
    virtual ~IDecisionRule() = default;
  };
}
```

```cpp
// core_cpp/include/revia/core/IActionExecutor.h
namespace revia::core {
  class IActionExecutor {
  public:
    virtual ExecutionResult Execute(const IAction& a) = 0;
    virtual ActionType Handles() const = 0;
    virtual ~IActionExecutor() = default;
  };
}
```

Python decision plugins conform to the same shape over IPC (JSON contracts), loaded by the existing `plugin_manager.cpp` — this is why user explicitly said *"optional decision plugins if isolated safely"*.

---

## 3. Build/Update — Module-by-Module Plan (Strangler-Fig)

The strategy is **additive first, migrative second, deletive last.** We build the Core layer in C++ alongside existing code. We route one event type through it end-to-end. We migrate the rest module-by-module. We delete dead code only after the last caller is gone.

### Phase 0 — Foundation (no behavior change)

**P0.1 Create `revia_core_cpp/src/core/` package with headers only.**
- Files: `IEvent.h`, `IAction.h`, `IContextProvider.h`, `IDecisionRule.h`, `IActionExecutor.h`, `RuntimeState.h`, `ContextPackage.h`, `DecisionResult.h`, `ExecutionResult.h`, `Enums.h`.
- No `.cpp` yet. Interfaces and POD structs only.
- Update `CMakeLists.txt` to add `core/` as its own library target.

**P0.2 Introduce `StructuredLogger`.**
- Every Core module logs: event, state, ranked context, influences, overrides, timing mods, action, safety, execution, learning. One JSON line per step.
- Python `logging` handler bridge so existing py logs join the same stream.

**P0.3 Freeze legacy runtime behavior behind a feature flag.**
- Add `REVIA_CORE_V2_ENABLED` env var, default off.
- When off, everything runs as today. When on, events route through the new orchestrator (empty pass-through initially).

### Phase 1 — EventBus + StateManager

**P1.1 `EventBus` (cpp)**
- Owner: `core_cpp/Core/EventBus`.
- Internally: a lock-free MPMC queue + topic subscribe table.
- Must: normalize (fill id/timestamp/source if missing), route to subscribers, never block producers.
- Pseudocode:
  ```pseudo
  class EventBus {
      publish(IEvent e) {
          if (e.id.empty()) e.id = uuid()
          if (e.created_at == null) e.created_at = now()
          queue.push(e)
      }
      subscribe(EventType t, handler) { ... }
      private worker_loop() {
          while (running) {
              IEvent e = queue.pop_blocking()
              for h in subscribers[e.type]: h(e)
          }
      }
  }
  ```
- **Migration bridge:** existing `revia_controller_py/app/event_bus.py` (Qt signals) stays, but gains a forwarding adapter that republishes selected signals onto the cpp bus over WS.

**P1.2 `StateManager` (cpp)**
- Owner: `core_cpp/Core/StateManager`.
- Owns `RuntimeState` (spec §Example Core Data Models).
- Transition table explicit; invalid transitions throw, do not silently accept.
- Exposes snapshot to BackendSync (read-only copy).
- **Retires:** `_set_runtime_state()` in `core_server.py:3981`, `revia_states.py` constants (kept as deprecation aliases), `EAssistantState` enum in `assistant_status_manager.py` (becomes a thin mapping to core state).

**P1.3 First end-to-end route**
- Route **only `EventType::UserText`** from Discord → new EventBus → empty orchestrator → no-op action.
- Success criterion: structured log shows the event flowing through every stage with no regressions in user-visible behavior.

### Phase 2 — Orchestrator + Context + Decision skeleton

**P2.1 `CoreOrchestrator` (cpp)**
- Owner: `core_cpp/Core/CoreOrchestrator`.
- Exactly the pseudocode from `REVIA_CORE.md` §Example Pseudocode — Core Orchestrator.
- Depends only on interfaces, not concrete providers/rules/executors. Registrations happen at boot.

**P2.2 `ContextManager` + providers (cpp, providers may call py over IPC)**
- Owner: `core_cpp/Core/ContextManager`.
- First-day providers (C++ side, talk to py adapters as needed):
  - `ConversationContextProvider` — reads recent conversation window
  - `ProfileContextProvider` — reads active profile snapshot
  - `PlatformContextProvider` — reads platform flags (Discord/Twitch constraints)
  - `EmotionContextProvider` — pulls from `EmotionManager`
- Add `RelationshipContextProvider`, `MemoryContextProvider`, `VoiceContextProvider`, `VisionContextProvider`, `CapabilityContextProvider` in Phase 4.
- `ContextRanker` is a separate class, swappable.

**P2.3 `DecisionEngine` + initial rules**
- Owner: `core_cpp/Core/DecisionEngine`.
- First-day rules:
  - `ResponseEligibilityRule`
  - `IntentConfidenceRule`
  - `InterruptionRule`
  - `PlatformConstraintRule`
- Rules are **small objects** implementing `IDecisionRule`. No rule may mutate state directly.
- **Retires:** decision branches inside `core_server.py:process_pipeline` lines ~3415–3839.

**P2.4 `PriorityResolver`**
- Owner: `core_cpp/Core/PriorityResolver`.
- Uses explicit precedence table: `Interruption > Safety > PlatformConstraint > Profile > Emotion > Relationship > Timing`.
- Logs every override applied.

### Phase 3 — TimingEngine + ActionDispatcher + SafetyGateway

**P3.1 `TimingEngine`**
- Owner: `core_cpp/Core/TimingEngine`.
- Absorbs: `anti_loop_engine.py`, pacing logic in `human_feel_layer.py`, silence timeouts in `core_server.py`, cooldowns in `conversation_policy.py`, live timers in `assistant_status_manager.py`.
- Exposes: `response_delay`, `silence_threshold`, `proactive_followup_timer`, `burst_window`, `anti_loop_cooldown`, `recovery_backoff`.
- **Hard rule:** no other module may own a timer that influences behavior. UI-only display timers are fine locally.

**P3.2 `ActionDispatcher`**
- Owner: `core_cpp/Core/ActionDispatcher`.
- Registry: `Map<ActionType, IActionExecutor>`.
- Executors:
  - `SpeakResponseExecutor` → coordinates `VoicePipelineCoordinator` (STT/LLM/TTS)
  - `SendTextResponseExecutor` → writes to integration adapter
  - `InterruptCurrentOutputExecutor` → cancels active speech
  - `WaitSilentlyExecutor`, `IgnoreEventExecutor`, `FollowUpExecutor`, `AskClarifyingQuestionExecutor`, `QueueProactiveMessageExecutor`, `TriggerFallbackExecutor`, `UpdateInternalStateOnlyExecutor`.
- Executors are **thin** — they call into adapters (py) and Identity/Intelligence managers (cpp). They never decide.

**P3.3 `SafetyGateway`**
- Owner: `core_cpp/Core/SafetyGateway`.
- Chain pattern: `HardFilter → AIFilter → PlatformRules → ModeRules → ProfileOverrides → FailSafe`.
- Each layer is a small class with a single `Validate(decision, ctx) : SafetyResult` method.
- Absorbs: `answer_validation.py` (moves to filter layer classes), filter toggles in `FiltersTab` (now read from config, applied here).

### Phase 4 — Remaining context providers, FeedbackManager, BackendSync

**P4.1 Remaining providers** — Memory, Relationship, Voice, Vision, Capability.
- Memory provider calls py `MemoryAdapter` over IPC.
- Vision provider reads from camera_service snapshot (controller side) via BackendSync.

**P4.2 `FeedbackManager`**
- Owner: `core_cpp/Core/FeedbackManager`.
- Drives: `EmotionManager.ApplyFailureSignal`, `RelationshipManager.ApplyPositiveSignal`, `LearningManager.ApplyOutcome`, `MetricsCollector.Record`.
- **Retires:** direct calls from `process_pipeline` into `reinforcement_learner.py`.
- `reinforcement_learner.py` itself moves to `revia_core_py/analytics/` as a pure analytic module called by `FeedbackManager` through a `ILearningSink` interface.

**P4.3 `BackendSyncService` (cpp) ↔ `BackendSyncClient` (py controller)**
- cpp side: publish RuntimeState, DecisionResult, ExecutionResult, theme state, config snapshot.
- py controller side: subscribe to all of the above, push user config changes through a validated apply path (`ConfigChange` event on the bus, never direct mutation).
- **Retires:** scattered WS broadcast helpers in `core_server.py`.

### Phase 5 — Theme System (full spec compliance)

**P5.1 `ThemeManager` (controller-side, full impl)**
- File: `revia_controller_py/app/theme_manager.py` — **rewrite** the 22-line stub.
- Owns: `ThemeDefinition` registry, active selection, validation, token generation.
- Implements spec §Theme System: built-in Light/Dark, custom themes, per-token editor, live preview, save/load/delete custom themes, contrast validation, per-user persistence.
- Tokens: `PrimaryBackground, SecondaryBackground, Surface, SurfaceAlt, PrimaryText, SecondaryText, Accent, AccentHover, AccentActive, Border, ButtonPrimary, ButtonSecondary, Success, Warning, Error, Info, Disabled`.
- Validation: contrast ratio ≥ 4.5:1 for body text (WCAG AA), ≥ 3:1 for large text; warn otherwise.

**P5.2 Widget refactor**
- All widgets under `revia_controller_py/gui/widgets/` and `gui/tabs/` must stop reading hardcoded colors.
- Generate Qt stylesheet dynamically from active `ThemeDefinition` tokens. Single `StyleComposer` builds the QSS string.
- `.qss` files become templates with `${token}` placeholders, or are retired in favor of composed styles.

**P5.3 Theme sync through Core**
- Theme changes flow: `UI editor → ThemeManager.SaveCustomTheme → BackendSyncClient → ConfigChange event → cpp BackendSyncService → validated apply → published ThemeState → UI refresh`.
- **Rule:** UI never mutates active theme state on the cpp side directly.

**P5.4 Theme editor UI**
- New tab or panel in controller: selector, preset list, color editor with live preview, reset, save-as, edit, delete, accessibility warnings.
- Reuses existing tab infrastructure under `gui/tabs/`.

### Phase 6 — Cleanup and deletion

Only after the last caller is migrated:
- Retire `core_server.py` god module. Split residue into: `revia_core_py/adapters/llm.py`, `…/adapters/memory.py`, `…/adapters/web_search.py`, `…/admin/profile_io.py`. Delete what's obsolete.
- Retire `conversation_runtime.py` (duplicate runtime).
- Retire `assistant_status_manager.py` god Qt object. Split into:
  - `AssistantStatusView` (read-only snapshot consumer, ~150 LOC)
  - `RequestTimingCollector` (if still needed client-side; probably moves to TimingEngine on cpp)
  - `StatusFormatter` (pure formatting helpers)
- Retire `anti_loop_engine.py`, `human_feel_layer.py`, `interruption_handler.py` — their logic now lives inside TimingEngine/DecisionEngine/PriorityResolver.
- Retire module-level `_set_runtime_state`, `_mark_user_activity`, `_seconds_since_user_activity` etc. in `core_server.py`.

### Phase-exit checklist (applied at each phase boundary)

1. No new hidden dependencies introduced (search for module-level state mutation).
2. Every Core module has exactly one owner class.
3. Every public method has an interface or a `# pragma: no-interface` comment justifying why.
4. Structured logs emitted at every required point.
5. Feature flag `REVIA_CORE_V2_ENABLED` still toggles old vs. new cleanly.
6. Tests: orchestrator unit tests, state transition table tests, rule evaluation tests, executor registration tests, timing rule tests, safety chain tests.

---

## 4. Validate — Checklist against REVIA_CORE.md and Modular Coding Skill

### 4.1 Core Responsibilities checklist (spec §Core Responsibilities)

| Responsibility | Current | After plan |
|---|---|---|
| Receive and normalize events | Partial (Qt UI only) | ✅ EventBus (cpp) |
| Track runtime state | Scattered | ✅ StateManager single-owner |
| Gather relevant context | Ad-hoc inside pipeline | ✅ ContextManager + providers |
| Score context importance | Absent | ✅ ContextRanker |
| Resolve conflicts between systems | Absent | ✅ PriorityResolver with explicit precedence |
| Choose actions | Inlined in god function | ✅ DecisionEngine + rules |
| Enforce timing and pacing | Scattered across 5 modules | ✅ TimingEngine single-owner |
| Pass outputs through safety systems | Ad-hoc | ✅ SafetyGateway chain |
| Dispatch execution | Inline side effects | ✅ ActionDispatcher + executors |
| Collect feedback | Partial, out-of-band | ✅ FeedbackManager orchestrated |
| Update learning signals | Direct calls | ✅ ILearningSink via FeedbackManager |

### 4.2 Core Design Goals (spec §Core Design Goals)

| Goal | Currently | After |
|---|---|---|
| Modular and swappable | No | Yes — interfaces on every boundary |
| Explicit ownership | No | Yes — §2.2 ownership table |
| Event-driven | No (call-driven) | Yes |
| Debuggable | Weak | Yes — structured log at every step |
| Explainable | No | Yes — `ReasonSummary` in `DecisionResult` |
| Stable under failure | No | Yes — Recovering state + fallback actions |
| Profile-aware | Partial | Yes — ProfileContextProvider |
| Learning-compatible | Bolted on | Yes — FeedbackManager |
| Safe but expressive | Filters bolted on | Yes — SafetyGateway layered |

### 4.3 Modular Coding Skill rules (MODULAR_CODING_SKILL.md §RULES)

| Rule | Compliance |
|---|---|
| Prioritize correctness | ✅ Phase 1 routes *one* event end-to-end before expanding |
| Prioritize maintainability | ✅ god modules split; single-owner rule enforced |
| Avoid overengineering | ✅ Interfaces exist where boundaries are needed; executors are thin; no speculative patterns |
| Avoid hidden dependencies | ✅ Module-level globals retired in Phase 6; all state passes through owners |
| Enforce modular design | ✅ Interfaces + DI at orchestrator boot |

### 4.4 Spec design warnings (spec §Design Warnings)

| Avoid | Currently present? | Addressed by |
|---|---|---|
| Giant god classes | Yes — core_server.py, assistant_status_manager.py | Phase 6 split |
| Emotion logic scattered in multiple systems | Yes | EmotionManager single-owner + EmotionContextProvider |
| Memory retrieval directly deciding action | Yes (inline in process_pipeline) | Goes through DecisionEngine |
| UI bypassing core ownership rules | Yes (runtime_state_sync directly mutates) | Theme + config flow via ConfigChange event |
| Action execution containing decision logic | Yes (pipeline stages decide in-line) | Executors are thin; decisions in DecisionEngine |
| Duplicated timing logic | Yes (5 modules) | TimingEngine single-owner |

### 4.5 Theme System requirements (spec §Theme System Requirements)

| Requirement | Plan reference |
|---|---|
| Built-in themes (Light, Dark, Custom) | P5.1 |
| Allow user-created themes | P5.1 |
| Edit all required tokens | P5.1 token list |
| Live preview | P5.4 |
| Validate readability (contrast) | P5.1 WCAG AA check |
| Per-user persistence | P5.1 persistence; P5.3 sync |
| Widgets read tokens, no hardcoded colors | P5.2 StyleComposer |
| Validated save flow | P5.3 (goes through BackendSync) |
| Centralized ownership | P5.1 ThemeManager single-owner |

---

## 5. Refactor/Recheck — Risks, Hidden Issues, Extensibility

### 5.1 Risks (ordered by severity)

**R1 — Performance regression from IPC for context providers.**
*Cause:* Several providers will want Python adapters (memory, model metadata, reinforcement stats) while living in the cpp core.
*Mitigation:* Providers run on a thread pool; Context assembly has a hard timeout (e.g. 150 ms); expired fragments are dropped with a log signal `context_fragment_timeout`. Orchestrator is never blocked by a slow provider.

**R2 — Feature flag decay.**
*Cause:* `REVIA_CORE_V2_ENABLED` off-path and on-path diverge over time; the off-path becomes untested.
*Mitigation:* CI runs both paths. Off-path is deleted at Phase 6 exit. Any commit after Phase 2 must update both paths or the test suite flags it.

**R3 — Qt signal bus vs. cpp EventBus confusion.**
*Cause:* Developers may emit into the wrong bus.
*Mitigation:* Qt signal bus renamed to `UiEventBus` and scoped strictly to UI-local events (redraws, widget state). All behavior-affecting events go through the cpp bus via `BackendSyncClient`.

**R4 — Rule proliferation.**
*Cause:* Spec lists 10 rule categories; easy to add a hundred ad-hoc rules.
*Mitigation:* Every rule requires a one-page RFC stub in `docs/rules/`. Rules grouped by category; category ordering in PriorityResolver is explicit.

**R5 — Theme contrast false positives.**
*Cause:* WCAG contrast rules are non-trivial for semi-transparent surfaces.
*Mitigation:* Validation warns, does not hard-block. Users can override with a logged acknowledgement. Phase 5.1 ships a minimal validator; a fuller one comes in Phase 5 follow-up.

**R6 — Hidden dependencies not detected by eye.**
*Cause:* Module globals are easy to miss.
*Mitigation:* Add a lint rule (ruff / custom check) that fails if `revia_core_py/adapters/` or `controller_py/app/` modules define top-level mutable state beyond explicit constants.

**R7 — Voice pipeline latency.**
*Cause:* Adding Orchestrator + Safety + Dispatch layers to STT→LLM→TTS risks audible delay.
*Mitigation:* Voice pipeline keeps its own fast path inside `VoicePipelineCoordinator` for token streaming; Safety runs per-sentence (already the unit for TTS) not per-token. Measured latency budget: ≤ 30 ms overhead on top of current.

**R8 — Integration adapters emitting unnormalized events.**
*Cause:* Discord/Twitch bots currently push straight into the Qt signal bus.
*Mitigation:* `PlatformEventAdapter` in `core_cpp/Integrations/` normalizes every external input. Discord/Twitch py bots emit via a single WS endpoint that the adapter subscribes to.

**R9 — Plugin-based decision rules (py) becoming a trust boundary.**
*Cause:* Python plugin rules running inside the decision loop is a safety risk (crashes, bad output).
*Mitigation:* Plugins isolated in subprocesses via existing `plugin_manager.cpp`; contract is JSON over stdio; plugin errors degrade to `WaitSilently` action, never block the loop. This matches user's *"if isolated safely"* constraint.

**R10 — Theme editor can brick the UI.**
*Cause:* A saved theme with unreadable contrast or missing tokens can make controls unusable.
*Mitigation:* `RequiredTokensPresent()` + contrast validator before `SetActiveTheme`; auto-rollback to last-known-good theme on failure (ThemeManager keeps previous active). Emergency keyboard shortcut resets to built-in Dark.

### 5.2 Hidden coupling audit (things the review surfaced that aren't obvious)

1. `runtime_state_sync.py` reads directly from `model_tab`, `voice_tab`, `filters_tab`, etc., then pushes to the server — this is a UI-widgets-directly-feeding-runtime-config path. Plan fixes by routing through `ConfigChange` events.
2. `_PipelineInterrupted` exception (`core_server.py:3409`) is thrown from deep stages and caught at the top — effectively a hidden control-flow signal replacing what should be a state transition.
3. `sing_mode.py` (839 LOC) is a parallel state machine (sing states, queue, progress). It should become an `IActionExecutor` chain plus a `SingController` in Intelligence, not a peer to the main runtime.
4. `TelemetryEngine` is a singleton accessed globally. It should be an injected dependency on the orchestrator; tests cannot run without polluting global metrics today.
5. `LLMBackend` (~1,000 LOC inside `core_server.py`) mixes request shaping, streaming, retry logic, prompt injection, and response cleanup. The prompt/injection/cleanup parts belong in `ModelRouterAdapter` (py); the streaming/retry parts belong in a `LlmClient` inside the cpp core.
6. Emotion writes happen from *at least three* places (`core_server.py:_record_emotion`, `reinforcement_learner.py`, `conversation_runtime.py`). Single-owner rule requires only `EmotionManager` writes.

### 5.3 Extensibility — did we preserve the spec's extension rules?

| Extension rule | Plan satisfies it? |
|---|---|
| Add new decision rules without rewriting orchestrator | Yes — `DecisionEngine` iterates `List<IDecisionRule>` |
| Add new context providers without breaking flow | Yes — `ContextManager` iterates `List<IContextProvider>` |
| Add new action types by registering executors | Yes — `ActionDispatcher` registry by `ActionType` |
| Add new integrations emitting events into same bus | Yes — `PlatformEventAdapter` + EventBus |
| Backend UI can inspect all critical runtime signals | Yes — `BackendSyncService` publishes everything |

### 5.4 Recheck against Modular Coding Skill default workflow

- **Review** — done in §1.
- **Architect** — done in §2.
- **Build/Update** — planned in §3 (strangler-fig).
- **Validate** — done in §4.
- **Refactor** — Phase 6 (and §5.2 audits will feed smaller refactors across phases).
- **Recheck** — performed here in §5; phase-exit checklist in §3 keeps rechecking.
- **Output** — this document.

No step is skipped. No stage depends on a later stage in a way that creates a cycle.

---

## 6. Final Validated Structure

```text
REVIA/
├── revia_core_cpp/                         (CANONICAL RUNTIME)
│   ├── CMakeLists.txt
│   └── src/
│       ├── main.cpp                         (boots Core + adapters + servers)
│       ├── core/
│       │   ├── CoreOrchestrator.{h,cpp}
│       │   ├── EventBus.{h,cpp}
│       │   ├── StateManager.{h,cpp}
│       │   ├── ContextManager.{h,cpp}
│       │   ├── ContextRanker.{h,cpp}
│       │   ├── DecisionEngine.{h,cpp}
│       │   ├── PriorityResolver.{h,cpp}
│       │   ├── TimingEngine.{h,cpp}
│       │   ├── ActionDispatcher.{h,cpp}
│       │   ├── SafetyGateway.{h,cpp}
│       │   ├── FeedbackManager.{h,cpp}
│       │   └── StructuredLogger.{h,cpp}
│       ├── interfaces/
│       │   ├── IEvent.h  IAction.h  IContextProvider.h
│       │   ├── IDecisionRule.h  IActionExecutor.h  ILearningSink.h
│       ├── models/
│       │   ├── RuntimeState.h  ContextPackage.h
│       │   ├── DecisionResult.h  ExecutionResult.h  Enums.h
│       ├── rules/
│       │   ├── ResponseEligibilityRule.{h,cpp}
│       │   ├── IntentConfidenceRule.{h,cpp}
│       │   ├── InterruptionRule.{h,cpp}
│       │   ├── EmotionInfluenceRule.{h,cpp}
│       │   ├── ProfileBehaviorRule.{h,cpp}
│       │   ├── RelationshipAffinityRule.{h,cpp}
│       │   ├── SilenceFollowUpRule.{h,cpp}
│       │   ├── PlatformConstraintRule.{h,cpp}
│       │   ├── SafetyPreparationRule.{h,cpp}
│       │   └── RecoveryModeRule.{h,cpp}
│       ├── providers/
│       │   ├── ConversationContextProvider.{h,cpp}
│       │   ├── MemoryContextProvider.{h,cpp}
│       │   ├── ProfileContextProvider.{h,cpp}
│       │   ├── EmotionContextProvider.{h,cpp}
│       │   ├── RelationshipContextProvider.{h,cpp}
│       │   ├── PlatformContextProvider.{h,cpp}
│       │   ├── VoiceContextProvider.{h,cpp}
│       │   ├── VisionContextProvider.{h,cpp}
│       │   └── CapabilityContextProvider.{h,cpp}
│       ├── executors/
│       │   ├── SpeakResponseExecutor.{h,cpp}
│       │   ├── SendTextResponseExecutor.{h,cpp}
│       │   ├── InterruptCurrentOutputExecutor.{h,cpp}
│       │   ├── AskClarifyingQuestionExecutor.{h,cpp}
│       │   ├── WaitSilentlyExecutor.{h,cpp}
│       │   ├── FollowUpExecutor.{h,cpp}
│       │   ├── IgnoreEventExecutor.{h,cpp}
│       │   ├── QueueProactiveMessageExecutor.{h,cpp}
│       │   ├── TriggerFallbackExecutor.{h,cpp}
│       │   └── UpdateInternalStateOnlyExecutor.{h,cpp}
│       ├── safety/
│       │   ├── HardFilter.{h,cpp}    AiFilter.{h,cpp}
│       │   ├── PlatformRules.{h,cpp}  ModeRules.{h,cpp}
│       │   ├── ProfileOverrides.{h,cpp}  FailSafeRules.{h,cpp}
│       ├── identity/
│       │   ├── EmotionManager.{h,cpp}
│       │   ├── RelationshipManager.{h,cpp}
│       │   └── ProfileSnapshot.{h,cpp}   (read-model; admin writes live in py)
│       ├── intelligence/
│       │   ├── VoicePipelineCoordinator.{h,cpp}
│       │   ├── LlmClient.{h,cpp}         (streaming/retry only)
│       │   └── ModelRouterBridge.{h,cpp} (thin bridge to py adapter)
│       ├── integrations/
│       │   └── PlatformEventAdapter.{h,cpp}
│       ├── ui_sync/
│       │   ├── BackendSyncService.{h,cpp}
│       │   └── MonitoringPublisher.{h,cpp}
│       ├── api/                  rest_server, ws_server (kept)
│       ├── neural/               EmotionNet, RouterClassifier (kept)
│       ├── plugins/              plugin_manager (kept; now hosts py decision plugins)
│       └── telemetry/            TelemetryEngine (kept; now DI-injected)
│
├── revia_core_py/                         (SCRIPTING / EXPERIMENTATION)
│   ├── adapters/
│   │   ├── llm_adapter.py        (vLLM + API routing; was LLMBackend)
│   │   ├── memory_adapter.py     (was MemoryStore)
│   │   ├── web_search_adapter.py (was WebSearchEngine)
│   │   └── model_router_adapter.py
│   ├── admin/
│   │   ├── profile_admin.py
│   │   ├── persona_manager.py
│   │   └── prompt_library.py
│   ├── analytics/
│   │   ├── reinforcement_learner.py   (now implements ILearningSink over IPC)
│   │   ├── metrics_export.py
│   │   └── eval_harness.py
│   ├── decision_plugins/                 (optional py-side IDecisionRule plugins,
│   │                                       isolated subprocess, JSON contract)
│   ├── experiments/
│   │   ├── prompt_assembly_v2.py         (was prompt_assembly)
│   │   └── reply_planner_prototypes.py   (was reply_planner)
│   ├── integrations/
│   │   ├── discord_bot.py    twitch_bot.py    integration_manager.py
│   │   (all emit via PlatformEventAdapter WS endpoint)
│   ├── ipc/
│   │   ├── event_publisher.py
│   │   └── rule_plugin_runner.py
│   └── tests/
│
├── revia_controller_py/                   (BACKEND UI)
│   ├── app/
│   │   ├── controller_client.py           (thin IPC client; shrunk from 707 LOC)
│   │   ├── assistant_status_view.py       (read-only; replaces AssistantStatusManager)
│   │   ├── backend_sync_client.py         (subscribes to cpp Core publish stream)
│   │   ├── theme_manager.py               (FULL spec impl; tokens, validation, save/load)
│   │   ├── style_composer.py              (builds Qt stylesheet from active ThemeDefinition)
│   │   ├── ui_event_bus.py                (renamed event_bus.py; UI-local only)
│   │   ├── audio_service.py  continuous_audio.py  tts_backend.py
│   │   ├── voice_manager.py  voice_library.py  voice_profile.py
│   │   ├── camera_service.py
│   │   ├── sing_controller.py             (shrunk; emits Events, registers executors)
│   │   └── …
│   └── gui/
│       ├── main_window.py
│       ├── tabs/       (consume theme tokens via StyleComposer)
│       ├── widgets/    (consume theme tokens; no hardcoded colors)
│       ├── theme_editor/ (new — selector, editor, preview, accessibility warnings)
│       └── qss/        (templates with ${token} placeholders, or retired)
│
└── docs/
    ├── REVIA_CORE_REVIEW.md              (this document)
    ├── rules/                            (one-page RFC per decision rule)
    ├── adrs/                             (architecture decision records)
    └── runbooks/                         (operational docs)
```

### 6.1 Deliverable summary

- **Review of gaps:** §1.2 mapping matrix + §1.3 smells list.
- **Architecture changes:** §2 ownership split, single-owner table, runtime flow, key interfaces.
- **Implementation/update plan:** §3 six-phase strangler-fig plan with phase-exit checklist.
- **Risks and hidden issues:** §5.1 ten risks with mitigations + §5.2 hidden coupling audit.
- **Final validated structure:** §6 target tree.

### 6.2 Definition of done (for the overall migration, not this doc)

1. `REVIA_CORE_V2_ENABLED` flag removed; only Core path exists.
2. `core_server.py` no longer contains any runtime logic; only adapters remain (or it is deleted entirely).
3. `assistant_status_manager.py` is gone; replaced by `AssistantStatusView`.
4. Every module in §6 tree exists and has a single owner.
5. Structured log shows every step of spec §Core Runtime Flow.
6. Theme editor ships with built-in Light, built-in Dark, at least one custom theme round-trip, contrast warnings, and live preview.
7. CI runs orchestrator, state, rule, executor, timing, safety, and theme test suites.
8. No top-level mutable module state in `core_py/adapters/` or `controller_py/app/` (lint-enforced).

---

*End of review.*
