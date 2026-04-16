# REVIA Core — Phase 0 scaffolding

<!-- Phase 1 update lives at the end of this file. -->

This folder is the foundation of the strangler-fig migration described in
`../../../REVIA_CORE_REVIEW.md`. Phase 0 is **non-behavioral**: nothing here
is wired into the runtime yet.

## What Phase 0 delivers

| File | Purpose |
|---|---|
| `StructuredLogger.{h,cpp}` | Process-wide JSONL logger. Writes to `logs/revia_core.jsonl` and (optionally) stderr. Thread-safe. Batched flush every 50 entries. |
| `FeatureFlags.{h,cpp}` | Env-var-driven toggles. Master switch: `REVIA_CORE_V2_ENABLED`. |
| `../interfaces/I*.h` | Pure-virtual contracts: `IEvent`, `IAction`, `IContextProvider`, `IDecisionRule`, `IActionExecutor`, `ILearningSink`. |
| `../models/*.h` | POD types: `Enums`, `RuntimeState`, `ContextFragment`, `ContextPackage`, `ToneProfile`, `DelayProfile`, `SafetyProfile`, `SafetyResult`, `FeedbackSignal`, `DecisionInfluence`, `DecisionResult`, `ExecutionResult`. |

The existing CMake glob `file(GLOB_RECURSE SOURCES src/*.cpp)` picks up the
new `.cpp` files automatically. No CMakeLists.txt edit is required for
Phase 0.

## Environment variables

| Var | Default | Effect |
|---|---|---|
| `REVIA_CORE_V2_ENABLED` | `false` | Master switch. When off, legacy paths run unchanged. When on, Phase 1+ routes events through the new orchestrator. |
| `REVIA_CORE_V2_LOG_STDERR` | `true` | If false, suppresses the human-readable stderr line (JSONL file is unaffected). |
| `REVIA_CORE_LOG_PATH` | `logs/revia_core.jsonl` | Override the JSONL sink path (Python side; cpp uses its hardcoded default unless `StructuredLogger::set_file_path()` is called). |

## Log schema

One JSON object per line, UTF-8:

```json
{
  "ts":     "2026-04-16T12:34:56.789Z",
  "level":  "info",
  "stage":  "dotted.identifier",
  "thread": "<native thread id>",
  "fields": { "k": "v" }
}
```

Both the cpp `StructuredLogger` and the Python `revia_core_py.ipc.structured_log`
write this exact shape so the file can be tailed/grepped without caring which
runtime produced the line.

## How to use from existing code (Phase 0 — read-only)

**cpp:**

```cpp
#include "core/StructuredLogger.h"
#include "core/FeatureFlags.h"

using revia::core::StructuredLogger;
using revia::core::FeatureFlags;

if (FeatureFlags::instance().core_v2_enabled()) {
    StructuredLogger::instance().info("boot.v2_path", {{"note", "v2 on"}});
} else {
    StructuredLogger::instance().info("boot.legacy_path", {});
}
```

**Python:**

```python
from revia_core_py.ipc import core_log, is_core_v2_enabled, StructuredJsonlHandler
import logging

# Join legacy logging calls to the same JSONL stream:
logging.getLogger().addHandler(StructuredJsonlHandler())

core_log("boot.py_layer", {"v2_enabled": is_core_v2_enabled()})
```

## What Phase 0 intentionally does NOT do

- Create any engine (`CoreOrchestrator`, `EventBus`, `StateManager`, …).
- Wire any existing code to the logger or flags.
- Build a Python subprocess bridge for decision plugins.
- Touch `core_server.py`, `assistant_status_manager.py`, or any integration.

Those come in Phase 1+. Phase 0 is deliberately leaf-level so it can land
without any behavior risk.

## Exit criteria for Phase 0

- [x] `src/core/StructuredLogger.{h,cpp}` compile into the existing target.
- [x] `src/core/FeatureFlags.{h,cpp}` compile into the existing target.
- [x] `src/interfaces/` + `src/models/` headers compile standalone.
- [x] Python `revia_core_py.ipc` package importable; writes the same JSONL shape.
- [ ] Manual smoke test: set `REVIA_CORE_V2_ENABLED=1`, run any entry point,
      observe `feature_flags.loaded` in `logs/revia_core.jsonl` on both sides.
- [ ] CI: add a build job that compiles cpp headers + runs `pytest` on a
      minimal Python smoke test.

Last two items are Phase 0 → Phase 1 handoff work; the rest is shipped.

---

## Phase 1 - EventBus + StateManager

Phase 1 starts the behavioral spine without moving response ownership yet.
The legacy pipeline still handles visible replies. The new path accepts only
`EventType::UserText`, normalizes and queues it, logs the route, and completes a
no-op `UpdateInternalStateOnly` probe.

| File | Purpose |
|---|---|
| `EventBus.{h,cpp}` | Typed async event bus. Fills missing event id, timestamp, and correlation id; dispatches on a worker thread; logs handler errors without crashing producers. |
| `StateManager.{h,cpp}` | Authoritative `RuntimeState` owner. Holds the explicit transition table and rejects invalid transitions. |
| `../api/rest_server.{h,cpp}` | Adds `/api/core/events` for Phase 1 event ingress and `/api/core/state` for read-only snapshots. |
| `../main.cpp` | Wires a feature-flagged `UserText` subscriber that logs `orchestrator.phase1.*` and returns state to `Idle`. |
| `../../../revia_core_py/ipc/event_publisher.py` | Best-effort Python bridge to the cpp event endpoint. |
| `../../../revia_core_py/integrations/discord_bot.py` | Fire-and-forget Discord `UserText` forwarding; legacy Discord response flow remains unchanged. |

## Phase 1 smoke test

```powershell
$env:REVIA_CORE_V2_ENABLED='1'
cd revia_core_cpp
cmake --build build --config Release
.\build\Release\revia_core.exe

Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8123/api/core/events `
  -ContentType 'application/json' `
  -Body '{"type":"UserText","source":"Discord","payload":{"text":"phase 1 smoke"}}'

Invoke-RestMethod http://127.0.0.1:8123/api/core/state
Get-Content logs/revia_core.jsonl -Tail 20
```

Expected log stages:

- `feature_flags.loaded`
- `event_bus.started`
- `state.transition.accepted`
- `api.core_event.accepted`
- `event_bus.event_dequeued`
- `orchestrator.phase1.event_received`
- `orchestrator.phase1.noop_action`

## Phase 1 exit criteria

- [x] `EventBus` exists as the single owner for normalization, queueing, and dispatch.
- [x] `StateManager` exists as the single owner for runtime state transitions.
- [x] Invalid state transitions are rejected by an explicit table.
- [x] `/api/core/events` accepts only `UserText` in Phase 1.
- [x] Discord can forward `UserText` into the cpp event path without changing legacy response behavior.
- [ ] Manual smoke test: observe the expected log stages with `REVIA_CORE_V2_ENABLED=1`.
- [ ] CI: build cpp core and import `revia_core_py.ipc.event_publisher`.

---

## Phase 2 - Orchestrator + Context + Decision skeleton

Phase 2 moves the feature-flagged `UserText` path out of the Phase 1 inline
probe and into the first real Core orchestration loop. It still does not execute
actions. Timing, safety, dispatch, and feedback remain Phase 3+.

| File | Purpose |
|---|---|
| `CoreOrchestrator.{h,cpp}` | Owns the event -> state -> context -> decision -> priority flow. Subscribes to `UserText` and publishes structured decisions. |
| `ContextManager.{h,cpp}` | Owns provider coordination and builds `ContextPackage`. |
| `ContextRanker.{h,cpp}` | Scores and sorts context signals. |
| `../providers/DefaultContextProviders.{h,cpp}` | First-day providers: conversation, profile, platform, emotion. |
| `DecisionEngine.{h,cpp}` | Owns rule evaluation and initial action selection. |
| `../rules/DefaultDecisionRules.{h,cpp}` | First-day rules: response eligibility, intent confidence, interruption, platform constraints. |
| `PriorityResolver.{h,cpp}` | Applies explicit precedence overrides. |
| `../models/CoreAction.h` | Concrete action plan value used before Phase 3 executors exist. |

## Phase 2 runtime flow

```text
EventBus(UserText)
  -> CoreOrchestrator.OnEventReceived
  -> StateManager snapshot / Thinking
  -> ContextManager.BuildContext
  -> ContextRanker.Score
  -> DecisionEngine.Evaluate
  -> PriorityResolver.Resolve
  -> action plan logged
  -> StateManager Idle
```

Expected new log stages:

- `orchestrator.event_received`
- `context.build_started`
- `context.provider_collected`
- `context.build_completed`
- `decision.evaluate_started`
- `decision.rule_evaluated`
- `decision.evaluate_completed`
- `priority.resolve_completed`
- `orchestrator.action_planned`
- `orchestrator.event_finished`

## Phase 2 exit criteria

- [x] `CoreOrchestrator` owns the feature-flagged event path.
- [x] Context collection is provider-based and ranked.
- [x] Decision behavior is rule-based and side-effect free.
- [x] Priority conflict resolution is explicit and logged.
- [x] Phase 2 stops before execution; Phase 3 remains the owner for timing, safety, dispatch, and feedback.
- [ ] Manual smoke test: observe the expected Phase 2 log stages with `REVIA_CORE_V2_ENABLED=1`.
- [ ] CI: compile cpp core and run a focused orchestrator/rules smoke test.

---

## Phase 3 - TimingEngine + SafetyGateway + ActionDispatcher

Phase 3 makes the Core loop pass through timing, safety, and dispatch. The
registered executors are safe no-op executors for now, so legacy visible
behavior remains unchanged until adapter-backed executors are implemented.
Feedback processing remains Phase 4.

| File | Purpose |
|---|---|
| `TimingEngine.{h,cpp}` | Owns pacing decisions, recovery backoff, burst flags, and interruption timing overrides. |
| `SafetyGateway.{h,cpp}` | Owns ordered safety validation and layer registration. |
| `../safety/DefaultSafetyLayers.{h,cpp}` | First safety chain: HardFilter, AiFilter, PlatformRules, ModeRules, ProfileOverrides, FailSafe. |
| `ActionDispatcher.{h,cpp}` | Owns executor registry, action id fill, dispatch logging, and executor failure handling. |
| `../executors/DefaultActionExecutors.{h,cpp}` | Phase 3 no-op executors for all `ActionType` values. |
| `CoreOrchestrator.{h,cpp}` | Now runs `TimingEngine -> SafetyGateway -> ActionDispatcher` after priority resolution. |

## Phase 3 runtime flow

```text
EventBus(UserText)
  -> CoreOrchestrator.OnEventReceived
  -> StateManager snapshot / Thinking
  -> ContextManager.BuildContext
  -> DecisionEngine.Evaluate
  -> PriorityResolver.Resolve
  -> TimingEngine.ApplyTiming
  -> SafetyGateway.Validate
  -> ActionDispatcher.Dispatch
  -> StateManager Idle
```

Expected new log stages:

- `timing.apply_completed`
- `safety.validate_started`
- `safety.layer_evaluated`
- `safety.validate_completed`
- `orchestrator.action_ready`
- `action.id_filled`
- `action.dispatch_started`
- `executor.noop_executed`
- `action.dispatch_completed`
- `orchestrator.execution_completed`

## Phase 3 exit criteria

- [x] Timing ownership is centralized in `TimingEngine`.
- [x] Safety ownership is centralized in `SafetyGateway`.
- [x] Action dispatch ownership is centralized in `ActionDispatcher`.
- [x] Executors are registered by `ActionType`.
- [x] Response-producing executors remain no-op stubs until adapter contracts exist.
- [x] Orchestrator executes through timing, safety, and dispatch.
- [ ] Manual smoke test: observe the expected Phase 3 log stages with `REVIA_CORE_V2_ENABLED=1`.
- [ ] CI: compile cpp core and run focused timing/safety/dispatcher tests.

---

## Phase 4 - Remaining Context, Feedback, Backend Sync

Phase 4 completes the Core loop after execution by adding feedback processing
and backend sync publication. It also registers the remaining context providers
as lightweight placeholders until IPC adapters are implemented.

| File | Purpose |
|---|---|
| `FeedbackManager.{h,cpp}` | Owns execution outcome processing and learning sink fanout. |
| `../feedback/DefaultLearningSinks.{h,cpp}` | Structured-log learning sink for safe first feedback pass. |
| `../ui_sync/BackendSyncService.{h,cpp}` | Publishes state/decision/execution/config/theme snapshots and receives `ConfigChange`. |
| `../providers/DefaultContextProviders.*` | Adds relationship, memory, voice, vision, and capability providers. |

## Phase 5 - Theme System

Controller-side theme ownership moved from the old QSS toggle stub to:

- `revia_controller_py/app/theme_manager.py`
- `revia_controller_py/app/style_composer.py`
- `revia_controller_py/app/backend_sync_client.py`

The new theme manager owns built-in themes, custom theme persistence, token
generation, contrast validation, preview, reset, save, and delete flow.

## Phase 6 - Cleanup Boundaries

Cleanup is now bounded by replacement modules rather than destructive deletion.
Live legacy modules remain until imports are migrated.

- `revia_controller_py/app/assistant_status_view.py`
- `revia_controller_py/app/status_formatter.py`
- `revia_controller_py/app/request_timing_collector.py`
- `revia_core_py/adapters/*`
- `revia_core_py/admin/profile_io.py`
- `docs/phase6_cleanup_boundaries.md`

## Phase 4-6 exit criteria

- [x] Feedback flow exists after dispatch.
- [x] Learning updates are behind `ILearningSink`.
- [x] Backend sync receives `ConfigChange` events and publishes Core snapshots.
- [x] Remaining context providers are registered.
- [x] Theme system owns tokens, validation, persistence, and generated QSS.
- [x] Phase 6 replacement boundaries exist without deleting live callers.
- [ ] Manual smoke test with `REVIA_CORE_V2_ENABLED=1`.
- [ ] CI: compile cpp core and run feedback/sync/theme tests.
