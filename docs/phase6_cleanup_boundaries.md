# Phase 6 Cleanup Boundaries

Phase 6 is a deletion phase only after callers have moved. In this pass the
replacement boundaries exist, but the legacy modules remain because the current
controller and Python runtime still import them.

## Legacy Module -> Replacement Boundary

| Legacy module | Replacement boundary |
|---|---|
| `revia_core_py/core_server.py` | `revia_core_py/adapters/*`, `revia_core_py/admin/profile_io.py`, C++ Core owners |
| `revia_core_py/conversation_runtime.py` | `revia_core_cpp/src/core/CoreOrchestrator.*` |
| `revia_core_py/anti_loop_engine.py` | `revia_core_cpp/src/core/TimingEngine.*` |
| `revia_core_py/human_feel_layer.py` | `TimingEngine`, `DecisionEngine`, future tone providers |
| `revia_core_py/interruption_handler.py` | `PriorityResolver`, `TimingEngine`, `InterruptCurrentOutputExecutor` |
| `revia_controller_py/app/assistant_status_manager.py` | `assistant_status_view.py`, `status_formatter.py`, `request_timing_collector.py` |
| hardcoded theme `.qss` ownership | `ThemeManager` + `StyleComposer` token system |

## Deletion Gate

Do not delete a legacy module until:

- all imports are removed,
- equivalent behavior is covered by the new owner,
- structured logs prove the new path is active,
- tests cover the new owner,
- `REVIA_CORE_V2_ENABLED` no longer needs to preserve that legacy path.
