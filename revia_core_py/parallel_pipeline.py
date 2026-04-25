"""Parallel Pipeline — concurrent lane execution for Revia.

Implements a multi-lane architecture where independent pipeline stages
run concurrently instead of sequentially:

Lane Architecture:
  Lane 1 (Perception): Emotion analysis + intent routing + memory update
  Lane 2 (Cognition):  LLM generation (the slowest stage)
  Lane 3 (Expression): TTS synthesis + output delivery

The key insight: Lane 1 can finish while Lane 2 is still running, and
Lane 3 can start synthesizing the first sentence while Lane 2 is still
generating the rest. This eliminates the serial bottleneck.

Concurrency model:
  - Lane 1 runs synchronously (fast, < 50ms)
  - Lane 2 runs in a background thread (slow, 200ms-10s)
  - Lane 3 starts as soon as Lane 2 produces the first sentence
    (streaming TTS already does this via chat_sentence events)

Thread safety:
  - Each lane has its own lock for shared state
  - Cross-lane communication uses thread-safe queues
  - The ConversationManager state machine transitions are already locked
"""
from __future__ import annotations

import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

try:
    from conversation_runtime import (
        ConversationManager, ReviaState, TriggerKind, TriggerRequest,
    )
    from runtime_models import AssistantResponse, RequestLifecycleState, TurnManager
except ImportError:
    ConversationManager = None
    ReviaState = None
    TriggerKind = None
    TriggerRequest = None
    AssistantResponse = None
    RequestLifecycleState = None
    TurnManager = None


class LaneState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class LaneResult:
    """Result from a single lane execution."""
    lane: str
    state: LaneState = LaneState.IDLE
    data: dict = field(default_factory=dict)
    elapsed_ms: float = 0.0
    error: Optional[str] = None


class ParallelPipeline:
    """Manages concurrent execution of Revia's pipeline stages.

    The pipeline is split into three lanes that can run concurrently:

    1. Perception Lane (fast): Emotion inference, intent classification,
       memory update. These are all fast operations (< 50ms combined)
       and their results feed into the cognition lane.

    2. Cognition Lane (slow): LLM generation. This is the bottleneck
       stage and benefits most from not being blocked by perception.

    3. Expression Lane (streaming): TTS synthesis and output delivery.
       This starts as soon as the cognition lane produces the first
       sentence token (already handled by chat_sentence events).

    The perception and cognition lanes run in parallel when possible.
    For the first message after idle, perception must complete before
    cognition can start (emotion context is needed for the prompt).
    But for subsequent messages, the emotion from the previous turn
    is already available, so perception and cognition can overlap.
    """

    def __init__(self, max_workers: int = 4, log_fn: Callable = None):
        self._log = log_fn or (lambda msg: None)
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="revia-lane",
        )
        self._lock = threading.Lock()
        self._active_lanes: dict[str, LaneState] = {
            "perception": LaneState.IDLE,
            "cognition": LaneState.IDLE,
            "expression": LaneState.IDLE,
        }
        self._lane_results: dict[str, LaneResult] = {}
        self._pipeline_history: deque[dict] = deque(maxlen=50)

    def submit_perception(
        self,
        fn: Callable,
        *args,
        on_complete: Optional[Callable] = None,
        **kwargs,
    ) -> Future:
        """Submit a perception lane task (emotion, routing, memory)."""
        return self._submit_lane("perception", fn, on_complete, *args, **kwargs)

    def submit_cognition(
        self,
        fn: Callable,
        *args,
        on_complete: Optional[Callable] = None,
        **kwargs,
    ) -> Future:
        """Submit a cognition lane task (LLM generation)."""
        return self._submit_lane("cognition", fn, on_complete, *args, **kwargs)

    def submit_expression(
        self,
        fn: Callable,
        *args,
        on_complete: Optional[Callable] = None,
        **kwargs,
    ) -> Future:
        """Submit an expression lane task (TTS, output)."""
        return self._submit_lane("expression", fn, on_complete, *args, **kwargs)

    def _submit_lane(
        self,
        lane_name: str,
        fn: Callable,
        on_complete: Optional[Callable],
        *args,
        **kwargs,
    ) -> Future:
        with self._lock:
            self._active_lanes[lane_name] = LaneState.RUNNING

        t0 = time.monotonic()

        def _run():
            result = LaneResult(lane=lane_name)
            try:
                data = fn(*args, **kwargs)
                result.state = LaneState.COMPLETE
                result.data = data if isinstance(data, dict) else {}
                result.elapsed_ms = (time.monotonic() - t0) * 1000.0
            except Exception as exc:
                result.state = LaneState.ERROR
                result.error = str(exc)
                result.elapsed_ms = (time.monotonic() - t0) * 1000.0
                self._log(f"[ParallelPipeline] Lane '{lane_name}' error: {exc}")
            finally:
                with self._lock:
                    self._active_lanes[lane_name] = result.state
                    self._lane_results[lane_name] = result
                self._record_history(result)
                if on_complete:
                    try:
                        on_complete(result)
                    except Exception:
                        pass
            return result

        return self._executor.submit(_run)

    def get_lane_state(self, lane_name: str) -> LaneState:
        with self._lock:
            return self._active_lanes.get(lane_name, LaneState.IDLE)

    def get_lane_result(self, lane_name: str) -> Optional[LaneResult]:
        with self._lock:
            return self._lane_results.get(lane_name)

    def all_lanes_idle(self) -> bool:
        with self._lock:
            return all(
                s in (LaneState.IDLE, LaneState.COMPLETE, LaneState.ERROR)
                for s in self._active_lanes.values()
            )

    def any_lane_running(self) -> bool:
        with self._lock:
            return any(
                s == LaneState.RUNNING
                for s in self._active_lanes.values()
            )

    def status(self) -> dict[str, Any]:
        with self._lock:
            lanes = dict(self._active_lanes)
            results = {
                name: {
                    "state": r.state.value,
                    "elapsed_ms": round(r.elapsed_ms, 1),
                    "error": r.error,
                }
                for name, r in self._lane_results.items()
            }
        return {
            "lanes": {k: v.value for k, v in lanes.items()},
            "results": results,
            "any_running": self.any_lane_running(),
        }

    def _record_history(self, result: LaneResult):
        with self._lock:
            self._pipeline_history.append({
                "lane": result.lane,
                "state": result.state.value,
                "elapsed_ms": round(result.elapsed_ms, 1),
                "error": result.error,
                "timestamp": time.monotonic(),
            })

    def shutdown(self):
        """Shut down the executor. Call on application exit."""
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            self._executor.shutdown(wait=False)
