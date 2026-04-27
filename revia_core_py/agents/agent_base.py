"""Common types for the REVIA parallel-agent system.

The orchestrator runs many agents in parallel.  Each one accepts the same
``AgentContext`` and returns an ``AgentResult`` so the rest of the pipeline
(final-response builder, quality gate, interface router) can treat them
uniformly without knowing what model or backend produced the value.

Core rules from the architecture spec:
    1. Agents are internal thoughts; only the FinalResponseBuilder speaks.
    2. Every agent has ONE job and returns a structured ``AgentResult``.
    3. Every agent honours its ``CancellationToken`` and per-agent timeout.
"""
from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class CancelledError(Exception):
    """Raised when a cancellation token fires inside an agent."""


class CancellationToken:
    """Lightweight cooperative cancel signal keyed to a turn_id.

    Python's ThreadPoolExecutor cannot pre-empt running work, so agents
    must check ``cancelled`` (or call ``raise_if_cancelled``) at safe
    yield points.  The orchestrator flips the token when the user
    interrupts or starts a new turn.
    """

    __slots__ = ("_event", "turn_id")

    def __init__(self, turn_id: str = ""):
        self.turn_id = turn_id
        self._event = threading.Event()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        self._event.set()

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise CancelledError(f"turn {self.turn_id} cancelled")


@dataclass
class AgentContext:
    """Shared, read-only context passed to every agent in a turn."""

    user_text: str
    turn_id: str = ""
    conversation_id: str = ""
    user_profile: str = ""
    response_threshold: float = 0.70
    cancel_token: CancellationToken = field(default_factory=CancellationToken)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_text": self.user_text[:200],
            "turn_id": self.turn_id,
            "conversation_id": self.conversation_id,
            "user_profile": self.user_profile,
            "response_threshold": round(self.response_threshold, 4),
            "metadata": dict(self.metadata),
        }


@dataclass
class AgentRequest:
    """Internal envelope produced by the orchestrator for each agent run."""

    agent_name: str
    context: AgentContext
    timeout_ms: int = 5000
    priority: str = "normal"  # critical | high | normal | low


@dataclass
class AgentResult:
    """Structured agent output consumed by the orchestrator."""

    agent: str
    success: bool = False
    confidence: float = 0.0
    result: dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "success": self.success,
            "confidence": round(float(self.confidence), 4),
            "result": self.result,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "error": self.error,
        }


class Agent(ABC):
    """Base class for every parallel agent.

    Subclasses implement :meth:`run` and set :attr:`name` /
    :attr:`default_timeout_ms`.  The base class wraps :meth:`run` in
    ``execute`` to enforce uniform timing, error-capture and cancellation
    semantics so the orchestrator never has to special-case any agent.
    """

    name: str = "Agent"
    default_timeout_ms: int = 5000

    def execute(self, context: AgentContext) -> AgentResult:
        t0 = time.monotonic()
        try:
            context.cancel_token.raise_if_cancelled()
            payload = self.run(context)
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            confidence = float(payload.pop("_confidence", 0.0)) if isinstance(payload, dict) else 0.0
            return AgentResult(
                agent=self.name,
                success=True,
                confidence=max(0.0, min(1.0, confidence)),
                result=payload if isinstance(payload, dict) else {"value": payload},
                elapsed_ms=elapsed_ms,
            )
        except CancelledError as exc:
            return AgentResult(
                agent=self.name,
                success=False,
                confidence=0.0,
                elapsed_ms=(time.monotonic() - t0) * 1000.0,
                error=f"cancelled: {exc}",
            )
        except Exception as exc:
            return AgentResult(
                agent=self.name,
                success=False,
                confidence=0.0,
                elapsed_ms=(time.monotonic() - t0) * 1000.0,
                error=f"{type(exc).__name__}: {exc}",
            )

    @abstractmethod
    def run(self, context: AgentContext) -> dict[str, Any]:
        """Compute the agent's structured payload.

        Implementations may put ``_confidence`` (0-1 float) into the
        returned dict; the base class will pop it out and place it on
        the ``AgentResult.confidence`` field.
        """
        raise NotImplementedError
