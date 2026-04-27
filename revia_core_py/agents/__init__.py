"""REVIA Parallel Agents — V1 spine.

Public flow::

    text input
      -> AgentOrchestrator
      -> [MemoryAgent | EmotionAgent | ReasoningAgent]  (parallel)
      -> FinalResponseBuilder (one canonical answer)
      -> QualityGate          (threshold from ProfileEngine)
      -> console / web output

The agent layer is purely additive: it does not replace any existing
``revia_core_py`` subsystem.  Each agent is a thin wrapper around an
already-trusted module (EmotionNet, MemoryStore, ReplyPlanner, AVS, HFL)
so all profile knobs and unit tests keep working.
"""

from .agent_base import (
    Agent,
    AgentContext,
    AgentRequest,
    AgentResult,
    CancellationToken,
    CancelledError,
)
from .model_router import ModelRouter, ModelRoute, NoRouteAdmittedError

# Re-export the runtime-side requirements dataclass so agent callers can
# declare resource needs without importing from two packages.
try:  # pragma: no cover - import guard
    from ..runtime.runtime_scheduler import ModelRequirements
except ImportError:  # pragma: no cover
    from runtime.runtime_scheduler import ModelRequirements  # type: ignore[no-redef]
from .memory_agent import MemoryAgent
from .emotion_agent import EmotionAgent
from .intent_agent import IntentAgent
from .reasoning_agent import ReasoningAgent
from .voice_style_agent import VoiceStyleAgent
from .critic_agent import CriticAgent
from .quality_gate import QualityGate, QualityVerdict
from .final_response import FinalResponse, FinalResponseBuilder
from .orchestrator import AgentOrchestrator, OrchestratorOutput

__all__ = [
    "Agent",
    "AgentContext",
    "AgentRequest",
    "AgentResult",
    "AgentOrchestrator",
    "CancellationToken",
    "CancelledError",
    "CriticAgent",
    "EmotionAgent",
    "FinalResponse",
    "FinalResponseBuilder",
    "IntentAgent",
    "MemoryAgent",
    "ModelRequirements",
    "ModelRoute",
    "ModelRouter",
    "NoRouteAdmittedError",
    "OrchestratorOutput",
    "QualityGate",
    "QualityVerdict",
    "ReasoningAgent",
    "VoiceStyleAgent",
]
