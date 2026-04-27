"""Console demo for the V1 parallel-agents spine.

Usage::

    cd revia_core_py
    python -m agents.cli_demo "How should Revia run parallel agents?"

The demo wires the orchestrator with a stub MemoryStore and stub
ReplyPlanner so it works without a running LLM server, but if you
already have ``revia_core_py`` running (memory_store, emotion_net,
llm_backend) you can also import :func:`build_default_orchestrator`
and pass live singletons.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

# Support both "python -m revia_core_py.agents.cli_demo" and direct run
# from inside the revia_core_py directory.
try:  # package context
    from .agent_base import AgentContext, CancellationToken
    from .critic_agent import CriticAgent
    from .emotion_agent import EmotionAgent
    from .final_response import FinalResponseBuilder
    from .intent_agent import IntentAgent
    from .memory_agent import MemoryAgent
    from .model_router import ModelRouter
    from .orchestrator import AgentOrchestrator
    from .quality_gate import QualityGate
    from .reasoning_agent import ReasoningAgent
    from .voice_style_agent import VoiceStyleAgent
except ImportError:  # direct context
    from agent_base import AgentContext, CancellationToken  # type: ignore
    from critic_agent import CriticAgent  # type: ignore
    from emotion_agent import EmotionAgent  # type: ignore
    from final_response import FinalResponseBuilder  # type: ignore
    from intent_agent import IntentAgent  # type: ignore
    from memory_agent import MemoryAgent  # type: ignore
    from model_router import ModelRouter  # type: ignore
    from orchestrator import AgentOrchestrator  # type: ignore
    from quality_gate import QualityGate  # type: ignore
    from reasoning_agent import ReasoningAgent  # type: ignore
    from voice_style_agent import VoiceStyleAgent  # type: ignore


class _StubMemory:
    """Minimal in-memory MemoryStore stand-in for the demo."""

    def __init__(self):
        self.short_term: list[dict[str, str]] = [
            {"role": "user", "content": "I want Revia to feel alive."},
            {"role": "assistant", "content": "Okay — focusing on personality and pacing."},
        ]
        self.long_term: list[dict[str, str]] = [
            {"content": "User is building Revia as a local AI companion with voice, vision and parallel agents."},
            {"content": "User prefers detailed architecture explanations over surface-level overviews."},
        ]

    def get_short_term(self, limit: int = 50):
        return list(self.short_term[-limit:])

    def search(self, query: str, max_results: int = 5):
        q = (query or "").lower()
        if not q:
            return []
        return [e for e in self.long_term if q.split()[0] in e["content"].lower()][:max_results]


class _StubEmotionNet:
    enabled = True

    def infer(self, text, recent_messages=None, prev_emotion=None,
              profile_name=None, profile_state=None):
        lowered = (text or "").lower()
        if any(w in lowered for w in ("frustrated", "angry", "broken", "hate")):
            label, valence = "Frustrated", -0.6
        elif any(w in lowered for w in ("excited", "amazing", "love", "great")):
            label, valence = "Excited", 0.7
        else:
            label, valence = "Focused", 0.2
        return {
            "label": label,
            "secondary_label": "Neutral",
            "confidence": 0.78,
            "uncertainty": 0.22,
            "valence": valence,
            "arousal": 0.4,
            "dominance": 0.3,
            "emotion_probs": {label: 0.78, "Neutral": 0.22},
            "top_emotions": [{"label": label, "prob": 0.78}],
            "signals": {},
            "temporal": {},
            "model": "stub_emotion_v0",
            "inference_ms": 4.0,
        }


def build_default_orchestrator(
    memory_store=None,
    emotion_net=None,
    reply_planner=None,
    profile_engine=None,
    hfl=None,
) -> AgentOrchestrator:
    """Wire orchestrator + agents from REVIA singletons (or stubs).

    Pass live REVIA components from ``core_server.py`` to plug into the
    real backends; pass nothing to use stubs (used by the CLI demo).
    """
    memory_store = memory_store or _StubMemory()
    emotion_net = emotion_net or _StubEmotionNet()

    router = ModelRouter()
    router.register(
        "emotion_classify",
        backend_name="emotion_net",
        handler=emotion_net.infer,
        description="EmotionNet rule-based affective fusion",
    )

    agents = [
        MemoryAgent(memory_store=memory_store, model_router=router),
        EmotionAgent(emotion_net=emotion_net, model_router=router,
                     profile_engine=profile_engine),
        IntentAgent(model_router=router),
        VoiceStyleAgent(profile_engine=profile_engine, model_router=router),
        ReasoningAgent(reply_planner=reply_planner, model_router=router),
    ]
    post_agents = [
        CriticAgent(model_router=router, profile_engine=profile_engine),
    ]
    final_builder = FinalResponseBuilder(hfl=hfl)
    quality_gate = QualityGate(profile_engine=profile_engine)
    return AgentOrchestrator(
        agents=agents,
        final_builder=final_builder,
        quality_gate=quality_gate,
        post_agents=post_agents,
        max_regen=1,
        agent_timeouts_ms={
            "MemoryAgent": 1500,
            "EmotionAgent": 500,
            "IntentAgent": 250,
            "VoiceStyleAgent": 250,
            "ReasoningAgent": 8000,
            "CriticAgent": 400,
        },
    )


def _format_table(output) -> str:
    lines = []
    for r in output.agent_results:
        line = (
            f"[{r.agent:<14}] success={str(r.success):<5} "
            f"conf={r.confidence:0.2f} elapsed={r.elapsed_ms:7.1f}ms"
        )
        if r.error:
            line += f"  error={r.error}"
        lines.append(line)
    lines.append(
        f"[Quality       ] score={output.quality.score:0.2f} "
        f"threshold={output.quality.threshold:0.2f} approved={output.quality.approved}"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="revia-agents",
        description="Run one parallel-agents turn over text input.",
    )
    parser.add_argument("text", nargs="?", default=None, help="user message")
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    parser.add_argument("--threshold", type=float, default=0.70,
                        help="quality threshold (default 0.70)")
    args = parser.parse_args(argv)

    if not args.text:
        if sys.stdin.isatty():
            print("Type a message (Ctrl+D to send):", file=sys.stderr)
        text = sys.stdin.read().strip()
    else:
        text = args.text.strip()

    if not text:
        print("ERROR: empty input", file=sys.stderr)
        return 2

    orch = build_default_orchestrator()
    ctx = AgentContext(
        user_text=text,
        turn_id="demo-1",
        conversation_id="cli",
        user_profile="demo",
        response_threshold=args.threshold,
        cancel_token=CancellationToken(turn_id="demo-1"),
        metadata={},
    )

    output = orch.run_turn(ctx)
    orch.shutdown()

    if args.json:
        print(json.dumps(output.to_dict(), indent=2))
        return 0

    print(_format_table(output))
    print("---")
    print(f"Final: {output.final.text}")
    print(f"(elapsed_ms={output.elapsed_ms:.1f}, cancelled={output.cancelled})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
