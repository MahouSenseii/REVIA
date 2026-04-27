"""EmotionAgent — wraps EmotionNet for parallel-agent execution.

The existing :class:`EmotionNet.infer` already returns a rich dict
(label, valence/arousal/dominance, top emotions, etc.).  This agent just
makes its output conform to the shared :class:`AgentResult` schema and
provides a confidence value derived from EmotionNet's certainty.
"""
from __future__ import annotations

from typing import Any

from .agent_base import Agent, AgentContext


class EmotionAgent(Agent):
    name = "EmotionAgent"
    default_timeout_ms = 500

    def __init__(self, emotion_net=None, model_router=None, profile_engine=None):
        self._emotion_net = emotion_net
        self._router = model_router
        self._pe = profile_engine

    def run(self, context: AgentContext) -> dict[str, Any]:
        if self._emotion_net is None:
            return {
                "_confidence": 0.0,
                "label": "Neutral",
                "secondary_label": "Neutral",
                "valence": 0.0,
                "arousal": 0.0,
                "dominance": 0.0,
                "uncertainty": 1.0,
                "note": "emotion_net unavailable",
            }

        context.cancel_token.raise_if_cancelled()

        recent_messages = context.metadata.get("recent_messages") or []
        prev_emotion = context.metadata.get("prev_emotion")
        profile_state = context.metadata.get("profile_state") or {}

        # Use the registered route if available; otherwise call EmotionNet
        # directly.  Both must produce the same result schema.
        if self._router is not None and self._router.has("emotion_classify"):
            inferred = self._router.call(
                "emotion_classify",
                context.user_text,
                recent_messages=recent_messages,
                prev_emotion=prev_emotion,
                profile_name=context.user_profile or None,
                profile_state=profile_state,
            )
        else:
            inferred = self._emotion_net.infer(
                context.user_text,
                recent_messages=recent_messages,
                prev_emotion=prev_emotion,
                profile_name=context.user_profile or None,
                profile_state=profile_state,
            )

        label = str(inferred.get("label", "Neutral"))
        confidence = float(inferred.get("confidence", 0.0))
        uncertainty = float(inferred.get("uncertainty", 1.0))
        # If EmotionNet doesn't expose explicit confidence, fall back to
        # (1 - uncertainty), clipped to [0, 1].
        if confidence <= 0.0 and uncertainty <= 1.0:
            confidence = max(0.0, min(1.0, 1.0 - uncertainty))

        return {
            "_confidence": confidence,
            "label": label,
            "secondary_label": str(inferred.get("secondary_label", "Neutral")),
            "valence": float(inferred.get("valence", 0.0)),
            "arousal": float(inferred.get("arousal", 0.0)),
            "dominance": float(inferred.get("dominance", 0.0)),
            "uncertainty": uncertainty,
            "top_emotions": inferred.get("top_emotions", []),
        }
