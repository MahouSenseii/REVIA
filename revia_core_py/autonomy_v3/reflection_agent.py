"""V3.2 ReflectionAgent — post-turn metacognition.

Runs *after* the orchestrator returns its canonical answer.  Inspects
the turn (user_text, candidate, quality, critic, regen) and emits a
short *lesson* that gets attached to the corresponding
:class:`Episode`.

Lessons follow a fixed taxonomy so downstream RL tuning can react:

    "low_quality"          -- quality_score < threshold despite regen
    "regen_overhead"       -- regen happened but reply still mediocre
    "critic_repetition"    -- critic flagged repetition
    "critic_refusal"       -- critic flagged refusal_spam
    "fact_uncertainty"     -- critic flagged uncertain_facts on Q/cmd
    "intent_mismatch"      -- critic flagged off_intent
    "user_negative_share"  -- intent=emotional_share + polarity=negative
    "user_complaint"       -- intent=complaint
    "user_compliment"      -- intent=compliment
    "stable_chat"          -- normal happy path

V3 stays heuristic; V4+ can plug a small LLM behind ``model_router``
under the ``reflect`` task type without changing the agent contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    from ..agents.agent_base import Agent, AgentContext
except ImportError:  # pragma: no cover
    from agents.agent_base import Agent, AgentContext  # type: ignore[no-redef]


@dataclass
class ReflectionVerdict:
    label: str = "stable_chat"
    notes: list[str] = field(default_factory=list)
    rl_hint: dict[str, float] = field(default_factory=dict)
    lesson: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "notes": list(self.notes),
            "rl_hint": dict(self.rl_hint),
            "lesson": self.lesson,
        }


class ReflectionAgent(Agent):
    """Lightweight post-turn reflection.

    Reads the orchestrator output via ``context.metadata`` keys:

        candidate_text, quality, critic, regen_attempts,
        intent, emotion_label, recent_episodes, threshold
    """

    name = "ReflectionAgent"
    default_timeout_ms = 250

    def __init__(self, model_router=None, episode_store=None):
        self._router = model_router
        self._store = episode_store

    def run(self, context: AgentContext) -> dict[str, Any]:
        context.cancel_token.raise_if_cancelled()

        # Optional plug-in via router.
        if self._router is not None and self._router.has("reflect"):
            try:
                inferred = self._router.call("reflect", context)
                if isinstance(inferred, dict) and inferred.get("label"):
                    inferred.setdefault("_confidence", 0.7)
                    return inferred
            except Exception:
                pass

        verdict = self._heuristic(context)
        if self._store is not None:
            episode_id = str(context.metadata.get("episode_id") or "")
            if episode_id:
                try:
                    self._store.attach_lesson(episode_id, verdict.lesson)
                except Exception:
                    pass
        return {
            "_confidence": 0.7,
            **verdict.to_dict(),
        }

    # ------------------------------------------------------------------
    # Heuristic
    # ------------------------------------------------------------------

    @staticmethod
    def _heuristic(ctx: AgentContext) -> ReflectionVerdict:
        meta = ctx.metadata or {}
        quality = meta.get("quality") or {}
        critic = meta.get("critic") or {}
        intent = meta.get("intent") or {}

        threshold = float(meta.get("threshold") or 0.70)
        score = float(quality.get("score") or 0.0)
        approved = bool(quality.get("approved"))
        regen = int(meta.get("regen_attempts") or 0)

        critic_issues = critic.get("issues") or {}
        critic_recommendation = str(critic.get("recommendation") or "accept")

        intent_label = str(intent.get("label") or "chat").lower()
        polarity = str(intent.get("polarity") or "neutral").lower()

        notes: list[str] = []
        rl_hint: dict[str, float] = {}

        if intent_label == "complaint":
            label = "user_complaint"
            notes.append("user_was_complaining")
            rl_hint["formality"] = 0.6
            rl_hint["verbosity"] = 0.4
        elif intent_label == "compliment":
            label = "user_compliment"
            notes.append("positive_user_signal")
            rl_hint["playfulness"] = 0.7
        elif intent_label == "emotional_share" and polarity == "negative":
            label = "user_negative_share"
            notes.append("user_shared_negative_emotion")
            rl_hint["formality"] = 0.5
            rl_hint["verbosity"] = 0.6
        elif critic_issues.get("refusal_spam"):
            label = "critic_refusal"
            notes.append("answer_contained_refusal_spam")
        elif critic_issues.get("repetition"):
            label = "critic_repetition"
            notes.append("answer_was_repetitive")
            rl_hint["self_correction_rate"] = 0.7
        elif critic_issues.get("off_intent"):
            label = "intent_mismatch"
            notes.append("answer_did_not_address_intent")
        elif critic_issues.get("uncertain_facts"):
            label = "fact_uncertainty"
            notes.append("answer_hedged_on_facts")
        elif (not approved) and regen > 0:
            label = "regen_overhead"
            notes.append(f"regen_used_{regen}x_still_below_threshold")
        elif score < threshold:
            label = "low_quality"
            notes.append(f"score_{score:.2f}_below_{threshold:.2f}")
        else:
            label = "stable_chat"
            notes.append("happy_path")

        lesson_bits = [label]
        if notes:
            lesson_bits.append("|".join(notes))
        if critic_recommendation != "accept":
            lesson_bits.append(f"critic={critic_recommendation}")
        lesson = " · ".join(lesson_bits)
        return ReflectionVerdict(
            label=label,
            notes=notes,
            rl_hint=rl_hint,
            lesson=lesson,
        )
