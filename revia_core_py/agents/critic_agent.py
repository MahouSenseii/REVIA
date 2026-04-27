"""CriticAgent — final-pass critic on the candidate reply.

Operates *after* the ReasoningAgent has produced ``candidate_text``
(passed in via ``context.metadata``).  The critic returns a structured
verdict with a recommendation that the orchestrator's regen loop reads.

The critic is intentionally lighter than the existing AVS:
    * AVS = full multi-criterion scoring (relevance, naturalness,
      profile alignment, etc.).
    * CriticAgent = quick gate that catches the common failure modes
      (empty / repetitive / refusal-spam / wrong-intent / bad-length)
      and recommends ``regen`` when something is clearly off.

Output schema::

    {
        "score": 0..1,
        "recommendation": "accept" | "regen" | "clarify",
        "reasons": [str, ...],
        "issues": {
            "empty": bool, "too_short": bool, "too_long": bool,
            "refusal_spam": bool, "repetition": bool,
            "off_intent": bool, "profile_violation": bool,
            "uncertain_facts": bool,
        },
    }

V1 = pure heuristics; V2 plug-in via the ``critic`` task type on the
:class:`ModelRouter` (e.g. a small classifier or reranker).
"""
from __future__ import annotations

import re
from typing import Any

from .agent_base import Agent, AgentContext


_REFUSAL_MARKERS = (
    "as an ai", "as a language model", "i cannot", "i can't help with",
    "i'm sorry, but i can", "i am unable to", "i apologize, but i",
    "it is not possible for me", "i'm just an ai",
)

_VAGUE_FACT_PATTERNS = (
    "i think", "i believe", "i guess", "probably", "maybe", "kind of",
    "sort of", "not sure", "i'm not certain", "could be",
)


class CriticAgent(Agent):
    """Lightweight post-generation critic.

    The orchestrator passes the candidate reply via
    ``context.metadata['candidate_text']`` and the intent payload via
    ``context.metadata['intent']``.  All other inputs are optional.
    """

    name = "CriticAgent"
    default_timeout_ms = 400

    def __init__(self, model_router=None, profile_engine=None):
        self._router = model_router
        self._pe = profile_engine

    # ------------------------------------------------------------------
    # Agent protocol
    # ------------------------------------------------------------------

    def run(self, context: AgentContext) -> dict[str, Any]:
        context.cancel_token.raise_if_cancelled()

        candidate = str(context.metadata.get("candidate_text") or "").strip()
        intent = context.metadata.get("intent") or {}
        recent = list(context.metadata.get("recent_replies") or [])
        banned = list(context.metadata.get("banned_phrases") or [])

        # Optional override via router (kept best-effort).
        if self._router is not None and self._router.has("critic"):
            try:
                inferred = self._router.call(
                    "critic", candidate, context.user_text, intent,
                )
                if isinstance(inferred, dict) and "recommendation" in inferred:
                    inferred.setdefault("_confidence",
                                        float(inferred.get("score", 0.5)))
                    return inferred
            except Exception:
                pass

        return self._heuristic(candidate, context.user_text, intent, recent, banned)

    # ------------------------------------------------------------------
    # Heuristic critic
    # ------------------------------------------------------------------

    def _heuristic(
        self,
        candidate: str,
        user_text: str,
        intent: dict[str, Any],
        recent: list[str],
        banned: list[str],
    ) -> dict[str, Any]:
        issues = {
            "empty": False, "too_short": False, "too_long": False,
            "refusal_spam": False, "repetition": False,
            "off_intent": False, "profile_violation": False,
            "uncertain_facts": False,
        }
        reasons: list[str] = []
        score = 1.0

        if not candidate:
            issues["empty"] = True
            reasons.append("empty_candidate")
            return _verdict(0.0, "regen", reasons, issues)

        lowered = candidate.lower()
        words = re.findall(r"\w+", lowered)
        n_words = len(words)
        intent_label = str(intent.get("label") or "chat").lower()
        is_question = bool(intent.get("is_question"))
        is_imperative = bool(intent.get("is_imperative"))

        # Length checks (intent-aware).
        if intent_label in ("greeting", "farewell", "affirmation",
                            "negation", "small_talk", "compliment"):
            if n_words > 60:
                issues["too_long"] = True
                reasons.append(f"too_long_for_{intent_label} ({n_words} words)")
                score -= 0.20
        elif intent_label in ("question", "command", "emotional_share"):
            if n_words < 4:
                issues["too_short"] = True
                reasons.append(f"too_short_for_{intent_label} ({n_words} words)")
                score -= 0.30
            if n_words > 220:
                issues["too_long"] = True
                reasons.append(f"too_long_for_{intent_label} ({n_words} words)")
                score -= 0.10

        # Refusal-spam (catches "as an AI..." regressions).
        if any(marker in lowered for marker in _REFUSAL_MARKERS):
            issues["refusal_spam"] = True
            reasons.append("refusal_spam")
            score -= 0.40

        # Repetition: candidate too similar to a recent reply.
        if recent:
            rep_score = max(_jaccard_words(candidate, r) for r in recent if r)
            if rep_score >= 0.85:
                issues["repetition"] = True
                reasons.append(f"repetition_jaccard_{rep_score:.2f}")
                score -= 0.30

        # Off-intent: question/command got a meta-answer instead.
        if (is_question or is_imperative) and self._looks_meta(lowered):
            issues["off_intent"] = True
            reasons.append("off_intent_meta_reply")
            score -= 0.20

        # Profile violation: banned phrase appears verbatim.
        for phrase in banned:
            if phrase and phrase.lower() in lowered:
                issues["profile_violation"] = True
                reasons.append(f"banned_phrase:{phrase[:30]}")
                score -= 0.30
                break

        # Uncertain facts: hedging on a fact-bearing intent.
        if (is_question or is_imperative or
                intent.get("expects_facts")) and any(
                p in lowered for p in _VAGUE_FACT_PATTERNS):
            issues["uncertain_facts"] = True
            reasons.append("uncertain_facts_hedging")
            score -= 0.10

        # Recommendation:
        score = max(0.0, min(1.0, score))
        if issues["empty"] or issues["refusal_spam"] or issues["repetition"]:
            rec = "regen"
        elif issues["off_intent"] or issues["too_short"]:
            rec = "regen"
        elif issues["profile_violation"]:
            rec = "regen"
        elif issues["uncertain_facts"] and (is_question or is_imperative):
            rec = "clarify"
        else:
            rec = "accept"

        return _verdict(score, rec, reasons, issues)

    @staticmethod
    def _looks_meta(lowered: str) -> bool:
        """Heuristic: candidate is *talking about* answering instead of answering."""
        meta_starts = (
            "i can help", "let me know", "could you tell me more",
            "what would you like", "do you want me to",
        )
        return any(lowered.startswith(s) for s in meta_starts) and len(lowered) < 80


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _verdict(
    score: float,
    recommendation: str,
    reasons: list[str],
    issues: dict[str, bool],
) -> dict[str, Any]:
    return {
        "_confidence": float(score),
        "score": round(float(score), 4),
        "recommendation": recommendation,
        "reasons": list(reasons),
        "issues": dict(issues),
    }


def _jaccard_words(a: str, b: str) -> float:
    aw = set(re.findall(r"\w+", (a or "").lower()))
    bw = set(re.findall(r"\w+", (b or "").lower()))
    if not aw or not bw:
        return 0.0
    return len(aw & bw) / float(len(aw | bw))
