"""IntentAgent — fast rule-based classification of the user's utterance.

Output schema (kept minimal so downstream agents can rely on it)::

    {
        "label": "question" | "command" | "chat" | "greeting" | "farewell"
                 | "affirmation" | "negation" | "emotional_share"
                 | "clarification" | "compliment" | "complaint"
                 | "small_talk",
        "confidence": 0..1,
        "is_question": bool,
        "is_imperative": bool,
        "expects_facts": bool,         # the reply should be grounded
        "ends_open": bool,             # ends with "..." or no terminator
        "polarity": "positive"|"neutral"|"negative",
        "topic_hint": str,             # rough subject extracted heuristically
    }

V1 = pure heuristics (zero deps).  V2 plug-in is via the
``intent_classify`` task_type on the :class:`ModelRouter` — when wired to
a small classifier it overrides the rule output.  Both paths produce the
same schema.
"""
from __future__ import annotations

import re
from typing import Any

from .agent_base import Agent, AgentContext


# Word lists are tuned for casual English chat with REVIA.
_QUESTION_STARTERS = (
    "what", "why", "how", "when", "where", "who", "which", "whom", "whose",
    "can", "could", "will", "would", "should", "shall",
    "do", "does", "did", "is", "are", "was", "were", "am",
    "may", "might", "have", "has", "had",
)

_COMMAND_VERBS = (
    "write", "find", "search", "look", "open", "close", "run", "execute",
    "make", "create", "build", "generate", "draw", "paint", "compose",
    "explain", "describe", "summarize", "translate", "convert",
    "list", "show", "display", "tell", "give", "send", "post",
    "play", "stop", "pause", "start", "save", "load", "delete", "remove",
    "edit", "fix", "debug", "compile", "deploy", "install",
)

_GREETINGS = (
    "hi", "hii", "hello", "hey", "yo", "sup", "howdy",
    "good morning", "good afternoon", "good evening", "morning", "evening",
)

_FAREWELLS = (
    "bye", "byee", "goodbye", "see ya", "see you", "later", "g2g",
    "good night", "night", "ttyl",
)

_AFFIRMATIONS = (
    "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "alright", "sounds good",
    "absolutely", "of course", "right", "true", "agreed", "fine", "cool",
)

_NEGATIONS = (
    "no", "nope", "nah", "never", "not really", "negative", "don't think so",
)

_FEELING_PHRASES = (
    "i feel", "i'm feeling", "im feeling", "i am feeling",
    "i'm sad", "im sad", "i feel sad", "i feel down",
    "i'm happy", "im happy", "i'm excited", "im excited",
    "i love", "i hate", "i miss", "i'm scared", "i'm angry",
    "i'm worried", "im worried", "i'm tired", "i'm lonely",
)

_COMPLIMENT_HINTS = (
    "thank you", "thanks", "thx", "appreciate", "you're great", "love it",
    "good job", "well done", "awesome", "amazing", "you rock",
)

_COMPLAINT_HINTS = (
    "this sucks", "you suck", "broken", "doesn't work", "stop it",
    "annoying", "frustrating", "wrong again", "fail", "useless",
    "garbage", "terrible",
)

_NEGATIVE_TOKENS = (
    "sad", "angry", "mad", "upset", "hate", "tired", "lonely",
    "fail", "broken", "wrong", "bad", "worst", "awful", "stuck",
    "scared", "worried", "anxious", "depressed",
)

_POSITIVE_TOKENS = (
    "happy", "love", "great", "good", "nice", "cool", "awesome",
    "fantastic", "wonderful", "excellent", "amazing", "best",
    "yay", "lol", "haha", "thanks",
)

_FILLER_TOKENS = {
    "the", "a", "an", "to", "of", "in", "on", "for", "with", "by",
    "and", "or", "but", "is", "are", "was", "were", "be", "been",
    "i", "you", "we", "they", "he", "she", "it",
    "do", "did", "does", "have", "has", "had",
    "this", "that", "these", "those",
    "very", "really", "just", "so", "much", "more", "than",
}


class IntentAgent(Agent):
    """Fast, deterministic intent classifier.

    Optionally delegates to a router task type ``intent_classify`` if one
    is registered (e.g. a small ONNX classifier later).
    """

    name = "IntentAgent"
    default_timeout_ms = 250

    def __init__(self, model_router=None):
        self._router = model_router

    def run(self, context: AgentContext) -> dict[str, Any]:
        context.cancel_token.raise_if_cancelled()

        text = (context.user_text or "").strip()
        if not text:
            return {
                "_confidence": 0.0,
                "label": "chat",
                "confidence": 0.0,
                "is_question": False,
                "is_imperative": False,
                "expects_facts": False,
                "ends_open": False,
                "polarity": "neutral",
                "topic_hint": "",
            }

        # Optional override via router (kept silent on errors so the
        # heuristic path is always available as a safety net).
        if self._router is not None and self._router.has("intent_classify"):
            try:
                inferred = self._router.call("intent_classify", text)
                if isinstance(inferred, dict) and inferred.get("label"):
                    inferred.setdefault("_confidence", float(inferred.get("confidence", 0.0)))
                    return inferred
            except Exception:
                pass

        return self._classify_heuristic(text)

    # ------------------------------------------------------------------
    # Heuristic classifier
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_heuristic(text: str) -> dict[str, Any]:
        lowered = text.lower().strip()
        first_word = lowered.split(" ", 1)[0] if lowered else ""

        is_question = (
            text.endswith("?")
            or first_word in _QUESTION_STARTERS
            or " or " in lowered and lowered.endswith(("?", "."))
        )
        is_imperative = first_word in _COMMAND_VERBS or lowered.startswith("please ")
        ends_open = text.endswith("...") or text.endswith("…")

        # Greeting / farewell take precedence.
        if any(lowered == g or lowered.startswith(g + " ") or lowered.startswith(g + ",") for g in _GREETINGS):
            return _result("greeting", 0.9, is_question, is_imperative, ends_open,
                           polarity="positive", topic=_topic_hint(lowered))
        if any(lowered == g or lowered.startswith(g + " ") or lowered.startswith(g + ",") for g in _FAREWELLS):
            return _result("farewell", 0.9, is_question, is_imperative, ends_open,
                           polarity="positive", topic="")

        if lowered in _AFFIRMATIONS or any(lowered.startswith(a + " ") or lowered == a for a in _AFFIRMATIONS):
            return _result("affirmation", 0.85, is_question, is_imperative, ends_open,
                           polarity="positive", topic=_topic_hint(lowered))
        if lowered in _NEGATIONS or any(lowered.startswith(n + " ") or lowered == n for n in _NEGATIONS):
            return _result("negation", 0.8, is_question, is_imperative, ends_open,
                           polarity="negative", topic=_topic_hint(lowered))

        if any(p in lowered for p in _COMPLIMENT_HINTS):
            return _result("compliment", 0.75, is_question, is_imperative, ends_open,
                           polarity="positive", topic="")
        if any(p in lowered for p in _COMPLAINT_HINTS):
            return _result("complaint", 0.75, is_question, is_imperative, ends_open,
                           polarity="negative", topic="")

        if any(p in lowered for p in _FEELING_PHRASES):
            polarity = "negative" if any(t in lowered for t in _NEGATIVE_TOKENS) else "positive"
            return _result("emotional_share", 0.7, is_question, is_imperative, ends_open,
                           polarity=polarity, topic=_topic_hint(lowered))

        if is_imperative:
            return _result("command", 0.8, is_question, is_imperative, ends_open,
                           polarity=_polarity(lowered), topic=_topic_hint(lowered),
                           expects_facts=True)

        if is_question:
            # Distinguish factual question from clarification (very short
            # follow-up like "really?" or "why?").
            label = "clarification" if len(lowered.split()) <= 3 else "question"
            confidence = 0.85 if label == "question" else 0.65
            return _result(label, confidence, is_question, is_imperative, ends_open,
                           polarity=_polarity(lowered), topic=_topic_hint(lowered),
                           expects_facts=True)

        # Default: chat / small_talk depending on length & content density.
        words = [w for w in re.findall(r"[a-z']+", lowered) if w not in _FILLER_TOKENS]
        if len(words) <= 3:
            return _result("small_talk", 0.55, is_question, is_imperative, ends_open,
                           polarity=_polarity(lowered), topic=_topic_hint(lowered))
        return _result("chat", 0.65, is_question, is_imperative, ends_open,
                       polarity=_polarity(lowered), topic=_topic_hint(lowered))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _polarity(lowered: str) -> str:
    pos = sum(1 for t in _POSITIVE_TOKENS if t in lowered)
    neg = sum(1 for t in _NEGATIVE_TOKENS if t in lowered)
    if pos > neg + 0:
        return "positive"
    if neg > pos:
        return "negative"
    return "neutral"


def _topic_hint(lowered: str) -> str:
    """Rough topic = first content word that isn't a filler."""
    for w in re.findall(r"[a-z']+", lowered):
        if w not in _FILLER_TOKENS and len(w) > 2:
            return w
    return ""


def _result(
    label: str, confidence: float, is_question: bool, is_imperative: bool,
    ends_open: bool, polarity: str, topic: str, expects_facts: bool = False,
) -> dict[str, Any]:
    return {
        "_confidence": float(confidence),
        "label": label,
        "confidence": float(confidence),
        "is_question": bool(is_question),
        "is_imperative": bool(is_imperative),
        "expects_facts": bool(expects_facts or is_question or is_imperative),
        "ends_open": bool(ends_open),
        "polarity": polarity,
        "topic_hint": topic,
    }
