"""VoiceStyleAgent — derives style/prosody hints from profile + intent + emotion.

Output is a *metadata-only* contribution; this agent never rewrites the
candidate text directly.  The :class:`FinalResponseBuilder` (and TTS layer
downstream) read the style block to decide:

* how terse / verbose the answer should be
* whether to include a signature interjection / quirk
* what TTS rate / pitch to suggest

V1 keeps it deterministic and cheap (no LLM call).  Later versions can
plug into the router via the ``voice_style`` task type.

Output schema::

    {
        "formality": 0..1          (lower = casual)
        "verbosity": 0..1          (higher = longer)
        "tone": "warm"|"neutral"|"playful"|"focused"|"reassuring",
        "speech_rate": 0.85..1.15  (TTS rate hint, 1.0 = profile default)
        "pitch_shift_st": -2..+2   (semitones)
        "signature_phrase": str    (optional interjection)
        "max_sentences": int       (soft cap)
    }
"""
from __future__ import annotations

from typing import Any

from .agent_base import Agent, AgentContext


_TONE_BY_EMOTION = {
    # Negative / vulnerable user emotions -> reassuring tone.
    "sadness": "reassuring", "fear": "reassuring", "anxious": "reassuring",
    "grief": "reassuring",   "lonely": "reassuring",
    # Positive / playful.
    "joy": "playful", "amusement": "playful", "excitement": "playful",
    "love": "warm",   "gratitude": "warm",     "pride": "warm",
    # Anger / frustration -> calm + focused.
    "anger": "focused", "annoyance": "focused", "disgust": "focused",
    # Default.
    "neutral": "warm", "calm": "warm",
    "surprise": "warm", "curiosity": "warm",
}


_TONE_BY_INTENT = {
    "command": "focused",
    "question": "warm",
    "clarification": "focused",
    "greeting": "warm",
    "farewell": "warm",
    "affirmation": "playful",
    "negation": "neutral",
    "small_talk": "playful",
    "compliment": "playful",
    "complaint": "reassuring",
    "emotional_share": "reassuring",
    "chat": "warm",
}


class VoiceStyleAgent(Agent):
    name = "VoiceStyleAgent"
    default_timeout_ms = 250

    def __init__(self, profile_engine=None, model_router=None):
        self._pe = profile_engine
        self._router = model_router

    def run(self, context: AgentContext) -> dict[str, Any]:
        context.cancel_token.raise_if_cancelled()

        # Optional plug-in via router (kept best-effort; if it fails or
        # returns garbage we fall back to deterministic defaults).
        if self._router is not None and self._router.has("voice_style"):
            try:
                inferred = self._router.call("voice_style", context)
                if isinstance(inferred, dict) and "tone" in inferred:
                    inferred.setdefault("_confidence",
                                        float(inferred.get("confidence", 0.6)))
                    return inferred
            except Exception:
                pass

        intent_label = self._intent_label(context)
        emotion_label = self._emotion_label(context)
        polarity = self._polarity(context)

        # Profile-driven baselines.
        formality, verbosity = self._profile_baseline()

        tone = _TONE_BY_INTENT.get(intent_label, "warm")
        # Emotion overrides intent for negative / vulnerable cases.
        emo_tone = _TONE_BY_EMOTION.get(emotion_label.lower(), None)
        if emo_tone == "reassuring":
            tone = "reassuring"
        elif tone == "warm" and emo_tone:
            tone = emo_tone

        # Verbosity adjustments.
        max_sentences = self._max_sentences(verbosity, intent_label)

        speech_rate, pitch_shift = self._prosody(tone, polarity)

        signature = self._signature_phrase(intent_label, tone)

        return {
            "_confidence": 0.7,
            "formality": round(float(formality), 3),
            "verbosity": round(float(verbosity), 3),
            "tone": tone,
            "speech_rate": round(float(speech_rate), 3),
            "pitch_shift_st": round(float(pitch_shift), 2),
            "signature_phrase": signature,
            "max_sentences": int(max_sentences),
            "intent_label": intent_label,
            "emotion_label": emotion_label,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _intent_label(ctx: AgentContext) -> str:
        intent = ctx.metadata.get("intent") or {}
        if isinstance(intent, dict):
            return str(intent.get("label") or "chat").lower()
        return "chat"

    @staticmethod
    def _emotion_label(ctx: AgentContext) -> str:
        return str(ctx.metadata.get("emotion_label") or "neutral").lower()

    @staticmethod
    def _polarity(ctx: AgentContext) -> str:
        intent = ctx.metadata.get("intent") or {}
        if isinstance(intent, dict):
            return str(intent.get("polarity") or "neutral").lower()
        return "neutral"

    def _profile_baseline(self) -> tuple[float, float]:
        # Pull from the profile engine when available; otherwise sane defaults
        # (slightly casual, medium verbosity).
        formality = 0.4
        verbosity = 0.5
        if self._pe is not None:
            try:
                verbosity = float(self._pe.verbosity)
            except Exception:
                pass
            try:
                # ProfileEngine doesn't expose formality directly; reach into
                # the active profile if it provides one.
                cur = self._pe.current or {}
                bp = (cur.get("behavior") or {})
                if "formality" in bp:
                    formality = float(bp.get("formality", formality))
            except Exception:
                pass
        return (
            max(0.0, min(1.0, formality)),
            max(0.0, min(1.0, verbosity)),
        )

    @staticmethod
    def _max_sentences(verbosity: float, intent_label: str) -> int:
        # Map [0,1] verbosity to a sentence cap, then bias by intent.
        base = 1 + int(round(verbosity * 5))   # 1..6
        if intent_label in ("greeting", "farewell", "affirmation",
                            "negation", "small_talk"):
            return max(1, min(2, base))
        if intent_label in ("compliment", "clarification"):
            return max(1, min(3, base))
        if intent_label in ("question", "command", "emotional_share"):
            return max(2, min(6, base + 1))
        return base

    @staticmethod
    def _prosody(tone: str, polarity: str) -> tuple[float, float]:
        # speech_rate = 1.0 baseline; tone / polarity nudges.
        rate = 1.0
        pitch = 0.0
        if tone == "playful":
            rate = 1.05
            pitch = 0.5
        elif tone == "reassuring":
            rate = 0.92
            pitch = -0.5
        elif tone == "focused":
            rate = 1.0
            pitch = 0.0
        elif tone == "warm":
            rate = 0.98
            pitch = 0.2
        if polarity == "negative":
            rate = max(0.85, rate - 0.05)
        elif polarity == "positive":
            rate = min(1.15, rate + 0.02)
        return rate, pitch

    @staticmethod
    def _signature_phrase(intent_label: str, tone: str) -> str:
        if intent_label == "greeting":
            return "" if tone == "focused" else "hey there"
        if intent_label == "farewell":
            return "talk soon"
        if intent_label == "compliment" and tone in ("playful", "warm"):
            return "thanks!"
        if intent_label == "complaint" or tone == "reassuring":
            return ""    # reassuring tone shouldn't add filler
        return ""
