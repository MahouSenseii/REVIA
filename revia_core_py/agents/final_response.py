"""FinalResponseBuilder — assemble ONE canonical answer from the agents.

Per spec rule:
    "Many agents think.  One Revia speaks."

The ReasoningAgent produces the candidate text; Memory and Emotion
agents act as soft conditioning data (already folded into the
ReasoningAgent's prompt via the system_prompt_provider) and as audit
metadata in the final output packet.

V2.2: a ``voice_style`` block (from :class:`VoiceStyleAgent`) is honoured
non-destructively:

* A soft ``max_sentences`` cap trims long answers (only when the
  reasoning text exceeds the cap by a clear margin).
* ``speech_rate`` / ``pitch_shift_st`` / ``tone`` are surfaced via the
  ``prosody`` payload so the TTS / interface layer can use them.
* A ``signature_phrase`` is appended only when the candidate doesn't
  already include one similar.

HFL post-processing remains optional: when the ReplyPlanner already ran
HFL we skip it here, otherwise we run it now so the output always carries
prosody hints for the voice layer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FinalResponse:
    """Canonical Revia output packet produced after the parallel run."""

    text: str
    emotion_label: str = "neutral"
    confidence: float = 0.0
    prosody: dict[str, Any] = field(default_factory=dict)
    agent_results: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    voice_style: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "emotion_label": self.emotion_label,
            "confidence": round(float(self.confidence), 4),
            "prosody": dict(self.prosody),
            "agent_results": list(self.agent_results),
            "notes": list(self.notes),
            "voice_style": dict(self.voice_style),
        }


class FinalResponseBuilder:
    """Combine agent outputs into a single Revia response.

    ``hfl`` is optional: if ReasoningAgent's planner already ran HFL we
    skip it here, otherwise we run it now so the output always carries
    prosody hints for the voice layer.
    """

    def __init__(self, hfl=None):
        self._hfl = hfl

    def build(
        self,
        agent_results: list,
        reasoning_result_payload: dict[str, Any] | None,
        emotion_label_default: str = "neutral",
        voice_style: dict[str, Any] | None = None,
    ) -> FinalResponse:
        notes: list[str] = []
        voice_style = dict(voice_style or {})

        if reasoning_result_payload is None:
            notes.append("reasoning_unavailable")
            return FinalResponse(
                text="",
                emotion_label=emotion_label_default,
                confidence=0.0,
                agent_results=[r.to_dict() for r in agent_results],
                notes=notes,
                voice_style=voice_style,
            )

        text = str(reasoning_result_payload.get("text") or "").strip()
        emotion_label = str(
            reasoning_result_payload.get("emotion_label") or emotion_label_default
        )
        confidence = float(reasoning_result_payload.get("avs_best_score") or 0.0)
        if confidence <= 0.0:
            for r in agent_results:
                if r.agent == "ReasoningAgent" and r.success:
                    confidence = float(r.confidence)
                    break

        prosody: dict[str, Any] = {}
        # HFL only when reasoning didn't already (planner runs it).
        if self._hfl is not None and not reasoning_result_payload.get("notes", []):
            try:
                hfl_result = self._hfl.process(text, emotion_label)
                text = hfl_result.processed
                prosody = (
                    hfl_result.prosody.to_dict()
                    if hasattr(hfl_result.prosody, "to_dict") else {}
                )
                notes.append("hfl_post_processed")
            except Exception as exc:
                notes.append(f"hfl_skipped: {type(exc).__name__}")

        # Apply voice-style hints (additive, non-destructive).
        if voice_style:
            text, style_notes = self._apply_voice_style(text, voice_style)
            notes.extend(style_notes)
            prosody = self._merge_prosody(prosody, voice_style)

        return FinalResponse(
            text=text,
            emotion_label=emotion_label,
            confidence=confidence,
            prosody=prosody,
            agent_results=[r.to_dict() for r in agent_results],
            notes=notes,
            voice_style=voice_style,
        )

    # ------------------------------------------------------------------
    # Voice-style application
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_voice_style(
        text: str,
        voice_style: dict[str, Any],
    ) -> tuple[str, list[str]]:
        notes: list[str] = []
        if not text:
            return text, notes

        # 1) Soft sentence cap.
        max_sentences = int(voice_style.get("max_sentences") or 0)
        if max_sentences > 0:
            sentences = _split_sentences(text)
            if len(sentences) > max_sentences + 1:
                # Only trim when we exceed by 2+; otherwise leave it.
                trimmed = " ".join(sentences[:max_sentences]).strip()
                if trimmed:
                    text = trimmed
                    notes.append(f"voice_style_sentence_cap_{max_sentences}")

        # 2) Optional signature interjection.
        sig = str(voice_style.get("signature_phrase") or "").strip()
        if sig:
            lowered = text.lower()
            if sig.lower() not in lowered:
                # Place at the start for greetings, otherwise as a tail.
                tone = str(voice_style.get("tone") or "").lower()
                intent = str(voice_style.get("intent_label") or "").lower()
                if intent in ("greeting", "compliment") and tone in ("warm", "playful"):
                    text = f"{sig.capitalize()} — {text}"
                    notes.append("voice_style_signature_lead")
                elif intent == "farewell":
                    text = f"{text} {sig.capitalize()}."
                    notes.append("voice_style_signature_tail")
        return text, notes

    @staticmethod
    def _merge_prosody(
        existing: dict[str, Any],
        voice_style: dict[str, Any],
    ) -> dict[str, Any]:
        merged = dict(existing or {})
        for key in ("speech_rate", "pitch_shift_st", "tone"):
            if key in voice_style and key not in merged:
                merged[key] = voice_style[key]
        return merged


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(text or "") if p.strip()]
    return parts or ([text.strip()] if text and text.strip() else [])
