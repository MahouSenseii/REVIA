"""Prosody computation for REVIA's TTS pipeline.

WS-4: Text-mutation transforms (_apply_thinking_pause, _apply_self_correction,
inject_quirks, inject_vocalizations) have been deleted.  They inserted "Hmm…",
"ngl,", "*sigh*" etc. into finished LLM text via regex, producing ungrammatical
seams and random inconsistency.  A modern LLM produces disfluency correctly
when the *prompt* asks for it — which is where it now lives (persona_manager.py
speech_quirks + prompt_assembly.py natural-speech block).

What remains is legitimate:
  - Prosody hints (pitch / rate / energy) for TTS from emotion label.
  - A hard verbosity cap as a safety guard (not a "feel" feature).

The class name HumanFeelLayer is kept for import compatibility.
See prosody.py for the forward alias (ProsodyLayer).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

_log = logging.getLogger(__name__)

# Hard safety caps for verbosity.  Source of truth is ProfileEngine when
# available; these are genuine fallback constants (no longer falsely claimed
# to be "zero hardcoded values").
_VERBOSITY_MIN_WORDS: int = 25
_VERBOSITY_MAX_WORDS: int = 220


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ProsodyHints:
    """TTS control hints derived from emotion state."""
    pitch_semitones: float = 0.0
    rate_multiplier: float = 1.0
    energy_db: float = 0.0
    affect_mode: str = "natural"

    def to_dict(self) -> dict[str, Any]:
        return {
            "pitch_semitones": round(self.pitch_semitones, 3),
            "rate_multiplier": round(self.rate_multiplier, 3),
            "energy_db":       round(self.energy_db, 3),
            "affect_mode":     self.affect_mode,
        }


@dataclass
class HFLResult:
    """Result of a prosody pass (WS-4: text is no longer mutated)."""
    original:   str          = ""
    processed:  str          = ""   # identical to original unless trimmed
    prosody:    ProsodyHints = field(default_factory=ProsodyHints)
    trimmed:    bool         = False
    elapsed_ms: float        = 0.0
    # original_len / processed_len in WORD counts for quick sanity checks
    original_len:  int = field(init=False)
    processed_len: int = field(init=False)

    def __post_init__(self) -> None:
        self.original_len  = len(self.original.split())
        self.processed_len = len(self.processed.split())

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_len":  self.original_len,
            "processed_len": self.processed_len,
            "prosody":       self.prosody.to_dict(),
            "trimmed":       self.trimmed,
            "elapsed_ms":    round(self.elapsed_ms, 2),
        }


# ---------------------------------------------------------------------------
# Main class (name kept for import compat; see prosody.py for ProsodyLayer)
# ---------------------------------------------------------------------------

class HumanFeelLayer:
    """Prosody + safety-cap layer for REVIA's expression pipeline.

    WS-4: Text-mutation transforms removed.  This class now only:
      1. Computes TTS prosody hints from the emotion label.
      2. Applies a hard word-count cap as a runaway-length safety guard.

    Disfluency, quirks, and pacing are driven by the persona prompt.
    """

    def __init__(self, profile_engine=None) -> None:
        self._pe = profile_engine
        _log.debug("[Prosody] Initialized (text-mutation removed, WS-4)")

    @classmethod
    def create(cls, profile_engine=None) -> "HumanFeelLayer":
        return cls(profile_engine)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        reply: str,
        emotion_label: str = "neutral",
        rng_seed: int | None = None,   # kept for call-site compat; now a no-op
    ) -> HFLResult:
        """Compute prosody hints and apply the safety length cap.

        Parameters
        ----------
        reply:
            Raw LLM output.  NOT mutated — returned as-is unless the word
            count exceeds the hard cap.
        emotion_label:
            Emotion label from EmotionNet (e.g. "happy", "sad").
        rng_seed:
            Legacy parameter, kept for call-site compatibility.  No longer
            has any effect (text-mutation RNG was removed in WS-4).
        """
        t0 = time.monotonic()

        if not reply or not reply.strip():
            return HFLResult(
                original=reply or "",
                processed=reply or "",
                elapsed_ms=(time.monotonic() - t0) * 1000,
            )

        result = HFLResult(original=reply, processed=reply)

        # 1. Prosody hints (pure math — no mutation)
        result.prosody = self._compute_prosody(emotion_label)

        # 2. Verbosity trim (hard safety cap only)
        result = self._apply_verbosity_trim(result)

        result.elapsed_ms = (time.monotonic() - t0) * 1000
        _log.debug(
            "[Prosody] trimmed=%s elapsed=%.1f ms",
            result.trimmed, result.elapsed_ms,
        )
        return result

    def compute_prosody(self, emotion_label: str) -> ProsodyHints:
        """Public alias for _compute_prosody (used by tests and TTS callers)."""
        return self._compute_prosody(emotion_label)

    # ------------------------------------------------------------------
    # Verbosity cap (safety guard — NOT a "feel" feature)
    # ------------------------------------------------------------------

    def _apply_verbosity_trim(self, result: HFLResult) -> HFLResult:
        """Hard safety cap: truncate runaway-long replies at the word limit."""
        max_words = self._get_verbosity()
        words = result.processed.split()
        if len(words) <= max_words:
            return result
        trimmed_text = " ".join(words[:max_words])
        # Try to end on a sentence boundary
        for punct in (".", "!", "?"):
            idx = trimmed_text.rfind(punct)
            if idx > len(trimmed_text) // 2:
                trimmed_text = trimmed_text[: idx + 1]
                break
        result.processed = trimmed_text
        result.trimmed = True
        result.processed_len = len(result.processed.split())
        return result

    # ------------------------------------------------------------------
    # Prosody computation
    # ------------------------------------------------------------------

    def _compute_prosody(self, emotion_label: str) -> ProsodyHints:
        """Map emotion label → TTS pitch / rate / energy hints."""
        intensity = self._get_emotion_intensity()
        mode      = self._get_affect_display_mode()
        valence   = self._get_baseline_valence()
        rate_mod  = self._get_speech_rate_modifier()

        emotion = (emotion_label or "neutral").lower()

        # Emotion → raw prosody values
        _emotion_map: dict[str, tuple[float, float, float]] = {
            # label         pitch_st  rate    energy_db
            "happy":        (1.1,     1.07,   1.1),
            "excited":      (1.8,     1.15,   2.0),
            "joyful":       (1.5,     1.10,   1.5),
            "sad":          (-1.2,    0.90,  -1.2),
            "melancholy":   (-0.8,    0.93,  -0.8),
            "angry":        (0.5,     1.05,   2.5),
            "frustrated":   (0.3,     1.02,   1.5),
            "fearful":      (-0.3,    1.08,   0.5),
            "surprised":    (1.2,     1.10,   1.2),
            "disgusted":    (-0.5,    0.95,   0.8),
            "contemptuous": (-0.8,    0.88,   0.5),
            "confused":     (0.2,     0.97,   0.3),
            "curious":      (0.6,     1.03,   0.6),
            "calm":         (-0.2,    0.95,  -0.3),
            "bored":        (-0.5,    0.88,  -0.8),
            "empathetic":   (-0.3,    0.93,   0.2),
            "playful":      (1.3,     1.08,   1.3),
            "neutral":      (0.0,     1.00,   0.0),
        }

        pitch_raw, rate_raw, energy_raw = _emotion_map.get(
            emotion, (0.0, 1.0, 0.0)
        )

        # Scale by profile intensity, valence, and speech-rate modifier
        pitch  = pitch_raw  * intensity * (1.0 + valence * 0.1)
        rate   = 1.0 + (rate_raw - 1.0) * intensity * rate_mod
        energy = energy_raw * intensity

        return ProsodyHints(
            pitch_semitones=round(pitch,  3),
            rate_multiplier=round(rate,   3),
            energy_db=      round(energy, 3),
            affect_mode=mode,
        )

    # ------------------------------------------------------------------
    # Profile-sourced parameters (with fallbacks)
    # ------------------------------------------------------------------

    def _get_verbosity(self) -> int:
        if self._pe:
            try:
                v = float(self._pe.get_verbosity())
                # Map [0.0, 1.0] → [_VERBOSITY_MIN_WORDS, _VERBOSITY_MAX_WORDS]
                span = _VERBOSITY_MAX_WORDS - _VERBOSITY_MIN_WORDS
                return int(_VERBOSITY_MIN_WORDS + v * span)
            except Exception:
                pass
        return _VERBOSITY_MAX_WORDS

    def _get_emotion_intensity(self) -> float:
        if self._pe:
            try:
                return float(self._pe.get_emotion_intensity())
            except Exception:
                pass
        return 1.0

    def _get_affect_display_mode(self) -> str:
        if self._pe:
            try:
                return str(self._pe.get_affect_display_mode())
            except Exception:
                pass
        return "natural"

    def _get_baseline_valence(self) -> float:
        if self._pe:
            try:
                return float(self._pe.get_baseline_valence())
            except Exception:
                pass
        return 0.0

    def _get_speech_rate_modifier(self) -> float:
        if self._pe:
            try:
                return float(self._pe.get_speech_rate_modifier())
            except Exception:
                pass
        return 1.0
