"""
Human Feel Layer (HFL) — PRD §13
==================================
Post-processes the raw LLM reply to make it feel more natural, warm, and
human-paced.  All probabilities and rates are consumed from ProfileEngine.

Transformations applied (in order)
------------------------------------
1. Thinking pause prefix     — "Hmm…" / "Let me think…" prepended with
                               probability = profile.timing.thinking_pause_probability
2. Self-correction insertion — "—actually, …" / "wait, let me rephrase…"
                               injected at clause boundaries with rate =
                               profile.behavior.self_correction_rate
3. Prosody computation       — derive SSML-style hints (pitch, rate, energy)
                               from profile.emotion params for the TTS layer
4. Verbosity trim            — truncate over-length replies to keep them
                               within the verbosity budget

The layer is deterministic when a fixed ``random_seed`` is supplied (useful
for testing) and stochastic in production (seed=None).

All thresholds consumed through ProfileEngine — zero hardcoded values.
"""
from __future__ import annotations

import logging
import random
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Thinking pause candidates (PRD section 13.1)
# ---------------------------------------------------------------------------
_THINKING_PAUSES = [
    "Hmm… ",
    "Let me think… ",
    "Okay… ",
    "Right, so… ",
    "Let's see… ",
    "Good question — ",
    "Interesting… ",
    "One sec — ",
]

# Self-correction injection phrases (PRD section 13.2)
_SELF_CORRECTIONS = [
    "actually, ", "well, ", "I mean, ", "wait, ", "okay so, ",
    "honestly, ", "here's the thing, ", "look, ", "ngl, ", "like, ",
    "you know what, ", "hmm, ", "let me think... ", "so basically, ",
    "oh wait, ", "real talk, ",
]

# Clause-boundary regex: comma, semicolon, dash, "and", "but"
_CLAUSE_BOUNDARY_RE = re.compile(
    r"(,|;| — | and | but | though | although | however)"
)

# Verbosity -> target word-count mapping
# verbosity 0.0 -> 25 words, 1.0 -> 220 words
_VERBOSITY_MIN_WORDS = 25
_VERBOSITY_MAX_WORDS = 220


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ProsodyHints:
    """SSML-compatible prosody parameters for the TTS subsystem."""
    pitch_semitones: float = 0.0       # relative pitch shift (± semitones)
    rate_multiplier: float = 1.0       # speech rate (0.5–2.0)
    energy_db:       float = 0.0       # loudness shift in dB
    affect_mode:     str   = "natural" # suppressed | natural | amplified

    def to_dict(self) -> dict[str, Any]:
        return {
            "pitch_semitones": round(self.pitch_semitones, 2),
            "rate_multiplier": round(self.rate_multiplier, 3),
            "energy_db":       round(self.energy_db, 2),
            "affect_mode":     self.affect_mode,
        }


@dataclass
class HFLResult:
    """Full record of all HFL transformations applied to a reply."""
    original:          str
    processed:         str
    pause_added:       bool                    = False
    correction_added:  bool                    = False
    trimmed:           bool                    = False
    prosody:           ProsodyHints            = field(default_factory=ProsodyHints)
    elapsed_ms:        float                   = 0.0
    notes:             list[str]               = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_len":    len(self.original.split()),
            "processed_len":   len(self.processed.split()),
            "pause_added":     self.pause_added,
            "correction_added":self.correction_added,
            "trimmed":         self.trimmed,
            "prosody":         self.prosody.to_dict(),
            "elapsed_ms":      round(self.elapsed_ms, 2),
            "notes":           self.notes,
        }


# ---------------------------------------------------------------------------
# HFL engine
# ---------------------------------------------------------------------------

class HumanFeelLayer:
    """
    PRD §13 — Human Feel Layer

    Usage::

        hfl = HumanFeelLayer(profile_engine)

        result = hfl.process(
            reply         = raw_llm_text,
            emotion_label = "happy",
            rng_seed      = None,   # None = stochastic
        )
        tts.set_prosody(result.prosody)
        deliver(result.processed)
    """

    def __init__(self, profile_engine=None, rng_seed: int | None = None):
        self._pe = profile_engine
        self._rng = random.Random(rng_seed)
        self._rng_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Thread-safe RNG helpers — lock only wraps the actual RNG call so
    # the rest of the pipeline (profile lookups, prosody math, etc.) can
    # run concurrently from other threads.
    # ------------------------------------------------------------------

    def _rng_random(self) -> float:
        with self._rng_lock:
            return self._rng.random()

    def _rng_choice(self, seq):
        with self._rng_lock:
            return self._rng.choice(seq)

    def _rng_seed(self, seed: int | None) -> None:
        with self._rng_lock:
            self._rng.seed(seed)

    # Public API

    def process(
        self,
        reply: str,
        emotion_label: str = "neutral",
        rng_seed: int | None = None,
    ) -> HFLResult:
        """
        Apply all HFL transformations and return an :class:`HFLResult`.

        Parameters
        ----------
        reply :
            Raw LLM output to process.
        emotion_label :
            The active emotion label from EmotionNet (e.g. "happy", "sad").
        rng_seed :
            Override the instance RNG seed for this call (useful for tests).
        """
        t0 = time.monotonic()

        # Guard: empty / whitespace-only replies pass through untouched
        if not reply or not reply.strip():
            _log.debug("[HFL] Empty reply — skipping all transformations")
            return HFLResult(
                original=reply or "",
                processed=reply or "",
                elapsed_ms=(time.monotonic() - t0) * 1000,
            )

        if rng_seed is not None:
            self._rng_seed(rng_seed)

        result = HFLResult(original=reply, processed=reply)

        # 1. Thinking pause
        result = self._apply_thinking_pause(result)

        # 2. Self-correction
        result = self._apply_self_correction(result)

        # 3. Prosody (pure math — no RNG, no lock needed)
        result.prosody = self._compute_prosody(emotion_label)

        # 4. Verbosity trim (deterministic — no RNG)
        result = self._apply_verbosity_trim(result)

        # 5. Inject personality quirks from profile
        if self._pe:
            quirks = self._pe.get_speech_quirks()
            freq = self._pe.get_quirk_frequency()
        else:
            quirks = ["honestly", "here's the thing", "ngl", "okay so", "like"]
            freq = 0.15
        result.processed = self.inject_quirks(result.processed, quirks, freq)

        # 6. Inject emotional vocalizations
        if emotion_label and emotion_label.lower() not in ("neutral", "disabled", "---", ""):
            result.processed = self.inject_vocalizations(result.processed, emotion_label)

        result.elapsed_ms = (time.monotonic() - t0) * 1000
        _log.debug(
            "[HFL] pause=%s correction=%s trimmed=%s elapsed=%.1f ms",
            result.pause_added, result.correction_added,
            result.trimmed, result.elapsed_ms,
        )
        return result

    def compute_prosody(self, emotion_label: str) -> ProsodyHints:
        """Public convenience accessor for prosody-only use."""
        return self._compute_prosody(emotion_label)

    # Transformation steps

    def _apply_thinking_pause(self, result: HFLResult) -> HFLResult:
        prob = self._get_thinking_pause_probability()
        if self._rng_random() < prob:
            pause = self._rng_choice(_THINKING_PAUSES)
            result.processed  = pause + result.processed
            result.pause_added = True
            result.notes.append(f"thinking_pause: {pause!r}")
        return result

    def _apply_self_correction(self, result: HFLResult) -> HFLResult:
        rate = self._get_self_correction_rate()
        if self._rng_random() >= rate:
            return result   # skip with probability (1 - rate)

        text = result.processed
        # Find all clause boundaries
        boundaries = [m.start() for m in _CLAUSE_BOUNDARY_RE.finditer(text)]
        # Only inject mid-reply (not within first or last 20% of text)
        lo = int(len(text) * 0.20)
        hi = int(len(text) * 0.80)
        eligible = [pos for pos in boundaries if lo <= pos <= hi]

        if not eligible:
            return result

        pos        = self._rng_choice(eligible)
        correction = self._rng_choice(_SELF_CORRECTIONS)

        # Insert correction phrase AFTER the delimiter, preserving the original char
        new_text = text[:pos + 1] + " " + correction + text[pos + 1:].lstrip()
        result.processed       = new_text
        result.correction_added = True
        result.notes.append(f"self_correction at pos {pos}: {correction!r}")
        return result

    def _apply_verbosity_trim(self, result: HFLResult) -> HFLResult:
        verbosity  = self._get_verbosity()
        max_words  = int(
            _VERBOSITY_MIN_WORDS + verbosity * (_VERBOSITY_MAX_WORDS - _VERBOSITY_MIN_WORDS)
        )
        words = result.processed.split()
        if len(words) <= max_words:
            return result

        # Trim to max_words at the nearest sentence end
        trimmed_words = words[:max_words]
        trimmed_text  = " ".join(trimmed_words)

        # Try to end at a sentence boundary (. ! ?)
        last_sentence_end = max(
            trimmed_text.rfind("."),
            trimmed_text.rfind("!"),
            trimmed_text.rfind("?"),
        )
        cutoff = int(len(trimmed_text) * 0.70)
        if last_sentence_end > cutoff and last_sentence_end >= 0:
            trimmed_text = trimmed_text[:last_sentence_end + 1]
        else:
            trimmed_text = trimmed_text.rstrip(",;:—") + "."

        result.processed = trimmed_text
        result.trimmed   = True
        result.notes.append(
            f"verbosity_trim: {len(words)} → {len(result.processed.split())} words "
            f"(max={max_words})"
        )
        return result

    def inject_quirks(self, text: str, quirks: list, frequency: float = 0.15) -> str:
        """Randomly inject personality quirks at sentence boundaries."""
        if not quirks or not text:
            return text
        # Split on sentence-ending punctuation while preserving the delimiter
        parts = re.split(r'(?<=[.!?]) +', text)
        result = []
        for i, sent in enumerate(parts):
            if i > 0 and sent and self._rng_random() < frequency and quirks:
                quirk = self._rng_choice(quirks)
                sent = f"{quirk}, {sent[0].lower()}{sent[1:]}"
            result.append(sent)
        return ' '.join(result)

    def inject_vocalizations(self, text: str, emotion: str) -> str:
        """Add emotional vocalizations based on current mood."""
        markers = {
            "happy": ["haha", "lol", "heh"],
            "excited": ["omg", "yooo", "wait wait wait"],
            "nervous": ["um", "uh", "ehh"],
            "angry": ["ugh", "seriously", "wow okay"],
            "sad": ["sigh", "*sigh*", "..."],
        }
        if emotion in markers and text and self._rng_random() < 0.25:
            marker = self._rng_choice(markers[emotion])
            if self._rng_random() < 0.5:
                # Prepend marker before text
                if len(text) > 1:
                    text = f"{marker}, {text[0].lower()}{text[1:]}"
                else:
                    text = f"{marker}, {text}"
            else:
                # Insert after first sentence
                dot = text.find('. ')
                if dot > 0:
                    text = text[:dot+2] + f"{marker}... " + text[dot+2:]
        return text

    # Prosody computation

    def _compute_prosody(self, emotion_label: str) -> ProsodyHints:
        """
        Derive SSML prosody hints from the emotion label and profile params.

        Emotion → voice mapping (approximate):
          happy/excited   → higher pitch, faster rate, more energy
          sad/concerned   → lower pitch, slower rate, less energy
          neutral         → near-baseline
          empathetic      → slightly lower pitch, slower rate, softer energy
        """
        intensity    = self._get_emotion_intensity()
        affect_mode  = self._get_affect_display_mode()
        speech_rate  = self._get_speech_rate_modifier()

        # Base adjustments per emotion label
        emotion_map: dict[str, tuple[float, float, float]] = {
            # (pitch_st, rate_mult, energy_db)
            "happy":     ( 2.0,  0.05,  2.0),
            "excited":   ( 3.5,  0.12,  4.0),
            "positive":  ( 1.5,  0.03,  1.5),
            "sad":       (-2.0, -0.10, -3.0),
            "concerned": (-1.0, -0.05, -1.5),
            "negative":  (-1.5, -0.08, -2.0),
            "neutral":   ( 0.0,  0.00,  0.0),
            "empathetic":(-0.5, -0.08, -1.0),
            "surprised": ( 2.5,  0.08,  3.0),
            "angry":     (-1.0,  0.10,  4.0),
        }

        base_pitch, base_rate, base_energy = emotion_map.get(
            emotion_label.lower(), (0.0, 0.0, 0.0)
        )

        # Scale by emotion_intensity (from profile)
        # affect_mode further amplifies or suppresses
        amp_map = {"suppressed": 0.3, "natural": 1.0, "amplified": 1.6}
        amp     = amp_map.get(affect_mode, 1.0)

        pitch  = base_pitch  * intensity * amp
        rate   = speech_rate + base_rate * intensity * amp
        energy = base_energy * intensity * amp

        # Clamp to reasonable SSML ranges
        pitch  = max(-6.0, min(pitch, 6.0))
        rate   = max(0.50, min(rate, 2.0))
        energy = max(-6.0, min(energy, 6.0))

        return ProsodyHints(
            pitch_semitones = round(pitch, 2),
            rate_multiplier = round(rate, 3),
            energy_db       = round(energy, 2),
            affect_mode     = affect_mode,
        )

    # Profile parameter accessors

    def _get_thinking_pause_probability(self) -> float:
        if self._pe:
            return float(self._pe.thinking_pause_probability)
        return 0.20

    def _get_self_correction_rate(self) -> float:
        if self._pe:
            return float(self._pe.self_correction_rate)
        return 0.15

    def _get_verbosity(self) -> float:
        if self._pe:
            return float(self._pe.verbosity)
        return 0.50

    def _get_emotion_intensity(self) -> float:
        if self._pe:
            return float(self._pe.emotion_intensity)
        return 0.55

    def _get_affect_display_mode(self) -> str:
        if self._pe:
            return str(self._pe.get_emotion_param("affect_display_mode", "natural"))
        return "natural"

    def _get_baseline_valence(self) -> float:
        if self._pe:
            return float(self._pe.get_emotion_param("baseline_valence", 0.55))
        return 0.55

    def _get_speech_rate_modifier(self) -> float:
        if self._pe:
            return float(self._pe.get_timing_param("speech_rate_modifier", 1.00))
        return 1.00
