"""
Answer Validation System (AVS) — PRD §10
==========================================
Multi-dimensional scoring rubric with a hard gate against
``minimum_answer_threshold`` sourced from the active ProfileEngine.

Scoring dimensions
------------------
  intent_coverage       35 %  — did the reply address the user's intent?
  factual_coherence     20 %  — internal consistency, no contradiction
  emotional_alignment   20 %  — matches the emotional tone required
  novelty               15 %  — not a near-duplicate of a recent reply
  length_adherence      10 %  — within verbosity-appropriate token range

The composite score is a weighted sum in [0, 1].  The reply is accepted
only if composite >= profile.minimum_answer_threshold.  If not, a regen
cycle is triggered up to profile.regen_patience times before the best
candidate is accepted via fallback.

All thresholds are consumed through ProfileEngine — zero hardcoded values.
"""
from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scoring weights  (must sum to 1.0)
# ---------------------------------------------------------------------------
_WEIGHTS: dict[str, float] = {
    "intent_coverage":     0.32,
    "factual_coherence":   0.18,
    "emotional_alignment": 0.18,
    "novelty":             0.12,
    "length_adherence":    0.08,
    "entertainment_value": 0.12,
}
assert abs(sum(_WEIGHTS.values()) - 1.0) < 1e-9, "AVS weight sum != 1.0"

# Pre-compiled regexes used in scoring (avoid re-compiling on every call)
_FILLER_RE = re.compile(r"\.\.\.|<[^>]+>|\[INST\]|\[/INST\]|###")
_WORD_WITH_APOSTROPHE_RE = re.compile(r"[a-z']+")   # intent coverage keyword extraction
_ALPHA_WORD_RE = re.compile(r"[a-z]+")              # emotional alignment token extraction
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")          # sentence boundary split

# Contradiction pairs for factual coherence — compiled once, reused every call
_CONTRADICTION_PAIRS: tuple[tuple, ...] = (
    (re.compile(r"\byes\b"),        re.compile(r"\bno\b")),
    (re.compile(r"\balways\b"),     re.compile(r"\bnever\b")),
    (re.compile(r"\beveryone\b"),   re.compile(r"\bno\s+one\b")),
    (re.compile(r"\bimpossible\b"), re.compile(r"\bpossible\b")),
)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
_STOPWORDS = {
    "a","an","the","is","are","was","were","be","been","being",
    "have","has","had","do","does","did","will","would","could",
    "should","may","might","shall","can","need","dare","ought",
    "i","you","he","she","it","we","they","me","him","her","us",
    "my","your","his","its","our","their","what","how","why",
    "when","where","who","which","that","this","these","those",
    "and","or","but","if","so","yet","for","nor","both","either",
    "neither","not","no","yes","just","also","about","up","out",
    "in","on","at","to","from","with","by","of","as","into","onto",
}

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AVSScores:
    intent_coverage:     float = 0.0
    factual_coherence:   float = 0.0
    emotional_alignment: float = 0.0
    novelty:             float = 0.0
    length_adherence:    float = 0.0
    entertainment_value: float = 0.0

    @property
    def composite(self) -> float:
        return (
            self.intent_coverage     * _WEIGHTS["intent_coverage"]
            + self.factual_coherence   * _WEIGHTS["factual_coherence"]
            + self.emotional_alignment * _WEIGHTS["emotional_alignment"]
            + self.novelty             * _WEIGHTS["novelty"]
            + self.length_adherence    * _WEIGHTS["length_adherence"]
            + self.entertainment_value * _WEIGHTS["entertainment_value"]
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "intent_coverage":     round(self.intent_coverage, 4),
            "factual_coherence":   round(self.factual_coherence, 4),
            "emotional_alignment": round(self.emotional_alignment, 4),
            "novelty":             round(self.novelty, 4),
            "length_adherence":    round(self.length_adherence, 4),
            "entertainment_value": round(self.entertainment_value, 4),
            "composite":           round(self.composite, 4),
        }


@dataclass
class AVSResult:
    """Full audit record for one validation pass."""
    reply:           str
    scores:          AVSScores
    passed:          bool
    threshold:       float
    regen_attempt:   int       = 0
    elapsed_ms:      float     = 0.0
    fallback_accept: bool      = False
    notes:           list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reply_preview":   self.reply[:120],
            "scores":          self.scores.to_dict(),
            "passed":          self.passed,
            "threshold":       round(self.threshold, 4),
            "regen_attempt":   self.regen_attempt,
            "elapsed_ms":      round(self.elapsed_ms, 2),
            "fallback_accept": self.fallback_accept,
            "notes":           self.notes,
        }


# ---------------------------------------------------------------------------
# AVS engine
# ---------------------------------------------------------------------------

class AnswerValidationSystem:
    """
    PRD §10 — Answer Validation System

    Usage::

        avs = AnswerValidationSystem(profile_engine)

        result = avs.validate(
            reply         = candidate_text,
            user_utterance= user_text,
            emotion_label = "neutral",
            recent_replies= last_n_replies,
            regen_attempt = 0,
        )
        if result.passed:
            deliver(result.reply)
        else:
            request_regen()
    """

    def __init__(self, profile_engine=None):
        """
        Parameters
        ----------
        profile_engine : ProfileEngine | None
            The active ProfileEngine instance.  When None, PRD defaults are
            used so the AVS can operate in unit-test / standalone mode.
        """
        self._pe = profile_engine
        self._history: list[AVSResult] = []   # audit trail

    # Public API

    def validate(
        self,
        reply: str,
        user_utterance: str,
        emotion_label: str = "neutral",
        recent_replies: list[str] | None = None,
        regen_attempt: int = 0,
    ) -> AVSResult:
        """
        Score ``reply`` on all five dimensions and compare to the profile
        threshold.  Returns an :class:`AVSResult` with full audit detail.
        """
        t0 = time.monotonic()
        recent_replies = recent_replies or []
        threshold = self._get_threshold()
        verbosity = self._get_verbosity()

        scores = AVSScores(
            intent_coverage     = self._score_intent_coverage(reply, user_utterance),
            factual_coherence   = self._score_factual_coherence(reply),
            emotional_alignment = self._score_emotional_alignment(reply, emotion_label),
            novelty             = self._score_novelty(reply, recent_replies),
            length_adherence    = self._score_length_adherence(reply, verbosity),
            entertainment_value = self._score_entertainment(reply, emotion_label),
        )

        passed = scores.composite >= threshold
        elapsed = (time.monotonic() - t0) * 1000

        result = AVSResult(
            reply         = reply,
            scores        = scores,
            passed        = passed,
            threshold     = threshold,
            regen_attempt = regen_attempt,
            elapsed_ms    = elapsed,
        )

        _log.debug(
            "[AVS] attempt=%d composite=%.3f threshold=%.3f passed=%s",
            regen_attempt, scores.composite, threshold, passed,
        )

        self._history.append(result)
        return result

    def select_best(self, candidates: list[AVSResult]) -> AVSResult:
        """
        Given a list of regen candidates, pick the highest-scoring one and
        mark it as fallback_accept=True (since none exceeded the threshold).
        """
        if not candidates:
            raise ValueError("AVS.select_best(): empty candidate list")
        best = max(candidates, key=lambda r: r.scores.composite)
        best.fallback_accept = True
        best.notes.append(
            f"Fallback accept after {len(candidates)} attempt(s); "
            f"best composite={best.scores.composite:.3f}"
        )
        _log.warning(
            "[AVS] Fallback accept — best composite=%.3f after %d attempts",
            best.scores.composite, len(candidates),
        )
        return best

    def should_regen(self, result: AVSResult) -> bool:
        """True when the reply failed and regen patience has not been exhausted."""
        if result.passed:
            return False
        patience = self._get_regen_patience()
        return result.regen_attempt < patience

    def last_result(self) -> AVSResult | None:
        return self._history[-1] if self._history else None

    def audit_trail(self) -> list[dict]:
        return [r.to_dict() for r in self._history[-50:]]   # last 50 entries

    # Scoring helpers

    def _score_intent_coverage(self, reply: str, utterance: str) -> float:
        """
        Proxy for semantic intent coverage.

        We measure how many of the non-stop keywords from the utterance
        appear in the reply.  When a proper embedding model is available
        the caller should override this with cosine similarity.
        """
        if not utterance.strip():
            return 1.0   # no utterance → trivially covered

        def _keywords(text: str) -> set[str]:
            words = _WORD_WITH_APOSTROPHE_RE.findall(text.lower())
            return {w for w in words if w not in _STOPWORDS and len(w) > 2}

        utt_kw   = _keywords(utterance)
        reply_kw = _keywords(reply)

        if not utt_kw:
            return 1.0

        overlap = len(utt_kw & reply_kw)
        # Soft coverage: at least half the keywords earns full marks
        raw = overlap / len(utt_kw)
        # Sigmoid-like curve: 50% overlap -> score ~0.80
        score = 1.0 - math.exp(-3.0 * raw)
        return round(min(score, 1.0), 4)

    def _score_factual_coherence(self, reply: str) -> float:
        """
        Lightweight internal consistency check.

        Flags replies that:
        - are empty or trivially short (< 3 words)
        - contain explicit self-contradictions ("yes … no", "always … never")
        - repeat the same sentence verbatim 3+ times
        - consist mostly of ellipsis / placeholder tokens
        """
        if not reply.strip():
            return 0.0

        words = reply.split()
        if len(words) < 3:
            return 0.30   # suspiciously short

        score = 1.0
        notes_deductions: list[float] = []

        # Contradiction signals
        lower = reply.lower()
        for pat_a, pat_b in _CONTRADICTION_PAIRS:
            if pat_a.search(lower) and pat_b.search(lower):
                notes_deductions.append(0.15)

        # Repeated sentence penalty
        sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(reply) if s.strip()]
        seen: dict[str, int] = {}
        for s in sentences:
            key = s.lower()[:120]
            seen[key] = seen.get(key, 0) + 1
        max_repeats = max(seen.values(), default=1)
        if max_repeats >= 3:
            notes_deductions.append(0.25)
        elif max_repeats == 2:
            notes_deductions.append(0.10)

        # Placeholder / filler ratio
        filler_count = len(_FILLER_RE.findall(reply))
        if filler_count > 2:
            notes_deductions.append(0.15)

        for d in notes_deductions:
            score -= d
        return round(max(score, 0.0), 4)

    def _score_emotional_alignment(self, reply: str, emotion_label: str) -> float:
        """
        Check whether the reply's surface tone matches the required emotion.

        Uses simple lexical polarity signals.  In production this should be
        replaced by an EmotionNet forward pass over the reply tokens.
        """
        POSITIVE_WORDS = {
            "glad", "happy", "great", "wonderful", "love", "excited", "joy",
            "thanks", "appreciate", "pleasure", "absolutely", "certainly",
            "fantastic", "amazing", "excellent",
        }
        NEGATIVE_WORDS = {
            "sorry", "unfortunately", "sad", "difficult", "problem", "issue",
            "hard", "tough", "worried", "concerned", "apologize",
        }
        NEUTRAL_WORDS  = {
            "okay", "sure", "understand", "noted", "right", "alright", "fine",
        }

        tokens = set(_ALPHA_WORD_RE.findall(reply.lower()))
        pos_hits = len(tokens & POSITIVE_WORDS)
        neg_hits = len(tokens & NEGATIVE_WORDS)
        neu_hits = len(tokens & NEUTRAL_WORDS)
        total    = max(pos_hits + neg_hits + neu_hits, 1)

        pos_ratio = pos_hits / total
        neg_ratio = neg_hits / total

        emotion_map: dict[str, float] = {
            "happy":     pos_ratio,
            "excited":   pos_ratio,
            "positive":  pos_ratio,
            "sad":       neg_ratio,
            "concerned": neg_ratio,
            "negative":  neg_ratio,
            "neutral":   0.6 + 0.4 * (neu_hits / max(len(tokens), 1)),
            "empathetic": 0.5 + 0.3 * neg_ratio + 0.2 * pos_ratio,
        }

        score = emotion_map.get(emotion_label.lower(), 0.70)
        # Apply emotion_intensity scaling from profile
        intensity = self._get_emotion_intensity()
        # Low intensity profile -> less strict about alignment
        adjusted = score * intensity + (1.0 - intensity) * 0.80
        return round(min(adjusted, 1.0), 4)

    def _score_novelty(self, reply: str, recent_replies: list[str]) -> float:
        """
        Penalise replies that are near-duplicates of recent ones.

        We use 4-gram character overlap as a fast similarity proxy.
        """
        if not recent_replies:
            return 1.0

        def _ngrams(text: str, n: int = 4) -> set[str]:
            t = text.lower()
            return {t[i:i+n] for i in range(len(t) - n + 1)}

        reply_ng = _ngrams(reply)
        if not reply_ng:
            return 1.0

        max_overlap = 0.0
        for prev in recent_replies[-6:]:          # look back up to 6 turns
            prev_ng  = _ngrams(prev)
            if not prev_ng:
                continue
            overlap  = len(reply_ng & prev_ng) / len(reply_ng | prev_ng)
            max_overlap = max(max_overlap, overlap)

        # Jaccard 0 -> novelty 1.0; Jaccard 1 -> novelty 0.0
        novelty = 1.0 - max_overlap
        return round(novelty, 4)

    def _score_length_adherence(self, reply: str, verbosity: float) -> float:
        """
        Reward replies whose length aligns with the profile verbosity setting.

        verbosity 0.0 = terse  (~20 words ideal)
        verbosity 1.0 = expansive (~200 words ideal)
        """
        word_count = len(reply.split())
        ideal_words = 20 + verbosity * 180      # 20 … 200

        # Gaussian-like penalty around ideal
        deviation   = abs(word_count - ideal_words) / ideal_words
        score = math.exp(-2.5 * deviation)
        return round(min(score, 1.0), 4)

    def _score_entertainment(self, response: str, emotion: str = "") -> float:
        """Score response for entertainment/personality value."""
        score = 0.0
        # Brevity bonus for witty one-liners (under 80 chars)
        if len(response) < 80 and len(response) > 10:
            score += 0.3
        # Merge hardcoded humor patterns with profile speech quirks
        profile_quirks = []
        if self._pe:
            profile_quirks = self._pe.get_speech_quirks()
        humor_patterns = ["lol", "haha", "lmao", "bruh", "ngl", "imagine", "literally"] + profile_quirks
        resp_lower = response.lower()
        if any(p and p.lower() in resp_lower for p in humor_patterns):
            score += 0.15
        # Personality expression (questions, exclamations, ellipsis)
        if response.count('?') >= 1:
            score += 0.1
        if response.count('!') >= 1:
            score += 0.1
        if '...' in response:
            score += 0.1
        # Emotional expressiveness
        if emotion and emotion.lower() not in ('neutral', 'none', ''):
            score += 0.2
        return min(1.0, score)

    # Profile parameter accessors

    def _get_threshold(self) -> float:
        if self._pe:
            return float(self._pe.minimum_answer_threshold)
        return 0.72   # PRD §4.2 default

    def _get_regen_patience(self) -> int:
        if self._pe:
            return int(self._pe.regen_patience)
        return 3

    def _get_verbosity(self) -> float:
        if self._pe:
            return float(self._pe.verbosity)
        return 0.50

    def _get_emotion_intensity(self) -> float:
        if self._pe:
            return float(self._pe.emotion_intensity)
        return 0.55
