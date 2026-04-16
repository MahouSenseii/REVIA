"""
Anti-Loop Engine (ALE) — PRD §12
==================================
Detects repetitive / stuck generation and computes a loop_risk_score in
[0, 1].  When the score exceeds the profile threshold the engine signals
the pipeline to regen with a chosen recovery strategy.

Detection techniques (per PRD §12)
-----------------------------------
1. N-gram overlap scan       — 4-gram that appears >1× in the reply
2. Starter phrase fingerprint — first 8 words repeated across last N turns
3. Semantic echo check        — word-overlap cosine proxy (no heavy deps)
4. Regen divergence check     — new candidate vs previous; < 0.10 overlap = stuck

Recovery modes (sourced from profile.behavior.loop_recovery_mode)
-------------------------------------------------------------------
  rephrase     — inject a rephrasing cue at the start of the prompt
  topic_shift  — explicit "Let me approach this differently" pivot phrase
  silence      — return empty string so the system falls silent

All thresholds consumed through ProfileEngine — zero hardcoded values.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Repair phrases injected into the LLM context when a loop is detected
# ---------------------------------------------------------------------------
_REPAIR_PHRASES: dict[str, str] = {
    "rephrase": (
        "Your previous reply appears to be repeating itself. "
        "Please rephrase your answer using completely different "
        "sentence structures and word choices. Do NOT start with the same "
        "sentence as before."
    ),
    "topic_shift": (
        "The conversation seems stuck. Shift the topic slightly or offer "
        "a new angle. Begin with something like 'Let me try a different "
        "approach...' or a similar natural pivot."
    ),
    "silence": "",
}

# ---------------------------------------------------------------------------
# Personality trait whitelist (phrases allowed to repeat)
# ---------------------------------------------------------------------------
_PERSONALITY_WHITELIST: frozenset = frozenset()  # Atomically replaced; thread-safe
_PERSONALITY_WHITELIST_LOCK = threading.Lock()

# Default score thresholds (PRD §12 — overridden by profile in practice)
_DEFAULT_LOOP_RISK_TRIGGER = 0.65   # score above this → repair action
_DEFAULT_LOOP_DETECTION_WINDOW = 80  # token lookback


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ALEReport:
    """Diagnostic record for one ALE evaluation pass."""
    reply:              str
    loop_risk_score:    float
    triggered:          bool
    recovery_mode:      str
    repair_phrase:      str
    elapsed_ms:         float         = 0.0
    signals:            dict[str, float] = field(default_factory=dict)
    notes:              list[str]     = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reply_preview":   self.reply[:80],
            "loop_risk_score": round(self.loop_risk_score, 4),
            "triggered":       self.triggered,
            "recovery_mode":   self.recovery_mode,
            "repair_phrase":   self.repair_phrase[:80] if self.repair_phrase else "",
            "elapsed_ms":      round(self.elapsed_ms, 2),
            "signals":         {k: round(v, 4) for k, v in self.signals.items()},
            "notes":           self.notes,
        }


# ---------------------------------------------------------------------------
# ALE engine
# ---------------------------------------------------------------------------

class AntiLoopEngine:
    """
    PRD §12 — Anti-Loop Engine

    Usage::

        ale = AntiLoopEngine(profile_engine)

        report = ale.check(
            reply          = candidate_text,
            recent_replies = last_n_replies,
            previous_reply = prior_candidate_on_this_turn,  # for regen check
        )
        if report.triggered:
            prompt_ctx += report.repair_phrase
            request_regen()
    """

    def __init__(self, profile_engine=None):
        self._pe = profile_engine
        self._history: list[ALEReport] = []

    @classmethod
    def set_personality_whitelist(cls, phrases: list):
        """Set phrases that are allowed to repeat (catchphrases, quirks)."""
        global _PERSONALITY_WHITELIST
        # Atomically replace with a frozenset — safe for concurrent readers
        with _PERSONALITY_WHITELIST_LOCK:
            _PERSONALITY_WHITELIST = frozenset(p.lower().strip() for p in phrases)

    # ── Public API ────────────────────────────────────────────────────────

    def check(
        self,
        reply: str,
        recent_replies: list[str] | None = None,
        previous_reply: str | None = None,
    ) -> ALEReport:
        """
        Evaluate ``reply`` for looping patterns.

        Parameters
        ----------
        reply :
            The candidate reply to evaluate.
        recent_replies :
            List of recent delivered replies (oldest first), used for
            starter-phrase and semantic-echo detection.
        previous_reply :
            The immediately preceding regen candidate on this same turn,
            used for regen-divergence check.

        Returns
        -------
        ALEReport
            Full diagnostic record including loop_risk_score and repair phrase.
        """
        t0 = time.monotonic()
        recent_replies = recent_replies or []

        signals: dict[str, float] = {}

        # 1. Internal n-gram repetition
        signals["ngram_repetition"] = self._check_ngram_repetition(reply)

        # 2. Starter phrase fingerprint
        signals["starter_fingerprint"] = self._check_starter_fingerprint(
            reply, recent_replies
        )

        # 3. Semantic echo (word-overlap cosine proxy)
        signals["semantic_echo"] = self._check_semantic_echo(
            reply, recent_replies
        )

        # 4. Regen divergence (stuck generation check)
        signals["regen_divergence"] = self._check_regen_divergence(
            reply, previous_reply
        )

        # Weighted composite loop_risk_score
        # Weights: ngram 30%, starter 25%, semantic 25%, regen_div 20%
        loop_risk = (
            0.30 * signals["ngram_repetition"]
            + 0.25 * signals["starter_fingerprint"]
            + 0.25 * signals["semantic_echo"]
            + 0.20 * signals["regen_divergence"]
        )
        loop_risk = round(max(0.0, min(loop_risk, 1.0)), 4)

        trigger_threshold = self._get_trigger_threshold()
        triggered         = loop_risk >= trigger_threshold
        recovery_mode     = self._get_recovery_mode()
        repair_phrase     = _REPAIR_PHRASES.get(recovery_mode, "")

        elapsed = (time.monotonic() - t0) * 1000

        report = ALEReport(
            reply           = reply,
            loop_risk_score = loop_risk,
            triggered       = triggered,
            recovery_mode   = recovery_mode,
            repair_phrase   = repair_phrase,
            elapsed_ms      = elapsed,
            signals         = signals,
        )

        if triggered:
            report.notes.append(
                f"Loop triggered: risk={loop_risk:.3f} >= threshold={trigger_threshold:.3f}; "
                f"mode={recovery_mode}"
            )
            _log.warning(
                "[ALE] Loop detected — risk=%.3f threshold=%.3f mode=%s",
                loop_risk, trigger_threshold, recovery_mode,
            )
        else:
            _log.debug("[ALE] No loop — risk=%.3f", loop_risk)

        self._history.append(report)
        return report

    def audit_trail(self) -> list[dict]:
        return [r.to_dict() for r in self._history[-50:]]

    # ── Detection helpers ─────────────────────────────────────────────────

    def _check_ngram_repetition(self, reply: str) -> float:
        """
        Count 4-grams that appear more than once within ``reply``.
        Returns proportion of reply tokens involved in repeated 4-grams.
        """
        # Before scoring repetition, check whitelist
        for phrase in _PERSONALITY_WHITELIST:
            if phrase in reply.lower():
                return 0.0  # Not a loop, it's a personality trait

        if not reply or not reply.strip():
            return 0.0  # Empty reply handled elsewhere; don't flag as loop
        tokens = reply.lower().split()
        n = 4
        if len(tokens) < n:
            return 0.0

        seen: dict[tuple, int] = {}
        for i in range(len(tokens) - n + 1):
            gram = tuple(tokens[i:i+n])
            seen[gram] = seen.get(gram, 0) + 1

        repeated_start_positions = sum(
            1 for cnt in seen.values() if cnt > 1
        )
        # Normalise by number of possible n-gram positions
        total_positions = len(tokens) - n + 1
        score = repeated_start_positions / max(total_positions, 1)
        # Only amplify for longer replies — short replies (< 20 tokens) are
        # legitimately brief and shouldn't be over-penalised as loops.
        scale = 1.5 if len(tokens) >= 20 else 1.0
        return round(min(score * scale, 1.0), 4)

    def _check_starter_fingerprint(
        self, reply: str, recent_replies: list[str]
    ) -> float:
        """
        Check if the first 8 words of ``reply`` closely match a recent reply's
        opening.  High similarity → high score.
        """
        window = self._get_detection_window()
        if not recent_replies:
            return 0.0

        def _starter(text: str) -> list[str]:
            return text.lower().split()[:8]

        my_start = _starter(reply)
        if not my_start:
            return 0.0

        # Truncate to the configured lookback window
        lookback = recent_replies[-window:]
        max_sim = 0.0
        for prev in lookback:
            prev_start = _starter(prev)
            if not prev_start:
                continue
            common = sum(1 for a, b in zip(my_start, prev_start) if a == b)
            sim    = common / max(len(my_start), len(prev_start))
            max_sim = max(max_sim, sim)

        return round(min(max_sim, 1.0), 4)

    def _check_semantic_echo(
        self, reply: str, recent_replies: list[str]
    ) -> float:
        """
        Word-overlap Jaccard similarity between ``reply`` and recent replies.
        High overlap across the last N turns signals semantic repetition.
        """
        if not recent_replies:
            return 0.0

        def _token_set(text: str) -> set[str]:
            return set(re.findall(r"[a-z]+", text.lower()))

        my_tokens = _token_set(reply)
        if not my_tokens:
            return 0.0

        window = max(4, self._get_detection_window() // 2)
        lookback = recent_replies[-window:]
        all_prev: set[str] = set()
        for r in lookback:
            all_prev |= _token_set(r)

        if not all_prev:
            return 0.0

        jaccard = len(my_tokens & all_prev) / len(my_tokens | all_prev)
        # 70%+ overlap is a strong echo signal
        score   = max(0.0, (jaccard - 0.40) / 0.30)   # [0,1] above 40%
        return round(min(score, 1.0), 4)

    def _check_regen_divergence(
        self, reply: str, previous_reply: str | None
    ) -> float:
        """
        If a previous regen candidate exists, check how different this one is.
        Very low divergence (< 0.10) means the model is stuck and keeps
        producing the same output regardless of the repair prompt.
        """
        if not previous_reply:
            return 0.0   # no prior regen → not a regen-stuck situation

        def _token_set(text: str) -> set[str]:
            return set(re.findall(r"[a-z]+", text.lower()))

        a = _token_set(reply)
        b = _token_set(previous_reply)
        if not a or not b:
            return 0.0

        jaccard   = len(a & b) / len(a | b)
        # High Jaccard = low divergence = stuck → high risk score
        # Below 0.20 divergence → risk = 1.0; above 0.80 → risk = 0.0
        divergence = 1.0 - jaccard
        stuck_score = max(0.0, 1.0 - (divergence / 0.40))
        return round(min(stuck_score, 1.0), 4)

    # ── Profile parameter accessors ───────────────────────────────────────

    def _get_trigger_threshold(self) -> float:
        """
        There is no explicit ALE trigger threshold in PRD §4.  We derive it
        from the profile's minimum_answer_threshold (lower quality tolerance
        → also lower loop tolerance) or fall back to the module default.
        """
        if self._pe:
            # Slightly below MAT so loops are caught before AVS fails
            mat = float(self._pe.minimum_answer_threshold)
            return max(0.50, mat - 0.10)
        return _DEFAULT_LOOP_RISK_TRIGGER

    def _get_recovery_mode(self) -> str:
        if self._pe:
            return str(self._pe.loop_recovery_mode)
        return "rephrase"

    def _get_detection_window(self) -> int:
        if self._pe:
            return int(self._pe.loop_detection_window)
        return _DEFAULT_LOOP_DETECTION_WINDOW
