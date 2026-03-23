"""
Interruption Handling System (IHS) — PRD §8
=============================================
Classifies incoming interruptions and selects the appropriate recovery
path.  All sensitivity parameters are consumed from the active ProfileEngine.

Interruption types (PRD §8.1)
------------------------------
  TOPICAL      — user redirects the topic mid-response
  CORRECTION   — user corrects a factual error
  SOCIAL       — casual interjection ("yeah", "right", "mm-hmm")
  EMERGENT     — new high-priority information
  URGENT       — urgent request (keyword-triggered)
  UNKNOWN      — classifier confidence below threshold

Recovery actions (PRD §8.3)
----------------------------
  RESUME        — continue from where speech was paused
  RESTART       — discard partial, regenerate from scratch
  ACKNOWLEDGE   — short social acknowledgement + resume
  ABSORB        — absorb the new info into context and regen
  YIELD         — stop speaking entirely; wait for user to finish

Profile influence
-----------------
  interrupt_sensitivity (0-1)  — high → more readily interrupted
  verbosity                    — high → prefer RESUME over RESTART
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class InterruptionType(str, Enum):
    TOPICAL    = "TOPICAL"
    CORRECTION = "CORRECTION"
    SOCIAL     = "SOCIAL"
    EMERGENT   = "EMERGENT"
    URGENT     = "URGENT"
    UNKNOWN    = "UNKNOWN"


class RecoveryAction(str, Enum):
    RESUME      = "RESUME"
    RESTART     = "RESTART"
    ACKNOWLEDGE = "ACKNOWLEDGE"
    ABSORB      = "ABSORB"
    YIELD       = "YIELD"


# ---------------------------------------------------------------------------
# Keyword banks (PRD §8.1 — classification signals)
# ---------------------------------------------------------------------------

_CORRECTION_PATTERNS = [
    r"\b(no|wrong|incorrect|actually|that'?s? not right|wait)\b",
    r"\b(you said|you mentioned|that's wrong|not quite)\b",
    r"\b(actually|in fact|to be precise|correction)\b",
]

_SOCIAL_PATTERNS = [
    r"^(yeah|yep|yes|ok|okay|mhm|uh-?huh|right|sure|got it|i see|oh|ah|hmm|mm)\.?$",
    r"^(cool|great|nice|interesting|wow|alright|understood)\.?$",
]

_URGENT_PATTERNS = [
    r"\b(stop|halt|wait|hold on|emergency|urgent|quick|asap|help|now)\b",
    r"\b(stop talking|be quiet|quiet|shush|shut up)\b",
]

_EMERGENT_PATTERNS = [
    r"\b(actually|wait|by the way|also|one more thing|i forgot|important)\b",
    r"\b(oh i should mention|i just realized|quick question|real quick)\b",
]

_TOPICAL_PATTERNS = [
    r"\b(what about|how about|speaking of|changing the subject|different question)\b",
    r"\b(actually i wanted to ask|let me ask you|can you tell me|forget that)\b",
    r"\b(never mind|let's talk about|i meant to ask)\b",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class InterruptionEvent:
    """Audit record for one interruption classification + recovery decision."""
    utterance:           str
    partial_spoken:      str                   # what Revia had said so far
    interruption_type:   InterruptionType
    recovery_action:     RecoveryAction
    confidence:          float
    sensitivity_used:    float
    elapsed_ms:          float                 = 0.0
    signals:             dict[str, float]      = field(default_factory=dict)
    notes:               list[str]             = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "utterance":         self.utterance[:120],
            "partial_spoken":    self.partial_spoken[:120],
            "interruption_type": self.interruption_type.value,
            "recovery_action":   self.recovery_action.value,
            "confidence":        round(self.confidence, 4),
            "sensitivity_used":  round(self.sensitivity_used, 4),
            "elapsed_ms":        round(self.elapsed_ms, 2),
            "signals":           {k: round(v, 4) for k, v in self.signals.items()},
            "notes":             self.notes,
        }


# ---------------------------------------------------------------------------
# IHS engine
# ---------------------------------------------------------------------------

class InterruptionHandler:
    """
    PRD §8 — Interruption Handling System

    Usage::

        ihs = InterruptionHandler(profile_engine)

        event = ihs.classify_interruption(
            utterance      = user_text_during_speech,
            partial_spoken = what_revia_said_so_far,
        )

        if event.recovery_action == RecoveryAction.RESTART:
            pipeline.restart(new_context=utterance)
        elif event.recovery_action == RecoveryAction.ACKNOWLEDGE:
            pipeline.say_ack()
            pipeline.resume(from_position=partial_spoken)
        ...
    """

    def __init__(self, profile_engine=None):
        self._pe = profile_engine
        self._history: list[InterruptionEvent] = []

    # ── Public API ────────────────────────────────────────────────────────

    def classify_interruption(
        self,
        utterance: str,
        partial_spoken: str = "",
    ) -> InterruptionEvent:
        """
        Classify the interruption and determine the recovery action.

        Parameters
        ----------
        utterance :
            The user's speech detected during Revia's response.
        partial_spoken :
            The portion of Revia's reply that has already been delivered.

        Returns
        -------
        InterruptionEvent
        """
        t0 = time.monotonic()
        sensitivity = self._get_sensitivity()

        signals: dict[str, float] = {}
        signals["correction"] = self._match_score(utterance, _CORRECTION_PATTERNS)
        signals["social"]     = self._match_score(utterance, _SOCIAL_PATTERNS, full_line=True)
        signals["urgent"]     = self._match_score(utterance, _URGENT_PATTERNS)
        signals["emergent"]   = self._match_score(utterance, _EMERGENT_PATTERNS)
        signals["topical"]    = self._match_score(utterance, _TOPICAL_PATTERNS)

        itype, confidence = self._pick_type(signals, sensitivity)
        action            = self.get_recovery_action(itype, partial_spoken, sensitivity)
        elapsed           = (time.monotonic() - t0) * 1000

        event = InterruptionEvent(
            utterance          = utterance,
            partial_spoken     = partial_spoken,
            interruption_type  = itype,
            recovery_action    = action,
            confidence         = confidence,
            sensitivity_used   = sensitivity,
            elapsed_ms         = elapsed,
            signals            = signals,
        )

        _log.debug(
            "[IHS] type=%s action=%s conf=%.3f sensitivity=%.2f utterance=%r",
            itype.value, action.value, confidence, sensitivity,
            utterance[:60],
        )

        self._history.append(event)
        return event

    def get_recovery_action(
        self,
        itype: InterruptionType,
        partial_spoken: str = "",
        sensitivity: float | None = None,
    ) -> RecoveryAction:
        """
        Map interruption type → recovery action, modulated by profile settings.

        PRD §8.3 recovery table:
          TOPICAL    → RESTART  (topic has changed; old reply irrelevant)
          CORRECTION → RESTART  (factual error; must regenerate)
          SOCIAL     → ACKNOWLEDGE  (brief ack, then resume)
          EMERGENT   → ABSORB   (incorporate new info, regen)
          URGENT     → YIELD    (stop immediately, wait for user)
          UNKNOWN    → depends on sensitivity
        """
        if sensitivity is None:
            sensitivity = self._get_sensitivity()
        verbosity   = self._get_verbosity()

        # URGENT always yields
        if itype == InterruptionType.URGENT:
            return RecoveryAction.YIELD

        # SOCIAL: social acks when sensitivity allows
        if itype == InterruptionType.SOCIAL:
            return RecoveryAction.ACKNOWLEDGE

        # CORRECTION: always restart
        if itype == InterruptionType.CORRECTION:
            return RecoveryAction.RESTART

        # EMERGENT: absorb new information
        if itype == InterruptionType.EMERGENT:
            return RecoveryAction.ABSORB

        # TOPICAL: restart (topic changed)
        if itype == InterruptionType.TOPICAL:
            return RecoveryAction.RESTART

        # UNKNOWN: use sensitivity to decide
        # High sensitivity + low verbosity → RESTART (don't fight it)
        # Low sensitivity + anything        → ACKNOWLEDGE (treat like social)
        if sensitivity >= 0.65:
            if verbosity >= 0.60:
                return RecoveryAction.ABSORB    # talkative profile: absorb and continue
            return RecoveryAction.RESTART
        return RecoveryAction.ACKNOWLEDGE

    def is_barge_in_allowed(self, current_sensitivity: float | None = None) -> bool:
        """
        Return True if the current sensitivity setting permits barge-in.
        Barge-in is always permitted for URGENT; for others it depends on
        the profile interrupt_sensitivity threshold (> 0.30 enables it).
        """
        if current_sensitivity is None:
            current_sensitivity = self._get_sensitivity()
        return current_sensitivity > 0.30

    def last_event(self) -> InterruptionEvent | None:
        return self._history[-1] if self._history else None

    def audit_trail(self) -> list[dict]:
        return [e.to_dict() for e in self._history[-50:]]

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _match_score(
        text: str,
        patterns: list[str],
        full_line: bool = False,
    ) -> float:
        """
        Return a [0, 1] match score for the given pattern list.
        ``full_line=True`` requires the ENTIRE text to match (for social acks).
        """
        lower = text.lower().strip()
        if not lower:
            return 0.0

        hits = 0
        for pat in patterns:
            if full_line:
                if re.fullmatch(pat, lower):
                    hits += 1
            else:
                if re.search(pat, lower):
                    hits += 1

        # Normalise against list length; any single hit earns at least 0.5
        if hits == 0:
            return 0.0
        return min(0.50 + 0.50 * (hits / len(patterns)), 1.0)

    def _pick_type(
        self,
        signals: dict[str, float],
        sensitivity: float,
    ) -> tuple[InterruptionType, float]:
        """
        Pick the highest-scoring type if above the sensitivity-gated minimum.
        """
        # Map signal keys to enum values
        type_map: dict[str, InterruptionType] = {
            "correction": InterruptionType.CORRECTION,
            "social":     InterruptionType.SOCIAL,
            "urgent":     InterruptionType.URGENT,
            "emergent":   InterruptionType.EMERGENT,
            "topical":    InterruptionType.TOPICAL,
        }

        # Minimum confidence required, scaled by sensitivity
        # High sensitivity → accept lower-confidence classifications
        min_confidence = max(0.20, 0.60 - sensitivity * 0.40)

        best_type  = InterruptionType.UNKNOWN
        best_score = 0.0

        for key, itype in type_map.items():
            score = signals.get(key, 0.0)
            if score > best_score:
                best_score = score
                best_type  = itype

        if best_score < min_confidence:
            return InterruptionType.UNKNOWN, best_score

        return best_type, best_score

    # ── Profile parameter accessors ───────────────────────────────────────

    def _get_sensitivity(self) -> float:
        if self._pe:
            return float(self._pe.interrupt_sensitivity)
        return 0.55   # PRD §4.2 default

    def _get_verbosity(self) -> float:
        if self._pe:
            return float(self._pe.verbosity)
        return 0.50
