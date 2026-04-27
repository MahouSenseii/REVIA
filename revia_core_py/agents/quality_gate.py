"""QualityGate — wraps the existing AVS to decide if Revia may speak.

Per spec rule 7: Revia only speaks if the answer's quality reaches a
threshold (default 0.70).  We re-use the
:class:`AnswerValidationSystem` (AVS) so the score and notes match what
the ReplyPlanner regen loop already used.

If the ReasoningAgent already produced an AVS-passed candidate, we can
short-circuit: trust the upstream score.  Otherwise we re-validate here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:  # pragma: no cover - import guard for both package and direct run
    from ..answer_validation import AnswerValidationSystem
except ImportError:  # pragma: no cover
    from answer_validation import AnswerValidationSystem  # type: ignore[no-redef]


@dataclass
class QualityVerdict:
    """Outcome of a single QualityGate check."""

    score: float
    threshold: float
    approved: bool
    elapsed_ms: float = 0.0
    reasons: list[str] = None  # type: ignore[assignment]
    source: str = "avs"

    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(float(self.score), 4),
            "threshold": round(float(self.threshold), 4),
            "approved": bool(self.approved),
            "elapsed_ms": round(float(self.elapsed_ms), 2),
            "reasons": list(self.reasons),
            "source": self.source,
        }


class QualityGate:
    def __init__(self, profile_engine=None, avs: AnswerValidationSystem | None = None):
        self._pe = profile_engine
        self._avs = avs or AnswerValidationSystem(profile_engine)

    def check(
        self,
        reply: str,
        user_utterance: str,
        emotion_label: str = "neutral",
        recent_replies: list[str] | None = None,
        upstream_score: float | None = None,
        threshold: float | None = None,
    ) -> QualityVerdict:
        eff_threshold = self._resolve_threshold(threshold)
        reply = str(reply or "").strip()
        if not reply:
            return QualityVerdict(
                score=0.0,
                threshold=eff_threshold,
                approved=False,
                reasons=["empty_reply"],
                source="empty",
            )

        # Trust upstream AVS score when present (avoids double work).
        if upstream_score is not None and upstream_score > 0.0:
            approved = upstream_score >= eff_threshold
            return QualityVerdict(
                score=float(upstream_score),
                threshold=eff_threshold,
                approved=approved,
                reasons=[] if approved else [
                    f"upstream_score_below_threshold ({upstream_score:.2f} < {eff_threshold:.2f})"
                ],
                source="upstream_avs",
            )

        result = self._avs.validate(
            reply=reply,
            user_utterance=user_utterance,
            emotion_label=emotion_label,
            recent_replies=recent_replies or [],
            regen_attempt=0,
        )
        approved = result.passed and result.scores.composite >= eff_threshold
        reasons: list[str] = []
        if not approved:
            reasons.append(
                f"avs_composite_below_threshold ({result.scores.composite:.2f} < {eff_threshold:.2f})"
            )
        return QualityVerdict(
            score=float(result.scores.composite),
            threshold=eff_threshold,
            approved=bool(approved),
            elapsed_ms=float(result.elapsed_ms),
            reasons=reasons,
            source="avs",
        )

    def _resolve_threshold(self, threshold: float | None) -> float:
        if threshold is not None:
            return float(threshold)
        if self._pe is not None:
            try:
                return float(self._pe.minimum_answer_threshold)
            except AttributeError:
                pass
        return 0.70
