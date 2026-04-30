"""V4.1 SkillRegistry — discovery, ranking, dispatch.

The registry is intentionally simple: it holds an ordered list of
:class:`Skill` instances, ranks them per-request via :meth:`Skill.match`,
and dispatches the user message to the highest-scoring one whose score
exceeds a confidence threshold.

The :class:`ToolUseAgent` (V4.2) is the *agent* in front of this layer:
it consults a single :class:`SkillRegistry` and produces a structured
``AgentResult`` containing the skill output for the ReasoningAgent to
fold into the prompt.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Iterable

from .base import Skill, SkillMatch, SkillRequest, SkillResponse

_log = logging.getLogger(__name__)


@dataclass
class SkillDispatchResult:
    """Result of one registry dispatch (best skill + its response)."""

    chosen: str = ""
    score: float = 0.0
    response: SkillResponse | None = None
    candidates: list[SkillMatch] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chosen": self.chosen,
            "score": round(float(self.score), 4),
            "response": self.response.to_dict() if self.response else None,
            "candidates": [
                {"name": c.skill_name, "score": round(c.score, 4),
                 "reasons": list(c.reasons)} for c in self.candidates
            ],
        }


class SkillRegistry:
    DEFAULT_MIN_SCORE = 0.3

    def __init__(
        self,
        skills: Iterable[Skill] = (),
        min_score: float = DEFAULT_MIN_SCORE,
        log_fn=None,
    ):
        self._lock = threading.Lock()
        self._skills: dict[str, Skill] = {}
        for s in skills:
            self._skills[s.name] = s
        self._min_score = float(min_score)
        self._log = log_fn or _log.info

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def add(self, skill: Skill) -> None:
        with self._lock:
            self._skills[skill.name] = skill

    def remove(self, name: str) -> Skill | None:
        with self._lock:
            return self._skills.pop(name, None)

    def get(self, name: str) -> Skill | None:
        with self._lock:
            return self._skills.get(name)

    def names(self) -> list[str]:
        with self._lock:
            return list(self._skills.keys())

    def set_enabled(self, name: str, on: bool) -> bool:
        with self._lock:
            s = self._skills.get(name)
            if s is None:
                return False
            s.set_enabled(on)
            return True

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "skills": [s.info() for s in self._skills.values()],
                "min_score": self._min_score,
                "count": len(self._skills),
            }

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def rank(self, req: SkillRequest) -> list[SkillMatch]:
        with self._lock:
            skills = list(self._skills.values())
        ranked = [s.match(req) for s in skills]
        ranked.sort(key=lambda m: m.score, reverse=True)
        return ranked

    def best_match(self, req: SkillRequest) -> SkillMatch | None:
        ranked = self.rank(req)
        if not ranked:
            return None
        if ranked[0].score < self._min_score:
            return None
        return ranked[0]

    def dispatch(
        self,
        req: SkillRequest,
        explicit_skill: str | None = None,
    ) -> SkillDispatchResult:
        """Pick the best skill (or use ``explicit_skill``) and run it."""
        candidates = self.rank(req)

        chosen_name = ""
        chosen_score = 0.0
        if explicit_skill:
            chosen_name = explicit_skill
            for c in candidates:
                if c.skill_name == explicit_skill:
                    chosen_score = c.score
                    break
        else:
            if not candidates or candidates[0].score < self._min_score:
                return SkillDispatchResult(candidates=candidates)
            chosen_name = candidates[0].skill_name
            chosen_score = candidates[0].score

        with self._lock:
            skill = self._skills.get(chosen_name)
        if skill is None:
            return SkillDispatchResult(
                chosen=chosen_name,
                score=chosen_score,
                response=SkillResponse(success=False, error="skill_not_found"),
                candidates=candidates,
            )
        if not skill.enabled:
            return SkillDispatchResult(
                chosen=chosen_name,
                score=chosen_score,
                response=SkillResponse(success=False, error="skill_disabled"),
                candidates=candidates,
            )
        try:
            response = skill.handle(req)
        except Exception as exc:  # pragma: no cover - defensive
            response = SkillResponse(
                success=False, error=f"{type(exc).__name__}: {exc}",
            )
            self._log(f"[SkillRegistry] {chosen_name} raised: {exc}")
        return SkillDispatchResult(
            chosen=chosen_name,
            score=chosen_score,
            response=response,
            candidates=candidates,
        )
