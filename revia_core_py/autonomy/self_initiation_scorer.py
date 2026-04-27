from __future__ import annotations

from dataclasses import dataclass, field

from .candidate_generator import SelfInitiationCandidate


@dataclass
class ScoredCandidate:
    candidate: SelfInitiationCandidate
    timing: float
    relevance: float
    usefulness: float
    user_interest: float
    personality_fit: float
    emotional_fit: float
    novelty: float
    confidence: float
    interruption_risk: float
    repetition_risk: float
    safety: float
    final_score: float
    chance_to_speak: float
    active_cooldowns: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.candidate.candidate_id,
            "type": self.candidate.type,
            "topic": self.candidate.topic,
            "score": round(self.final_score, 4),
            "chance_to_speak": round(self.chance_to_speak, 3),
            "timing": round(self.timing, 3),
            "relevance": round(self.relevance, 3),
            "usefulness": round(self.usefulness, 3),
            "interest": round(self.user_interest, 3),
            "interruption_risk": round(self.interruption_risk, 3),
            "repetition_risk": round(self.repetition_risk, 3),
            "cooldowns": dict(self.active_cooldowns),
            "reason": self.candidate.reason,
        }


class SelfInitiationScorer:
    _USEFULNESS_BY_TYPE = {
        "continue_old_topic": 0.68,
        "ask_followup": 0.72,
        "start_new_topic": 0.42,
        "make_observation": 0.38,
        "suggest_action": 0.82,
        "emotional_checkin": 0.76,
        "memory_reflection": 0.74,
        "silence": 0.50,
    }

    _PERSONALITY_BY_TYPE = {
        "continue_old_topic": 0.78,
        "ask_followup": 0.74,
        "start_new_topic": 0.52,
        "make_observation": 0.70,
        "suggest_action": 0.86,
        "emotional_checkin": 0.76,
        "memory_reflection": 0.82,
        "silence": 0.90,
    }

    def score_all(self, candidates, state, topics, cooldown_manager) -> list[ScoredCandidate]:
        topic_interest = {t.topic: float(t.user_interest or t.score or 0.0) for t in topics or []}
        topic_relevance = {t.topic: float(t.score or 0.0) for t in topics or []}
        return [
            self.score(candidate, state, topic_interest, topic_relevance, cooldown_manager)
            for candidate in candidates
        ]

    def score(self, candidate, state, topic_interest, topic_relevance, cooldown_manager) -> ScoredCandidate:
        if candidate.type == "silence":
            return self._score_silence(candidate, state)

        cooldown = cooldown_manager.snapshot_for(candidate, state)
        timing = self._timing_score(state)
        relevance = max(
            float(topic_relevance.get(candidate.topic, 0.0)),
            0.35 if candidate.topic else 0.20,
        )
        usefulness = self._USEFULNESS_BY_TYPE.get(candidate.type, 0.45)
        user_interest = max(
            float(topic_interest.get(candidate.topic, 0.0)),
            0.60 if candidate.type in {"ask_followup", "emotional_checkin"} else 0.30,
        )
        personality_fit = self._PERSONALITY_BY_TYPE.get(candidate.type, 0.60)
        mood = str(state.user_mood or "").lower()
        emotional_fit = 0.80 if candidate.type == "emotional_checkin" and mood not in {"neutral", "disabled"} else 0.55
        novelty = max(0.0, 1.0 - cooldown.repetition_risk)
        confidence = 0.72
        if candidate.requires_deep_processing:
            confidence -= 0.08
        if candidate.type == "start_new_topic":
            confidence -= 0.12
        safety = 1.0
        if cooldown.active.get("quiet_request"):
            safety = 0.0
        elif state.user_appears_focused and state.current_mode in {"quiet", "work"}:
            safety = 0.35

        final = (
            timing * 0.20
            + relevance * 0.20
            + usefulness * 0.20
            + user_interest * 0.15
            + personality_fit * 0.10
            + emotional_fit * 0.05
            + novelty * 0.05
            + confidence * 0.05
        )
        final -= state.interruption_risk * 0.25
        final -= cooldown.repetition_risk * 0.20
        final *= safety
        final = max(0.0, min(1.0, final))

        return ScoredCandidate(
            candidate=candidate,
            timing=timing,
            relevance=relevance,
            usefulness=usefulness,
            user_interest=user_interest,
            personality_fit=personality_fit,
            emotional_fit=emotional_fit,
            novelty=novelty,
            confidence=confidence,
            interruption_risk=state.interruption_risk,
            repetition_risk=cooldown.repetition_risk,
            safety=safety,
            final_score=final,
            chance_to_speak=self._chance_to_speak(final),
            active_cooldowns=cooldown.active,
        )

    def _score_silence(self, candidate: SelfInitiationCandidate, state) -> ScoredCandidate:
        score = state.profile.base_silence_score
        if state.seconds_since_last_user_activity < state.profile.min_idle_seconds:
            score += 0.18
        if state.user_is_speaking or state.user_is_typing or state.revia_is_speaking:
            score += 0.25
        if state.response_in_progress:
            score += 0.25
        if state.quiet_requested:
            score += 0.30
        if state.current_mode == "stream" and state.seconds_since_last_user_activity > 120:
            score -= 0.12
        score = max(0.0, min(1.0, score))
        return ScoredCandidate(
            candidate=candidate,
            timing=1.0,
            relevance=1.0,
            usefulness=0.70,
            user_interest=0.50,
            personality_fit=0.95,
            emotional_fit=0.70,
            novelty=1.0,
            confidence=0.95,
            interruption_risk=state.interruption_risk,
            repetition_risk=0.0,
            safety=1.0,
            final_score=score,
            chance_to_speak=0.0,
            active_cooldowns={},
        )

    @staticmethod
    def _timing_score(state) -> float:
        idle = max(state.seconds_since_last_user_message, state.seconds_since_last_user_activity)
        if idle < state.profile.min_idle_seconds:
            return 0.15
        if idle < state.profile.min_idle_seconds * 1.5:
            return 0.55
        if idle < state.profile.min_idle_seconds * 3:
            return 0.78
        return 0.92

    @staticmethod
    def _chance_to_speak(final_score: float) -> float:
        if final_score >= 0.90:
            return 0.95
        if final_score >= 0.80:
            return 0.70
        if final_score >= 0.70:
            return 0.35
        return 0.0
