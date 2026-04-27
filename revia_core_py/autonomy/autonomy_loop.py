from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any

from .candidate_generator import CandidateGenerator, create_silence_candidate
from .cooldown_manager import CooldownManager
from .deep_processing_policy import should_use_deep_processing
from .memory_retriever import MemoryRetriever
from .personality_rewrite import build_self_initiation_prompt
from .self_initiation_scorer import ScoredCandidate, SelfInitiationScorer
from .state_tracker import StateTracker
from .topic_manager import TopicManager


@dataclass
class AutonomyDecision:
    should_speak: bool
    reason: str
    state_mode: str
    selected: ScoredCandidate | None = None
    prompt: str = ""
    scored_candidates: list[dict[str, Any]] = field(default_factory=list)
    do_not_talk_reasons: list[str] = field(default_factory=list)
    rng_roll: float = 0.0
    deep_processing: bool = False
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        selected = self.selected.to_dict() if self.selected else None
        return {
            "should_speak": self.should_speak,
            "reason": self.reason,
            "mode": self.state_mode,
            "selected": selected,
            "prompt": self.prompt[:240],
            "candidates": self.scored_candidates,
            "do_not_talk_reasons": list(self.do_not_talk_reasons),
            "rng_roll": round(self.rng_roll, 4),
            "deep_processing": self.deep_processing,
            "created_at": self.created_at,
        }


class ReviaAutonomyLoop:
    def __init__(
        self,
        *,
        state_tracker: StateTracker,
        topic_manager: TopicManager,
        memory_retriever: MemoryRetriever,
        candidate_generator: CandidateGenerator,
        scorer: SelfInitiationScorer,
        cooldown_manager: CooldownManager,
        log_fn=None,
        rng: random.Random | None = None,
    ):
        self._state_tracker = state_tracker
        self._topic_manager = topic_manager
        self._memory_retriever = memory_retriever
        self._candidate_generator = candidate_generator
        self._scorer = scorer
        self._cooldowns = cooldown_manager
        self._log = log_fn or (lambda _msg: None)
        self._rng = rng or random.Random()
        self._last_decision: AutonomyDecision | None = None

    @property
    def last_decision(self) -> AutonomyDecision | None:
        return self._last_decision

    def evaluate_once(
        self,
        *,
        source: str = "IdleTimer",
        reason: str = "autonomy check",
        force: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> AutonomyDecision:
        metadata = dict(metadata or {})
        state = self._state_tracker.get_state(
            metadata=metadata,
            active_topic=self._topic_manager.active_topic,
        )
        self._log(
            "[Autonomy] Check started | "
            f"source={source} | mode={state.current_mode} | "
            f"idle_user={state.seconds_since_last_user_message:.1f}s | "
            f"idle_activity={state.seconds_since_last_user_activity:.1f}s | "
            f"runtime={state.runtime_state}"
        )

        if state.quiet_requested:
            self._cooldowns.note_quiet_request(state.profile.quiet_request_cooldown_seconds)

        blocked = [] if force else state.do_not_talk_reasons()
        if blocked:
            decision = AutonomyDecision(
                should_speak=False,
                reason="blocked before candidate generation",
                state_mode=state.current_mode,
                do_not_talk_reasons=blocked,
            )
            self._last_decision = decision
            self._log("[Autonomy] Staying silent | " + "; ".join(blocked))
            return decision

        topics = self._topic_manager.get_relevant_topics(state)
        memories = self._memory_retriever.retrieve(
            active_topic=self._topic_manager.active_topic,
            recent_topics=topics,
            mood=state.user_mood,
        )
        candidates = self._candidate_generator.generate(
            state=state,
            topics=topics,
            memories=memories,
        )
        candidates.append(create_silence_candidate())

        scored = self._scorer.score_all(candidates, state, topics, self._cooldowns)
        scored_sorted = sorted(scored, key=lambda item: item.final_score, reverse=True)
        top = scored_sorted[:4]
        self._log(
            "[Autonomy] Candidates scored | "
            + " | ".join(
                f"{item.candidate.type}:{item.final_score:.2f}"
                for item in top
            )
        )

        best = scored_sorted[0]
        if best.candidate.type == "silence":
            decision = AutonomyDecision(
                should_speak=False,
                reason="silence won",
                state_mode=state.current_mode,
                selected=best,
                scored_candidates=[item.to_dict() for item in scored_sorted],
            )
            self._last_decision = decision
            self._log(
                "[Autonomy] Staying silent | silence score "
                f"{best.final_score:.2f} beat spoken candidates"
            )
            return decision

        if best.final_score < state.profile.speak_threshold:
            decision = AutonomyDecision(
                should_speak=False,
                reason="best candidate below speak threshold",
                state_mode=state.current_mode,
                selected=best,
                scored_candidates=[item.to_dict() for item in scored_sorted],
            )
            self._last_decision = decision
            self._log(
                "[Autonomy] Staying silent | "
                f"best={best.candidate.type}:{best.final_score:.2f} "
                f"threshold={state.profile.speak_threshold:.2f}"
            )
            return decision

        roll = self._rng.random()
        if roll > best.chance_to_speak:
            decision = AutonomyDecision(
                should_speak=False,
                reason="weighted randomness chose silence",
                state_mode=state.current_mode,
                selected=best,
                scored_candidates=[item.to_dict() for item in scored_sorted],
                rng_roll=roll,
            )
            self._last_decision = decision
            self._log(
                "[Autonomy] Staying silent | weighted roll "
                f"{roll:.2f} > chance {best.chance_to_speak:.2f}"
            )
            return decision

        deep = should_use_deep_processing(best.candidate, best)
        prompt = build_self_initiation_prompt(best.candidate, state, memories, best)
        decision = AutonomyDecision(
            should_speak=True,
            reason="selected spoken candidate",
            state_mode=state.current_mode,
            selected=best,
            prompt=prompt,
            scored_candidates=[item.to_dict() for item in scored_sorted],
            rng_roll=roll,
            deep_processing=deep,
        )
        self._last_decision = decision
        self._log(
            "[Autonomy] Speaking allowed | "
            f"type={best.candidate.type} | topic={best.candidate.topic or 'none'} | "
            f"score={best.final_score:.2f} | chance={best.chance_to_speak:.2f} | "
            f"roll={roll:.2f} | deep={deep}"
        )
        return decision

    def register_spoken_decision(self, decision: AutonomyDecision) -> None:
        if not decision or not decision.selected:
            return
        candidate = decision.selected.candidate
        previous_topic = self._topic_manager.active_topic
        self._cooldowns.register_spoken_candidate(
            candidate,
            type("_StateProxy", (), {
                "active_topic": previous_topic,
                "profile": None,
            })(),
        )
        if candidate.topic:
            self._topic_manager.register_topic_used(
                candidate.topic,
                user_interest=decision.selected.user_interest,
            )
        self._log(
            "[Autonomy] Registered spoken candidate | "
            f"type={candidate.type} | topic={candidate.topic or 'none'}"
        )

    def note_quiet_request(self, duration_s: float) -> None:
        self._cooldowns.note_quiet_request(duration_s)
        self._log(f"[Autonomy] Quiet request cooldown started | {duration_s:.1f}s")

    def note_interruption(self, duration_s: float) -> None:
        self._cooldowns.note_interruption(duration_s)
        self._log(f"[Autonomy] Interruption cooldown started | {duration_s:.1f}s")

    def status(self) -> dict[str, Any]:
        decision = self._last_decision.to_dict() if self._last_decision else None
        return {
            "last_decision": decision,
            "cooldowns": self._cooldowns.active_cooldowns(),
        }
