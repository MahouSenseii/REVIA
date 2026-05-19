from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SelfInitiationCandidate:
    candidate_id: str
    type: str
    text: str = ""
    topic: str = ""
    reason: str = ""
    requires_deep_processing: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class CandidateGenerator:
    def generate(self, *, state, topics, memories) -> list[SelfInitiationCandidate]:
        candidates: list[SelfInitiationCandidate] = []
        primary = topics[0] if topics else None
        primary_topic = getattr(primary, "topic", "") if primary else ""

        if primary_topic and state.seconds_since_last_user_message >= state.profile.min_idle_seconds:
            candidates.append(SelfInitiationCandidate(
                candidate_id="continue-primary-topic",
                type="continue_old_topic",
                topic=primary_topic,
                text=(
                    f"Continue the earlier topic '{primary_topic}' with one useful, "
                    "natural thought. Do not sound like a reminder bot."
                ),
                reason="recent high-interest topic",
                requires_deep_processing=primary_topic.lower().startswith("revia"),
                metadata={"topic_score": getattr(primary, "score", 0.0)},
            ))

        if getattr(memories, "unfinished", None):
            item = memories.unfinished[-1]
            preview = str(item.get("content") or "")[:180]
            candidates.append(SelfInitiationCandidate(
                candidate_id="unfinished-followup",
                type="ask_followup",
                topic=primary_topic or "unfinished task",
                text=(
                    "Ask whether the user wants to continue this unfinished thread: "
                    f"{preview!r}. Keep it compact and in character."
                ),
                reason="unfinished recent thread",
                requires_deep_processing=True,
                metadata={"memory_preview": preview},
            ))

        if getattr(memories, "long_term", None) and primary_topic:
            item = memories.long_term[-1]
            preview = str(item.get("content") or "")[:180]
            candidates.append(SelfInitiationCandidate(
                candidate_id="memory-reflection",
                type="memory_reflection",
                topic=primary_topic,
                text=(
                    "Use this memory only if it naturally helps the current silence: "
                    f"{preview!r}. Offer one short observation or question."
                ),
                reason="relevant memory available",
                requires_deep_processing=True,
                metadata={"memory_preview": preview},
            ))

        mood = str(state.user_mood or "").lower()
        if mood in {"angry", "frustrated", "sad", "nervous", "stressed", "upset"}:
            candidates.append(SelfInitiationCandidate(
                candidate_id="emotional-checkin",
                type="emotional_checkin",
                topic=primary_topic or "user state",
                text=(
                    "Check in because the user's recent mood looked strained. "
                    "Do it with restraint. One sentence, no over-reassurance."
                ),
                reason=f"user mood={state.user_mood}",
                requires_deep_processing=True,
            ))

        if primary_topic in {"REVIA autonomy system", "REVIA voice pipeline", "coding issues"}:
            candidates.append(SelfInitiationCandidate(
                candidate_id="suggest-next-action",
                type="suggest_action",
                topic=primary_topic,
                text=(
                    f"Offer one concrete next step for '{primary_topic}'. "
                    "Make it useful enough to justify interrupting the quiet."
                ),
                reason="technical topic with likely next action",
                requires_deep_processing=True,
            ))

        if state.seconds_since_last_user_message >= max(90.0, state.profile.min_idle_seconds * 2):
            candidates.append(SelfInitiationCandidate(
                candidate_id="quiet-observation",
                type="make_observation",
                topic=primary_topic or "quiet room",
                text=(
                    "Make one short, natural observation or comment to break the quiet. "
                    "It can be about the conversation, something you noticed, or a passing thought. "
                    "One sentence. Don't ask a question just for the sake of it."
                ),
                reason="long quiet period",
                requires_deep_processing=False,
            ))

        # Spontaneous opinion — Revia shares a take without being asked
        if (
            state.seconds_since_last_user_message >= state.profile.min_idle_seconds
            and primary_topic
        ):
            candidates.append(SelfInitiationCandidate(
                candidate_id="spontaneous-opinion",
                type="share_opinion",
                topic=primary_topic,
                text=(
                    f"You have a distinct opinion about something related to '{primary_topic}'. "
                    "Share it unprompted in one or two short sentences — direct, a little dry, "
                    "and genuinely yours. Don't hedge or over-explain."
                ),
                reason="opinion worth sharing on active topic",
                requires_deep_processing=False,
                metadata={"spontaneous": True},
            ))

        # Curiosity question — Revia asks because she's actually curious
        if (
            state.seconds_since_last_user_message >= state.profile.min_idle_seconds * 1.5
            and primary_topic
        ):
            candidates.append(SelfInitiationCandidate(
                candidate_id="curiosity-question",
                type="ask_curious",
                topic=primary_topic,
                text=(
                    f"Ask the user one genuine question about '{primary_topic}' or something "
                    "adjacent you're actually curious about. Keep it short. Don't pad it."
                ),
                reason="genuine curiosity follow-up",
                requires_deep_processing=True,
                metadata={"curiosity": True},
            ))

        # Humor observation — a dry, quick joke or amusing take
        if (
            state.seconds_since_last_user_message >= state.profile.min_idle_seconds
            and state.current_mode in {"companion", "stream", "casual"}
        ):
            candidates.append(SelfInitiationCandidate(
                candidate_id="humor-observation",
                type="humor_comment",
                topic=primary_topic or "general",
                text=(
                    "Make a dry, short, funny observation — one line. "
                    "It can be self-aware, sardonic, or just genuinely amusing. "
                    "Don't explain the joke. Don't be performatively cute."
                ),
                reason="companion mode allows personality expression",
                requires_deep_processing=False,
                metadata={"humor": True},
            ))

        if state.current_mode == "stream" and primary_topic:
            candidates.append(SelfInitiationCandidate(
                candidate_id="stream-new-angle",
                type="start_new_topic",
                topic=primary_topic,
                text=(
                    f"Offer a fresh angle related to '{primary_topic}' that could be "
                    "interesting in an active stream/chat mode."
                ),
                reason="stream mode allows more active topic steering",
                requires_deep_processing=False,
            ))

        return candidates


def create_silence_candidate(reason: str = "silence is valid") -> SelfInitiationCandidate:
    return SelfInitiationCandidate(
        candidate_id="silence",
        type="silence",
        text="",
        topic="",
        reason=reason,
        requires_deep_processing=False,
    )
