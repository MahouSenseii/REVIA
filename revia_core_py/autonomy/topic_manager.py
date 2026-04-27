from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any


TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "REVIA autonomy system": (
        "revia",
        "autonomy",
        "self-initiation",
        "proactive",
        "diana",
        "conversation system",
        "core x",
    ),
    "REVIA voice pipeline": (
        "voice",
        "tts",
        "stt",
        "speech",
        "barge-in",
        "qwen",
        "whisper",
    ),
    "Project Hunter": (
        "project hunter",
        "projecthunter",
        "unreal",
        "gameplay",
        "mob",
        "spawner",
        "tags",
    ),
    "coding issues": (
        "code",
        "coding",
        "bug",
        "debug",
        "test",
        "repo",
        "python",
        "compile",
    ),
    "memory and continuity": (
        "memory",
        "remember",
        "recall",
        "unfinished",
        "topic",
        "history",
    ),
    "music": (
        "music",
        "song",
        "sing",
        "lyrics",
        "audio",
    ),
    "work": (
        "work",
        "ojt",
        "job",
        "task",
        "focus",
    ),
    "art": (
        "art",
        "drawing",
        "image",
        "design",
        "visual",
    ),
}

_UNFINISHED_RE = re.compile(
    r"\b(todo|next|later|unfinished|come back|continue|still need|we should|need to)\b",
    re.IGNORECASE,
)


@dataclass
class TopicSignal:
    topic: str
    score: float
    reason: str = ""
    user_interest: float = 0.0
    last_discussed_s: float = 9999.0
    metadata: dict[str, Any] = field(default_factory=dict)


class TopicManager:
    def __init__(self, *, memory_store, log_fn=None):
        self._memory_store = memory_store
        self._log = log_fn or (lambda _msg: None)
        self._topic_state: dict[str, dict[str, Any]] = {}
        self._active_topic = ""

    @property
    def active_topic(self) -> str:
        return self._active_topic

    def get_relevant_topics(self, state) -> list[TopicSignal]:
        recent = self._memory_store.get_short_term(limit=30)
        now = time.monotonic()
        scored: dict[str, TopicSignal] = {}
        for topic, keywords in TOPIC_KEYWORDS.items():
            score = 0.0
            user_hits = 0
            assistant_hits = 0
            unfinished = False
            for msg in recent[-16:]:
                role = str(msg.get("role") or "").lower()
                text = str(msg.get("content") or "").lower()
                hits = sum(1 for kw in keywords if kw in text)
                if hits <= 0:
                    continue
                weight = 1.0 if role == "user" else 0.45
                score += hits * weight
                if role == "user":
                    user_hits += hits
                else:
                    assistant_hits += hits
                if _UNFINISHED_RE.search(text):
                    unfinished = True
            stored = self._topic_state.get(topic, {})
            last_used = float(stored.get("last_discussed_at", 0.0) or 0.0)
            age = max(0.0, now - last_used) if last_used else 9999.0
            if score <= 0.0 and topic == self._active_topic:
                score = 0.6
            if score <= 0.0:
                continue
            if unfinished:
                score += 1.4
            interest = max(
                float(stored.get("user_interest", 0.0) or 0.0),
                min(1.0, user_hits / 4.0),
            )
            scored[topic] = TopicSignal(
                topic=topic,
                score=min(1.0, score / 5.0),
                reason="recent conversation" + ("; unfinished" if unfinished else ""),
                user_interest=interest,
                last_discussed_s=age,
                metadata={
                    "user_hits": user_hits,
                    "assistant_hits": assistant_hits,
                    "unfinished": unfinished,
                },
            )

        topics = sorted(scored.values(), key=lambda t: (t.score, t.user_interest), reverse=True)
        if topics:
            self._active_topic = topics[0].topic
        elif not self._active_topic:
            self._active_topic = str(getattr(state, "active_topic", "") or "")
        return topics[:5]

    def register_topic_used(self, topic: str, *, user_interest: float = 0.0) -> None:
        clean = str(topic or "").strip()
        if not clean:
            return
        now = time.monotonic()
        item = self._topic_state.setdefault(clean, {})
        item["last_discussed_at"] = now
        item["user_interest"] = max(float(item.get("user_interest", 0.0) or 0.0), float(user_interest or 0.0))
        self._active_topic = clean

    def topic_last_discussed_s(self, topic: str) -> float:
        item = self._topic_state.get(str(topic or "").strip(), {})
        last = float(item.get("last_discussed_at", 0.0) or 0.0)
        if not last:
            return 9999.0
        return max(0.0, time.monotonic() - last)
