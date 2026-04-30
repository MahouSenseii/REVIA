"""V3.1 EpisodicMemoryStore — structured per-turn memory with time-decay search.

Each turn produces one :class:`Episode` containing the user message, the
final reply, the intent / emotion / confidence triple, and any lesson
the ReflectionAgent later attached.  The store keeps a rolling window
in memory and persists periodically to ``data/episodic_memory.jsonl``
so cross-session continuity survives restarts.

Search is keyword + time-decayed scoring (recent + topical episodes
score highest).  This is intentionally cheaper than a vector store
(no embedding cost) so it works on CPU-only profiles too.  V3.x
later can plug a real embedder behind the same :meth:`search` surface.
"""
from __future__ import annotations

import json
import math
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


_FILLER = frozenset({
    "the", "a", "an", "to", "of", "in", "on", "for", "with", "and", "or",
    "but", "is", "are", "was", "were", "be", "been", "i", "you", "we",
    "this", "that", "very", "just", "so", "much",
})

_WORD_RE = re.compile(r"[a-z']+")


@dataclass
class Episode:
    """A single conversational turn with rich audit metadata."""

    id: str
    timestamp: float
    user_text: str
    reply_text: str
    intent_label: str = "chat"
    emotion_label: str = "neutral"
    polarity: str = "neutral"
    confidence: float = 0.0
    quality_score: float = 0.0
    regen_attempts: int = 0
    critic_recommendation: str = "accept"
    session_id: str = ""
    turn_id: str = ""
    keywords: list[str] = field(default_factory=list)
    lesson: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Episode":
        return Episode(
            id=str(data.get("id") or ""),
            timestamp=float(data.get("timestamp") or 0.0),
            user_text=str(data.get("user_text") or ""),
            reply_text=str(data.get("reply_text") or ""),
            intent_label=str(data.get("intent_label") or "chat"),
            emotion_label=str(data.get("emotion_label") or "neutral"),
            polarity=str(data.get("polarity") or "neutral"),
            confidence=float(data.get("confidence") or 0.0),
            quality_score=float(data.get("quality_score") or 0.0),
            regen_attempts=int(data.get("regen_attempts") or 0),
            critic_recommendation=str(data.get("critic_recommendation") or "accept"),
            session_id=str(data.get("session_id") or ""),
            turn_id=str(data.get("turn_id") or ""),
            keywords=list(data.get("keywords") or []),
            lesson=str(data.get("lesson") or ""),
        )


@dataclass
class SearchResult:
    episode: Episode
    score: float
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode": self.episode.to_dict(),
            "score": round(float(self.score), 4),
            "reason": self.reason,
        }


class EpisodicMemoryStore:
    """In-memory rolling store + JSONL persistence + time-decayed search."""

    DEFAULT_PATH = Path("data/episodic_memory.jsonl")
    DEFAULT_MAX_EPISODES = 2000
    DEFAULT_HALF_LIFE_S = 14 * 24 * 3600.0   # 14 days

    def __init__(
        self,
        path: Path | None = None,
        max_episodes: int = DEFAULT_MAX_EPISODES,
        half_life_s: float = DEFAULT_HALF_LIFE_S,
    ):
        self._path = path or self.DEFAULT_PATH
        self._max = max(1, int(max_episodes))
        self._half_life_s = max(60.0, float(half_life_s))
        self._episodes: list[Episode] = []
        self._lock = threading.Lock()
        self._dirty = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> int:
        """Load persisted episodes from JSONL.  Returns count loaded."""
        try:
            raw = self._path.read_text(encoding="utf-8")
        except (FileNotFoundError, OSError):
            return 0
        loaded: list[Episode] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                loaded.append(Episode.from_dict(data))
            except Exception:
                continue
        with self._lock:
            self._episodes = loaded[-self._max:]
            self._dirty = False
        return len(loaded)

    def save(self) -> int:
        """Persist all episodes to JSONL.  Returns bytes written."""
        with self._lock:
            payload = "\n".join(
                json.dumps(e.to_dict(), ensure_ascii=False) for e in self._episodes
            )
            self._dirty = False
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(payload + ("\n" if payload else ""), encoding="utf-8")
        return len(payload)

    def add(
        self,
        *,
        user_text: str,
        reply_text: str,
        intent_label: str = "chat",
        emotion_label: str = "neutral",
        polarity: str = "neutral",
        confidence: float = 0.0,
        quality_score: float = 0.0,
        regen_attempts: int = 0,
        critic_recommendation: str = "accept",
        session_id: str = "",
        turn_id: str = "",
        lesson: str = "",
    ) -> Episode:
        keywords = self._extract_keywords(user_text + " " + reply_text)
        ep = Episode(
            id=str(uuid.uuid4()),
            timestamp=time.time(),
            user_text=user_text,
            reply_text=reply_text,
            intent_label=intent_label,
            emotion_label=emotion_label,
            polarity=polarity,
            confidence=float(confidence),
            quality_score=float(quality_score),
            regen_attempts=int(regen_attempts),
            critic_recommendation=critic_recommendation,
            session_id=session_id,
            turn_id=turn_id,
            keywords=keywords,
            lesson=lesson,
        )
        with self._lock:
            self._episodes.append(ep)
            if len(self._episodes) > self._max:
                self._episodes = self._episodes[-self._max:]
            self._dirty = True
        return ep

    def attach_lesson(self, episode_id: str, lesson: str) -> bool:
        with self._lock:
            for ep in reversed(self._episodes):
                if ep.id == episode_id:
                    ep.lesson = str(lesson or "")[:1000]
                    self._dirty = True
                    return True
        return False

    def recent(self, limit: int = 10) -> list[Episode]:
        with self._lock:
            return list(self._episodes[-max(0, int(limit)):])

    def all(self) -> list[Episode]:
        with self._lock:
            return list(self._episodes)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            n = len(self._episodes)
            if not n:
                return {"count": 0, "intents": {}, "emotions": {}}
            intents: dict[str, int] = {}
            emotions: dict[str, int] = {}
            quality_sum = 0.0
            for ep in self._episodes:
                intents[ep.intent_label] = intents.get(ep.intent_label, 0) + 1
                emotions[ep.emotion_label] = emotions.get(ep.emotion_label, 0) + 1
                quality_sum += ep.quality_score
            return {
                "count": n,
                "avg_quality": round(quality_sum / n, 4),
                "intents": intents,
                "emotions": emotions,
            }

    def search(
        self,
        query: str,
        limit: int = 5,
        intent_label: str | None = None,
        min_score: float = 0.05,
    ) -> list[SearchResult]:
        """Time-decayed keyword search.

        Score = jaccard(keywords) * (alpha) + recency_decay * (1 - alpha)
        where alpha = 0.7 if query has tokens, else 0.0 (purely recency).
        """
        q_tokens = set(self._extract_keywords(query))
        now = time.time()
        results: list[SearchResult] = []

        with self._lock:
            episodes = list(self._episodes)

        if not episodes:
            return []

        alpha = 0.7 if q_tokens else 0.0
        for ep in episodes:
            if intent_label and ep.intent_label != intent_label:
                continue
            kw_set = set(ep.keywords)
            jacc = (
                len(q_tokens & kw_set) / float(len(q_tokens | kw_set))
                if q_tokens and kw_set else 0.0
            )
            age_s = max(0.0, now - ep.timestamp)
            decay = math.exp(-math.log(2.0) * age_s / self._half_life_s)
            score = alpha * jacc + (1.0 - alpha) * decay
            if score < min_score:
                continue
            reason_bits = []
            if jacc:
                reason_bits.append(f"jacc={jacc:.2f}")
            reason_bits.append(f"age_d={age_s/86400.0:.1f}")
            results.append(SearchResult(
                episode=ep, score=score, reason=",".join(reason_bits)
            ))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:max(0, int(limit))]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_keywords(text: str) -> list[str]:
        if not text:
            return []
        seen: dict[str, None] = {}
        for w in _WORD_RE.findall(text.lower()):
            if w in _FILLER or len(w) <= 2:
                continue
            if w not in seen:
                seen[w] = None
        return list(seen.keys())[:24]
