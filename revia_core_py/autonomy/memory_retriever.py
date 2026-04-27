from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


_AVOID_RE = re.compile(r"\b(don't bring up|do not bring up|stop talking about|avoid)\b", re.IGNORECASE)
_UNFINISHED_RE = re.compile(r"\b(todo|next|later|unfinished|come back|continue|still need|we should|need to)\b", re.IGNORECASE)


@dataclass
class RetrievedMemories:
    short_term: list[dict[str, Any]] = field(default_factory=list)
    long_term: list[dict[str, Any]] = field(default_factory=list)
    unfinished: list[dict[str, Any]] = field(default_factory=list)
    avoid: list[str] = field(default_factory=list)
    summary: str = ""


class MemoryRetriever:
    def __init__(self, *, memory_store):
        self._memory_store = memory_store

    def retrieve(
        self,
        *,
        active_topic: str = "",
        recent_topics: list[Any] | None = None,
        mood: str = "",
        limit: int = 6,
    ) -> RetrievedMemories:
        recent = self._memory_store.get_short_term(limit=24)
        topics = [str(active_topic or "").strip()]
        for topic in recent_topics or []:
            name = str(getattr(topic, "topic", "") or "").strip()
            if name and name not in topics:
                topics.append(name)
        topics = [t for t in topics if t][:4]

        long_term: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for topic in topics:
            for item in self._memory_store.search(topic, max_results=max(2, limit // 2)):
                item_id = str(item.get("id") or item.get("timestamp") or item.get("content") or "")
                if item_id in seen_ids:
                    continue
                seen_ids.add(item_id)
                long_term.append(item)
                if len(long_term) >= limit:
                    break
            if len(long_term) >= limit:
                break

        if not long_term:
            long_term = self._memory_store.get_long_term(limit=min(4, limit))

        unfinished = [
            dict(m)
            for m in recent[-16:]
            if _UNFINISHED_RE.search(str(m.get("content") or ""))
        ]
        avoid: list[str] = []
        for msg in recent[-12:]:
            content = str(msg.get("content") or "")
            if _AVOID_RE.search(content):
                avoid.append(content[:160])

        summary_parts = []
        if topics:
            summary_parts.append("topics=" + ", ".join(topics[:3]))
        if mood:
            summary_parts.append(f"mood={mood}")
        if unfinished:
            summary_parts.append(f"unfinished={len(unfinished)}")
        if long_term:
            summary_parts.append(f"long_term={len(long_term)}")

        return RetrievedMemories(
            short_term=recent,
            long_term=long_term,
            unfinished=unfinished[-4:],
            avoid=avoid[-4:],
            summary="; ".join(summary_parts),
        )
