"""MemoryAgent -- produces relevant facts for the current turn.

V1 uses keyword retrieval over the existing :class:`MemoryStore` (plus a
small lookback over short-term turns).  V2 can swap in vector retrieval
without changing this class's contract.

Output payload::

    {
        "_confidence": 0.0 - 1.0,
        "relevant_facts": ["short-term: ...", "long-term: ...", "episodic-lesson: ..."],
        "short_term_count": int,
        "long_term_hits":   int,
        "episodic_lesson_count": int,
    }
"""
from __future__ import annotations

import re
from typing import Any

from .agent_base import Agent, AgentContext

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]{3,}")


class MemoryAgent(Agent):
    name = "MemoryAgent"
    default_timeout_ms = 1500

    def __init__(
        self,
        memory_store=None,
        model_router=None,
        max_facts: int = 5,
        short_term_window: int = 6,
        episodic_store=None,
        max_episodic_lessons: int = 3,
    ):
        self._memory = memory_store
        self._router = model_router
        self._max_facts = max(1, int(max_facts))
        self._short_term_window = max(0, int(short_term_window))
        # Issue #4 (REVIA_DEEP_DIVE) -- pull lessons from the EpisodicMemoryStore
        # so cross-session learnings reach the LLM context, not just the
        # autonomy loop.  Search is keyword + time-decayed (CPU-local), so
        # the latency cost is negligible.
        self._episodic = episodic_store
        self._max_episodic_lessons = max(0, int(max_episodic_lessons))

    # ------------------------------------------------------------------
    # Override execute via the standard run() contract
    # ------------------------------------------------------------------

    def run(self, context: AgentContext) -> dict[str, Any]:
        if self._memory is None:
            return {
                "_confidence": 0.0,
                "relevant_facts": [],
                "short_term_count": 0,
                "long_term_hits": 0,
                "episodic_lesson_count": 0,
                "note": "memory_store unavailable",
            }

        context.cancel_token.raise_if_cancelled()

        # Short-term: just include the recent turns themselves so the
        # reasoning agent can ground its reply.
        short_term = []
        try:
            short_term = list(self._memory.get_short_term(self._short_term_window))
        except Exception:
            short_term = []

        # Long-term: use the existing keyword search; pick salient tokens
        # from the user text to widen recall.
        keywords = self._extract_keywords(context.user_text)
        long_term_hits: list[dict[str, Any]] = []
        seen: set[str] = set()
        try:
            primary = self._memory.search(context.user_text, max_results=self._max_facts)
            for entry in primary or []:
                content = (entry.get("content") or "").strip()
                if content and content not in seen:
                    seen.add(content)
                    long_term_hits.append(entry)

            # Augment with single-keyword searches if we still have room.
            for kw in keywords:
                if len(long_term_hits) >= self._max_facts:
                    break
                context.cancel_token.raise_if_cancelled()
                more = self._memory.search(kw, max_results=self._max_facts)
                for entry in more or []:
                    content = (entry.get("content") or "").strip()
                    if not content or content in seen:
                        continue
                    seen.add(content)
                    long_term_hits.append(entry)
                    if len(long_term_hits) >= self._max_facts:
                        break
        except Exception:
            long_term_hits = []

        relevant: list[str] = []
        for turn in short_term[-self._short_term_window:]:
            role = (turn.get("role") or "user").lower()
            text = (turn.get("content") or "").strip()
            if text:
                relevant.append(f"short-term {role}: {text}")
            if len(relevant) >= self._max_facts:
                break

        for entry in long_term_hits:
            text = (entry.get("content") or "").strip()
            if text:
                relevant.append(f"long-term: {text}")
            if len(relevant) >= self._max_facts * 2:
                break

        # Issue #4 -- Episodic lessons.  Search the EpisodicMemoryStore for
        # past turns whose attached lesson is relevant to the current user
        # text.  Lessons survive across sessions, so this is the path by
        # which cross-session learning reaches the LLM context.
        episodic_lessons: list[str] = []
        if self._episodic is not None and self._max_episodic_lessons > 0:
            try:
                context.cancel_token.raise_if_cancelled()
                results = self._episodic.search(
                    context.user_text,
                    limit=self._max_episodic_lessons,
                )
                for result in results or []:
                    ep = getattr(result, "episode", None)
                    lesson = (getattr(ep, "lesson", "") or "").strip() if ep else ""
                    if not lesson:
                        continue
                    episodic_lessons.append(lesson)
                    relevant.append(f"episodic-lesson: {lesson}")
            except Exception:
                # Episodic search is a *boost*, never a hard dependency for
                # the turn -- never let its failures starve memory retrieval.
                episodic_lessons = []

        # Confidence heuristic: how many distinct sources fired.
        confidence = 0.0
        if long_term_hits:
            confidence += 0.5
        if short_term:
            confidence += 0.3
        if keywords:
            confidence += 0.2
        if episodic_lessons:
            confidence = min(1.0, confidence + 0.1)
        confidence = min(1.0, confidence)

        return {
            "_confidence": confidence,
            "relevant_facts": relevant,
            "short_term_count": len(short_term),
            "long_term_hits": len(long_term_hits),
            "episodic_lesson_count": len(episodic_lessons),
            "keywords": keywords,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_keywords(text: str, limit: int = 5) -> list[str]:
        if not text:
            return []
        tokens = _TOKEN_RE.findall(text.lower())
        # Preserve order but de-dupe; drop trivial English stop-ish tokens.
        STOP = {
            "the", "and", "you", "your", "with", "from", "this", "that",
            "have", "has", "for", "but", "are", "was", "were", "what",
            "how", "why", "can", "could", "would", "should", "into",
            "about", "there", "their", "they", "them",
        }
        seen: set[str] = set()
        out: list[str] = []
        for tok in tokens:
            if tok in STOP or tok in seen:
                continue
            seen.add(tok)
            out.append(tok)
            if len(out) >= limit:
                break
        return out
