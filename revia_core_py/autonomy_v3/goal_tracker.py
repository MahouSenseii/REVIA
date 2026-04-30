"""V3.3 GoalTracker — multi-turn goal state machine.

A *goal* represents an open task or commitment: the user asked a question
that wasn't fully answered, requested a follow-up, or shared an emotion
that warrants a future check-in.  The tracker holds a small queue of
open goals with deadlines, status transitions, and tags.

Lifecycle: ``open -> in_progress -> resolved`` (or ``expired`` /
``abandoned``).  Each transition is timestamped.

Heuristic detection lets the orchestrator wire this in without needing
an LLM call:

    intent_label == "command"  + reply_text < 12 words   -> "follow_up_command"
    intent_label == "question" + critic.uncertain_facts  -> "answer_pending"
    intent_label == "emotional_share" + polarity=neg     -> "well_being_check_in"
    user_text contains "remind me" / "later" / "tomorrow" -> "reminder"
"""
from __future__ import annotations

import json
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


_REMIND_PATTERNS = (
    re.compile(r"\bremind\s+me\b", re.IGNORECASE),
    re.compile(r"\b(tomorrow|later|tonight|next\s+week)\b", re.IGNORECASE),
    re.compile(r"\bcheck[-\s]?in\b", re.IGNORECASE),
)


@dataclass
class Goal:
    """A single open goal."""

    id: str
    kind: str                    # "follow_up_command" | "answer_pending" | "well_being_check_in" | "reminder" | "user_defined"
    title: str
    detail: str = ""
    status: str = "open"         # "open" | "in_progress" | "resolved" | "expired" | "abandoned"
    tags: list[str] = field(default_factory=list)
    created_at: float = 0.0
    updated_at: float = 0.0
    deadline_at: float = 0.0     # 0 = no deadline
    related_episode_id: str = ""
    session_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Goal":
        return Goal(
            id=str(data.get("id") or ""),
            kind=str(data.get("kind") or "user_defined"),
            title=str(data.get("title") or ""),
            detail=str(data.get("detail") or ""),
            status=str(data.get("status") or "open"),
            tags=list(data.get("tags") or []),
            created_at=float(data.get("created_at") or 0.0),
            updated_at=float(data.get("updated_at") or 0.0),
            deadline_at=float(data.get("deadline_at") or 0.0),
            related_episode_id=str(data.get("related_episode_id") or ""),
            session_id=str(data.get("session_id") or ""),
        )


class GoalTracker:
    """In-memory goal queue with optional JSON persistence."""

    DEFAULT_PATH = Path("data/open_goals.json")
    DEFAULT_MAX_OPEN = 64
    DEFAULT_AUTO_EXPIRE_S = 7 * 24 * 3600.0   # 7 days

    def __init__(
        self,
        path: Path | None = None,
        max_open: int = DEFAULT_MAX_OPEN,
        auto_expire_s: float = DEFAULT_AUTO_EXPIRE_S,
    ):
        self._path = path or self.DEFAULT_PATH
        self._max = max(1, int(max_open))
        self._auto_expire_s = max(0.001, float(auto_expire_s))
        self._goals: dict[str, Goal] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> int:
        try:
            raw = self._path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return 0
        if not isinstance(data, list):
            return 0
        loaded: dict[str, Goal] = {}
        for item in data:
            if isinstance(item, dict):
                try:
                    g = Goal.from_dict(item)
                    if g.id:
                        loaded[g.id] = g
                except Exception:
                    continue
        with self._lock:
            self._goals = loaded
        return len(loaded)

    def save(self) -> int:
        with self._lock:
            payload = [g.to_dict() for g in self._goals.values()]
        self._path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(payload, indent=2, ensure_ascii=False)
        self._path.write_text(text, encoding="utf-8")
        return len(text)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(
        self,
        kind: str,
        title: str,
        detail: str = "",
        tags: list[str] | None = None,
        deadline_at: float = 0.0,
        related_episode_id: str = "",
        session_id: str = "",
    ) -> Goal:
        now = time.time()
        g = Goal(
            id=str(uuid.uuid4()),
            kind=kind,
            title=title.strip()[:160],
            detail=detail.strip()[:600],
            status="open",
            tags=list(tags or []),
            created_at=now,
            updated_at=now,
            deadline_at=float(deadline_at or 0.0),
            related_episode_id=related_episode_id,
            session_id=session_id,
        )
        with self._lock:
            self._goals[g.id] = g
            self._enforce_capacity_locked()
        return g

    def update_status(self, goal_id: str, status: str) -> bool:
        if status not in {"open", "in_progress", "resolved", "expired", "abandoned"}:
            return False
        with self._lock:
            g = self._goals.get(goal_id)
            if g is None:
                return False
            g.status = status
            g.updated_at = time.time()
            return True

    def remove(self, goal_id: str) -> bool:
        with self._lock:
            return self._goals.pop(goal_id, None) is not None

    def get(self, goal_id: str) -> Goal | None:
        with self._lock:
            return self._goals.get(goal_id)

    def open_goals(self, session_id: str | None = None) -> list[Goal]:
        with self._lock:
            out = [g for g in self._goals.values() if g.status in ("open", "in_progress")]
        if session_id is not None:
            out = [g for g in out if g.session_id == session_id]
        out.sort(key=lambda g: (g.deadline_at or 1e18, -g.updated_at))
        return out

    def all(self) -> list[Goal]:
        with self._lock:
            return list(self._goals.values())

    def stats(self) -> dict[str, Any]:
        with self._lock:
            counts: dict[str, int] = {}
            for g in self._goals.values():
                counts[g.status] = counts.get(g.status, 0) + 1
            return {"total": len(self._goals), "by_status": counts}

    # ------------------------------------------------------------------
    # Heuristic detection
    # ------------------------------------------------------------------

    def detect_from_turn(
        self,
        user_text: str,
        reply_text: str,
        intent: dict[str, Any] | None = None,
        critic: dict[str, Any] | None = None,
        episode_id: str = "",
        session_id: str = "",
    ) -> list[Goal]:
        """Scan a finished turn for goal triggers.  Returns added goals."""
        added: list[Goal] = []
        intent = intent or {}
        critic = critic or {}
        intent_label = str(intent.get("label") or "").lower()
        polarity = str(intent.get("polarity") or "neutral").lower()
        critic_issues = critic.get("issues") or {}

        # Reminder pattern in user_text -> always create a "reminder" goal.
        if any(p.search(user_text or "") for p in _REMIND_PATTERNS):
            added.append(self.add(
                kind="reminder",
                title=user_text.strip()[:140] or "user reminder",
                detail="Detected reminder phrase in user message",
                tags=["reminder"],
                related_episode_id=episode_id,
                session_id=session_id,
            ))

        # Command with very short reply -> open follow-up.
        if intent_label == "command":
            words = len((reply_text or "").split())
            if words < 12:
                added.append(self.add(
                    kind="follow_up_command",
                    title=f"follow up on: {user_text.strip()[:80]}",
                    detail="Initial reply was very short; user may need more detail.",
                    tags=["command", "follow_up"],
                    related_episode_id=episode_id,
                    session_id=session_id,
                ))

        # Question + critic-flagged uncertainty -> open answer-pending.
        if intent_label == "question" and critic_issues.get("uncertain_facts"):
            added.append(self.add(
                kind="answer_pending",
                title=f"answer pending: {user_text.strip()[:80]}",
                detail="Critic flagged uncertain_facts; verify and follow up.",
                tags=["question", "uncertain"],
                related_episode_id=episode_id,
                session_id=session_id,
            ))

        # Negative emotional share -> well-being check-in.
        if intent_label == "emotional_share" and polarity == "negative":
            added.append(self.add(
                kind="well_being_check_in",
                title="check in on user wellbeing",
                detail=f"User shared negative emotion: {user_text.strip()[:120]}",
                tags=["emotional", "well_being"],
                deadline_at=time.time() + 24 * 3600.0,
                related_episode_id=episode_id,
                session_id=session_id,
            ))

        return added

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def expire_overdue(self) -> int:
        """Mark overdue goals as ``expired``.  Returns count expired."""
        now = time.time()
        expired = 0
        with self._lock:
            for g in self._goals.values():
                if g.status != "open" and g.status != "in_progress":
                    continue
                if g.deadline_at and now >= g.deadline_at:
                    g.status = "expired"
                    g.updated_at = now
                    expired += 1
                elif (now - g.created_at) > self._auto_expire_s:
                    g.status = "expired"
                    g.updated_at = now
                    expired += 1
        return expired

    def _enforce_capacity_locked(self) -> None:
        if len(self._goals) <= self._max:
            return
        # Drop oldest *resolved* / *expired* / *abandoned* first.
        finished = [g for g in self._goals.values()
                     if g.status in ("resolved", "expired", "abandoned")]
        finished.sort(key=lambda g: g.updated_at)
        while len(self._goals) > self._max and finished:
            self._goals.pop(finished.pop(0).id, None)
        # If still over, drop oldest by created_at.
        if len(self._goals) > self._max:
            ordered = sorted(self._goals.values(), key=lambda g: g.created_at)
            for g in ordered:
                if len(self._goals) <= self._max:
                    break
                self._goals.pop(g.id, None)
