"""REVIA V3 — Autonomy & Persistence Layer.

Public surface::

    EpisodicMemoryStore  — V3.1 long-term per-turn store with time-decay search
    Episode              — single turn record (immutable after add)
    SearchResult         — scored match returned by EpisodicMemoryStore.search
    ReflectionAgent      — V3.2 post-turn metacognition + lesson tagging
    ReflectionVerdict    — structured reflection output (label + rl_hint)
    Goal                 — V3.3 single multi-turn goal record
    GoalTracker          — V3.3 in-memory goal queue + heuristic detection
    AutonomyScheduler    — V3.4 background loop with pressure awareness
    AutonomyTask         — single registered task entry
    register_default_tasks — wire standard tasks (persist memory, expire goals)
"""
from __future__ import annotations

from .episodic_memory import Episode, EpisodicMemoryStore, SearchResult
from .reflection_agent import ReflectionAgent, ReflectionVerdict
from .goal_tracker import Goal, GoalTracker
from .autonomy_scheduler import (
    AutonomyScheduler,
    AutonomyTask,
    register_default_tasks,
)

__all__ = [
    "AutonomyScheduler",
    "AutonomyTask",
    "Episode",
    "EpisodicMemoryStore",
    "Goal",
    "GoalTracker",
    "ReflectionAgent",
    "ReflectionVerdict",
    "SearchResult",
    "register_default_tasks",
]
