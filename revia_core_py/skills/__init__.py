"""REVIA V4 Skills — Tool-Use plugin layer.

Public surface::

    Skill                  — abstract base class for any capability
    SkillRequest           — structured input
    SkillResponse          — structured output
    SkillMatch             — candidate score from Skill.match
    SkillDispatchResult    — registry dispatch outcome
    SkillRegistry          — registry of skills with rank + dispatch
    CalculatorSkill        — built-in safe arithmetic
    ClockSkill             — built-in current-time / date
    EchoSkill              — built-in debug passthrough
    MemoryRecallSkill      — built-in EpisodicMemoryStore search
"""
from __future__ import annotations

from .base import (
    Skill,
    SkillMatch,
    SkillRequest,
    SkillResponse,
)
from .builtin import (
    CalculatorSkill,
    ClockSkill,
    EchoSkill,
    MemoryRecallSkill,
)
from .registry import SkillDispatchResult, SkillRegistry

__all__ = [
    "CalculatorSkill",
    "ClockSkill",
    "EchoSkill",
    "MemoryRecallSkill",
    "Skill",
    "SkillDispatchResult",
    "SkillMatch",
    "SkillRegistry",
    "SkillRequest",
    "SkillResponse",
]
