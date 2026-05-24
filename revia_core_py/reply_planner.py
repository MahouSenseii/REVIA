"""reply_planner — RETIRED (WS-2).

This module was orphaned: ReasoningAgent always wired it as
``reply_planner=None``, making the 4-stage RPS loop permanently dead.
AVS (answer_validation.py) and ALE (anti_loop_engine.py) — the only
components worth keeping — are already standalone modules. Everything
else has been removed from the active path.

The file is left as an inert placeholder because the mount does not
permit deletion. You can safely delete reply_planner.py from your IDE.

DO NOT import from this module. DO NOT resurrect these classes.
"""
from __future__ import annotations

__all__: list[str] = []
