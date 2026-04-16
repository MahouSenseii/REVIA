"""Shared runtime state string constants for the REVIA controller layer.

Both ``conversation_policy.py`` and ``assistant_status_manager.py`` reference
these strings.  Keeping them in one place prevents silent mismatches when a
state name changes.
"""

from __future__ import annotations

# Core runtime states (must match the values emitted by the core server)
STATE_IDLE = "Idle"
STATE_THINKING = "Thinking"
STATE_SPEAKING = "Speaking"
STATE_COOLDOWN = "Cooldown"
STATE_BOOTING = "Booting"
STATE_LISTENING = "Listening"
STATE_GENERATING = "Generating"
STATE_ERROR = "Error"

# States that indicate the assistant is actively processing/interruptible
INTERRUPTIBLE_STATES = frozenset({STATE_SPEAKING, STATE_THINKING})

# States that block a new non-forced response
BUSY_STATES = frozenset({STATE_THINKING, STATE_SPEAKING, STATE_COOLDOWN})
