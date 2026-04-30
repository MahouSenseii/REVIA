from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any


_SIMPLE_GREETING_RE = re.compile(r"^(hi|hey|hello|yo|sup|good morning|good evening|good night)[!. ]*$", re.I)

# Default greeting pool used when the profile does not provide
# greeting_variants. The reflex responder picks one at random so Revia
# does not echo the exact same greeting every session.
_DEFAULT_GREETING_POOL: tuple[str, ...] = (
    "I'm here.",
    "Back online.",
    "Okay, I'm with you.",
    "Hey. I'm listening.",
    "Right here. What do we have?",
    "Online. Ready when you are.",
    "I caught the signal. Go ahead.",
    "Standing by.",
    "Mm. I'm awake.",
    "Yeah, I'm here. What's up?",
)
_LAST_GREETING: dict[int, str] = {}


def _pick_greeting(profile: dict[str, Any] | None) -> str:
    pool: list[str] = []
    if isinstance(profile, dict):
        variants = profile.get("greeting_variants")
        if not variants:
            persona_def = profile.get("persona_definition") or {}
            interaction = persona_def.get("interaction_style") or {}
            variants = interaction.get("greeting_variants")
        if isinstance(variants, str):
            variants = [variants]
        if isinstance(variants, (list, tuple)):
            pool = [str(v).strip() for v in variants if str(v).strip()]

    if not pool:
        legacy = ""
        if isinstance(profile, dict):
            legacy = str(profile.get("greeting") or "").strip()
        if legacy:
            pool = [legacy]

    if not pool:
        pool = list(_DEFAULT_GREETING_POOL)

    if len(pool) == 1:
        return pool[0]

    key = id(profile) if profile is not None else 0
    last = _LAST_GREETING.get(key)
    candidates = [p for p in pool if p != last] or pool
    choice = random.choice(candidates)
    _LAST_GREETING[key] = choice
    return choice
_THANKS_RE = re.compile(r"^(thanks|thank you|ty|thx|appreciate it)[!. ]*$", re.I)
_AFFIRM_RE = re.compile(r"^(yes|yeah|yep|sure|ok|okay|alright|got it|mhm)[!. ]*$", re.I)
_NEGATE_RE = re.compile(r"^(no|nah|nope)[!. ]*$", re.I)
_QUIET_RE = re.compile(r"^(stop|wait|hold on|quiet|be quiet|shush|not now|pause)[!. ]*$", re.I)
_REPEAT_RE = re.compile(r"^(repeat|say that again|what did you say)[?.! ]*$", re.I)


@dataclass
class ReflexReply:
    text: str
    reason: str
    quiet_request: bool = False
    speakable: bool = True
    commit_to_memory: bool = True


def get_reflex_reply(text: str, *, memory_store=None, profile: dict[str, Any] | None = None) -> ReflexReply | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    if len(raw.split()) > 4 and not _REPEAT_RE.match(raw):
        return None

    if _QUIET_RE.match(raw):
        return ReflexReply(
            text="Okay. I'll stay quiet.",
            reason="quiet_request",
            quiet_request=True,
            speakable=True,
        )

    if _REPEAT_RE.match(raw):
        previous = _latest_assistant(memory_store)
        if previous:
            return ReflexReply(
                text=previous,
                reason="repeat_last_assistant",
                speakable=True,
                commit_to_memory=False,
            )
        return ReflexReply(
            text="I do not have a previous line to repeat yet.",
            reason="repeat_no_history",
            speakable=True,
            commit_to_memory=False,
        )

    if _SIMPLE_GREETING_RE.match(raw):
        greeting = _pick_greeting(profile)
        return ReflexReply(
            text=greeting or "I'm here.",
            reason="simple_greeting",
            speakable=True,
        )

    if _THANKS_RE.match(raw):
        return ReflexReply(
            text="Mm. You're welcome.",
            reason="thanks",
            speakable=True,
        )

    if _AFFIRM_RE.match(raw):
        return ReflexReply(
            text="Got it.",
            reason="affirmation",
            speakable=True,
        )

    if _NEGATE_RE.match(raw):
        return ReflexReply(
            text="Understood.",
            reason="negation",
            speakable=True,
        )

    return None


def _latest_assistant(memory_store) -> str:
    if memory_store is None:
        return ""
    try:
        for msg in reversed(memory_store.get_short_term(limit=12)):
            if str(msg.get("role") or "").lower() == "assistant":
                return str(msg.get("content") or "").strip()
    except Exception:
        return ""
    return ""
