"""REVIA Output Interfaces — V2.4.

The :class:`InterfaceRouter` takes the single canonical answer the
agent orchestrator produces and fans it out to every active output
channel: chat, voice (TTS), avatar, audit log, and OS notifications.

Public surface::

    Interface                — abstract base class for any channel
    InterfaceContext         — read-only context for one dispatch
    InterfaceDecision        — per-channel result (delivered / skipped)
    InterfaceRouter          — registry + parallel dispatcher
    DispatchOutput           — aggregate result of one router call

    TextChatInterface        — always-on; text bubble in the UI
    VoiceInterface           — opt-in; routes to TTS engine
    VisionInterface          — opt-in; drives an avatar / visual cue
    LogInterface             — always-on; one-line audit per turn
    NotificationInterface    — opt-in; OS notification on key intents
"""
from __future__ import annotations

from .base import (
    Interface,
    InterfaceContext,
    InterfaceDecision,
)
from .builtin import (
    LogInterface,
    NotificationInterface,
    TextChatInterface,
    VisionInterface,
    VoiceInterface,
)
from .router import DispatchOutput, InterfaceRouter

__all__ = [
    "DispatchOutput",
    "Interface",
    "InterfaceContext",
    "InterfaceDecision",
    "InterfaceRouter",
    "LogInterface",
    "NotificationInterface",
    "TextChatInterface",
    "VisionInterface",
    "VoiceInterface",
]
