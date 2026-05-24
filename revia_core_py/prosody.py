"""prosody — canonical name for the TTS prosody layer (WS-4 rename).

The implementation lives in human_feel_layer.py (name kept for import
compatibility with core_server.py and other callers).  New code should
import from here.

Usage::

    from prosody import ProsodyLayer, ProsodyHints, ProsodyResult

"""
from __future__ import annotations

from human_feel_layer import HumanFeelLayer as ProsodyLayer
from human_feel_layer import HFLResult      as ProsodyResult
from human_feel_layer import ProsodyHints

__all__ = ["ProsodyLayer", "ProsodyResult", "ProsodyHints"]
