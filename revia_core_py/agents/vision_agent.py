"""V4.3 VisionAgent — surfaces a description of the active vision frame.

The agent runs in parallel with the other pre-agents.  When a vision
frame is available (delivered via ``context.metadata['vision_frame']``)
it produces a short structured description the :class:`ReasoningAgent`
can fold into its prompt.

Two paths:

    * Path A: a vision-capable provider is registered on the
      :class:`ModelRouter` under the ``describe_image`` task type.
      Used when REVIA has a multi-modal model loaded (e.g. a Qwen-VL
      backend).
    * Path B: heuristic stub that just acknowledges the frame's
      presence and any externally-attached caption metadata.

Output schema::

    {
        "has_frame": bool,
        "description": str,
        "tags": [str, ...],
        "source": "router" | "stub" | "metadata",
    }
"""
from __future__ import annotations

from typing import Any

from .agent_base import Agent, AgentContext


class VisionAgent(Agent):
    name = "VisionAgent"
    default_timeout_ms = 1200

    def __init__(self, model_router=None):
        self._router = model_router

    def run(self, context: AgentContext) -> dict[str, Any]:
        context.cancel_token.raise_if_cancelled()

        meta = context.metadata or {}
        frame = meta.get("vision_frame")
        attached_caption = str(meta.get("vision_caption") or "").strip()
        attached_tags = list(meta.get("vision_tags") or [])

        if frame is None and not attached_caption:
            return {
                "_confidence": 0.0,
                "has_frame": False,
                "description": "",
                "tags": [],
                "source": "none",
            }

        # Path A — router.
        if frame is not None and self._router is not None and self._router.has("describe_image"):
            try:
                result = self._router.call(
                    "describe_image",
                    frame,
                    user_text=context.user_text,
                    hints=attached_tags,
                )
                if isinstance(result, dict) and result.get("description"):
                    return {
                        "_confidence": float(result.get("confidence", 0.7)),
                        "has_frame": True,
                        "description": str(result.get("description") or "")[:600],
                        "tags": list(result.get("tags") or attached_tags),
                        "source": "router",
                    }
                if isinstance(result, str) and result.strip():
                    return {
                        "_confidence": 0.7,
                        "has_frame": True,
                        "description": result.strip()[:600],
                        "tags": list(attached_tags),
                        "source": "router",
                    }
            except Exception:
                pass

        # Path B — metadata-only fallback.
        if attached_caption:
            return {
                "_confidence": 0.5,
                "has_frame": frame is not None,
                "description": attached_caption[:600],
                "tags": list(attached_tags),
                "source": "metadata",
            }

        # Path C — pure stub.
        size_hint = self._size_hint(frame)
        desc = (
            "A vision frame was captured; no descriptor model is wired so "
            "no detailed caption is available."
        )
        if size_hint:
            desc = f"{desc} ({size_hint})"
        return {
            "_confidence": 0.2,
            "has_frame": True,
            "description": desc,
            "tags": list(attached_tags),
            "source": "stub",
        }

    @staticmethod
    def _size_hint(frame: Any) -> str:
        if frame is None:
            return ""
        # numpy ndarray-ish?
        shape = getattr(frame, "shape", None)
        if shape is not None:
            return f"shape={tuple(shape)[:3]}"
        # bytes / bytearray
        if isinstance(frame, (bytes, bytearray)):
            return f"bytes={len(frame)}"
        return ""
