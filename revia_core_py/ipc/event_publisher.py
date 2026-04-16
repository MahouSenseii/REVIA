"""Phase 1+ bridge: publish selected Python-side events to the cpp EventBus.

Single ownership:
    CoreEventPublisher owns the HTTP shape for pushing events into
    ``revia_core_cpp/src/core/EventBus``. It does not decide behavior and it
    never changes legacy pipeline behavior.

Failure policy:
    This bridge is best-effort during the strangler-fig migration. If the cpp
    core is not running, v2 is disabled, or the request times out, callers get
    ``False`` and the legacy path continues unchanged.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Mapping, Optional

from .feature_flags import is_core_v2_enabled
from .structured_log import LogLevel, core_log


def _default_endpoint() -> str:
    port = os.environ.get("REVIA_REST_PORT", "8123").strip() or "8123"
    return os.environ.get(
        "REVIA_CORE_EVENT_ENDPOINT",
        f"http://127.0.0.1:{port}/api/core/events",
    )


class CoreEventPublisher:
    """Small REST client for the cpp core event endpoint."""

    def __init__(self, endpoint: Optional[str] = None, timeout_s: float = 0.35) -> None:
        self.endpoint = endpoint or _default_endpoint()
        self.timeout_s = timeout_s

    def publish_user_text(
        self,
        text: str,
        *,
        source: str,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        channel_id: Optional[str] = None,
        guild_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        """Publish a UserText event if Core v2 is enabled.

        Returns:
            True when the cpp endpoint accepted the event; False otherwise.
        """
        if not is_core_v2_enabled():
            return False

        clean_text = str(text or "").strip()
        if not clean_text:
            return False

        payload: dict[str, Any] = {
            "text": clean_text,
        }
        if user_id:
            payload["user_id"] = str(user_id)
        if username:
            payload["username"] = str(username)
        if channel_id:
            payload["channel_id"] = str(channel_id)
        if guild_id:
            payload["guild_id"] = str(guild_id)
        if metadata:
            payload["metadata"] = dict(metadata)

        body = {
            "type": "UserText",
            "source": source,
            "payload": payload,
        }
        return self._post(body)

    def publish_config_change(
        self,
        *,
        source: str = "ControllerUI",
        payload: Optional[Mapping[str, Any]] = None,
        correlation_id: Optional[str] = None,
    ) -> bool:
        """Publish a validated ConfigChange event if Core v2 is enabled."""
        if not is_core_v2_enabled():
            return False

        body: dict[str, Any] = {
            "type": "ConfigChange",
            "source": source,
            "payload": dict(payload or {}),
        }
        if correlation_id:
            body["correlation_id"] = str(correlation_id)
        return self._post(body)

    def _post(self, body: Mapping[str, Any]) -> bool:
        raw = json.dumps(body, separators=(",", ":")).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint,
            data=raw,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                response_body = resp.read(4096).decode("utf-8", errors="replace")
                response_json = json.loads(response_body) if response_body else {}
                accepted = 200 <= int(resp.status) < 300 and bool(response_json.get("ok"))
                core_log(
                    "event_publisher.posted",
                    {
                        "endpoint": self.endpoint,
                        "status": int(resp.status),
                        "accepted": accepted,
                        "event_type": body.get("type"),
                        "source": body.get("source"),
                    },
                    LogLevel.DEBUG,
                )
                return accepted
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            core_log(
                "event_publisher.failed",
                {
                    "endpoint": self.endpoint,
                    "error": str(exc),
                    "event_type": body.get("type"),
                    "source": body.get("source"),
                },
                LogLevel.DEBUG,
            )
            return False


_DEFAULT_PUBLISHER: Optional[CoreEventPublisher] = None


def get_event_publisher() -> CoreEventPublisher:
    global _DEFAULT_PUBLISHER
    if _DEFAULT_PUBLISHER is None:
        _DEFAULT_PUBLISHER = CoreEventPublisher()
    return _DEFAULT_PUBLISHER
