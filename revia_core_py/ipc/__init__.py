"""IPC helpers for the Python scripting layer to talk to the cpp core.

Phase 0 populates:
    structured_log : shared JSONL sink matching revia_core_cpp/src/core/StructuredLogger
    feature_flags  : env-var-driven toggles matching src/core/FeatureFlags

Phase 1 adds:
    event_publisher  : push selected events onto the cpp EventBus over REST
    rule_plugin_runner : host IDecisionRule plugins in an isolated subprocess
"""
from __future__ import annotations

from .structured_log import (
    StructuredLogger,
    StructuredJsonlHandler,
    get_logger,
    core_log,
    LogLevel,
)
from .feature_flags import FeatureFlags, is_core_v2_enabled
from .event_publisher import CoreEventPublisher, get_event_publisher

__all__ = [
    "StructuredLogger",
    "StructuredJsonlHandler",
    "get_logger",
    "core_log",
    "LogLevel",
    "FeatureFlags",
    "is_core_v2_enabled",
    "CoreEventPublisher",
    "get_event_publisher",
]
