"""Python mirror of revia_core_cpp/src/core/FeatureFlags.

Reads the same env vars with the same parse rules so both sides agree on
whether the new Core path is active.

Env vars
--------
  REVIA_CORE_V2_ENABLED      "1"/"true"/"on"/"yes"  -> on   (default off)
  REVIA_CORE_V2_LOG_STDERR   "0"/"false"/"off"/"no" -> off  (default on)
"""
from __future__ import annotations

import os
import threading
from typing import Optional


_TRUE_TOKENS  = {"1", "true", "on", "yes", "y", "t"}
_FALSE_TOKENS = {"0", "false", "off", "no", "n", "f"}


def _parse_bool(raw: Optional[str], default: bool) -> bool:
    if raw is None:
        return default
    s = raw.strip().lower()
    if not s:
        return default
    if s in _TRUE_TOKENS:
        return True
    if s in _FALSE_TOKENS:
        return False
    return default


class FeatureFlags:
    """Process-wide feature flag snapshot. Thread-safe getters and setters."""

    _inst: Optional["FeatureFlags"] = None
    _inst_lock = threading.Lock()

    @classmethod
    def instance(cls) -> "FeatureFlags":
        with cls._inst_lock:
            if cls._inst is None:
                cls._inst = cls()
            return cls._inst

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._core_v2_enabled = False
        self._log_stderr_enabled = True
        self.reload_from_env()

    def core_v2_enabled(self) -> bool:
        with self._lock:
            return self._core_v2_enabled

    def log_stderr_enabled(self) -> bool:
        with self._lock:
            return self._log_stderr_enabled

    def set_core_v2_enabled(self, value: bool) -> None:
        with self._lock:
            self._core_v2_enabled = bool(value)

    def set_log_stderr_enabled(self, value: bool) -> None:
        with self._lock:
            self._log_stderr_enabled = bool(value)

    def reload_from_env(self) -> None:
        v2 = _parse_bool(os.environ.get("REVIA_CORE_V2_ENABLED"), default=False)
        stderr_on = _parse_bool(
            os.environ.get("REVIA_CORE_V2_LOG_STDERR"), default=True
        )
        with self._lock:
            self._core_v2_enabled = v2
            self._log_stderr_enabled = stderr_on

        # Late import to avoid circular dependency at module import time.
        try:
            from .structured_log import StructuredLogger
            StructuredLogger.instance().set_stderr_sink_enabled(stderr_on)
            StructuredLogger.instance().event(
                "feature_flags.loaded",
                {
                    "core_v2_enabled":    v2,
                    "log_stderr_enabled": stderr_on,
                    "source":             "env",
                },
            )
        except Exception:
            # Logger must never block flag loading.
            pass


def is_core_v2_enabled() -> bool:
    """Module-level shortcut."""
    return FeatureFlags.instance().core_v2_enabled()
