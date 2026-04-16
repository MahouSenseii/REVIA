"""Python-side bridge to the cpp StructuredLogger's JSONL sink.

Phase 0 goals
-------------
* Give Python code a single, structured sink that writes the *same* JSONL
  format as ``revia_core_cpp/src/core/StructuredLogger.cpp``.
* Route ``logging`` records through this sink so existing ``logging.getLogger``
  usage joins the stream automatically.
* Stay leaf-level: do not import any other REVIA module.

File location
-------------
Default sink is ``<cwd>/logs/revia_core.jsonl`` (matches the cpp default).
The path can be overridden via:
  * constructor argument ``path=``
  * env var ``REVIA_CORE_LOG_PATH``
  * ``StructuredLogger.instance().set_file_path(...)``

Schema (one JSON object per line, matching the cpp side)
--------------------------------------------------------
{
  "ts":     "2026-04-16T12:34:56.789Z",
  "level":  "info",
  "stage":  "dotted.identifier",
  "thread": "<native thread id>",
  "fields": { ... arbitrary structured fields ... }
}

Thread-safety
-------------
All writes are serialized behind a ``threading.Lock``. Batched flush every
50 entries, matching the cpp side.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional


class LogLevel(str, Enum):
    TRACE = "trace"
    DEBUG = "debug"
    INFO  = "info"
    WARN  = "warn"
    ERROR = "error"


_LEVEL_ORDER = {
    LogLevel.TRACE: 0,
    LogLevel.DEBUG: 1,
    LogLevel.INFO:  2,
    LogLevel.WARN:  3,
    LogLevel.ERROR: 4,
}


def _iso8601_now() -> str:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


def _thread_id() -> str:
    return str(threading.get_ident())


def _level_enabled(actual: LogLevel, threshold: LogLevel) -> bool:
    return _LEVEL_ORDER[actual] >= _LEVEL_ORDER[threshold]


def _default_path() -> str:
    return os.environ.get("REVIA_CORE_LOG_PATH", "logs/revia_core.jsonl")


class StructuredLogger:
    """Process-wide singleton; matches the cpp StructuredLogger semantics."""

    _inst: Optional["StructuredLogger"] = None
    _inst_lock = threading.Lock()

    @classmethod
    def instance(cls) -> "StructuredLogger":
        with cls._inst_lock:
            if cls._inst is None:
                cls._inst = cls()
            return cls._inst

    def __init__(self, path: Optional[str] = None) -> None:
        self._lock = threading.Lock()
        self._path = path or _default_path()
        self._file = None        # type: ignore[assignment]
        self._unflushed = 0
        self._flush_every = 50

        self._file_sink_enabled   = True
        self._stderr_sink_enabled = True
        self._file_min_level      = LogLevel.TRACE
        self._stderr_min_level    = LogLevel.INFO

        self._total_events = 0

    # ---------- configuration ----------

    def set_file_sink_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._file_sink_enabled = enabled

    def set_stderr_sink_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._stderr_sink_enabled = enabled

    def set_file_path(self, path: str) -> None:
        with self._lock:
            if self._file is not None:
                try:
                    self._file.flush()
                    self._file.close()
                except Exception:
                    pass
            self._file = None
            self._path = path

    def set_stderr_min_level(self, level: LogLevel) -> None:
        with self._lock:
            self._stderr_min_level = level

    def set_file_min_level(self, level: LogLevel) -> None:
        with self._lock:
            self._file_min_level = level

    # ---------- emission ----------

    def event(
        self,
        stage: str,
        fields: Optional[Mapping[str, Any]] = None,
        level: LogLevel = LogLevel.INFO,
    ) -> None:
        entry = {
            "ts":     _iso8601_now(),
            "level":  level.value,
            "stage":  stage,
            "thread": _thread_id(),
            "fields": dict(fields) if fields else {},
        }
        with self._lock:
            self._write_locked(entry, level)

    def trace(self, stage: str, fields: Optional[Mapping[str, Any]] = None) -> None:
        self.event(stage, fields, LogLevel.TRACE)

    def debug(self, stage: str, fields: Optional[Mapping[str, Any]] = None) -> None:
        self.event(stage, fields, LogLevel.DEBUG)

    def info(self, stage: str, fields: Optional[Mapping[str, Any]] = None) -> None:
        self.event(stage, fields, LogLevel.INFO)

    def warn(self, stage: str, fields: Optional[Mapping[str, Any]] = None) -> None:
        self.event(stage, fields, LogLevel.WARN)

    def error(self, stage: str, fields: Optional[Mapping[str, Any]] = None) -> None:
        self.event(stage, fields, LogLevel.ERROR)

    def flush(self) -> None:
        with self._lock:
            if self._file is not None:
                try:
                    self._file.flush()
                except Exception:
                    pass
            self._unflushed = 0

    @property
    def total_events(self) -> int:
        return self._total_events

    # ---------- internals ----------

    def _open_file_locked(self) -> None:
        if self._file is not None:
            return
        try:
            p = Path(self._path)
            if p.parent and not p.parent.exists():
                p.parent.mkdir(parents=True, exist_ok=True)
            # line-buffered; binary write path would be faster but harder to
            # reason about when another process (the cpp core) may also append.
            self._file = open(self._path, "a", encoding="utf-8")
        except Exception as exc:
            # Logger must never raise. Degrade to stderr-only.
            sys.stderr.write(
                f"[StructuredLogger] could not open {self._path}: {exc} "
                f"— file sink disabled.\n"
            )
            self._file = None
            self._file_sink_enabled = False

    def _write_locked(self, entry: dict, level: LogLevel) -> None:
        self._total_events += 1

        if self._file_sink_enabled and _level_enabled(level, self._file_min_level):
            if self._file is None:
                self._open_file_locked()
            if self._file is not None:
                try:
                    self._file.write(json.dumps(entry, separators=(",", ":")) + "\n")
                    self._unflushed += 1
                    if self._unflushed >= self._flush_every:
                        self._file.flush()
                        self._unflushed = 0
                except Exception as exc:
                    sys.stderr.write(
                        f"[StructuredLogger] write failed: {exc} — disabling file sink.\n"
                    )
                    self._file_sink_enabled = False

        if self._stderr_sink_enabled and _level_enabled(level, self._stderr_min_level):
            parts = [entry["ts"], entry["level"], entry["stage"]]
            if entry["fields"]:
                parts.append(json.dumps(entry["fields"], separators=(",", ":")))
            sys.stderr.write("  ".join(parts) + "\n")


# -------- logging.Handler bridge --------

_PY_TO_REVIA_LEVEL = {
    logging.DEBUG:    LogLevel.DEBUG,
    logging.INFO:     LogLevel.INFO,
    logging.WARNING:  LogLevel.WARN,
    logging.ERROR:    LogLevel.ERROR,
    logging.CRITICAL: LogLevel.ERROR,
}


class StructuredJsonlHandler(logging.Handler):
    """Plug into ``logging`` so legacy ``getLogger(...).info(...)`` calls
    route through the StructuredLogger sink.

    Usage:
        handler = StructuredJsonlHandler()
        logging.getLogger().addHandler(handler)
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = _PY_TO_REVIA_LEVEL.get(record.levelno, LogLevel.INFO)
            # Stage: use the logger name (e.g. "revia_core.memory") as the stage.
            stage = record.name or "py"
            fields: dict[str, Any] = {
                "message": record.getMessage(),
                "module":  record.module,
                "func":    record.funcName,
                "line":    record.lineno,
            }
            if record.exc_info:
                fields["exc_info"] = self.format(record)

            StructuredLogger.instance().event(stage, fields, level)
        except Exception:
            # Never let logging kill the caller.
            self.handleError(record)


# -------- module-level convenience --------

def get_logger() -> StructuredLogger:
    """Return the process-wide StructuredLogger."""
    return StructuredLogger.instance()


def core_log(
    stage: str,
    fields: Optional[Mapping[str, Any]] = None,
    level: LogLevel = LogLevel.INFO,
) -> None:
    """Module-level shortcut: ``core_log("stage", {"k":"v"})``."""
    StructuredLogger.instance().event(stage, fields, level)
