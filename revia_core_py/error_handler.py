"""
REVIA Error Handler — fully audited implementation.

Fixes applied:
  BUG-01: Global singleton leaks state → ErrorStore.reset()
  BUG-02: sys._getframe(1) wrong depth → stack-walking _get_caller_frame()
  BUG-03: __str__ XML-unsafe → html.escape in as_dict(), backslash normalisation
  BUG-04: catch_exception swallows MemoryError → re-raises >= ERROR severity
  BUG-05: error_count dict KeyError → defaultdict(int)
  BUG-06: No thread safety on error_history → threading.Lock in ErrorStore

  PERF-01: Unbounded error_history → deque(maxlen=1000)
  PERF-02: traceback.format_exc() called without exception → guarded
  PERF-03: String concat in __str__ → join()
  PERF-04: No lazy message eval → check() accepts callable
  PERF-05: PerformanceTimer context manager
  PERF-06: TokenBuffer stub with __slots__
  PERF-07: Async acheck() support

  OOP-01: SRP split → ErrorReportFactory, ErrorStore, ErrorBackend, facade
  OOP-02: ErrorCategory extensible registry
  OOP-03: ErrorBackend ABC for future logging backends
  OOP-04: CatchException is a callable class
  OOP-05: ErrorReport frozen dataclass with slots

  SEC-01: _sanitize() scrubs secrets from stack traces
  SEC-02: ConsoleBackend token-bucket rate limiter
  SEC-03: REVIA_LOG_LEVEL env var controls minimum emit level
"""

from __future__ import annotations

import abc
import asyncio
import html
import inspect
import logging
import os
import re
import sys
import threading
import time
import traceback
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Iterable, Optional, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Severity enum (IntEnum so comparisons work naturally)
# ---------------------------------------------------------------------------

class ErrorSeverity(IntEnum):
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4


# ---------------------------------------------------------------------------
# Extensible category registry  (OOP-02)
# ---------------------------------------------------------------------------

class ErrorCategory:
    _registry: dict[str, str] = {
        "inference": "inference",
        "audio": "audio",
        "network": "network",
        "memory": "memory",
        "config": "config",
        "general": "general",
    }

    @classmethod
    def register(cls, name: str) -> str:
        key = name.lower().strip()
        cls._registry[key] = key
        return key

    @classmethod
    def get(cls, name: str) -> str:
        key = name.lower().strip()
        if key not in cls._registry:
            raise ValueError(f"Unknown error category: {name!r}. "
                             f"Register it first with ErrorCategory.register({name!r})")
        return cls._registry[key]

    @classmethod
    def all(cls) -> list[str]:
        return sorted(cls._registry.keys())

    @classmethod
    def _reset(cls) -> None:
        """Reset to defaults — for tests only."""
        cls._registry = {
            "inference": "inference",
            "audio": "audio",
            "network": "network",
            "memory": "memory",
            "config": "config",
            "general": "general",
        }


# ---------------------------------------------------------------------------
# Frozen, slotted ErrorReport  (OOP-05, BUG-03)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ErrorReport:
    timestamp: float
    severity: ErrorSeverity
    category: str
    message: str
    function_name: str
    file_name: str
    line_number: int
    stack_trace: str = ""

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe dict with XML-escaped file paths (BUG-03)."""
        return {
            "timestamp": self.timestamp,
            "severity": self.severity.name,
            "category": self.category,
            "message": html.escape(self.message),
            "function_name": self.function_name,
            "file_name": html.escape(self.file_name.replace("\\", "/")),
            "line_number": self.line_number,
            "stack_trace": html.escape(self.stack_trace),
        }


# ---------------------------------------------------------------------------
# Secret sanitiser  (SEC-01)
# ---------------------------------------------------------------------------

_SECRET_RE = re.compile(
    r"(api_key|token|password|secret|bearer)\s*[=:]\s*\S+|"
    r"(Bearer)\s+\S+",
    re.IGNORECASE,
)


def _sanitize(text: str) -> str:
    """Redact secrets from stack traces before storage."""
    def _redact(m: re.Match) -> str:
        if m.group(2):  # "Bearer <token>" pattern
            return "Bearer ***REDACTED***"
        return m.group(1) + "=***REDACTED***"
    return _SECRET_RE.sub(_redact, text)


# ---------------------------------------------------------------------------
# Caller-frame walker  (BUG-02)
# ---------------------------------------------------------------------------

# Only skip frames whose *basename* is exactly this file.
_THIS_FILENAME = os.path.basename(__file__)  # "error_handler.py"


def _get_caller_frame() -> inspect.FrameInfo:
    """Walk the stack to find the first frame outside this module."""
    stack = inspect.stack()
    for frame_info in stack[1:]:
        basename = os.path.basename(frame_info.filename)
        if basename == _THIS_FILENAME:
            continue
        return frame_info
    # Fallback: return immediate caller
    return stack[1] if len(stack) > 1 else stack[0]


# ---------------------------------------------------------------------------
# ErrorReportFactory  (OOP-01)
# ---------------------------------------------------------------------------

class ErrorReportFactory:
    """Creates ErrorReport instances with automatic frame detection."""

    @staticmethod
    def create(
        severity: ErrorSeverity,
        category: str,
        message: str,
        *,
        include_trace: bool = False,
    ) -> ErrorReport:
        caller = _get_caller_frame()
        trace = ""
        if include_trace and sys.exc_info()[0] is not None:  # PERF-02
            trace = _sanitize(traceback.format_exc())
        return ErrorReport(
            timestamp=time.time(),
            severity=severity,
            category=category,
            message=message,
            function_name=caller.function,
            file_name=caller.filename.replace("\\", "/"),
            line_number=caller.lineno,
            stack_trace=trace,
        )


# ---------------------------------------------------------------------------
# Thread-safe ErrorStore  (BUG-01, BUG-05, BUG-06, PERF-01)
# ---------------------------------------------------------------------------

class ErrorStore:
    """Thread-safe, bounded error history with per-severity counters."""

    def __init__(self, maxlen: int = 1000) -> None:
        self._lock = threading.Lock()
        self._history: deque[ErrorReport] = deque(maxlen=maxlen)
        self._counts: defaultdict[str, int] = defaultdict(int)  # BUG-05

    # BUG-01 - reset for test isolation
    def reset(self) -> None:
        with self._lock:
            self._history.clear()
            self._counts.clear()

    def append(self, report: ErrorReport) -> None:
        with self._lock:  # BUG-06
            self._history.append(report)
            self._counts[report.severity.name] += 1

    def get_history(self, last_n: int | None = None) -> list[ErrorReport]:
        with self._lock:
            items = list(self._history)
        if last_n is not None:
            items = items[-last_n:]
        return items

    def get_counts(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)

    @property
    def total(self) -> int:
        with self._lock:
            return sum(self._counts.values())


# ---------------------------------------------------------------------------
# Backend ABC + implementations  (OOP-03, SEC-02, SEC-03)
# ---------------------------------------------------------------------------

class ErrorBackend(abc.ABC):
    """Abstract base for error-reporting backends."""

    @abc.abstractmethod
    def emit(self, report: ErrorReport) -> None: ...

    def close(self) -> None:
        """Release backend resources, if any."""
        return None


class ConsoleBackend(ErrorBackend):
    """Logs to stderr with token-bucket rate limiting (SEC-02)."""

    def __init__(self, rate_limit: float = 100.0) -> None:
        self._rate = rate_limit          # max per second
        self._tokens = rate_limit
        self._last = time.monotonic()
        self._lock = threading.Lock()
        min_level_name = os.environ.get("REVIA_LOG_LEVEL", "DEBUG").upper()
        try:
            self._min_level = ErrorSeverity[min_level_name]
        except KeyError:
            self._min_level = ErrorSeverity.DEBUG

    def emit(self, report: ErrorReport) -> None:
        if report.severity < self._min_level:           # SEC-03
            return
        with self._lock:
            now = time.monotonic()
            self._tokens = min(self._rate, self._tokens + (now - self._last) * self._rate)
            self._last = now
            if self._tokens < 1.0:
                return  # rate limited
            self._tokens -= 1.0

        # PERF-03: join instead of concat
        parts = [
            f"[{report.severity.name}]",
            f"[{report.category}]",
            report.message,
            f"({report.function_name} @ {os.path.basename(report.file_name)}:{report.line_number})",
        ]
        logger.log(
            _severity_to_logging(report.severity),
            " ".join(parts),
        )


class WebSocketBackend(ErrorBackend):
    """Forwards error reports to the GUI via a WebSocket broadcast callable.

    The broadcaster is provided as a callable (typically
    :func:`core_server.broadcast_json`) so this backend has no hard import
    dependency on the server module.  The payload matches the existing
    ``log_entry`` envelope consumed by the controller's ``logs_tab``::

        {"type": "log_entry", "text": "[ERROR] [inference] ... (fn:line)"}

    A small in-memory buffer captures messages emitted before the
    broadcaster is wired (e.g. during startup); the buffer is replayed
    automatically on the first successful broadcast.
    """

    __slots__ = ("_broadcast", "_buffer", "_buffer_lock", "_min_level")

    def __init__(
        self,
        broadcaster_fn: Callable[[dict], None] | None = None,
        *,
        buffer_size: int = 200,
    ) -> None:
        self._broadcast = broadcaster_fn
        self._buffer: deque[str] = deque(maxlen=max(0, int(buffer_size)))
        self._buffer_lock = threading.Lock()
        min_level_name = os.environ.get("REVIA_LOG_LEVEL", "DEBUG").upper()
        try:
            self._min_level = ErrorSeverity[min_level_name]
        except KeyError:
            self._min_level = ErrorSeverity.DEBUG

    def set_broadcaster(self, broadcaster_fn: Callable[[dict], None] | None) -> None:
        """Wire (or rewire) the broadcaster.  Flushes any buffered lines."""
        self._broadcast = broadcaster_fn
        if broadcaster_fn is None:
            return
        # Drain the buffer on first wiring so startup errors reach the GUI.
        with self._buffer_lock:
            pending = list(self._buffer)
            self._buffer.clear()
        for line in pending:
            try:
                broadcaster_fn({"type": "log_entry", "text": line})
            except Exception:
                # If broadcasting fails mid-replay, stop — the rest will be
                # lost rather than recurse into the error path.
                break

    def emit(self, report: ErrorReport) -> None:
        if report.severity < self._min_level:
            return
        parts = [
            f"[{report.severity.name}]",
            f"[{report.category}]",
            report.message,
            f"({report.function_name}:{report.line_number})",
        ]
        line = " ".join(parts)
        broadcaster = self._broadcast
        if not callable(broadcaster):
            with self._buffer_lock:
                self._buffer.append(line)
            return
        try:
            broadcaster({"type": "log_entry", "text": line})
        except Exception:
            # WS layer not ready or transport error — buffer for retry on
            # the next broadcaster swap.  Never raise from a logging path.
            with self._buffer_lock:
                self._buffer.append(line)


class CompositeBackend(ErrorBackend):
    """Fan-out backend: emits each report to every wrapped backend.

    Failures in one backend never block another.  Used to attach
    WebSocketBackend alongside ConsoleBackend / FileBackend without
    changing the :class:`ReviaErrorHandler` single-backend façade.
    """

    __slots__ = ("_backends",)

    def __init__(self, backends: Iterable[ErrorBackend]) -> None:
        self._backends: list[ErrorBackend] = [b for b in backends if b is not None]

    def add(self, backend: ErrorBackend) -> None:
        if backend is None:
            return
        self._backends.append(backend)

    def emit(self, report: ErrorReport) -> None:
        for backend in self._backends:
            try:
                backend.emit(report)
            except Exception:
                # A misbehaving backend must never starve the others.
                continue

    def close(self) -> None:
        for backend in self._backends:
            try:
                backend.close()
            except Exception:
                continue

    def find(self, kind: type) -> ErrorBackend | None:
        """Return the first wrapped backend of the given type, if any."""
        for backend in self._backends:
            if isinstance(backend, kind):
                return backend
        return None


class FileBackend(ErrorBackend):
    """Appends JSON-line reports to a file using a persistent file handle.

    A single file handle is opened at construction time and reused for every
    subsequent write.  This avoids the open()/close() syscall overhead on
    every error report, which matters when errors burst (e.g. pipeline failures).

    The handle is flushed after each write so crash-time log lines are never
    buffered away.  Call ``close()`` during shutdown for a clean flush+close.
    """

    def __init__(self, path: str) -> None:
        import json as _json
        self._json = _json
        self._path = path
        self._lock = threading.Lock()
        # Open once — append mode, line-buffered on text channels
        self._fh = open(path, "a", encoding="utf-8", buffering=1)  # noqa: SIM115

    def emit(self, report: ErrorReport) -> None:
        line = self._json.dumps(report.as_dict()) + "\n"
        with self._lock:
            self._fh.write(line)
            self._fh.flush()

    def close(self) -> None:
        """Flush and close the underlying file handle.  Safe to call multiple times."""
        with self._lock:
            try:
                self._fh.flush()
                self._fh.close()
            except OSError:
                pass


def _severity_to_logging(sev: ErrorSeverity) -> int:
    return {
        ErrorSeverity.DEBUG: logging.DEBUG,
        ErrorSeverity.INFO: logging.INFO,
        ErrorSeverity.WARNING: logging.WARNING,
        ErrorSeverity.ERROR: logging.ERROR,
        ErrorSeverity.CRITICAL: logging.CRITICAL,
    }[sev]


# ---------------------------------------------------------------------------
# CatchException callable class  (OOP-04, BUG-04)
# ---------------------------------------------------------------------------

class CatchException:
    """Decorator that catches exceptions, logs them, and optionally re-raises.

    BUG-04 fix: exceptions at ERROR or above are re-raised by default so
    callers (especially MemoryError / CRITICAL) are never silently swallowed.
    """

    def __init__(
        self,
        handler: "ReviaErrorHandler",
        category: str = "general",
        *,
        reraise_above: ErrorSeverity = ErrorSeverity.WARNING,
    ) -> None:
        self._handler = handler
        self._category = category
        self._reraise_above = reraise_above

    def __call__(self, func: Callable) -> Callable:
        import functools

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                severity = ErrorSeverity.CRITICAL if isinstance(exc, MemoryError) else ErrorSeverity.ERROR
                self._handler.log(
                    severity,
                    self._category,
                    f"{type(exc).__name__}: {exc}",
                    include_trace=True,
                )
                if severity >= self._reraise_above:  # BUG-04
                    raise
                return None

        return wrapper


# ---------------------------------------------------------------------------
# PerformanceTimer  (PERF-05)
# ---------------------------------------------------------------------------

class PerformanceTimer:
    """Context manager that measures wall-clock time with monotonic clock."""

    __slots__ = ("label", "handler", "category", "_start")

    def __init__(self, label: str, handler: "ReviaErrorHandler", category: str = "general") -> None:
        self.label = label
        self.handler = handler
        self.category = category
        self._start = 0.0

    def __enter__(self) -> "PerformanceTimer":
        self._start = time.monotonic()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        elapsed = time.monotonic() - self._start
        self.handler.log(
            ErrorSeverity.DEBUG,
            self.category,
            f"[TIMER] {self.label}: {elapsed:.4f}s",
        )

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._start


# ---------------------------------------------------------------------------
# TokenBuffer stub  (PERF-06)
# ---------------------------------------------------------------------------

class TokenBuffer:
    """Lightweight buffer for inference-engine tokens, ready for C++ bridge."""

    __slots__ = ("_buf", "_pos", "_size")

    def __init__(self, size: int = 4096) -> None:
        self._buf = bytearray(size)
        self._pos = 0
        self._size = size

    def write(self, data: bytes) -> int:
        n = min(len(data), self._size - self._pos)
        self._buf[self._pos: self._pos + n] = data[:n]
        self._pos += n
        return n

    def read(self) -> bytes:
        out = bytes(self._buf[: self._pos])
        self._pos = 0
        return out

    @property
    def remaining(self) -> int:
        return self._size - self._pos


# ---------------------------------------------------------------------------
# ReviaErrorHandler  (facade)
# ---------------------------------------------------------------------------

class ReviaErrorHandler:
    """Thin façade wiring together factory → store → backend(s)."""

    _instance: Optional["ReviaErrorHandler"] = None
    _instance_lock = threading.Lock()

    def __init__(self, *, backend: ErrorBackend | None = None) -> None:
        self.store = ErrorStore()
        self._backend: ErrorBackend = backend or ConsoleBackend()

    # --- singleton accessor (optional) ---
    @classmethod
    def get_instance(cls) -> "ReviaErrorHandler":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    # BUG-01: reset for test isolation
    @classmethod
    def reset_instance(cls) -> None:
        with cls._instance_lock:
            if cls._instance is not None:
                cls._instance.close()
                cls._instance.store.reset()
            cls._instance = None

    def reset(self) -> None:
        """Reset store state (for tests)."""
        self.store.reset()

    def close(self) -> None:
        """Release resources held by the active backend."""
        self._backend.close()

    # --- core logging ---
    def log(
        self,
        severity: ErrorSeverity,
        category: str,
        message: str,
        *,
        include_trace: bool = False,
    ) -> ErrorReport:
        cat = ErrorCategory.get(category) if category else "general"
        report = ErrorReportFactory.create(severity, cat, message, include_trace=include_trace)
        self.store.append(report)
        self._backend.emit(report)
        return report

    # PERF-04: lazy message evaluation
    def check(
        self,
        condition: bool,
        message: Union[str, Callable[[], str]],
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        category: str = "general",
    ) -> bool:
        """Log only if condition is False. Message can be a lambda for lazy eval."""
        if condition:
            return True
        msg = message() if callable(message) else message
        self.log(severity, category, f"Check failed: {msg}")
        return False

    # PERF-07: async check
    async def acheck(
        self,
        condition: bool,
        message: Union[str, Callable[[], str]],
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        category: str = "general",
    ) -> bool:
        """Async-friendly check — does not block the event loop."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self.check, condition, message, severity, category,
        )

    # --- convenience shortcuts ---
    def debug(self, category: str, message: str) -> ErrorReport:
        return self.log(ErrorSeverity.DEBUG, category, message)

    def info(self, category: str, message: str) -> ErrorReport:
        return self.log(ErrorSeverity.INFO, category, message)

    def warning(self, category: str, message: str) -> ErrorReport:
        return self.log(ErrorSeverity.WARNING, category, message)

    def error(self, category: str, message: str, *, include_trace: bool = False) -> ErrorReport:
        return self.log(ErrorSeverity.ERROR, category, message, include_trace=include_trace)

    def critical(self, category: str, message: str, *, include_trace: bool = False) -> ErrorReport:
        return self.log(ErrorSeverity.CRITICAL, category, message, include_trace=include_trace)

    # --- decorator ---
    def catch_exception(
        self,
        category: str = "general",
        *,
        reraise_above: ErrorSeverity = ErrorSeverity.WARNING,
    ) -> CatchException:
        return CatchException(self, category, reraise_above=reraise_above)

    # --- backend swap ---
    def swap_backend(self, backend: ErrorBackend) -> None:
        self._backend.close()
        self._backend = backend

    # --- composite-backend helpers ---
    def attach_backend(self, backend: ErrorBackend) -> None:
        """Add an additional backend without losing the existing one(s).

        If the current backend is a :class:`CompositeBackend`, ``backend`` is
        appended to it.  Otherwise the existing backend is wrapped in a fresh
        composite alongside the new one.
        """
        if isinstance(self._backend, CompositeBackend):
            self._backend.add(backend)
            return
        existing = self._backend
        self._backend = CompositeBackend([existing, backend])

    def attach_websocket_broadcaster(
        self, broadcaster_fn: Callable[[dict], None] | None
    ) -> WebSocketBackend:
        """Attach (or rewire) a :class:`WebSocketBackend` for GUI delivery.

        Idempotent: if a WebSocketBackend is already attached, its broadcaster
        is rewired and any buffered lines are flushed.  Returns the active
        backend so callers can introspect it.
        """
        existing: WebSocketBackend | None = None
        if isinstance(self._backend, CompositeBackend):
            found = self._backend.find(WebSocketBackend)
            if isinstance(found, WebSocketBackend):
                existing = found
        elif isinstance(self._backend, WebSocketBackend):
            existing = self._backend

        if existing is not None:
            existing.set_broadcaster(broadcaster_fn)
            return existing

        ws = WebSocketBackend(broadcaster_fn)
        self.attach_backend(ws)
        return ws

    # --- timer ---
    def timer(self, label: str, category: str = "general") -> PerformanceTimer:
        return PerformanceTimer(label, self, category)
