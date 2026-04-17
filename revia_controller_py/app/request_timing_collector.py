"""UI-local timing collector for display-only metrics.

Behavior-affecting timing belongs to the C++ TimingEngine. This class may be
used by widgets to display request durations, but it must not influence Core
decisions.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass
class TimingSample:
    label: str
    started_at: float
    ended_at: float

    @property
    def duration_ms(self) -> float:
        return max(0.0, (self.ended_at - self.started_at) * 1000.0)


class RequestTimingCollector:
    """Thread-safe collector for per-request timing samples.

    ``start``/``stop`` may be called from any thread (audio pipeline, GUI
    thread, background workers). All shared state is protected by a single
    reentrant lock so callers never need external synchronisation.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: dict[str, float] = {}
        self._samples: list[TimingSample] = []

    def start(self, key: str) -> None:
        t = time.monotonic()
        with self._lock:
            self._active[key] = t

    def stop(self, key: str, label: str | None = None) -> TimingSample | None:
        t = time.monotonic()
        with self._lock:
            started = self._active.pop(key, None)
            if started is None:
                return None
            sample = TimingSample(label or key, started, t)
            self._samples.append(sample)
        return sample

    def recent(self, limit: int = 20) -> list[TimingSample]:
        with self._lock:
            return list(self._samples[-limit:])
