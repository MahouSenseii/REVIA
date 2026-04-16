"""UI-local timing collector for display-only metrics.

Behavior-affecting timing belongs to the C++ TimingEngine. This class may be
used by widgets to display request durations, but it must not influence Core
decisions.
"""
from __future__ import annotations

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
    def __init__(self) -> None:
        self._active: dict[str, float] = {}
        self._samples: list[TimingSample] = []

    def start(self, key: str) -> None:
        self._active[key] = time.monotonic()

    def stop(self, key: str, label: str | None = None) -> TimingSample | None:
        started = self._active.pop(key, None)
        if started is None:
            return None
        sample = TimingSample(label or key, started, time.monotonic())
        self._samples.append(sample)
        return sample

    def recent(self, limit: int = 20) -> list[TimingSample]:
        return self._samples[-limit:]
