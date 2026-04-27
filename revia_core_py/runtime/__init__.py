"""REVIA runtime layer — hardware-aware scheduling for the agent runtime.

Public surface::

    HardwareFingerprint  — boot-time hardware fingerprint (persisted)
    HardwareProfiler     — discovers + persists the fingerprint
    HardwareSnapshot     — live snapshot (CPU/RAM/GPU/VRAM + pressure)
    HardwareAgent        — periodic agent that produces snapshots
    ModelRequirements    — per-route resource requirements
    Reservation          — handle returned by the scheduler when admitted
    RuntimeScheduler     — priority queue + concurrency caps + reservations

V2.1 design rule:
    Model Router asks the scheduler for a Reservation before dispatching.
    If the reservation is denied, the router tries the route's fallback.
"""
from __future__ import annotations

from .hardware_profiler import (
    GpuInfo,
    HardwareFingerprint,
    HardwareProfiler,
)
from .hardware_agent import (
    HardwareAgent,
    HardwareSnapshot,
)
from .runtime_scheduler import (
    ModelRequirements,
    PRIORITY_CLASSES,
    Reservation,
    RuntimeScheduler,
    SchedulerStatus,
)
from .provider_registry import ProviderEntry, ProviderRegistry

__all__ = [
    "GpuInfo",
    "HardwareAgent",
    "HardwareFingerprint",
    "HardwareProfiler",
    "HardwareSnapshot",
    "ModelRequirements",
    "PRIORITY_CLASSES",
    "ProviderEntry",
    "ProviderRegistry",
    "Reservation",
    "RuntimeScheduler",
    "SchedulerStatus",
]
