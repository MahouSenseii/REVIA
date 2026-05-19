"""Shared controller helpers for REVIA hardware routing.

The core owns authoritative hardware detection.  The controller still needs a
small local helper so subprocess launchers can read the saved policy before the
core has fully started.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_SETTINGS_FILE = PROJECT_ROOT / "model_settings.json"
PROFILE_SETTINGS_FILE = PROJECT_ROOT / "profile_settings.json"

SUPPORT_GPU_TASKS = [
    "speech_to_text",
    "text_to_speech",
    "vision",
    "memory_embeddings",
    "emotion_classifier",
    "background_agents",
]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def default_hardware_policy(gpu_count: int = 0) -> dict[str, Any]:
    if gpu_count <= 0:
        return {
            "hardware_mode": "cpu_only",
            "main_llm_gpu": "cpu",
            "support_gpu": "cpu",
            "allow_model_splitting": False,
            "allow_parallel_agents": False,
            "fallback_to_cpu": True,
        }
    if gpu_count == 1:
        return {
            "hardware_mode": "single_gpu_auto",
            "main_llm_gpu": "auto_best",
            "support_gpu": "cpu",
            "allow_model_splitting": False,
            "allow_parallel_agents": False,
            "fallback_to_cpu": True,
        }
    return {
        "hardware_mode": "multi_gpu_auto",
        "main_llm_gpu": "auto_best",
        "support_gpu": "auto_secondary",
        "allow_model_splitting": False,
        "allow_parallel_agents": True,
        "fallback_to_cpu": True,
    }


def normalize_hardware_policy(
    policy: dict[str, Any] | None,
    gpu_count: int = 0,
) -> dict[str, Any]:
    normalized = default_hardware_policy(gpu_count)
    if isinstance(policy, dict):
        normalized.update(policy)

    mode = str(normalized.get("hardware_mode") or "").strip().lower()
    if mode in {"auto", "auto_recommended", "recommended"}:
        normalized["hardware_mode"] = default_hardware_policy(gpu_count)["hardware_mode"]
    elif mode in {"both", "use_both", "multi_gpu"}:
        normalized["hardware_mode"] = "multi_gpu_auto" if gpu_count != 1 else "single_gpu_auto"
    elif mode in {"advanced", "advanced_multi_gpu", "multi_gpu_advanced"}:
        normalized["hardware_mode"] = "advanced_multi_gpu"
        normalized["allow_model_splitting"] = True
        normalized["allow_parallel_agents"] = True
    elif mode in {"single", "single_gpu"}:
        normalized["hardware_mode"] = "single_gpu"
        normalized["support_gpu"] = "cpu"
        normalized["allow_model_splitting"] = False

    if gpu_count == 1 and normalized.get("hardware_mode") == "multi_gpu_auto":
        normalized["hardware_mode"] = "single_gpu_auto"
        normalized["support_gpu"] = "cpu"
        normalized["allow_model_splitting"] = False
        normalized["allow_parallel_agents"] = False
    elif gpu_count <= 0 and not isinstance(policy, dict):
        normalized["hardware_mode"] = "cpu_only"

    normalized["allow_model_splitting"] = bool(normalized.get("allow_model_splitting", False))
    normalized["allow_parallel_agents"] = bool(normalized.get("allow_parallel_agents", True))
    normalized["fallback_to_cpu"] = bool(normalized.get("fallback_to_cpu", True))
    return normalized


def read_saved_hardware_policy() -> dict[str, Any]:
    model = _read_json(MODEL_SETTINGS_FILE)
    policy = model.get("hardware_policy")
    if isinstance(policy, dict):
        return normalize_hardware_policy(policy)

    profile = _read_json(PROFILE_SETTINGS_FILE)
    hardware = profile.get("hardware")
    if isinstance(hardware, dict):
        policy = hardware.get("routing_policy") or hardware
        if isinstance(policy, dict):
            return normalize_hardware_policy(policy)
    return default_hardware_policy(0)


def write_profile_hardware_policy(
    policy: dict[str, Any],
    *,
    roles: dict[str, Any] | None = None,
    detected_gpus: list[dict[str, Any]] | None = None,
) -> None:
    profile = _read_json(PROFILE_SETTINGS_FILE)
    hardware = dict(profile.get("hardware") or {})
    hardware["routing_policy"] = normalize_hardware_policy(
        policy, len(detected_gpus or [])
    )
    if roles:
        hardware["resolved_roles"] = dict(roles)
    if detected_gpus is not None:
        hardware["detected_gpus"] = list(detected_gpus)
    profile["hardware"] = hardware
    _write_json(PROFILE_SETTINGS_FILE, profile)


def detect_nvidia_smi_gpus() -> list[dict[str, Any]]:
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            timeout=3,
        ).decode("utf-8", "ignore")
    except Exception:
        return []

    gpus: list[dict[str, Any]] = []
    for line in raw.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            index = int(parts[0])
            total = int(float(parts[2]))
            used = int(float(parts[3]))
            load = float(parts[4])
        except ValueError:
            continue
        gpus.append(
            {
                "index": index,
                "name": parts[1],
                "vram_total_mb": total,
                "vram_used_mb": used,
                "load_percent": load,
                "cuda_supported": True,
            }
        )
    return gpus


def _gpu_score(gpu: dict[str, Any]) -> tuple[int, int, float]:
    try:
        total = int(gpu.get("vram_total_mb", 0) or 0)
    except (TypeError, ValueError):
        total = 0
    try:
        used = int(gpu.get("vram_used_mb", 0) or 0)
    except (TypeError, ValueError):
        used = 0
    try:
        load = float(gpu.get("load_percent", gpu.get("gpu_percent", 0.0)) or 0.0)
    except (TypeError, ValueError):
        load = 0.0
    return total, max(0, total - used), -load


def _ranked_gpus(gpus: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(gpus, key=_gpu_score, reverse=True)


def _index_for_gpu(gpu: dict[str, Any] | None) -> int | None:
    if not gpu:
        return None
    try:
        return int(gpu.get("index"))
    except (TypeError, ValueError):
        return None


def _coerce_gpu_index(value: Any, gpus: list[dict[str, Any]]) -> int | None:
    indices = {_index_for_gpu(gpu) for gpu in gpus}
    indices.discard(None)
    if isinstance(value, int):
        return value if value in indices else None
    text = str(value or "").strip().lower()
    for prefix in ("gpu:", "cuda:"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    try:
        idx = int(text)
    except ValueError:
        return None
    return idx if idx in indices else None


def resolve_gpu_roles(
    gpus: list[dict[str, Any]],
    policy: dict[str, Any] | None,
) -> dict[str, Any]:
    policy = normalize_hardware_policy(policy, len(gpus))
    ranked = _ranked_gpus(gpus)
    best = ranked[0] if ranked else None
    secondary = ranked[1] if len(ranked) > 1 else None

    mode = str(policy.get("hardware_mode") or "").lower()
    main = _coerce_gpu_index(policy.get("main_llm_gpu"), gpus)
    if main is None and policy.get("main_llm_gpu") in (None, "", "auto_best"):
        main = _index_for_gpu(best)

    support: int | None = None
    support_raw = str(policy.get("support_gpu") or "").strip().lower()
    if support_raw not in {"", "cpu", "none"}:
        support = _coerce_gpu_index(policy.get("support_gpu"), gpus)
        if support is None and support_raw == "auto_secondary":
            support = _index_for_gpu(secondary)

    if mode in {"single_gpu", "single_gpu_auto", "cpu_only"}:
        support = None
    if main is not None and support == main:
        support = None

    model_splitting_allowed = bool(policy.get("allow_model_splitting")) and len(gpus) > 1
    return {
        "hardware_mode": policy.get("hardware_mode"),
        "main_llm_gpu_index": main,
        "support_gpu_index": support,
        "support_tasks": list(SUPPORT_GPU_TASKS) if support is not None else [],
        "visible_main_llm_devices": [main] if main is not None else [],
        "visible_support_devices": [support] if support is not None else [],
        "model_splitting": {
            "allowed_by_policy": model_splitting_allowed,
            "enabled": False,
            "reason": (
                "Disabled until model fit, runtime support, and performance checks pass."
                if model_splitting_allowed
                else "Disabled by policy."
            ),
        },
    }
