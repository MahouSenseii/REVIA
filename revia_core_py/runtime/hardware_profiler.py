"""Boot-time hardware fingerprint for REVIA.

Detection sources (best-effort, all optional):
    * ``pynvml`` (preferred) -> GPU name, VRAM, driver, compute capability
    * ``nvidia-smi`` shell out (fallback when pynvml is missing)
    * ``torch.cuda`` / ``torch.backends.mps`` for runtime CUDA / Apple Metal
    * ``psutil`` for CPU / RAM
    * ``importlib.util.find_spec("flash_attn")`` for FlashAttention presence
    * HEAD probes against the LOCAL_SERVERS dict to populate available
      providers (caller passes this dict in to avoid an import cycle)

The output is persisted to ``data/hw_fingerprint.json`` so subsequent boots
can show the recommended profile instantly.  The profile is recomputed
every boot regardless of cache to catch hardware/driver changes; the cache
exists only for fast UI display.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional dep
    import psutil as _psutil
except ImportError:  # pragma: no cover
    _psutil = None  # type: ignore[assignment]

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GpuInfo:
    index: int
    name: str
    vram_total_mb: int
    driver: str = ""
    compute_capability: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HardwareFingerprint:
    """Boot-time machine fingerprint used by the Model Router."""

    cpu_brand: str = ""
    cpu_cores_logical: int = 0
    cpu_cores_physical: int = 0
    ram_total_mb: int = 0

    has_cuda: bool = False
    cuda_runtime: str = ""              # "12.8" / ""
    has_rocm: bool = False
    has_mps: bool = False               # Apple Metal

    cuda_devices: list[GpuInfo] = field(default_factory=list)

    has_flash_attn: bool = False
    torch_version: str = ""
    python_version: str = ""
    platform: str = ""

    available_providers: dict[str, bool] = field(default_factory=dict)
    suggested_profile: str = "cpu_only"  # cpu_only | low_8gb | mid_12gb | high_24gb
    recommended_defaults: dict[str, Any] = field(default_factory=dict)
    detected_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["cuda_devices"] = [g.to_dict() for g in self.cuda_devices]
        return d

    @property
    def primary_gpu_vram_mb(self) -> int:
        if not self.cuda_devices:
            return 0
        return max(g.vram_total_mb for g in self.cuda_devices)


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------

class HardwareProfiler:
    """Discover machine fingerprint once at boot."""

    DEFAULT_FINGERPRINT_PATH = Path("data/hw_fingerprint.json")

    def __init__(self, log_fn=None, fingerprint_path: Path | None = None):
        self._log = log_fn or _log.info
        self.fingerprint_path = fingerprint_path or self.DEFAULT_FINGERPRINT_PATH

    def detect(self) -> HardwareFingerprint:
        """Run all probes; return a fresh fingerprint."""
        fp = HardwareFingerprint(
            cpu_brand=self._cpu_brand(),
            cpu_cores_logical=self._cpu_cores(logical=True),
            cpu_cores_physical=self._cpu_cores(logical=False),
            ram_total_mb=self._ram_total_mb(),
            torch_version=self._torch_version(),
            python_version=sys.version.split()[0],
            platform=platform.platform(),
            has_flash_attn=self._has_flash_attn(),
            detected_at=self._now(),
        )
        fp.has_cuda = self._has_cuda()
        fp.cuda_runtime = self._cuda_runtime() if fp.has_cuda else ""
        fp.has_rocm = self._has_rocm()
        fp.has_mps = self._has_mps()
        fp.cuda_devices = self._enumerate_gpus() if fp.has_cuda else []

        fp.suggested_profile = self._classify(fp)
        fp.recommended_defaults = self._defaults_for(fp.suggested_profile)
        return fp

    def discover_providers(
        self,
        provider_urls: dict[str, str] | None = None,
        timeout_s: float = 0.6,
    ) -> dict[str, bool]:
        """HEAD-probe each ``{name: base_url}`` and return reachability map.

        Caller passes the LOCAL_SERVERS dict from ``core_server`` so we
        avoid an import cycle.
        """
        out: dict[str, bool] = {}
        for name, url in (provider_urls or {}).items():
            out[name] = self._http_alive(url, timeout_s=timeout_s)
        # OpenAI-compatible cloud only counts as available when a key is set.
        if os.environ.get("OPENAI_API_KEY"):
            out.setdefault("OpenAI", True)
        return out

    def detect_and_persist(
        self,
        provider_urls: dict[str, str] | None = None,
    ) -> HardwareFingerprint:
        fp = self.detect()
        if provider_urls is not None:
            fp.available_providers = self.discover_providers(provider_urls)
        try:
            self.save(fp)
        except Exception as exc:  # pragma: no cover - persistence best-effort
            self._log(f"[HardwareProfiler] Could not persist fingerprint: {exc}")
        self._log(
            f"[HardwareProfiler] profile={fp.suggested_profile} "
            f"gpus={len(fp.cuda_devices)} primary_vram={fp.primary_gpu_vram_mb}MB "
            f"ram={fp.ram_total_mb}MB cores={fp.cpu_cores_logical} "
            f"flash_attn={fp.has_flash_attn}"
        )
        return fp

    def load(self) -> HardwareFingerprint | None:
        try:
            data = json.loads(self.fingerprint_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None
        try:
            gpus = [GpuInfo(**g) for g in data.get("cuda_devices", []) or []]
            data["cuda_devices"] = gpus
            return HardwareFingerprint(**data)
        except TypeError:
            return None

    def save(self, fp: HardwareFingerprint) -> None:
        self.fingerprint_path.parent.mkdir(parents=True, exist_ok=True)
        self.fingerprint_path.write_text(
            json.dumps(fp.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Probes (each one isolated; failures are silent)
    # ------------------------------------------------------------------

    @staticmethod
    def _now() -> float:
        import time
        return time.time()

    @staticmethod
    def _cpu_brand() -> str:
        try:
            return platform.processor() or platform.machine() or ""
        except Exception:
            return ""

    @staticmethod
    def _cpu_cores(logical: bool) -> int:
        if _psutil is not None:
            try:
                return int(_psutil.cpu_count(logical=logical) or 0)
            except Exception:
                pass
        try:
            return int(os.cpu_count() or 0)
        except Exception:
            return 0

    @staticmethod
    def _ram_total_mb() -> int:
        if _psutil is None:
            return 0
        try:
            return int(_psutil.virtual_memory().total // (1024 * 1024))
        except Exception:
            return 0

    @staticmethod
    def _torch_version() -> str:
        try:
            import torch
            return str(getattr(torch, "__version__", ""))
        except ImportError:
            return ""

    @staticmethod
    def _has_cuda() -> bool:
        try:
            import torch
            return bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
        except ImportError:
            return False
        except Exception:
            return False

    @staticmethod
    def _cuda_runtime() -> str:
        try:
            import torch
            return str(getattr(torch.version, "cuda", "") or "")
        except Exception:
            return ""

    @staticmethod
    def _has_rocm() -> bool:
        try:
            import torch
            return bool(getattr(torch.version, "hip", "") or "")
        except Exception:
            return False

    @staticmethod
    def _has_mps() -> bool:
        try:
            import torch
            return bool(
                getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()
            )
        except Exception:
            return False

    @staticmethod
    def _has_flash_attn() -> bool:
        try:
            return importlib.util.find_spec("flash_attn") is not None
        except (ImportError, ValueError):
            return False

    def _enumerate_gpus(self) -> list[GpuInfo]:
        # 1) Prefer pynvml (precise + no shell-out)
        gpus = self._enumerate_via_pynvml()
        if gpus:
            return gpus
        # 2) Fall back to nvidia-smi
        gpus = self._enumerate_via_nvidia_smi()
        if gpus:
            return gpus
        # 3) Final fallback: torch.cuda
        return self._enumerate_via_torch()

    @staticmethod
    def _enumerate_via_pynvml() -> list[GpuInfo]:
        try:
            import pynvml  # type: ignore
        except ImportError:
            return []
        try:
            pynvml.nvmlInit()
            try:
                count = pynvml.nvmlDeviceGetCount()
                out: list[GpuInfo] = []
                driver = pynvml.nvmlSystemGetDriverVersion()
                if isinstance(driver, bytes):
                    driver = driver.decode("utf-8", "ignore")
                for i in range(count):
                    h = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(h)
                    if isinstance(name, bytes):
                        name = name.decode("utf-8", "ignore")
                    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                    cc = ""
                    try:
                        major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(h)
                        cc = f"{major}.{minor}"
                    except Exception:
                        pass
                    out.append(GpuInfo(
                        index=i,
                        name=str(name),
                        vram_total_mb=int(mem.total // (1024 * 1024)),
                        driver=str(driver),
                        compute_capability=cc,
                    ))
                return out
            finally:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass
        except Exception:
            return []

    def _enumerate_via_nvidia_smi(self) -> list[GpuInfo]:
        if not shutil.which("nvidia-smi"):
            return []
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total,driver_version,compute_cap",
                    "--format=csv,noheader,nounits",
                ],
                timeout=3, stderr=subprocess.DEVNULL,
            ).decode("utf-8", "ignore")
        except Exception:
            return []
        gpus: list[GpuInfo] = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            try:
                idx = int(parts[0])
                vram = int(float(parts[2]))
            except ValueError:
                continue
            gpus.append(GpuInfo(
                index=idx,
                name=parts[1],
                vram_total_mb=vram,
                driver=parts[3] if len(parts) > 3 else "",
                compute_capability=parts[4] if len(parts) > 4 else "",
            ))
        return gpus

    @staticmethod
    def _enumerate_via_torch() -> list[GpuInfo]:
        try:
            import torch
        except ImportError:
            return []
        try:
            n = torch.cuda.device_count()
        except Exception:
            return 0  # type: ignore[return-value]
        out: list[GpuInfo] = []
        for i in range(n):
            try:
                props = torch.cuda.get_device_properties(i)
                cc = f"{props.major}.{props.minor}"
                out.append(GpuInfo(
                    index=i,
                    name=str(props.name),
                    vram_total_mb=int(props.total_memory // (1024 * 1024)),
                    driver="",
                    compute_capability=cc,
                ))
            except Exception:
                continue
        return out

    @staticmethod
    def _http_alive(base_url: str, timeout_s: float = 0.6) -> bool:
        try:
            import urllib.request
        except ImportError:
            return False
        url = (base_url or "").rstrip("/")
        if not url:
            return False
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                return 200 <= int(getattr(resp, "status", 200) or 200) < 500
        except Exception:
            try:
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                    return 200 <= int(getattr(resp, "status", 200) or 200) < 500
            except Exception:
                return False

    # ------------------------------------------------------------------
    # Profile classification
    # ------------------------------------------------------------------

    @staticmethod
    def _classify(fp: HardwareFingerprint) -> str:
        if not fp.has_cuda and not fp.has_mps and not fp.has_rocm:
            return "cpu_only"
        vram = fp.primary_gpu_vram_mb
        if vram <= 0:
            return "cpu_only"
        if vram < 9 * 1024:
            return "low_8gb"
        if vram < 17 * 1024:
            return "mid_12gb"
        return "high_24gb"

    @staticmethod
    def _defaults_for(profile: str) -> dict[str, Any]:
        """Suggested defaults the Model Router uses on first run."""
        if profile == "high_24gb":
            return {
                "tts_model_size": "1.7B",
                "reasoning_size": "14B",
                "allow_critic_llm": True,
                "max_concurrent_llms": 2,
                "allow_vision": True,
                "vram_safety_mb": 1024,
            }
        if profile == "mid_12gb":
            return {
                "tts_model_size": "0.6B",
                "reasoning_size": "9B",
                "allow_critic_llm": True,
                "max_concurrent_llms": 1,
                "allow_vision": True,
                "vram_safety_mb": 768,
            }
        if profile == "low_8gb":
            return {
                "tts_model_size": "0.6B",
                "reasoning_size": "7B",
                "allow_critic_llm": False,
                "max_concurrent_llms": 1,
                "allow_vision": False,
                "vram_safety_mb": 512,
            }
        # cpu_only
        return {
            "tts_model_size": "0.6B",
            "reasoning_size": "3B",
            "allow_critic_llm": False,
            "max_concurrent_llms": 1,
            "allow_vision": False,
            "vram_safety_mb": 0,
        }
