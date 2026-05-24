"""WS-0 — latency benchmark harness.

WS-3 (the latency workstream) is judged against numbers this harness produces.
It separates the two things that make a turn slow:

  1. ARCHITECTURE overhead — REVIA's own per-turn CPU cost (prompt assembly,
     profile normalization, post-processing). This must be small and is
     fully measurable here with no model and no network.
  2. MODEL/NETWORK latency — the LLM call itself. Measured only in --url mode
     against a running server.

Keeping them separate matters: if a turn is slow, this harness tells you
whether to fix the architecture (this plan's job) or the model/host (a
separate decision), instead of guessing.

Usage
-----
    # Architecture overhead only (no server, no model needed):
    python tests/bench_turn.py

    # Against a running core server:
    python tests/bench_turn.py --url http://127.0.0.1:8123 --runs 30

CI uses the architecture-overhead number as a regression gate (WS-10).
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

# Make revia_core_py/ importable whether run as a script or via pytest.
_CORE_DIR = Path(__file__).resolve().parent.parent
if str(_CORE_DIR) not in sys.path:
    sys.path.insert(0, str(_CORE_DIR))


def _percentiles(samples_ms: list[float]) -> dict[str, float]:
    ordered = sorted(samples_ms)
    n = len(ordered)

    def pct(p: float) -> float:
        if n == 0:
            return 0.0
        idx = min(n - 1, int(round(p / 100.0 * (n - 1))))
        return ordered[idx]

    return {
        "count": n,
        "min": round(ordered[0], 3) if n else 0.0,
        "p50": round(pct(50), 3),
        "p95": round(pct(95), 3),
        "p99": round(pct(99), 3),
        "max": round(ordered[-1], 3) if n else 0.0,
        "mean": round(statistics.fmean(ordered), 3) if n else 0.0,
    }


def bench_architecture_overhead(runs: int = 200) -> dict[str, float]:
    """Time REVIA's per-turn non-LLM work: prompt assembly + post-processing.

    Returns latency percentiles in milliseconds. This is REVIA's own overhead;
    a healthy value is single-digit milliseconds.
    """
    from persona_manager import normalize_profile
    from prompt_assembly import CharacterProfileManager, PromptAssemblyManager
    from human_feel_layer import HumanFeelLayer
    from runtime_models import ResponseMode

    profiles = CharacterProfileManager(log_fn=lambda *a, **k: None)
    assembler = PromptAssemblyManager(log_fn=lambda *a, **k: None, profile_manager=profiles)
    hfl = HumanFeelLayer(profile_engine=None)
    profile = normalize_profile(None)
    sample_reply = (
        "Here is what I would try first. Check the service logs, then confirm "
        "the port binding, then restart and watch the first real request."
    )

    samples: list[float] = []
    for i in range(runs):
        t0 = time.perf_counter()
        sys_text = assembler.build_full_prompt_context(
            profile=profile,
            runtime_context="Runtime: idle.",
            memory_context="User prefers concise answers.",
            emotion_context="Detected emotion: neutral.",
            response_mode=ResponseMode.NORMAL_RESPONSE.value,
            behavior_params={"verbosity": 0.5},
        )
        hfl.process(sample_reply, emotion_label="neutral", rng_seed=i)
        samples.append((time.perf_counter() - t0) * 1000.0)
        _ = sys_text  # keep the result alive

    return _percentiles(samples)


def bench_live_turn(base_url: str, runs: int = 20) -> dict[str, float]:
    """Time a real /api/chat round trip against a running core server.

    Note: /api/chat returns as soon as the background turn is queued, so this
    measures request-acceptance latency, not full turn completion. Full-turn
    timing requires the WebSocket `chat_complete` event and is added in WS-3.
    """
    try:
        import requests  # local import — only needed in --url mode
    except ImportError:
        print("  [skip] live mode needs `requests` installed", file=sys.stderr)
        return {}

    base_url = base_url.rstrip("/")
    samples: list[float] = []
    for i in range(runs):
        t0 = time.perf_counter()
        try:
            resp = requests.post(
                f"{base_url}/api/chat",
                json={"text": f"benchmark probe {i}"},
                timeout=15,
            )
            resp.raise_for_status()
        except Exception as exc:  # pragma: no cover - network dependent
            print(f"  [warn] run {i} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        samples.append((time.perf_counter() - t0) * 1000.0)

    return _percentiles(samples)


def _print_report(title: str, stats: dict[str, float]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    if not stats:
        print("  (no samples)")
        return
    for key in ("count", "min", "p50", "p95", "p99", "max", "mean"):
        unit = "" if key == "count" else " ms"
        print(f"  {key:6s}: {stats.get(key, 0)}{unit}")


def main() -> int:
    parser = argparse.ArgumentParser(description="REVIA turn latency benchmark")
    parser.add_argument("--url", default="", help="base URL of a running core server")
    parser.add_argument("--runs", type=int, default=200, help="iterations")
    args = parser.parse_args()

    arch = bench_architecture_overhead(runs=args.runs)
    _print_report("Architecture overhead (prompt assembly + post-processing)", arch)

    if args.url:
        live = bench_live_turn(args.url, runs=max(5, args.runs // 10))
        _print_report(f"Live /api/chat acceptance ({args.url})", live)

    return 0


# --- pytest entry point (deselected by default; run with `-m bench`) --------
import pytest  # noqa: E402


@pytest.mark.bench
def test_bench_architecture_overhead():
    """Sanity gate: REVIA's per-turn non-LLM overhead should be modest.

    This is a loose ceiling, not the WS-3 contract. WS-3 tightens it into a
    real regression gate once streaming lands. 50 ms p95 here only catches a
    gross regression (e.g. an accidental O(n^2) in prompt assembly).
    """
    stats = bench_architecture_overhead(runs=100)
    assert stats["count"] == 100
    assert stats["p95"] < 50.0, f"architecture overhead p95 too high: {stats['p95']} ms"


if __name__ == "__main__":
    raise SystemExit(main())
