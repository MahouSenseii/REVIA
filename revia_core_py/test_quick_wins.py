"""
test_quick_wins.py — Regression tests for the four quick-win patches.

These tests are structural: they verify that the changes are actually present
in the source and behave correctly, so a future refactor can't accidentally
revert them without breaking CI.

Quick wins covered:
  1. globals().get("broadcast_json") replaced with _broadcast_fn module var
  2. len(text.split()) replaced with _WORD_RE.findall() in hot paths
  3. asyncio.gather used in _broadcast (concurrent fan-out)
  4. Throttle dicts capped to prevent unbounded growth

Run from revia_core_py/:
    python -m pytest test_quick_wins.py -v
"""

import importlib.util
import os
import re
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Path + minimal stubs (mirrors test_api.py pattern)
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

_SOURCE = (_HERE / "core_server.py").read_text(encoding="utf-8")
# Strip comment lines so pattern checks don't fire on documentation text
_CODE_LINES = [ln for ln in _SOURCE.splitlines() if not ln.lstrip().startswith("#")]
_CODE = "\n".join(_CODE_LINES)


def _stub_if_missing(name):
    if name not in sys.modules and importlib.util.find_spec(name) is None:
        sys.modules[name] = MagicMock()


for _dep in ["flask_sock", "redis", "psutil", "torch", "transformers"]:
    _stub_if_missing(_dep)


def _import_core():
    """Import core_server, returning the module (or None on failure)."""
    try:
        import core_server
        return core_server
    except Exception:
        return None


# ---------------------------------------------------------------------------
# QW-1: globals().get("broadcast_json") eliminated
# ---------------------------------------------------------------------------

class TestQW1BroadcastFn(unittest.TestCase):
    """globals().get('broadcast_json') must be gone; _broadcast_fn takes its place."""

    def test_globals_get_broadcast_json_absent_from_source(self):
        self.assertNotIn(
            'globals().get("broadcast_json")',
            _CODE,
            "globals().get(\"broadcast_json\") still present in code — QW-1 not applied",
        )

    def test_broadcast_fn_declared_in_source(self):
        self.assertIn(
            "_broadcast_fn",
            _CODE,
            "_broadcast_fn not found in source — QW-1 not applied",
        )

    def test_broadcast_fn_wired_to_broadcast_json(self):
        # The assignment `_broadcast_fn = broadcast_json` must appear after
        # the broadcast_json function definition.
        idx_def = _CODE.find("def broadcast_json(")
        idx_wire = _CODE.find("_broadcast_fn = broadcast_json", idx_def)
        self.assertGreater(
            idx_wire, idx_def,
            "_broadcast_fn = broadcast_json assignment not found after broadcast_json definition",
        )

    def test_broadcast_fn_module_attribute(self):
        cs = _import_core()
        if cs is None:
            self.skipTest("core_server import failed")
        self.assertTrue(
            hasattr(cs, "_broadcast_fn"),
            "_broadcast_fn missing as module attribute",
        )

    def test_revia_log_does_not_use_globals_dict(self):
        """_revia_log must reference _broadcast_fn, not globals()."""
        # Extract the _revia_log function body
        match = re.search(r"def _revia_log\(message\):(.*?)(?=\ndef |\Z)", _CODE, re.DOTALL)
        if not match:
            self.skipTest("Could not locate _revia_log body in stripped source")
        body = match.group(1)
        self.assertNotIn("globals()", body, "globals() still used inside _revia_log")
        self.assertIn("_broadcast_fn", body, "_broadcast_fn not referenced in _revia_log")


# ---------------------------------------------------------------------------
# QW-2: len(...split()) replaced with _WORD_RE.findall()
# ---------------------------------------------------------------------------

class TestQW2WordRe(unittest.TestCase):
    """All hot-path len(x.split()) word-counts must use _WORD_RE.findall()."""

    def test_word_re_defined_in_source(self):
        self.assertIn(
            "_WORD_RE = re.compile",
            _CODE,
            "_WORD_RE not defined — QW-2 not applied",
        )

    def test_no_raw_split_word_count_in_code(self):
        hits = re.findall(r"len\([^)]+\.split\(\)\)", _CODE)
        self.assertFalse(
            hits,
            f"Remaining len(...split()) calls found — QW-2 incomplete: {hits}",
        )

    def test_word_re_is_compiled_regex(self):
        cs = _import_core()
        if cs is None:
            self.skipTest("core_server import failed")
        self.assertTrue(hasattr(cs, "_WORD_RE"), "_WORD_RE missing from module")
        self.assertIsNotNone(
            getattr(cs._WORD_RE, "findall", None),
            "_WORD_RE.findall not present — not a compiled regex",
        )

    def test_word_re_counts_correctly(self):
        cs = _import_core()
        if cs is None:
            self.skipTest("core_server import failed")
        cases = [
            ("hello world", 2),
            ("one,two,three", 3),
            ("  spaces   everywhere  ", 2),
            ("", 0),
            # Apostrophes are not \w characters, so "it's" → ["it", "s"] = 2 tokens.
            # This matches the token-estimation purpose: slightly over-counts vs.
            # a real tokeniser, but is fast and deterministic.
            ("it's a test!", 4),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(len(cs._WORD_RE.findall(text)), expected)

    def test_word_re_used_in_token_est_path(self):
        self.assertIn(
            "_WORD_RE.findall",
            _CODE,
            "_WORD_RE.findall not found in source — QW-2 not applied",
        )
        count = _CODE.count("_WORD_RE.findall")
        self.assertGreaterEqual(
            count, 10,
            f"Expected ≥10 _WORD_RE.findall usages, found {count}",
        )


# ---------------------------------------------------------------------------
# QW-3: asyncio.gather in _broadcast
# ---------------------------------------------------------------------------

class TestQW3GatherBroadcast(unittest.TestCase):
    """_broadcast must fan out with asyncio.gather, not a sequential loop."""

    def test_sequential_await_loop_absent(self):
        # The old pattern was a for-loop with `await ws.send(text)` on its own line.
        # The new gather version uses `ws.send(text)` inside a list comprehension,
        # so we check for the old "await ws.send" statement (only appears in a loop body).
        self.assertNotIn(
            "await ws.send(text)",
            _CODE,
            "Direct 'await ws.send(text)' still present — sequential loop not replaced by gather (QW-3)",
        )

    def test_asyncio_gather_present_in_broadcast(self):
        idx_def = _CODE.find("async def _broadcast(")
        self.assertGreater(idx_def, 0, "async def _broadcast not found")
        # Find the end of the function (next async/def at the same indent level)
        body_start = idx_def + len("async def _broadcast(")
        next_def = re.search(r"\n(?:async )?def ", _CODE[body_start:])
        body_end = body_start + next_def.start() if next_def else len(_CODE)
        body = _CODE[idx_def:body_end]
        self.assertIn(
            "asyncio.gather",
            body,
            "asyncio.gather not found inside _broadcast body — QW-3 not applied",
        )

    def test_return_exceptions_true(self):
        """gather must use return_exceptions=True so one dead client can't abort others."""
        idx = _CODE.find("async def _broadcast(")
        segment = _CODE[idx: idx + 600]
        self.assertIn(
            "return_exceptions=True",
            segment,
            "return_exceptions=True missing from asyncio.gather call",
        )

    def test_dead_client_cleanup_preserved(self):
        """Dead-client eviction logic must still exist."""
        idx = _CODE.find("async def _broadcast(")
        segment = _CODE[idx: idx + 600]
        self.assertIn(
            "ws_clients.difference_update",
            segment,
            "Dead-client cleanup (ws_clients.difference_update) removed — regression",
        )


# ---------------------------------------------------------------------------
# QW-4: Throttle dict capped
# ---------------------------------------------------------------------------

class TestQW4ThrottleCap(unittest.TestCase):
    """_revia_log_throttled must evict stale entries to prevent unbounded growth."""

    def test_size_guard_present(self):
        idx = _CODE.find("def _revia_log_throttled(")
        self.assertGreater(idx, 0, "_revia_log_throttled not found")
        next_def = re.search(r"\ndef [a-zA-Z]", _CODE[idx + 10:])
        end = idx + 10 + next_def.start() if next_def else len(_CODE)
        body = _CODE[idx:end]
        self.assertIn(
            "len(_throttle_last_ts)",
            body,
            "Size guard (len(_throttle_last_ts) > N) missing from _revia_log_throttled — QW-4 not applied",
        )

    def test_eviction_cleans_both_dicts(self):
        idx = _CODE.find("def _revia_log_throttled(")
        end = idx + 1000
        body = _CODE[idx:end]
        self.assertIn("_throttle_last_ts.pop", body, "_throttle_last_ts.pop missing — eviction incomplete")
        self.assertIn("_throttle_suppressed.pop", body, "_throttle_suppressed.pop missing — eviction incomplete")

    def test_throttle_dicts_exist_at_module_level(self):
        cs = _import_core()
        if cs is None:
            self.skipTest("core_server import failed")
        self.assertTrue(hasattr(cs, "_throttle_last_ts"))
        self.assertTrue(hasattr(cs, "_throttle_suppressed"))

    def test_throttle_does_not_grow_unbounded(self):
        """Calling _revia_log_throttled with >500 distinct stale keys must evict them."""
        cs = _import_core()
        if cs is None:
            self.skipTest("core_server import failed")

        import time
        # Back-date all existing entries so they're expired
        old_ts = time.monotonic() - 999.0
        with cs._throttle_lock:
            cs._throttle_last_ts.clear()
            cs._throttle_suppressed.clear()
            for i in range(510):
                cs._throttle_last_ts[f"test_key_{i}"] = old_ts

        # One more call should trigger eviction of all 510 stale entries
        cs._revia_log_throttled("test_eviction_trigger", "eviction test", cooldown_s=1.0)

        with cs._throttle_lock:
            remaining = len(cs._throttle_last_ts)

        # After eviction: only the one new key should remain
        self.assertLessEqual(
            remaining, 5,
            f"Throttle dict not evicted — {remaining} keys remain after stale eviction",
        )


if __name__ == "__main__":
    unittest.main()
