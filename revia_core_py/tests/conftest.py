"""WS-0 — shared pytest fixtures and path setup for the REVIA regression net.

REVIA's modules import each other by bare name (``import core_server``,
``from persona_manager import ...``) and expect ``revia_core_py/`` to be on
``sys.path``.  The legacy top-level ``test_*.py`` files rely on being run with
that directory as the working directory; this conftest makes the new
``tests/`` suite work the same way regardless of where pytest is invoked.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# --------------------------------------------------------------------------
# Path setup — make `revia_core_py/` importable as the project root.
# --------------------------------------------------------------------------
_CORE_DIR = Path(__file__).resolve().parent.parent  # .../revia_core_py
if str(_CORE_DIR) not in sys.path:
    sys.path.insert(0, str(_CORE_DIR))

_GOLDEN_DIR = Path(__file__).resolve().parent / "golden"


# --------------------------------------------------------------------------
# Golden-snapshot helper (approval testing).
#
# First run for a given `name` writes the snapshot and the test passes with a
# notice.  Every subsequent run compares against the stored snapshot.  This is
# how a structural refactor is proven to NOT change conversational behavior:
# the snapshot must stay byte-identical unless the change is intentional
# (in which case you delete the snapshot file and re-run to re-bless it).
# --------------------------------------------------------------------------
def assert_golden(name: str, data) -> None:
    """Compare `data` against the stored golden snapshot `name`.

    `data` must be JSON-serializable. On first run the snapshot is created.
    """
    _GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    path = _GOLDEN_DIR / f"{name}.json"
    serialized = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False, default=str)

    if not path.exists():
        path.write_text(serialized, encoding="utf-8")
        pytest.skip(
            f"golden snapshot '{name}' created at {path.name} — "
            f"review it, commit it, then this test will enforce it"
        )
        return

    expected = path.read_text(encoding="utf-8")
    assert serialized == expected, (
        f"golden snapshot '{name}' changed.\n"
        f"If this change is INTENTIONAL: delete tests/golden/{name}.json and "
        f"re-run to re-bless.\n"
        f"If it is NOT intentional: a refactor altered conversational behavior."
    )


@pytest.fixture
def golden():
    """Fixture form of :func:`assert_golden`."""
    return assert_golden


# --------------------------------------------------------------------------
# Server fixture — Flask test client over the live app object.
#
# Importing `core_server` runs every module-level singleton (LLM backend,
# memory store, emotion net, Redis reconnect threads). It does NOT call
# main(), so no sockets are bound. If the import fails (missing heavy deps in
# a minimal CI env) the dependent tests skip with a clear reason rather than
# producing a misleading failure.
# --------------------------------------------------------------------------
@pytest.fixture(scope="session")
def core_app():
    try:
        import core_server  # noqa: WPS433 (intentional in-fixture import)
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"core_server could not be imported in this environment: "
                    f"{type(exc).__name__}: {exc}")
    app = getattr(core_server, "app", None)
    if app is None:
        pytest.skip("core_server.app not found — Flask app object missing")
    app.config.update(TESTING=True)
    return app


@pytest.fixture
def client(core_app):
    return core_app.test_client()
