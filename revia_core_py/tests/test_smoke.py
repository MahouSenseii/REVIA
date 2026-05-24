"""WS-0 — smoke test: the regression fence for the route/import graph.

This is the cheapest possible "did a refactor break the wiring" check. It
imports the whole server and exercises a read route and the chat route via
Flask's in-process test client (no sockets bound, no real LLM needed —
`/api/chat` returns immediately after spawning a background turn thread).

Run after every structural move during WS-1/WS-2. If this goes red, the
import graph or a route registration broke.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.smoke


def test_status_route_responds(client):
    """GET /api/status must return 200 with a JSON body."""
    resp = client.get("/api/status")
    assert resp.status_code == 200
    body = resp.get_json()
    assert isinstance(body, dict), "/api/status should return a JSON object"


def test_chat_rejects_empty_message(client):
    """POST /api/chat with no text must be a clean 400, not a crash."""
    resp = client.post("/api/chat", json={"text": "   "})
    assert resp.status_code == 400
    body = resp.get_json()
    assert isinstance(body, dict)
    assert "error" in body


def test_chat_accepts_a_message(client):
    """POST /api/chat with text must be wired end-to-end.

    A healthy server returns 200 ('processing' — the turn runs on a
    background thread) or 503 ('conversation_not_ready' — the behavior
    controller gated it). Both prove the route, trigger evaluation, and
    turn manager are wired. Anything else (404/500) means broken wiring.
    """
    resp = client.post("/api/chat", json={"text": "hello, are you there?"})
    assert resp.status_code in (200, 503), (
        f"unexpected status {resp.status_code} — route/pipeline wiring broke"
    )
    body = resp.get_json()
    assert isinstance(body, dict)
    if resp.status_code == 200:
        # The contract of the production /api/chat handler.
        assert body.get("status") == "processing"
        assert "turn_id" in body and "request_id" in body
    else:
        assert body.get("error") == "conversation_not_ready"
        assert "decision" in body


def test_unknown_route_is_404(client):
    """A route that does not exist must 404 — confirms no catch-all masks errors."""
    resp = client.get("/api/__definitely_not_a_real_route__")
    assert resp.status_code == 404
