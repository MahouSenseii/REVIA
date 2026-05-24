# REVIA Remediation — Progress & Outstanding Work

**Last updated:** 2026-05-24
**Tracks:** `REVIA_Structural_Remediation_Plan.md` (the 10-workstream plan to take every scorecard dimension to 5/5).
**Companion docs:** `REVIA_vs_riko_Comparison.md` (the strict review this all came from), `CORE_SERVER_SPLIT_PLAN.md` (adopted as Workstream 1).

---

## Summary

| | |
|---|---|
| Workstreams complete | 0 of 10 |
| Workstreams started | 1 (WS-0, ~95%) |
| Code executed / verified | **None** — the build sandbox could not run Python this session |
| Net risk introduced | Minimal — all changes are additive or behavior-preserving |

The remediation is at the **start of Phase A**. The regression net exists but has not been run. Nothing structural has been touched yet.

---

## What was done

### 1. Analysis & planning (complete)

- **Strict comparison** of REVIA vs. `riko_project/server/process` — `REVIA_vs_riko_Comparison.md`. Produced the 10-dimension scorecard.
- **Structural remediation plan** — `REVIA_Structural_Remediation_Plan.md`. 10 workstreams, each with a binary 5/5 acceptance test.
- **Pipeline investigation** — traced `core_server.py`. Confirmed: `/api/chat` → `parallel_pipeline` (perception/cognition/expression lanes) is the production path; `/api/agents/chat` + `AgentOrchestrator` is an additive experiment; `ReplyPlanner` is orphaned (`reasoning_agent` wires `reply_planner=None`). **Decision: keep `parallel_pipeline` as the spine; retire the orchestrator + `ReplyPlanner`.** (Confirmed by owner.)

### 2. WS-0 — Regression net (~95% complete; needs first run)

All files added under `revia_core_py/tests/` plus repo-root CI. Purely additive — no production behavior changed.

| File | Purpose | State |
|------|---------|-------|
| `tests/conftest.py` | Path setup, `golden` snapshot helper, Flask test-client fixture | Written, unrun |
| `tests/test_smoke.py` | Imports server, exercises `/api/status` + `/api/chat` | Written, unrun |
| `tests/test_pipeline_components.py` | Golden + behavior tests over persona / prompt assembly / HFL | Written, unrun |
| `tests/bench_turn.py` | Latency benchmark (architecture overhead vs. live server) | Written, unrun |
| `tests/README.md` | How to run the suite + golden-snapshot workflow | Done |
| `pytest.ini` | Updated: `testpaths`, markers (`smoke`/`golden`/`bench`) | Done |
| `requirements-dev.txt` | `pytest`, `pytest-cov`, `ruff` | Done |
| `.github/workflows/ci.yml` | CI: ruff (non-blocking) + pytest + bench | Done |

### 3. Safe code-quality fix (complete)

- `prompt_assembly.py` — moved two inline imports (`random`, `re`) to module scope. Behavior-preserving.

### 4. Investigated but deliberately deferred

- **`adapters/llm_adapter.py` deletion** — found it is *not* a grep-safe deletion. It is one of three coordinated "Phase 6 adapter" stubs with a dedicated test class (`TestPhase6AdapterGuards`) and likely subclasses. Removing it is structural work (WS-2/WS-7), not a trivial cleanup.
- **Agent orchestrator retirement** — requires surgery inside the 7,551-line `core_server.py`. Not safe to do without a runnable test net. Decision stands; execution deferred to WS-2.

---

## Immediate next action (blocking everything else)

WS-0 is not "done" until the net runs green. From `revia_core_py/`:

1. `pip install -r requirements-dev.txt`
2. `pytest -m "not bench"`
3. **Run 1:** golden tests skip-and-create snapshots in `tests/golden/`. Review those files, commit them.
4. **Run 2:** snapshots now enforce. Smoke + golden should pass.
5. Report any real failure (not a skip) — that is signal.
6. `git tag pre-remediation`.

Until this is green, no structural workstream should start — the net is the safety fence for all of them.

---

## What else needs to be done

Ordered by the plan's phases. Each workstream's 5/5 acceptance test is in `REVIA_Structural_Remediation_Plan.md` §2.

### Phase A — Net
- [ ] **WS-0** — finish: run the suite, bless snapshots, tag baseline. *(~95% — see above.)*

### Phase B — Untangle
- [ ] **WS-2 — Unify the reply pipeline.** Make `parallel_pipeline` the only spine. Salvage `AnswerValidationSystem` (AVS) + `AntiLoopEngine` (ALE), then delete `agents/`, `/api/agents/chat`, `reply_planner.py`. Resolve `autonomy/` vs `autonomy_v3/` (keep v3). Remove the dead `llm_adapter.py` stub set.
- [ ] **WS-1 — Decompose `core_server.py`.** Execute `CORE_SERVER_SPLIT_PLAN.md`: `server/` sub-package, `AppContainer` (kills `globals().get(...)`), route blueprints. End state: `core_server.py` ≤ 40 lines, no file > 800 lines. *(Do WS-2's deletions first so the split doesn't relocate dead code.)*

### Phase C — Sound human (highest-impact for the actual goal)
- [ ] **WS-3 — Latency budget & streaming.** End-to-end token streaming; sentence-level TTS start; one LLM call on the critical path; regen = 0 for voice. Targets: time-to-first-audio p95 ≤ 1.2 s, time-to-first-token p95 ≤ 700 ms.
- [ ] **WS-4 — Fix the Human Feel Layer.** Delete the text-mutation transforms (regex-injected "Hmm…", "ngl,", "*sigh*"). Keep prosody hints only; rename module to `prosody.py`. Move disfluency/quirks into the persona prompt. *(This re-blesses the `golden/hfl_*` snapshots — the one intentional snapshot change.)*

### Phase D — Finish & lock
- [ ] **WS-5 — Persona & prompt consolidation.** One intent classifier (delete the other two). Voice-consistency eval harness ≥ 90%. Fold natural-speech block into persona presets.
- [ ] **WS-6 — Memory hardening.** Token-budget the history (summarize/window); `deque` caps on unbounded lists; keyword inverted index instead of O(n) scan; batched JSONL; Redis-failover integration test.
- [ ] **WS-7 — Code-quality sweep.** Delete dead code; fix the false "zero hardcoded values" docstring in HFL; move remaining inline imports up; `ruff` + `mypy` clean and CI-blocking.
- [ ] **WS-8 — Robustness.** Lane timeout+fallback contract; structured error taxonomy; FSM 100% branch coverage; chaos suite (kill LLM mid-turn, Redis down, TTS down, barge-in, malformed input).
- [ ] **WS-9 — Config & secrets.** Secret-scan tree + git history; install `gitleaks`/`detect-secrets` pre-commit hook; complete `.env.example`; fail-fast on missing required env.

### Phase E — Seal
- [ ] **WS-10 — Test suite & CI.** Coverage gate; latency regression gate; make `ruff` blocking; fold in legacy `test_*.py`; update `README.md` to a < 30-min onboarding runbook.

### Housekeeping
- [ ] Mark `REVIA_CORE_REVIEW.md` as **superseded** — its C++-core-first thesis contradicts the current README ("Python core is primary"). Following it would start a conflicting multi-month rewrite.

---

## Scorecard — current vs. target

| Dimension | Now | Target | Workstream(s) |
|-----------|-----|--------|---------------|
| Architecture & design | 3 | 5 | WS-1, WS-2 |
| Code quality | 3 | 5 | WS-7, WS-10 |
| Use-case fit (sounds human) | 3 | 5 | WS-3, WS-4 |
| Persona / voice control | 4.5 | 5 | WS-4, WS-5 |
| Conversational runtime | 4 | 5 | WS-2, WS-8 |
| Memory | 4 | 5 | WS-6 |
| Latency / responsiveness | 2 | 5 | WS-3 |
| Error handling / robustness | 4 | 5 | WS-8 |
| Maintainability / onboarding | 2 | 5 | WS-1, WS-7, WS-10 |
| Secrets / config hygiene | 4 | 5 | WS-9 |

No score moves until its workstream's acceptance test passes in CI.

---

## Constraints carried forward

- **No code has been executed.** Every file written this session is unverified by running. The first local `pytest` run is part of WS-0 acceptance.
- **Structural work needs a runnable test loop** — either a working sandbox, or the owner running `pytest` between steps and reporting results.
- **Destructive deletions are gated** on WS-0 being green and the pipeline decision (already confirmed).
