# REVIA — Structural Remediation Plan

**Goal:** take every dimension of the REVIA vs. riko scorecard to a true **5/5**.
**Date:** 2026-05-23
**Author:** Senior developer review.
**Mandate (confirmed with you):** full structural rewrite-in-place. Duplicate subsystems get deleted, the god-file gets split, the pipeline gets unified. Backward compatibility is *not* a protected constraint — correctness, latency, and a clear architecture are.
**Deliverable:** this plan only. No code changed yet. Nothing here executes until you approve.

---

## 0. Investigation result — which reply pipeline is authoritative

You asked me to trace `core_server.py` and decide. I did. Findings, with evidence:

- **`/api/chat` (line 5729) is the production turn endpoint.** It builds a `TriggerRequest`, runs `BehaviorController.evaluate`, starts a turn, and spawns `_run_pipeline_safe` on a thread. That pipeline path uses **`parallel_pipeline`** — `run_fanout`, `submit_perception`, `submit_cognition`, `submit_expression` (lines 4577–5157). This is the perception → cognition → expression lane model.
- **`/api/agents/chat` (line 6198) is a *separate, additive* endpoint.** The code says so verbatim: line 5788 — `# Parallel Agents V1 — additive endpoint (does not replace /api/chat)`. It calls `AgentOrchestrator.run_turn`.
- **`ReplyPlanner` is effectively orphaned.** The orchestrator wires `ReasoningAgent(reply_planner=None, model_router=router)` at line 6083 — it passes `None`. `ReplyPlanner` (the 4-stage "RPS" with AVS/ALE/HFL) is not on the production path and not on the orchestrator path either. It is dead weight with two live sub-components (AVS, ALE) worth salvaging.

**Decision:** the canonical turn spine is **`parallel_pipeline` (perception / cognition / expression lanes)** — because it is what actually serves users, and the three-lane split is the right abstraction for a low-latency voice turn. The `AgentOrchestrator` + `agents/` package and `ReplyPlanner` are **retired**: the genuinely useful pieces (AVS answer-scoring, ALE anti-loop) move into the expression lane as *optional, budgeted* validators; everything else is deleted. One endpoint, one path, one mental model.

---

## 1. Note on the two planning docs already in the repo

You already have `CORE_SERVER_SPLIT_PLAN.md` and `REVIA_CORE_REVIEW.md`. Before this plan builds on them, a correction you need to hear:

- **`CORE_SERVER_SPLIT_PLAN.md` is good and current.** This plan adopts it wholesale as **Workstream 1**. Do not rewrite it; execute it.
- **`REVIA_CORE_REVIEW.md` is partially stale and following it as-is would hurt you.** It is dated 2026-04-16 and its entire thesis rests on *"`revia_core_cpp` is the real core … `revia_core_py` is the scripting layer"*. Your current `README.md` says the opposite — *"REVIA now treats the Python core as the main runtime … the C++ core remains as experimental/secondary."* The direction flipped after that review was written. A six-phase strangler-fig migration to a C++ core (EventBus, CoreOrchestrator, etc.) is a multi-month rewrite that **contradicts your current runtime choice**. Do not start it. Mark `REVIA_CORE_REVIEW.md` as **superseded** at the top, or this plan will compete with it.

The good *ideas* in that review — single-owner subsystems, an orchestrator with explicit stages, structured logging, a decision/safety seam — are kept here, but realized **in Python**, in the runtime you actually ship.

---

## 2. The scorecard, and what 5/5 concretely means

A "5/5" that is just an opinion is worthless. Each dimension below gets an **acceptance test** — an objective, checkable condition. The plan is done when every one passes.

| # | Dimension | Now | Workstream(s) | 5/5 acceptance test |
|---|-----------|-----|---------------|---------------------|
| 1 | Architecture & design | 3 | WS1, WS2 | No file > 800 lines; exactly one turn pipeline; no duplicate subsystem (`autonomy` vs `autonomy_v3` resolved); dependency graph is acyclic and documented. |
| 2 | Code quality | 3 | WS7, WS10 | `ruff` + `mypy` clean in CI; zero dead modules; zero inline imports; zero docstring/comment contradicting code; one intent parser. |
| 3 | Use-case fit (sounds human) | 3 | WS3, WS4 | Blind A/B: 8/10 listeners rate REVIA ≥ riko on "sounds like a real person" on a fixed 12-turn script. |
| 4 | Persona / voice control | 4.5 | WS4, WS5 | Voice-consistency eval harness ≥ 90% on a golden persona set; disfluency is prompt-driven, not regex-injected. |
| 5 | Conversational runtime | 4 | WS2, WS8 | FSM has 100% branch coverage; barge-in + recovery covered by tests; one turn = one documented state path. |
| 6 | Memory | 4 | WS6 | History bounded (token budget enforced); retrieval O(log n)/indexed not O(n); Redis-down fallback covered by an integration test. |
| 7 | Latency / responsiveness | 2 | WS3 | Voice turn: time-to-first-audio p95 ≤ 1.2 s; text turn: time-to-first-token p95 ≤ 700 ms; regression-gated in CI. |
| 8 | Error handling / robustness | 4 | WS8 | Chaos suite passes: LLM killed mid-turn, Redis down, TTS down, malformed input — every case ends in a clean state, no stuck FSM. |
| 9 | Maintainability / onboarding | 2 | WS1, WS7, WS10 | A new dev runs the test suite and serves a turn in < 30 min from `README`; architecture diagram matches code; `core_server.py` ≤ 40 lines. |
| 10 | Secrets / config hygiene | 4 | WS9 | Secret-scanner pre-commit hook installed; no secret in any tracked file (verified by scan); server fail-fasts on missing required env. |

Note dimensions 4, 5, 6, 8, 10 are already close — they need *finishing and locking*, not rebuilding. The heavy structural work is dimensions 1, 2, 3, 7, 9.

---

## 3. Workstreams

Ten workstreams. Each lists its target dimensions, the concrete actions, and the exit gate. Sequencing is in §4.

### WS-0 — Guard rails (must come first)

**Targets:** enables every other workstream safely.
**Why:** "full rewrite-in-place" without a regression net is how you turn a 3/5 system into a 0/5 system. This is non-negotiable and comes before any structural change.

Actions:

1. Add `pytest` + `pytest-cov` to `requirements.txt`; create `revia_core_py/tests/`.
2. **Smoke test:** start the core in-process, assert `/api/status` 200, POST `/api/chat`, poll the turn to completion. This is the regression fence.
3. **Golden-transcript test:** freeze a 12-turn scripted conversation with a stubbed/deterministic LLM; snapshot the final assistant texts. Any structural change must keep this snapshot stable (or change it deliberately).
4. **Latency benchmark harness:** `tests/bench_turn.py` — runs N turns against a stub LLM with a fixed artificial token delay, reports time-to-first-token, time-to-first-audio, total turn, p50/p95. This is the instrument WS-3 is judged by.
5. `git tag pre-remediation`. Every workstream ends with its own tag.

**Exit gate:** smoke + golden + bench all run green from a clean checkout, wired into a CI workflow.

---

### WS-1 — Decompose `core_server.py` (the 7,551-line god-file)

**Targets:** Architecture (1), Maintainability (9).
**Source of truth:** `CORE_SERVER_SPLIT_PLAN.md` — adopt it as written. This workstream is its execution, with two amendments.

Actions: execute that document's Phases 0–8 — `server/` sub-package, `AppContainer` to kill the `globals().get(...)` anti-pattern, route blueprints, `core_server.py` reduced to a ~40-line entry point.

Amendments to that plan:

- **Amend A:** its Phase 6 ("Pipeline module") must land *after* WS-2 decides the pipeline shape, or it will faithfully relocate three competing pipelines into a tidy folder. Do WS-2's deletions first, then split what survives.
- **Amend B:** add a hard lint rule at Phase 8 — no module under `server/` may define top-level mutable state except declared constants. This prevents the god-file re-forming by accretion.

**Exit gate:** no file > 800 lines; `core_server.py` ≤ 40 lines; zero `globals().get(`; smoke + golden + bench still green.

---

### WS-2 — Unify the reply pipeline

**Targets:** Architecture (1), Latency (7), Maintainability (9).
**Decision rationale:** §0 above.

Actions:

1. **Make `parallel_pipeline` the only turn spine.** Document the canonical path: `/api/chat` → `_run_pipeline_safe` → perception lane → cognition lane → expression lane. One diagram, committed to `docs/`.
2. **Retire `AgentOrchestrator` + the `agents/` package.** Salvage first: lift `AnswerValidationSystem` (AVS) and `AntiLoopEngine` (ALE) out — they are the only parts with standalone value. Then delete `/api/agents/chat`, `orchestrator.py`, `final_response.py`, `quality_gate.py`, `critic_agent.py`, `reasoning_agent.py`, and the rest of `agents/`. If you want to *keep* the agent abstraction as optional enrichment, that is allowed — but it must be called *inside the cognition lane behind a feature flag*, never as a second endpoint.
3. **Delete `reply_planner.py`.** It is orphaned (`reply_planner=None`). Its 4-stage RPS is replaced by the three lanes. Its regex `_parse_intent` dies with it (see WS-7, single intent parser).
4. **Re-home AVS + ALE as optional, budgeted validators in the expression lane.** They run only if the latency budget (WS-3) has room. They never trigger a blocking regen on a voice turn — at most they annotate or, on a text turn, request *one* regeneration.
5. **Resolve `autonomy/` vs `autonomy_v3/`.** Pick `autonomy_v3` (it is the newer episodic-memory/goal-tracker design). Migrate any unique behavior out of `autonomy/`, then delete `autonomy/` entirely. One autonomy package.
6. **Delete the dead `adapters/llm_adapter.py` stub** (the `NotImplementedError` "Phase 6 boundary stub"), or finish it as the real adapter base. Do not leave it.

**Exit gate:** exactly one turn endpoint and one pipeline; `grep` finds zero references to `AgentOrchestrator`, `ReplyPlanner`, `autonomy/` (old); golden transcript still stable; turn latency unchanged or better.

---

### WS-3 — Latency budget & streaming (the single biggest lever for "sounds human")

**Targets:** Latency (7), Use-case fit (3).
**Why this matters most:** a conversation feels human through *timing*. A 4-second "thinking" gap reads as "a machine is computing." The current architecture stacks multiple sequential LLM calls per spoken turn (agent fan-out + Critic + QualityGate + regen loop). The fix is structural, not cosmetic.

Actions:

1. **Set explicit budgets** and treat them as contracts: voice turn time-to-first-audio p95 ≤ 1.2 s; text turn time-to-first-token p95 ≤ 700 ms. These are the WS-0 bench gates.
2. **End-to-end token streaming.** The cognition lane must stream LLM tokens, not wait for completion. The expression lane already has `_extract_complete_tts_sentences` — wire it so TTS synthesis starts on the *first complete sentence* while the LLM is still generating the rest.
3. **One LLM call on the critical path.** A voice turn does exactly one generation. AVS/ALE/Critic become *post-hoc, off-path* annotations (logged, fed to RL) — never a blocking regen before first audio.
4. **Regen policy by mode:** voice mode `max_regen = 0`; text mode `max_regen = 1` and only when AVS composite is far below threshold. The current `regen_patience + 1` candidate loop is deleted with `reply_planner.py`.
5. **Perception lane runs concurrent with generation, not before it.** Emotion/memory retrieval should not block the first token. Retrieve memory in parallel; if it lands after generation starts, fold it into the *next* turn. A slightly staler memory context is a far smaller naturalness cost than a 2-second gap.
6. **Delete the latency-apology stall tactic.** The HFL prepends "One sec —" / "Let me think…" — that exists to paper over latency this plan removes. It goes with WS-4.

**Exit gate:** bench harness shows p95 within budget with a realistic LLM token delay; CI fails any PR that regresses p95 by > 15%.

---

### WS-4 — Fix the Human Feel Layer (mostly deletion)

**Targets:** Use-case fit (3), Persona (4).
**Why:** `human_feel_layer.py` regex-injects "Hmm…", "ngl,", "*sigh*" into *finished* LLM text, lowercasing the next character to splice markers in. This is the single change most likely to make REVIA sound like a bot. A modern model produces disfluency correctly *when the prompt asks for it*; bolting it on afterward produces ungrammatical seams and random, inconsistent voice.

Actions:

1. **Delete all text-mutation transforms** from HFL: `_apply_thinking_pause`, `_apply_self_correction`, `inject_quirks`, `inject_vocalizations`. Delete `_THINKING_PAUSES`, `_SELF_CORRECTIONS`, the vocalization marker map.
2. **Keep the prosody computation** (`_compute_prosody`, `ProsodyHints`). Emotion → pitch/rate/energy hints for TTS is legitimate and the *right* place for it. Rename the module to `prosody.py` — "human feel layer" is no longer an accurate name.
3. **Keep the verbosity trim** only as a hard safety cap (runaway-length guard), not as a "feel" feature — and source its limit from `ProfileEngine` for real this time (see WS-7 on the hardcoded-values lie).
4. **Move disfluency, quirks, and pacing into the prompt.** `persona_manager.py` already has `speech_quirks` and a `style_prompt`. Add an explicit, optional "natural speech" instruction block in `prompt_assembly.py` that tells the model to use light disfluency and the persona's quirks *organically*. The model places them grammatically and in-context; the regex never could.

**Exit gate:** no code path mutates LLM output text except the safety length cap; golden transcript regenerated and reviewed; A/B naturalness test (dimension 3) run.

---

### WS-5 — Persona & prompt consolidation (finish the strongest part)

**Targets:** Persona (4 → 5).
**Why:** `persona_manager.py` + `prompt_assembly.py` is already REVIA's best work. It needs finishing and a regression net, not a rebuild.

Actions:

1. **Single intent system.** Three intent paths exist today (`IntentAgent`, `ReplyPlanner._parse_intent`, plus regex in `prompt_assembly`). Two die with WS-2. Keep exactly one intent classifier feeding the perception lane; document it.
2. **Voice-consistency eval harness.** For each persona preset, a fixed set of prompts + a rubric (in-character? right register? quirks organic? no generic-assistant drift?). Score with an LLM judge. This is dimension 4's acceptance test and it must run in CI.
3. **Fold WS-4's natural-speech block into the persona presets** so disfluency is per-persona (the `serious` preset stays crisp; `casual` gets more).
4. **Prompt-injection sanitization stays** (`_sanitize_profile_field`) — but move its inline `import re` to module scope (WS-7) and add a test with known injection payloads.

**Exit gate:** voice-consistency harness ≥ 90% across all presets; exactly one intent classifier in the tree.

---

### WS-6 — Memory hardening

**Targets:** Memory (6 → 5).
**Source:** `CORE_SERVER_SPLIT_PLAN.md` §7 already specifies the hot-path fixes — apply them when the memory module is extracted in WS-1 Phase 4.

Actions:

1. **Bound history with a token budget.** Conversation history fed to the LLM must be windowed or summarized to a configured token ceiling. Unbounded growth is riko's bug; REVIA must not inherit it. Add a rolling summarizer for older turns.
2. **Cap every unbounded list** with `deque(maxlen=...)` — `MemoryStore._short_term`, `AnswerValidationSystem._history`, `AntiLoopEngine._history`, `InterruptionHandler._history` (per SPLIT_PLAN §7a).
3. **Replace the O(n) memory scan** with the keyword inverted index (SPLIT_PLAN §7c) and **release the lock before scoring** (§7d).
4. **Batch JSONL flushes** (§7b).
5. **Integration test for the Redis-down path** — kill Redis mid-run, assert the JSONL fallback engages and the turn still completes. The README advertises this fallback; it must be proven.

**Exit gate:** memory retrieval is indexed (not full-scan); history token budget enforced and tested; Redis-failover integration test green.

---

### WS-7 — Code-quality sweep

**Targets:** Code quality (2 → 5), Maintainability (9).

Actions:

1. **Delete dead code** (much of this overlaps WS-2): `agents/` package, `reply_planner.py`, `autonomy/` (old), `llm_adapter.py` stub. Then `grep` for unreferenced modules and remove them.
2. **Fix contradictory comments.** `human_feel_layer.py` claims "zero hardcoded values" while hardcoding `_VERBOSITY_MAX_WORDS = 220`, the `emotion_map`, the `0.20/0.80` window, etc. Either make them genuinely profile-sourced or delete the false claim. A comment that lies is worse than none.
3. **Move every inline import to module scope** — `import random` and `import re as _re` inside function bodies in `prompt_assembly.py`, the `import random` in `human_feel_layer`, and others. `grep -rn "^\s\+import "` to find them.
4. **Remove dead imports** project-wide (the riko review found `gradio` unused there; sweep REVIA the same way with `ruff`).
5. **Adopt `ruff` + `mypy`.** Add configs, fix the findings, gate CI on both clean.
6. **Kill the `globals().get(...)` pattern** — done structurally by WS-1's `AppContainer`; this is the verification step.
7. **One module, one responsibility** — after WS-1, audit that no `server/` module mixes routes with business logic.

**Exit gate:** `ruff` + `mypy` clean in CI; zero dead modules (coverage + import-graph check); zero inline imports; zero `globals().get(`.

---

### WS-8 — Robustness & recovery

**Targets:** Error handling (8 → 5), Conversational runtime (5 → 5).
**Why:** the FSM watchdog (`force_recover_if_stuck`) already exists — good. This workstream proves it and closes the gaps.

Actions:

1. **Lane-level timeout + fallback contract.** Each lane (perception/cognition/expression) declares a timeout and a defined fallback result. No lane exception may escape into a stuck FSM.
2. **Structured error taxonomy.** Replace ad-hoc `[bracketed]` error strings with typed errors → in-character fallback text (the `_personality_error` map is a fine start; formalize it).
3. **FSM branch coverage to 100%** — every transition in `_ALLOWED_TRANSITIONS`, plus `INTERRUPTED`/`RECOVERING`, exercised by tests.
4. **Chaos suite:** automated tests that (a) kill the LLM mid-generation, (b) take Redis down, (c) take TTS down, (d) feed malformed/huge input, (e) fire a barge-in during `SPEAKING`. Every case must end in a valid FSM state with a coherent user-facing message.
5. **Verify the watchdog** under the chaos suite — the `THINKING`-stuck recovery must actually fire and be observed in logs.

**Exit gate:** chaos suite green; FSM 100% branch coverage; no test run leaves a stuck state.

---

### WS-9 — Config & secrets hygiene

**Targets:** Secrets (10 → 5).

Actions:

1. **Secret scan of the whole tree and git history.** `config.json`, `model_settings.json`, `profile_settings.json`, `*.example` — confirm no live token is committed. (riko's mistake: API key in a tracked YAML. REVIA already uses `.env` — this is verification, not rebuild.)
2. **Install a secret-scanner pre-commit hook** (`gitleaks` or `detect-secrets`) and run it in CI.
3. **Complete `.env.example`** so every required variable is documented.
4. **Fail-fast on missing required env** — the server must refuse to start with a clear message rather than half-booting.
5. If history scan finds a leaked secret, rotate it and document the rotation.

**Exit gate:** secret scanner clean on tree + history; pre-commit hook active; documented fail-fast on missing config.

---

### WS-10 — Test suite & CI (cross-cutting; locks in every 5/5)

**Targets:** all dimensions — this is what makes the scores *stay* at 5.

Actions:

1. Grow the suite started in WS-0: unit (FSM, prosody, persona, memory index), integration (full turn, Redis failover), chaos (WS-8), latency (WS-3), voice-consistency (WS-5).
2. **Coverage gate** — fail CI below an agreed threshold (suggest 75% on `revia_core_py/`, 100% on the FSM).
3. **Latency regression gate** — fail CI on > 15% p95 regression.
4. **CI pipeline:** `ruff` → `mypy` → unit → integration → bench → secret-scan, on every PR.
5. **Update `README.md`** with the real architecture, the one turn path, and a < 30-minute onboarding runbook. Update or supersede `REVIA_CORE_REVIEW.md` (see §1).

**Exit gate:** CI runs all gates on every PR; a new dev can serve a turn in < 30 min following the README.

---

## 4. Sequencing & dependency graph

```
WS-0  Guard rails ─────────────┐ (blocks everything)
                               │
WS-1  Split core_server  ◀─────┤
   │     (Phase 6 waits on WS-2)│
WS-2  Unify pipeline    ◀───────┤ (needs WS-0; informs WS-1 Ph6)
   │                            │
   ├── WS-3 Latency/streaming   │ (needs WS-2)
   ├── WS-4 Fix HFL             │ (needs WS-2; pairs with WS-3)
   │                            │
WS-5 Persona  ─ WS-6 Memory ─ WS-7 Quality ─ WS-8 Robustness ─ WS-9 Secrets
        (mostly independent; run alongside once WS-1/2 land)
                               │
WS-10 Tests & CI ──────────────┘ (grows continuously; finalized last)
```

Critical path: **WS-0 → WS-2 → WS-1 → WS-3 + WS-4**. WS-5/6/7/8/9 parallelize after WS-1/2. WS-10 accretes throughout and is sealed at the end.

### Phased rollout

| Phase | Workstreams | Outcome |
|-------|-------------|---------|
| **A — Net** | WS-0 | Regression fence + latency instrument exist. Nothing else is safe without this. |
| **B — Untangle** | WS-2, then WS-1 | One pipeline, one endpoint; god-file split; dead subsystems deleted. Architecture/Maintainability rise. |
| **C — Sound human** | WS-3, WS-4 | Streaming + latency budget; HFL text-mutation gone. The dimensions that actually move "sounds like a person." |
| **D — Finish & lock** | WS-5, WS-6, WS-7, WS-8, WS-9 | Persona harness, memory hardening, quality sweep, chaos suite, secrets. Each remaining dimension hits 5. |
| **E — Seal** | WS-10 | CI gates every dimension so the 5/5 doesn't decay. |

Each phase ends with a git tag and a full green run of smoke + golden + bench.

---

## 5. Risks & mitigations

1. **Rewrite-in-place breaks a working system.** → WS-0 first, always. Every workstream tagged; any red gate is `git revert` to the last tag.
2. **WS-1 relocates the mess instead of fixing it.** → Amendment A: WS-2's deletions land before the pipeline module is split, so only the surviving pipeline gets a folder.
3. **Deleting `agents/`/`ReplyPlanner` removes something load-bearing.** → Salvage AVS + ALE *before* deletion; the golden transcript catches behavior regressions.
4. **Latency budget proves unreachable with the current model/host.** → The budget is measured against a *stub* LLM with fixed token delay in WS-0, so it isolates *architecture* latency from *model* latency. If the model itself is the bottleneck, that is a separate, visible decision (smaller model, faster host) — not hidden by the architecture.
5. **Cutting the HFL makes REVIA briefly sound *more* generic.** → Expected and acceptable: WS-4 moves disfluency into the prompt in the *same* phase. The A/B test in dimension 3 is the judge, not intuition.
6. **Scope creep — "5/5 on everything" invites gold-plating.** → Every dimension has a *binary* acceptance test in §2. When the test passes, that dimension is done. Stop there.
7. **The stale `REVIA_CORE_REVIEW.md` keeps pulling toward a C++ rewrite.** → §1: explicitly supersede it on day one so no one executes two conflicting plans.

---

## 6. Definition of done

The remediation is complete when **all ten acceptance tests in §2 pass simultaneously in CI**, on one commit, from a clean checkout — and:

- `core_server.py` is ≤ 40 lines; no source file exceeds 800 lines.
- Exactly one chat endpoint and one turn pipeline exist.
- No `agents/` orchestrator, no `reply_planner.py`, no `autonomy/` (old), no `llm_adapter.py` stub.
- No code mutates LLM output text except a safety length cap.
- `ruff` + `mypy` + secret-scan are clean and CI-gated.
- The README's architecture section matches the code, and a new developer can serve a turn in under 30 minutes.

At that point the scorecard is a genuine 5/5 — not because the numbers were edited, but because each one has a passing test behind it.

---

## 7. What I need from you to proceed

This document is the plan only. Before any code changes, confirm:

1. **Pipeline call** — you asked me to decide; I chose `parallel_pipeline` as the spine and retiring `AgentOrchestrator`/`ReplyPlanner` (§0). If you have a reason the agent orchestrator must survive (e.g. you are actively building on it), say so now — it changes WS-2.
2. **`REVIA_CORE_REVIEW.md`** — confirm the Python core is the runtime going forward so I can formally supersede that document. If C++-core is still the real plan, this entire remediation changes shape.
3. **Start point** — Phase A (WS-0 guard rails) is the correct first move. Say go and I will begin with the test harness and latency bench.
