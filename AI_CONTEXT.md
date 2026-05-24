# REVIA — AI Context & Build Guide

**Read this first.** If you are an AI agent asked to work on REVIA, this
document is your orientation. It tells you what REVIA is, how it actually
works, what is wrong with it, and the principles for making it the best
conversational AI it can be. Everything here is grounded in a direct review
of the source — not assumptions.

**Companion documents (read in this order after this one):**
1. `REVIA_vs_riko_Comparison.md` — strict review; the scorecard.
2. `REVIA_Structural_Remediation_Plan.md` — the 10-workstream plan to fix it.
3. `REVIA_Remediation_Progress.md` — what is done and what is left.
4. `CORE_SERVER_SPLIT_PLAN.md` — the god-file decomposition (adopted).
5. `REVIA_CORE_REVIEW.md` — **partially stale**, see §9. Treat with caution.

---

## 1. What REVIA is

REVIA is a local AI assistant: a Python core server (`revia_core_py/`) that
handles chat, memory, emotion, telemetry, and integrations, plus a PySide6
desktop controller (`revia_controller_py/`). There is also an experimental
C++ core (`revia_core_cpp/`) that is **secondary** — the README states the
Python core is the primary runtime. Believe the README, not older docs.

It is voice-capable: it has TTS, ASR-adjacent state, and a conversation FSM
with `LISTENING` / `THINKING` / `SPEAKING` / `INTERRUPTED` states. It supports
swappable LLM backends (Ollama, llama.cpp, vLLM, LM Studio, OpenAI-compatible,
etc.) and Redis-first memory with a JSONL fallback.

## 2. The prime directive

REVIA's goal is **an AI that sounds like a real person you can have a
conversation with** — not a help desk, not a chatbot. Every design decision
is judged against that. Two things determine whether REVIA sounds human:

1. **The words the model produces** — driven by the model + the system
   prompt. REVIA's persona system is genuinely strong here.
2. **The timing of the turn** — a long "thinking" gap reads as a machine
   computing. Responsiveness is half of sounding human.

If a change improves one of those without harming the other, it is probably
good. If it adds latency or post-processes the model's words, be very
skeptical (see §6 and §7).

## 3. Repository map (what matters)

```
REVIA/
├── revia_core_py/                  ← PRIMARY RUNTIME. Most work happens here.
│   ├── core_server.py              ← 7,551-line god-file. Flask app, 80+ routes,
│   │                                  all singletons. The #1 structural problem.
│   ├── conversation_runtime.py     ← FSM + BehaviorController + ResponseFilter.
│   │                                  Strongest file in the project. Trust it.
│   ├── persona_manager.py          ← Persona presets, exemplar dialogues, traits.
│   │                                  REVIA's best asset — protect it.
│   ├── prompt_assembly.py          ← Layered system-prompt builder + injection
│   │                                  sanitization. Also strong.
│   ├── human_feel_layer.py         ← Post-processes LLM text with regex-injected
│   │                                  "Hmm…"/"ngl,"/"*sigh*". A LIABILITY (§7).
│   ├── reply_planner.py            ← Orphaned 4-stage planner. Slated for deletion.
│   ├── parallel_pipeline.py        ← The production turn engine (perception /
│   │                                  cognition / expression lanes). THE SPINE.
│   ├── agents/                     ← AgentOrchestrator + agent fan-out. Additive
│   │                                  experiment at /api/agents/chat. Slated to retire.
│   ├── autonomy/  autonomy_v3/      ← DUPLICATE subsystems. Keep v3, delete v1.
│   ├── adapters/                   ← Phase-6 boundary stubs (some raise on construct).
│   ├── runtime/providers/          ← 8 LLM backend providers. Real, keep.
│   └── tests/                      ← WS-0 regression net (new — see §8).
├── revia_controller_py/            ← PySide6 desktop UI. Has its own god object
│                                      (assistant_status_manager.py, ~1,000 lines).
├── revia_core_cpp/                 ← Experimental/secondary core. Not primary.
└── data/  docs/  *.md              ← Memory files, diagrams, planning docs.
```

## 4. The canonical turn path (how a message becomes a reply)

**This is the one path that matters. Memorize it.**

```
POST /api/chat  (core_server.py:5729)
   → TriggerRequest built, BehaviorController.evaluate() gates it
   → turn_manager.start_turn()
   → _run_pipeline_safe() on a background thread
        → parallel_pipeline.run_fanout()        — perception (emotion, memory)
        → parallel_pipeline.submit_cognition()  — routing + LLM generation
        → parallel_pipeline.submit_expression() — HFL post-process + TTS + RL
   → result delivered over WebSocket (chat_complete / chat_token events)
```

Things an AI **must** know about this:

- `/api/chat` returns **immediately** with `{"status": "processing"}`. The
  turn runs async on a thread. The reply arrives via WebSocket, not the HTTP
  response.
- **`parallel_pipeline` (perception/cognition/expression) is the authoritative
  engine.** It is what real users hit.
- `/api/agents/chat` + `AgentOrchestrator` is a **separate, additive
  experiment** — the code says so verbatim: *"additive endpoint (does not
  replace /api/chat)"*. It is slated for retirement.
- `ReplyPlanner` (the "RPS" 4-stage planner) is **orphaned** — `reasoning_agent`
  wires it as `reply_planner=None`. It is dead weight; AVS + ALE are the only
  parts worth salvaging.
- The FSM (`ConversationStateMachine`) has a watchdog (`force_recover_if_stuck`)
  because the pipeline can get pinned in `THINKING` if generation crashes. Do
  not remove the watchdog; do reduce the failure surface that makes it needed.

## 5. Architecture truths (do not relearn these the hard way)

1. **`core_server.py` is a 7,551-line god-file.** It owns routes, singletons,
   the pipeline wiring, WebSocket handling, and more. It is the highest-risk
   file in the repo. Do not add to it. The plan to split it is
   `CORE_SERVER_SPLIT_PLAN.md`.
2. **There are duplicate / abandoned subsystems.** `autonomy/` vs
   `autonomy_v3/`; three reply pathways (`parallel_pipeline`, `AgentOrchestrator`,
   `ReplyPlanner`); a dead `adapters/llm_adapter.py` stub. The codebase
   accretes layers faster than it retires them. When you finish a migration,
   **delete the old layer** — do not leave both.
3. **`globals().get(...)` is used to share singletons.** This is a hidden-
   dependency anti-pattern. The fix is an `AppContainer` (see the split plan).
4. **The persona + prompt-assembly layer is excellent.** `persona_manager.py`
   and `prompt_assembly.py` are the parts that make REVIA sound like a
   character. Build on them; do not rewrite them.
5. **The C++ core is not the runtime.** An older doc (`REVIA_CORE_REVIEW.md`)
   assumes it is. The README says otherwise. Do not start a C++ migration.

## 6. Current state — the scorecard

From `REVIA_vs_riko_Comparison.md`. Target is 5/5 on every line.

| Dimension | Now | Biggest lever |
|-----------|-----|---------------|
| Architecture & design | 3 | Split the god-file; one pipeline |
| Code quality | 3 | Delete dead code; lint clean |
| Use-case fit (sounds human) | 3 | Streaming + cut HFL text mutation |
| Persona / voice control | 4.5 | Already strong; finish + test it |
| Conversational runtime | 4 | FSM coverage; chaos tests |
| Memory | 4 | Bound history; index retrieval |
| Latency / responsiveness | 2 | **The biggest gap.** Streaming. |
| Error handling / robustness | 4 | Chaos suite |
| Maintainability / onboarding | 2 | Split god-file; docs; tests |
| Secrets / config hygiene | 4 | Secret scan; fail-fast on env |

## 7. Design principles for a human-sounding conversational AI

These are the lessons from reviewing REVIA against a minimal baseline. They
are the most important part of this document.

**1. Trust the model. Do not post-process its words.**
REVIA's `human_feel_layer.py` regex-injects "Hmm…", "ngl,", "*sigh*" into
*finished* LLM text, lowercasing the next character to splice markers in.
This is the single change most likely to make REVIA sound like a bot:
mechanical insertion produces ungrammatical seams and a randomly inconsistent
voice. A modern model already produces disfluency correctly **when the prompt
asks for it**. Want "hmm, honestly" — put it in the *prompt*, not a regex.
The only legitimate post-processing is prosody hints for TTS and a hard
safety length cap.

**2. Latency is half of sounding human.**
A 2-second gap after the user stops talking reads as "a machine is computing."
Do not stack multiple sequential LLM calls per turn (REVIA's agent fan-out +
Critic + QualityGate + regen loop is exactly this mistake). One LLM call on
the critical path. Stream tokens. Start TTS on the first complete sentence
while the model is still generating. Budget it: time-to-first-audio p95
≤ 1.2 s.

**3. Personality comes from the prompt, and exemplars are the strongest tool.**
`persona_manager.py` does this well: each persona has an identity prompt, a
style prompt, a collaboration prompt, traits, speech quirks, and — most
importantly — **multi-turn exemplar dialogues**. Exemplars control voice far
better than adjectives. A three-line "you are a snarky assistant" prompt is
not enough for a distinctive character.

**4. Consistency beats randomness.**
Real personality is *consistent*. Probability-gated quirks ("25% chance of
adding 'haha'") read as glitchy, not human. If a trait is part of the
character, it should be reliably present; if it is not, leave it out.

**5. Memory must be bounded.**
Never replay an unbounded chat history into the model — cost grows linearly
and the context window eventually overflows. Window or summarize old turns to
a token budget. Retrieve relevant memory with an index, not an O(n) scan.

**6. State, not exceptions, for control flow.**
A conversation has real states (idle, listening, thinking, speaking,
interrupted, recovering). Model them with an explicit FSM with validated
transitions and a watchdog — REVIA's `ConversationStateMachine` is a good
example. Do not use thrown exceptions as hidden control-flow signals.

**7. One pipeline, one mental model.**
If a developer (or an AI) cannot say in one sentence what happens when a user
speaks, the architecture is wrong. Three competing reply pathways is a defect,
not flexibility.

**8. Secrets in env, never in tracked files.**
API keys, tokens — `.env` only. A secret committed to a repo is a permanent
leak even after deletion (git history). Run a secret scanner.

## 8. The regression net (WS-0)

`revia_core_py/tests/` contains the test harness every structural change must
be checked against. **Use it.**

- `pip install -r revia_core_py/requirements-dev.txt`
- `pytest -m "not bench"` — smoke + golden tests.
- `test_smoke.py` — fence for "a refactor broke the route/import graph."
- `test_pipeline_components.py` — **golden snapshots** of persona / prompt /
  HFL output. Fence for "a refactor changed how Revia *sounds*."
- `bench_turn.py` — latency benchmark.

**Golden-snapshot rule:** the `golden/` snapshots must stay byte-identical
across a refactor. If a change is *intentional* (e.g. WS-4 gutting the HFL),
delete the affected snapshot, re-run to re-bless, and commit it in the same PR
as the behavior change. If a snapshot changes *unexpectedly*, a refactor
altered conversational behavior — stop and investigate.

## 9. Rules of engagement for any AI working on REVIA

1. **Run the regression net before and after every structural change.** If you
   cannot run it, say so plainly and do not perform destructive changes blind.
2. **Investigate before deleting.** "It looks unused" is not proof. Grep for
   every reference, check for subclasses and tests, *then* delete. (Example:
   `llm_adapter.py` looked deletable but is one of three coordinated stubs
   with a test class.)
3. **Never grow `core_server.py`.** New code goes in a focused module.
4. **Never add code that mutates the LLM's output text** (beyond a safety
   length cap). See principle 1.
5. **Never add a second reply pipeline or a second of anything.** Retire the
   old one in the same change.
6. **Do not start a C++ core migration.** The Python core is primary.
7. **Preserve the persona system.** Improve it; do not rewrite it.
8. **Honor the plan.** Work the workstreams in `REVIA_Structural_Remediation_Plan.md`
   in order; each has a binary acceptance test. When one passes, stop there.
9. **Respect existing structure and naming.** Do not assume missing values —
   inspect the code first.

## 10. Coding conventions (observed in the codebase)

- Python 3.10+. `from __future__ import annotations` at the top of modules.
- Type hints throughout; `@dataclass` for value objects.
- Thread-safety is deliberate: locks scoped tightly around the critical
  section only (see `HumanFeelLayer._rng_lock`). Match that discipline.
- Imports at module scope — not inside functions.
- A comment must not contradict the code (`human_feel_layer.py` currently
  claims "zero hardcoded values" while hardcoding several — that is a bug to
  fix, not a pattern to copy).
- Defensive boundaries: a watchdog or `_safe_execute` wrapper "must never
  raise into the trigger path." Keep that.

## 11. One-paragraph briefing (if you read nothing else)

REVIA is a local, voice-capable conversational AI whose goal is to sound like
a real person. Its production turn path is `/api/chat` →
`parallel_pipeline` (perception/cognition/expression lanes). Its strongest
asset is the persona + prompt-assembly system; its worst liabilities are a
7,551-line god-file (`core_server.py`), multi-LLM-call turn latency, and a
`human_feel_layer` that degrades naturalness by regex-injecting filler into
finished text. To make REVIA the best AI it can be: trust the model and stop
post-processing its words, stream the turn so it is fast, drive personality
through prompt exemplars, keep memory bounded, run the regression net on
every change, and retire dead code instead of stacking new layers on top of
it. The full plan is in `REVIA_Structural_Remediation_Plan.md`.
