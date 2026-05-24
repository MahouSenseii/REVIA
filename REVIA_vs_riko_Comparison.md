# REVIA vs. riko_project — Strict Code & Use Comparison

**Date:** 2026-05-23
**Reviewer brief:** Senior developer review. No sugar-coating. Focus on architecture & design, code quality, and use-case fit — specifically: *which codebase actually produces an AI that "sounds like a real person you can have a conversation with," and why.*

**Scope of code reviewed**

- **riko_project** — `server/main_chat.py` + everything under `server/process/` (`asr_func/asr_push_to_talk.py`, `llm_funcs/llm_scr.py`, `tts_func/sovits_ping.py`) plus `character_config.yaml`. ~250 lines total.
- **REVIA** — `revia_core_py/` core: `conversation_runtime.py`, `prompt_assembly.py`, `persona_manager.py`, `human_feel_layer.py`, `reply_planner.py`, `agents/orchestrator.py`, `adapters/llm_adapter.py`, plus structural inspection of `core_server.py` and the full module tree. ~100+ modules.

A note on honesty: these two projects are not the same weight class, and pretending otherwise would waste your time. riko is a 250-line hobby script. REVIA is a large application. So the interesting question is **not** "which is bigger" — it's "where is each one actually *right*, and where is REVIA's size working against its own goal." Both get hit hard below.

---

## 1. TL;DR verdict

**For the stated goal — an AI that sounds like a real person in conversation — neither codebase is clearly winning, and for opposite reasons.**

- **riko sounds human by getting out of the model's way.** One LLM call, fast turn, a thin system prompt. The naturalness you hear is 90% the model (`gpt-4.1-mini`) and the fact that riko adds almost nothing between the model output and your ears. That is a feature, not an accident.
- **REVIA has the *better persona engineering by far* but buries it under machinery that actively threatens naturalness.** `persona_manager.py` + `prompt_assembly.py` is genuinely strong work — multi-module personas, exemplar dialogues, layered system prompts. That is REVIA's real asset. But `human_feel_layer.py` post-processes finished LLM text with regex-injected "Hmm…", "ngl," and "*sigh*", and the multi-agent regen pipeline adds latency that breaks conversational flow. Both of those make REVIA sound *less* human, not more.

**Bottom line:** REVIA's *prompt layer* is the thing worth keeping and is clearly superior to riko. REVIA's *post-processing layer* (HFL) and its *turn latency* are liabilities that riko's simplicity avoids entirely. If you want REVIA to sound like a person, the fix is mostly **deletion**, not addition.

---

## 2. Scale & shape

| | riko_project (server/process) | REVIA (revia_core_py) |
|---|---|---|
| Files in scope | 4 Python files | 100+ Python modules |
| Total lines (process pipeline) | ~250 | tens of thousands |
| Largest single file | `llm_scr.py`, 91 lines | `core_server.py`, **7,551 lines** |
| LLM calls per turn | **1** | 2–8+ (agents fan-out + regen loop) |
| Tests | none | ~15 `test_*.py` files |
| Type hints | none | extensive, `from __future__ import annotations` everywhere |
| Concurrency | none (blocking loop) | thread pools, locks, futures, FSM |
| Persona definition | 3 lines of YAML | preset system with 5 personas, exemplars, quirks |
| Memory | raw history file replay | Redis-first + JSONL fallback, retrieval, episodic memory |
| Secrets handling | API key in tracked YAML | `.env`-based, env-aware |

---

## 3. Architecture & design

### 3.1 riko — strict assessment

**What is correct:** riko has a real architecture, small as it is. `main_chat.py` is the orchestrator; `process/` cleanly separates ASR / LLM / TTS into three modules with single-purpose functions (`record_and_transcribe`, `llm_response`, `sovits_gen`/`play_audio`). For a 250-line project that separation is appropriate and more than many hobby projects bother with. The data flow is linear and trivially understandable: record → transcribe → LLM → TTS → play. You can hold the entire system in your head.

**What is wrong:**

1. **`main_chat.py` is unstructured top-level script code.** The main loop is a bare `while True:` at module scope — no `main()` function, no `if __name__ == "__main__"` guard around the loop, no try/except anywhere in the cycle. **Any** exception — mic device error, network blip, SoVITS server down, malformed audio — kills the entire process. There is no recovery path. For something meant to run as a continuous companion, this is a real defect, not a nitpick.
2. **The pipeline has no abstraction boundary.** `llm_scr.py` reads `character_config.yaml` at *import time* (module top level). `sovits_ping.py` does the same. So the config path is hardcoded relative to CWD, the file must exist before import, and you cannot unit-test any module without the YAML present. Config loading belongs in a function, injected — not executed as an import side effect.
3. **No interface seam.** `llm_response` is hardwired to OpenAI's `responses.create`. Swapping models, adding a local model, or mocking for a test all require editing the function body. riko's TODO list even says "GUI / web interface" is wanted — that will be painful because nothing is behind an interface.

**Verdict:** Sound for its size, but it's a *script*, not an *application*. It has no error boundary and no seams. That ceiling is low.

### 3.2 REVIA — strict assessment

**What is genuinely good:**

- `conversation_runtime.py` is the strongest file in either project. The `ConversationStateMachine` has an explicit, validated transition table (`_ALLOWED_TRANSITIONS`), thread-safe state access, and a watchdog (`force_recover_if_stuck`) for stuck `THINKING`/`SPEAKING` states. The `BehaviorController` models cooldowns, startup grace periods, and auto-initiation gating. This is real engineering and it directly serves a *conversational* product (turn-taking, barge-in, pacing). riko has nothing like it and would need it the moment it grew.
- `persona_manager.py` and `prompt_assembly.py` are well-designed. Layered prompt assembly (identity + style + collaboration + behavioral + routing + memory + emotion), preset personas with multi-turn exemplar dialogues, profile normalization with deep-merge. This is the part of REVIA that *should* make it sound human, and it does the job well.
- Adapter boundaries exist on purpose (`adapters/`, `interfaces/`, `runtime/providers/` with eight provider implementations). Multi-backend support is real.

**What is wrong — and this is the core architectural problem:**

1. **`core_server.py` is 7,551 lines.** This is a god-file. It wires singletons, owns HTTP routes, builds the orchestrator, manages the parallel pipeline, holds locks, and registers `atexit` handlers. No single file in a maintainable codebase should be this large. It is the highest-risk file you own — every change touches it, every merge conflicts in it, and no one can hold it in their head. This alone caps REVIA's maintainability.
2. **There are at least three overlapping reply-generation pathways.** `core_server.py` imports and uses (a) `ParallelPipeline` with perception/cognition/expression *lanes*, (b) `AgentOrchestrator` with a Memory/Emotion/Intent/Reasoning/Voice/Hardware agent fan-out, and (c) `ReplyPlanner` (the "RPS" 4-stage planner) which is itself wrapped inside `ReasoningAgent`. So a turn goes: core_server → parallel pipeline lane → orchestrator → ReasoningAgent → ReplyPlanner → HFL/AVS/ALE → LLM. That is a *very* deep call graph for "answer one message." It is not obvious which path is authoritative, and the duplication (`autonomy/` **and** `autonomy_v3/` both exist as separate packages; `IntentAgent` exists **and** `ReplyPlanner._parse_intent` does its own regex intent parsing) is a strong sign of half-finished migrations.
3. **`adapters/llm_adapter.py` is an abstract stub that raises `NotImplementedError`** and is labeled "Phase 6 boundary stub … not migrated yet." So the codebase is mid-refactor with a dead boundary class sitting in the tree. Combined with `autonomy` vs `autonomy_v3`, this tells me REVIA accretes new layers faster than it retires old ones. That is technical debt compounding.
4. **The watchdog is a tell.** `force_recover_if_stuck` exists because the pipeline genuinely gets pinned in `THINKING` when something crashes mid-generation. That's a thoughtful band-aid — but the *need* for it is a symptom: a pipeline with this many stages and this many failure points will get stuck. riko's `while True` cannot get "stuck" because there is nothing to get stuck *in*. Complexity bought REVIA capability and bought it fragility at the same time.

**Verdict:** REVIA's architecture is genuinely strong in the conversation-runtime and persona layers and genuinely over-built everywhere else. The over-building is not free — it costs latency (Section 5) and it costs the one thing you cannot get back: a clear mental model of what happens when a user speaks.

---

## 4. Code quality

### 4.1 riko — strict assessment

- **No error handling** except one `try/except` in `sovits_gen` (which correctly catches and returns `None` — good). The LLM call and the entire main loop have none.
- **Dead imports:** `llm_scr.py` imports `gradio as gr` and never uses it. `main_chat.py` imports `time` and `os` and uses neither in the live path (`time` only in commented-out code).
- **Side-effecting list comprehension:** `[fp.unlink() for fp in Path("audio").glob("*.wav") if fp.is_file()]` builds a throwaway list purely for the `.unlink()` side effect. This is a recognized anti-pattern — it should be a `for` loop. It also **deletes the freshly generated output file** every iteration, including the one just played; fine here, but fragile.
- **Magic values inline:** `60 * samplerate` (60-second fixed buffer), `samplerate=44100`, `"http://127.0.0.1:9880/tts"`, `temperature=1`, `max_output_tokens=2048` — all hardcoded, none named.
- **No logging** — only `print()` with emoji. No log levels, no timestamps, nothing you can grep in production.
- **No type hints, no docstrings** beyond one.
- **Redundant assignment:** `conversation_recording = output_wav_path = Path("audio") / "conversation.wav"` — `output_wav_path` is immediately overwritten later. Confusing and pointless.

riko's code quality is "weekend hobby script" — readable because it's tiny, but it would not survive contact with real users.

### 4.2 REVIA — strict assessment

**Strong:** Consistent type hints, `@dataclass` usage, real docstrings, thread-safety done deliberately (locks scoped tightly — see `HumanFeelLayer._rng_lock` only wrapping the RNG call), defensive `try/except` at the right boundaries (`_safe_execute`, the watchdog "must never raise into the trigger path"). The orchestrator's audit-trail design (`OrchestratorOutput.to_dict`) is good observability practice. This is professional-grade code in the files reviewed.

**Strict criticisms:**

1. **The codebase contradicts its own claims.** `human_feel_layer.py` docstring says *"All thresholds consumed through ProfileEngine — zero hardcoded values."* That is false in the same file: `_VERBOSITY_MIN_WORDS = 25`, `_VERBOSITY_MAX_WORDS = 220`, the `emotion_map` pitch/rate/energy tuples, the `0.20`/`0.80` clause-window bounds, the `0.25` vocalization probability, and the `amp_map` values are all hardcoded literals. A comment that lies about the code is worse than no comment — it will mislead the next person.
2. **`import random` inside a function body.** `prompt_assembly.py._personality_error` does `import random` mid-function, and `_sanitize_profile_field` does `import re as _re` mid-function. Imports belong at module top. Minor individually, but it signals code added in a hurry without cleanup.
3. **Rule-based intent parsing presented as a real component.** `ReplyPlanner._parse_intent` is regex keyword matching (`_INTENT_GREET = re.compile(r"\b(hello|hi|hey...)\b")`). The docstring admits *"In production, replace with a proper NLU / embedding classifier."* Meanwhile a separate `IntentAgent` also exists. You are maintaining two intent systems and the cheaper one is shipped with a "replace me" note.
4. **Duplication / abandoned layers** (already noted): `autonomy/` vs `autonomy_v3/`, the dead `LlmAdapter` stub. Dead code is a quality defect — it costs reading time and creates "which one is real?" ambiguity on every visit.
5. **`core_server.py` at 7,551 lines** is itself the single biggest code-quality problem in the project. Everything about testability, review, and onboarding degrades because of it.

**Verdict:** REVIA's per-file craftsmanship is good-to-excellent. Its codebase-level hygiene — dead layers, contradictory comments, a 7.5k-line god-file, duplicated subsystems — is poor. The small files are written by someone who knows what they're doing; the project as a whole is not being *pruned*.

---

## 5. Use-case fit — "sounds like a real person you can converse with"

This is the question you actually asked. Breaking it into the things that determine it:

### 5.1 Raw output naturalness (the words the model produces)

This is mostly the **model + the system prompt**, and here REVIA wins decisively *on the prompt side*. Compare:

- **riko's entire personality:** three lines of YAML — *"You are a helpful assistant named Riko. You speak like a snarky anime girl. Always refer to the user as senpai."* That's it. It works only because `gpt-4.1-mini` is good enough to run with a thin prompt.
- **REVIA's persona layer:** `persona_manager.py` ships full personas with an `identity_prompt`, a `style_prompt` (*"Sound like a real person, not a product. Lead with your actual reaction…"*), a `collaboration_prompt`, trait lists, speech quirks, and — critically — **multi-turn exemplar dialogues** (see the `diana_inspired` preset's `technical examples`, `relational examples`, `introspection examples`). Exemplars are the single most effective prompt technique for controlling voice. This is real, competent prompt engineering and it is the best thing in REVIA.

**On prompt quality, REVIA is far ahead.** If riko's snarky-anime-girl voice feels natural to you, REVIA's persona system can hit any voice you want, far more reliably.

### 5.2 The Human Feel Layer — REVIA's self-inflicted wound

`human_feel_layer.py` takes the **finished** LLM reply and mutates it with regex:

- Prepends `"Hmm… "` / `"Let me think… "` with some probability.
- Injects `"actually, "`, `"ngl, "`, `"honestly, "` at a regex-detected clause boundary, mid-sentence.
- With 25% probability prepends or inserts `"haha"`, `"omg"`, `"*sigh*"`, `"..."` based on an emotion label, **lowercasing the next character** to splice it in.
- Injects "speech quirks" at sentence boundaries.

**This is the part of REVIA most likely to make it sound like a bot, not a person.** Reasons, bluntly:

1. **A modern LLM already produces disfluency far better than regex can.** If you want Revia to say "hmm, honestly," the *prompt* should ask for it — then it lands in grammatically correct, context-aware places. Bolting it on afterward means the marker has no idea what the sentence is about.
2. **Mechanical insertion produces ungrammatical seams.** Inserting `"ngl, "` after a comma and lowercasing the next word will regularly produce text no human would write. The `inject_vocalizations` splice (`f"{marker}, {text[0].lower()}{text[1:]}"`) is brittle by construction.
3. **It is random, so it is inconsistent.** Probability-gated quirks mean the same Revia sounds different turn to turn for no reason the user can perceive. Real personality is *consistent*; randomness reads as glitchy.
4. **It fights the persona layer.** You spent real effort in `persona_manager.py` getting the model to produce a specific voice — then HFL overwrites that voice with dice rolls.

riko has no HFL. That is why riko's raw output, whatever else is wrong with riko, does not have this specific failure mode. **This is a case where REVIA's extra code makes the product worse.**

### 5.3 Latency and turn-taking — the other half of "feeling human"

A conversation feels human partly through *timing*. Long gaps after you stop talking break the illusion harder than slightly-off wording does.

- **riko:** one LLM call, `stream=False`, then TTS, then playback. The LLM call itself is a single round trip — fast. The real latency cost in riko is non-streaming TTS + full playback before you can speak again, and a fixed 60-second record buffer. Turn latency is mediocre but the *thinking* part is quick.
- **REVIA:** a turn fans out 4–6 agents in parallel (good — parallel), but then runs **post-agents** (Critic), a **QualityGate**, and **up to `max_regen` regenerations**, where each regen re-runs the ReasoningAgent *and* the Critic. The `ReplyPlanner` separately loops up to `regen_patience + 1` LLM candidates with AVS scoring between them. **Worst case is many sequential LLM calls for one spoken reply.** For a text chat that's acceptable. For a *voice companion* — and REVIA's own FSM has `LISTENING`/`THINKING`/`SPEAKING` states and a TTS backend, so voice is the intended mode — multi-second thinking gaps will read as "the bot is computing," not "a person is replying."

REVIA even acknowledges this tension implicitly: it has a whole `human_feel_layer` that prepends *"One sec —"* / *"Let me think…"*. That is a stall tactic to paper over latency the architecture creates. The honest fix is to reduce the latency, not to script the AI saying "one sec."

**Net on use-case fit:** riko delivers a fast, simple turn with a thin voice. REVIA delivers a richly-controlled voice through a pipeline that risks slow turns and then partially corrupts the voice on the way out. Neither is the finished article. The *good* version is REVIA's persona layer + riko's single-fast-call discipline.

### 5.4 Memory — both are flawed, REVIA much less so

- **riko's "memory" is broken at scale.** `llm_scr.py` loads the *entire* `chat_history.json`, appends the new turn, sends *all of it* to the API, and saves it back — every turn, forever. No truncation, no summarization, no token budget. Cost grows linearly with conversation length and the context window will eventually overflow and error. The README advertises "remembers your conversations" — technically true, but it's an unbounded list that will break a long-running companion. This is a genuine bug.
- **REVIA** has Redis-first persistence with JSONL fallback, a `memory_retriever`, episodic memory (`autonomy_v3/episodic_memory.py`), and a `MemoryAgent`. This is the right shape. The risk here is the *opposite* of riko's: complexity and the question of whether all of it is wired and tested. But architecturally REVIA is correct and riko is not.

---

## 6. Scorecard

Ratings are 1–5, judged strictly against "a maintainable app that sounds like a real person in conversation" — not against each other's size.

| Dimension | riko | REVIA | Notes |
|---|---|---|---|
| Architecture & design | 2 / 5 | 3 / 5 | riko: clean but a script with no error boundary. REVIA: excellent runtime/persona layers, dragged down by a 7.5k-line god-file and 3 overlapping reply paths. |
| Code quality | 2 / 5 | 3.5 / 5 | riko: dead imports, no logging, no tests, anti-patterns. REVIA: strong per-file craft, poor codebase hygiene (dead layers, contradictory comments). |
| Use-case fit (sounds human) | 3 / 5 | 3 / 5 | riko: fast + thin voice, naturalness borrowed from the model. REVIA: superb persona prompts, undermined by HFL post-processing and turn latency. **Tie, for opposite reasons.** |
| Persona / voice control | 1.5 / 5 | 4.5 / 5 | REVIA's clear, decisive win. Exemplar-driven personas vs. 3 lines of YAML. |
| Conversational runtime (turn-taking, barge-in, pacing) | 1 / 5 | 4 / 5 | REVIA's FSM + BehaviorController is real and correct. riko has none. |
| Memory | 1 / 5 | 4 / 5 | riko's unbounded history replay is a bug. REVIA's design is correct. |
| Latency / responsiveness | 3.5 / 5 | 2 / 5 | riko: one call. REVIA: multi-agent + regen loops risk slow turns. |
| Error handling / robustness | 1.5 / 5 | 4 / 5 | riko: one exception kills the loop. REVIA: watchdogs, filters, defensive boundaries. |
| Maintainability / onboarding | 3 / 5 | 2 / 5 | riko is tiny so it's easy; REVIA's god-file + duplicate subsystems make it hard despite good intentions. |
| Secrets / config hygiene | 1 / 5 | 4 / 5 | riko stores the API key in a tracked YAML. REVIA uses `.env`. |

---

## 7. What each project should take from the other

**REVIA should steal riko's discipline:**

1. **Trust the model. Cut the Human Feel Layer back to almost nothing.** Move disfluency, quirks, and tone into the *prompt* (you already have the persona system to do it). Keep HFL only for things a prompt genuinely can't do — and even then, prosody hints for TTS are fine; *text mutation* is the part to delete. This is the single highest-impact change for "sounds human."
2. **Collapse the reply pipeline to one authoritative path.** Pick parallel-pipeline *or* orchestrator-with-agents. Delete the loser. Delete `autonomy/` or `autonomy_v3/` — keep one. Delete the dead `LlmAdapter` stub or finish it.
3. **Break up `core_server.py`.** 7,551 lines must become route modules + a wiring/bootstrap module + service modules. There is a `CORE_SERVER_SPLIT_PLAN.md` already in the repo — execute it.
4. **Budget your turn latency.** Measure worst-case LLM calls per spoken turn. If a voice reply can trigger 5+ sequential model calls, cap regen at 1 for voice mode and make the Critic/QualityGate optional under a latency budget.

**riko should steal REVIA's structure:**

1. **Wrap the main loop in a `main()` with try/except** so one bad turn doesn't kill the process.
2. **Bound the history** — summarize or window `chat_history.json` to a token budget. This is a correctness fix, not a nicety.
3. **Move the API key to `.env`** and out of any tracked file.
4. **Adopt something like REVIA's `persona_manager`** — even a single extra `style` block and two exemplar turns would noticeably sharpen Riko's voice over the current 3-line prompt.
5. Load config in a function, not at import; add real logging; drop the dead `gradio` import.

---

## 8. Final answer to your question

> *"Which is better, since the goal is an AI that sounds like an actual person you can converse with?"*

There is no single winner, and a strict review has to say so plainly:

- **The best *voice* (word choice, personality, consistency) comes from REVIA's `persona_manager` + `prompt_assembly`.** That layer is well ahead of anything in riko and is the part of REVIA you should protect and build on.
- **The most *natural-feeling turn* today probably comes from riko**, not because riko is good, but because riko doesn't do the two things that hurt REVIA: it doesn't post-process the model's words, and it doesn't stack multiple LLM calls before answering.
- **REVIA's `human_feel_layer.py` is misnamed.** Regex-injecting "hmm" and "ngl" into finished text is the *least* human-feeling thing in either codebase. It should be cut down hard.

So: REVIA is the better *foundation* by a wide margin — runtime, persona, memory, robustness are all real. But "sounds like a person" is being *spent down* by HFL post-processing and pipeline latency. Make REVIA sound human by **subtracting** (gut the HFL text mutation, collapse the duplicate pipelines, split the god-file) rather than adding. riko is a useful mirror precisely because its simplicity accidentally avoids REVIA's two worst naturalness bugs — but riko itself has a broken memory model, no error handling, and a leaked-secret config, so it is not a base to build on.
