# REVIA regression net (WS-0)

This is the test harness from **Workstream 0** of `REVIA_Structural_Remediation_Plan.md`.
It exists so the structural rewrite (WS-1/WS-2 onward) can be done safely —
every change is checked against this net.

## Running it

From `revia_core_py/`:

```bash
pip install -r requirements-dev.txt        # one-time
pytest -m "not bench"                      # the gate (smoke + golden)
pytest -m bench                            # latency benchmark only
python tests/bench_turn.py                 # architecture-overhead bench (CLI)
python tests/bench_turn.py --url http://127.0.0.1:8123   # live server bench
```

## What is in here

| File | Purpose |
|------|---------|
| `test_smoke.py` | Imports the whole server, exercises `/api/status` and `/api/chat` via Flask's test client. The fence for "a refactor broke the route/import graph." |
| `test_pipeline_components.py` | Golden + behavior tests over the deterministic conversational core (persona normalization, prompt assembly, Human Feel Layer with a fixed seed). The fence for "a refactor changed how Revia *sounds*." |
| `bench_turn.py` | Latency benchmark. Separates REVIA's own per-turn overhead from model/network latency. WS-3 is judged against this. |
| `conftest.py` | Path setup + the `golden` snapshot helper + the `client` fixture. |
| `golden/` | Stored snapshots (created on first run). |

## Golden snapshots — how they work

A golden test serializes its output and compares it to a stored snapshot in
`golden/`.

* **First run:** the snapshot is created and the test *skips* with a notice.
  Review the file, commit it — from then on it is enforced.
* **Later runs:** any difference fails the test.
* **Intentional change** (e.g. WS-4 gutting the Human Feel Layer will change
  the `hfl_*` snapshots on purpose): delete the relevant file in `golden/`
  and re-run to re-bless it. Commit the new snapshot in the same PR as the
  behavior change so the diff is reviewable.

## How this maps to the remediation

* Run `pytest -m "not bench"` **after every structural move** in WS-1/WS-2.
  Green = the move preserved behavior.
* The `golden/hfl_*` snapshots are *expected* to change exactly once, in WS-4.
* WS-10 tightens this net: coverage gate, a real WS-3 latency gate, the
  chaos suite (WS-8), and folding in the legacy top-level `test_*.py` files.

## Known limitation

The Linux sandbox used to author this harness could not execute Python, so
these files have **not been run yet**. Treat the first local `pytest` run as
part of WS-0 acceptance: expect the golden tests to skip-and-create on run 1,
then pass on run 2. Report any real failure — that is signal, not noise.
