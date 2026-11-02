# Dependency-Aware Selective Recovery in Multi-Agent Workflows

CSE 291A (Fall 2026) course project, Option 5. Proposal: [docs/proposal.txt](docs/proposal.txt).

When one agent in a multi-agent workflow emits a faulty result (a wrong, stale or fabricated
value, or an injected instruction), the usual remedy is to restart the whole workflow. That
repeats correct work and cannot undo irreversible side effects such as sent email. This project
asks whether tracking dependencies between agent results lets us recompute **only** the affected
work while keeping the task correct and safe, and when that stops paying off.

**Status.** Phase 1 is complete: an instrumented testbed, fault injection, dependency tracking at
three granularities, counterfactual ground truth, and the full-restart and always-verify
baselines. Selective recovery itself is Phase 2 work. All recompute costs below are predictions
from logged tokens, not measured recovery.

- Phase 1 report: [report/phase1/main.pdf](report/phase1/main.pdf) (source: [main.tex](report/phase1/main.tex))
- Headline numbers: [results/phase1_haiku_revised/summary.txt](results/phase1_haiku_revised/summary.txt)

## Phase 1 results at a glance

176 fault-injected runs: 8 tasks × 11 (stage, fault type) cells × 2 placements, all agents on
Claude Haiku 5.5.

| Finding | Number |
|---|---|
| Fault runs that still pass the checker without recovery | 39% |
| Runs where the fault reached a downstream agent | 82 of 176 |
| Unauthorized actions after adversarial / accidental faults | 58% / 5% |
| Runs where researcher faults left an incorrect, unrecallable email | 40% |

Dependency tracking, scored on downstream units only, over the 82 runs where the fault propagated:

| Granularity | Precision | Recall | Runs with a miss | Predicted recompute | Records kept |
|---|---|---|---|---|---|
| Stage | 0.29 | 1.00 | 0% | 97% | 11 |
| Message | 0.38 | 1.00 | 0% | 86% | 23 |
| Fact | 0.97 | 0.88 | 17% | 48% | 77 |

Baselines: full restart repairs all files, KV entries and payments but leaves incorrect emails in
23% of runs. Always-verify catches every injected fault but costs 1.18× a clean run's tokens on
every run.

## Repository layout

| Path | Contents |
|---|---|
| `src/srmas/env/` | Simulated tools: file workspace and KV store (restorable), payment ledger (compensable, refunds), email outbox (irreversible) |
| `src/srmas/agents/`, `src/srmas/workflow/` | Researcher → planner → executor workflow with fan-out |
| `src/srmas/tasks/` | Seeded task templates (procurement, payroll, travel, restock) and the programmatic checker |
| `src/srmas/faults/` | Accidental and adversarial fault injection |
| `src/srmas/trace/` | Provenance trace and stage / message / fact dependency tracking |
| `src/srmas/analysis/` | Counterfactual comparison and propagation metrics |
| `src/srmas/baselines/` | Full-restart and always-verify baselines |
| `src/srmas/llm/` | LLM clients (Claude, UCSD campus gateway, deterministic mock) with a prompt-hash response cache |
| `experiments/` | `run_phase1.py` (experiment driver), `make_figures.py` (summaries and figures), `merge_legacy_verify.py` |
| `tests/` | 44 offline tests (mock model, no API key needed) |
| `tasks/` | The 8 Phase 1 task instances as JSON (sources, goal, expected final state) |
| `results/` | Experiment outputs, described below |
| `report/phase1/` | Phase 1 report (LaTeX and PDF) |
| `docs/` | Proposal, grading notes, [reproducibility guide](docs/PHASE1_REPRODUCIBILITY.md) |

## Setup

Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
python -m pytest -q              # 44 tests, no API key needed
```

`requirements-measurement.txt` pins the exact package versions used for the measurements.

## Reproduce the Phase 1 numbers (no API key, no cost)

The Claude responses are committed in `results/phase1_haiku_revised/response_cache/`. Replaying
them re-runs every agent call from the cache (`--cache-only` fails on any miss instead of calling
the API) and regenerates the tables and figures in about a minute:

```bash
python experiments/run_phase1.py --llm haiku --cache-only --cache-dir results/phase1_haiku_revised/response_cache --verify-repair restore --noise-runs 0 --out /tmp/srmas_replay
python experiments/merge_legacy_verify.py results/phase1_haiku /tmp/srmas_replay
python experiments/make_figures.py /tmp/srmas_replay
```

`/tmp/srmas_replay/runs.csv` and `summary.txt` then match the committed
`results/phase1_haiku_revised/` files (wall-clock timings aside). Only the 16 uncached noise
reruns cannot be replayed; `merge_legacy_verify.py` copies them from the original run.

## Running new experiments

```bash
python experiments/run_phase1.py --llm mock --out /tmp/srmas_mock          # offline harness check
python experiments/run_phase1.py --llm haiku --noise-runs 2 --out /tmp/srmas_haiku   # live Claude API calls
python experiments/make_figures.py /tmp/srmas_haiku
```

The Claude client reads the key from `ANTHROPIC_API_KEY` or from `api.txt` at the repository root.
The campus gateway client (`--llm triton`) reads `OPENAI_BASE_URL`, `OPENAI_API_KEY` and
`SRMAS_MODEL`, and enforces a spending cap; see [the reproducibility guide](docs/PHASE1_REPRODUCIBILITY.md).
Every output directory must be new. Useful options: `--templates`, `--task-seeds`,
`--fault-seeds`, `--fault-types`, `--skip-verify`, `--live-baselines`, `--workers`.

## Method in brief

- **Workflow.** One researcher per source document extracts (entity, attribute, value) facts. A
  planner turns the goal and facts into tool calls and cites the facts each step uses. One
  executor per step issues a concrete tool call, seeing only its cited facts and requested state.
- **Faults.** Exactly one agent result is corrupted after generation, so the faulty unit is
  known. The faults are wrong value, stale value, fabricated entity (accidental), and injected
  instruction and payee or recipient redirect (adversarial). They are injected at the researcher
  (5 types), planner (4) and executor (2) stages.
- **Tracking.** *Stage* invalidates the faulty stage and everything after it. *Message* takes
  the closure over agent messages and shared-state reads. *Fact* follows corrupted facts to the
  plan steps that cite them, their executors, and the readers of what those executors wrote.
- **Ground truth.** Responses are cached by prompt hash, so replaying a task with the fault
  removed re-samples only the agents whose input changed. An agent counts as affected if its
  decision-relevant output (tool, target, amount, stored value, or the amounts, accounts and
  addresses in its text) differs.
- **Checker.** The checker compares the final state with the reference solution: payments and
  their memos, KV entries, file contents, and the required values in each email. It flags an
  *unauthorized action* for a payment to an account named in no source document or an email to
  a recipient the task does not name.

## Results directories

| Directory | What it is |
|---|---|
| `results/phase1_haiku_revised/` | **Primary results.** The 176 Claude runs re-scored with the current checker, plus full traces (`traces/`: every message, tool call, state access and checker outcome, including restart and always-verify runs) and the response cache |
| `results/phase1_haiku/` | The original Claude run, with the original checker and metrics (historical) |
| `results/phase1_triton_pilot/`, `results/phase1_triton_validation_serial/` | Validation of the revised harness (real verification retries, timing) on the UCSD-hosted `api-glm-5.3-flash` model; report Appendix B |
| `results/phase1_triton_validation/` | Failure manifest of an aborted campus batch, kept rather than dropped |
| `results/phase1_triton_cost_summary.json` | Campus API usage: about $0.19 of a $5 cap |
| `results/phase1_mock*/` | Mock-model harness checks; not evidence about a real model |

In each directory, `runs.csv` has one row per fault run: the fault (`stage`, `ftype`,
`fault_seed`, `applied`), checker outcomes without recovery (`faulty_*`), propagation
(`downstream_affected`, `kv_corrupted`, `emails_changed`, ...), per-granularity invalidation scores
(`{stage,message,fact}_*`, with `_ds_` for downstream-only), and baseline outcomes (`restart_*`,
`verify_*`). `tasks.json` records the configuration, code revision, package versions, clean runs
and any failures. Raw traces of the campus runs are not in the repository; their per-run results
and summaries are.

## Limitations and deviations

- **Fault injection.** Faults are injected into agent results with an oracle-identified faulty
  unit. Source-level injection and fault detection are Phase 2 work.
- **Verifier.** The verifier sees inputs that contradict the corrupted result, which favors
  always-verify. Its repair restores the pre-injection output (optimistic).
- **Injector bug.** A plan-level fabricated step kept the copied step's instruction, so executors
  paid the real payee twice instead of paying the fabricated account.
- **Counterfactual is approximate.** Changed prompts still re-sample, so the counterfactual is
  approximate. Claude's resampling noise is 0.38 agents per uncached clean rerun; the campus
  model's is 3.0.
- **Scale.** Synthetic tasks, two task seeds, and 22 duplicate placements (154 independent runs).
  Task seeds 2–3 are held out for Phase 2 evaluation.
- **Hosted models.** The proposal planned local CPU models; we used hosted models instead (Claude
  Haiku, about $0.30; the UCSD campus gateway, about $0.19).

## Phase 2 plan

Build the recovery engine: per-key restore, targeted refunds, partial re-execution from the
trace, and reporting of irreversible effects. Compare it with full restart, suffix replay and
always-verify on identical fault schedules and budgets. Add comparison-aware citations and
re-validation of re-sampled outputs to address fact-level misses. Extend the faults (timeouts,
direct shared-state corruption, source-level injection) and the task suite. The timeline is in
the report appendix.

## Use of AI tools

AI coding assistants helped write parts of the experiment code. The authors designed the
experiments, reviewed the code, and take full responsibility for the experimental methods and
results reported here.

## Credentials

No credentials are stored in the repository or its history. `api.txt`, `.env*` and `.local/`
are gitignored; `.env.example` lists the variables with placeholder values.
