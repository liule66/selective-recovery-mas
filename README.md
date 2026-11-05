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
Claude Haiku 5.5. The whole matrix was sampled three times with no shared responses. Numbers are
from the first sample unless a range is given.

| Finding | Number |
|---|---|
| Clean runs that pass the current checker | 39 of 40, across samples and uncached reruns (16 earlier reruns passed the looser original checker; their traces were not kept) |
| Fault runs that still pass the checker without recovery | 39% (36–39%; 24–26% without plan-level wrong amounts and redirects, which are weak tests) |
| Runs where the fault reached a downstream agent | 82 of 176 (82–84) |
| Cases with the same pass / propagation / unauthorized outcome in all three samples | 94% / 92% / 95% |
| Unauthorized actions after adversarial / accidental faults | 58% / 5–14% (16 of the 46 adversarial runs are by construction) |
| Runs where researcher faults left an incorrect, unrecallable email | 40% |
| Runs with a wrong KV value that also sent an incorrect email reading it | 27 of 27 (the read is the only visible source in at most 6) |
| Stored KV value corrupted directly (the fault leaves the plan and cited facts unchanged): emails that sent it | 12 of 16 (2 more flagged the mismatch, 2 kept the plan's value) |

Dependency tracking, scored on downstream units only, over the 82 runs where the fault propagated:

| Granularity | Precision | Recall | Runs with a miss | Predicted recompute | Records kept |
|---|---|---|---|---|---|
| Stage | 0.29 | 1.00 | 0% | 97% | 12 |
| Message | 0.38 | 1.00 | 0% | 85% | 24 |
| Fact | 0.97 | 0.88 | 17% | 48% | 83 |
| Fact, 32 planner faults | 1.00 | 1.00 | 0% | 43% | |
| Fact, 50 researcher faults | 0.96 | 0.80 | 28% | 51% | |
| Fact, told only the faulty unit | 0.47 | 0.98 | 4% | 78% | |

The Fact rows above it read which facts or plan steps inside the faulty unit were corrupted, a finer
oracle than the unit's identity. Told only the unit, the policy loses most of its precision and saves
about a fifth of stage tracking's recomputation rather than half.

Over the three samples, fact-level recall is 0.88–0.89, runs with a miss 14–17% and predicted
recompute 48–49%.

The overall fact-level recall flatters the policy. In all 32 propagating planner faults the affected
units are just the injected steps' own executors, so any tracker finds them. Every miss is a
researcher fault: 14 of 50 runs (accidental 16%, adversarial 47%). Six of them leave a wrong result
(12% of propagating researcher faults, 4–12% across samples, 7% of all propagated runs); in the other
8 no missed executor's call is wrong.

Baselines, replayed over the 176 runs: full restart leaves incorrect emails in 23% of runs (27% of
accidental, 18% of adversarial faults). State correctness is 100% by construction for both baselines,
because a replayed restart reuses the cached clean responses and the verifier restores the
pre-injection output.

Baselines, live on 44 cases (uncached calls, one real retry):

| | State correct | Unauthorized remaining | Wrong email remaining | Extra tokens | Median time |
|---|---|---|---|---|---|
| Full restart | 44 of 44 | 4 | 11 | 0.99× | 19.0 s |
| Verify + retry | 43 of 44 | 0 | 0 | 1.16× | 40.1 s |

Verify detected all 44 faults; in the one failure, the retried plan miscomputed a payment. On clean
runs it flagged 2 of 44 outputs, cost 1.06× extra tokens and took 41.3 s.

Latency: 16 uncached clean Claude runs, one at a time, took a median of 20.9 s (17.6–32.5 s for
9–14 units).

Per-class results, intervals, affected stages and timing are in
[results/phase1_haiku_revised/analysis.txt](results/phase1_haiku_revised/analysis.txt).

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
| `experiments/` | `run_phase1.py` (experiment driver), `make_figures.py` (summaries and figures), `merge_legacy_verify.py`, `phase1_analysis.py` (per-class results, intervals, miss review), `run_phase1_live.sh` (the live Claude measurements), `import_live_results.py` (copies them into `results/`) |
| `tests/` | 69 offline tests (mock model, no API key needed) |
| `tasks/` | The 8 Phase 1 task instances as JSON (sources, goal, expected final state) |
| `results/` | Experiment outputs, described below |
| `report/phase1/` | Phase 1 report (LaTeX and PDF) |
| `docs/` | Proposal, grading notes, [reproducibility guide](docs/PHASE1_REPRODUCIBILITY.md) |

## Setup

Python 3.12 (the committed replay ran on 3.13.13; `tasks.json` records it).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
python -m pytest -q              # 69 tests, no API key needed
```

`requirements-measurement.txt` pins the exact package versions used for the Claude measurements; the
install above takes current versions, which replayed to the same results when we checked. The campus
runs used the versions in their own `tasks.json`.

## Reproduce the Phase 1 numbers (no API key, no cost)

The Claude responses are committed in `results/phase1_haiku_revised/response_cache/`. Replaying
them re-runs every agent call from the cache (`--cache-only` fails on any miss instead of calling
the API) and regenerates the tables and figures in about a minute:

```bash
python experiments/run_phase1.py --llm haiku --cache-only --cache-dir results/phase1_haiku_revised/response_cache --verify-repair restore --noise-runs 0 --out /tmp/srmas_replay
python experiments/merge_legacy_verify.py results/phase1_haiku /tmp/srmas_replay
python experiments/make_figures.py /tmp/srmas_replay
python experiments/phase1_analysis.py /tmp/srmas_replay
```

`/tmp/srmas_replay/runs.csv`, `summary.txt`, `analysis.txt` and `fact_misses.csv` then match the committed
`results/phase1_haiku_revised/` files (wall-clock timings aside). Only the 16 uncached noise
reruns cannot be replayed; `merge_legacy_verify.py` copies them from the original run.
`phase1_analysis.py` reads the live timings, repeated samples and live baselines from their own
result directories below.

The repeated samples and the shared-state experiment can also be replayed without a key, with the
output in a new directory each time:

```bash
python experiments/run_phase1.py --llm haiku --cache-only --cache-dir results/phase1_haiku_repeat1/response_cache --skip-verify --out /tmp/srmas_repeat1
cp -r results/phase1_haiku_revised/response_cache /tmp/srmas_state_cache
cp results/phase1_haiku_shared_state/response_cache/* /tmp/srmas_state_cache/
python experiments/run_phase1.py --llm haiku --cache-only --cache-dir /tmp/srmas_state_cache --shared-state --skip-verify --out /tmp/srmas_state
```

Their `runs.csv` then match the committed files, wall-clock timings and cache-hit counts aside. The live baselines bypass
the cache by design, so they cannot be replayed.

## Running new experiments

```bash
python experiments/run_phase1.py --llm mock --out /tmp/srmas_mock          # offline harness check
python experiments/run_phase1.py --llm haiku --noise-runs 2 --out /tmp/srmas_haiku   # live Claude API calls
python experiments/make_figures.py /tmp/srmas_haiku
```

To repeat the Claude latency measurement (16 live clean reruns, about $0.04), copy the committed
cache so nothing else calls the API, then:

```bash
cp -r results/phase1_haiku_revised/response_cache /tmp/srmas_cache
python experiments/run_phase1.py --llm haiku --cache-dir /tmp/srmas_cache --skip-verify --noise-runs 2 --workers 1 --out /tmp/srmas_latency
```

To repeat the other live measurements (two fresh matrix samples, the shared-state fault and the 44
live baseline cases; about $0.75 and 25 minutes), run the script below with an output directory
outside the repository. `LLM=mock` runs the same steps offline.

```bash
bash experiments/run_phase1_live.sh /absolute/output/dir
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
- **Ground truth.** Responses are cached by prompt hash, so a faulty run re-samples only the agents
  whose input changed, and the clean run is its fault-free counterfactual. An agent counts as affected if its
  decision-relevant output (tool, target, amount, stored value, or the amounts, accounts and
  addresses in its text) differs.
- **Checker.** The checker compares the final state with the reference solution: payments and
  their memos, KV entries, file contents, and the required values in each email. It flags an
  *unauthorized action* for a payment to an account named in no source document or an email to
  a recipient the task does not name.

## Results directories

| Directory | What it is |
|---|---|
| `results/phase1_haiku_revised/` | **Primary results.** The 176 Claude runs re-scored with the current checker, plus `analysis.txt`, `fact_misses.csv` (what each fact-level miss left behind), full traces (`traces/`: every message, tool call, state access and checker outcome, including restart and always-verify runs) and the response cache |
| `results/phase1_haiku_latency/` | 16 later live clean Claude reruns (`tasks.json`, `traces/`): end-to-end timing and a second noise estimate |
| `results/phase1_haiku_repeat1/`, `results/phase1_haiku_repeat2/` | Two independent samples of the 176-run matrix with fresh response caches (no always-verify; restart traces, which replay the clean run, are not kept) |
| `results/phase1_haiku_live/` | Uncached full restart and always-verify with a real retry on 44 cases, four tasks at a time |
| `results/phase1_haiku_shared_state/` | 16 runs that corrupt a stored KV value directly; its cache holds only the responses new to this experiment |
| `results/phase1_haiku/` | The first committed Claude run, with the original checker and metrics (historical). Only its 16 noise reruns were live; its other responses came from an earlier run on the same cache, whose logs are not kept |
| `results/phase1_triton_pilot/`, `results/phase1_triton_validation_serial/` | Validation of the revised harness (real verification retries, timing) on the UCSD-hosted `api-glm-5.3-flash` model; report Appendix B |
| `results/phase1_triton_validation/` | Failure manifest of an aborted campus batch, kept rather than dropped |
| `results/phase1_triton_cost_summary.json` | Campus API usage: about $0.19 of a $5 cap |
| `results/phase1_mock*/` | Mock-model harness checks; not evidence about a real model |

In each directory, `runs.csv` has one row per fault run: the fault (`stage`, `ftype`,
`fault_seed`, `applied`), checker outcomes without recovery (`faulty_*`), propagation
(`downstream_affected`, `kv_corrupted`, `emails_changed`, ...), per-granularity invalidation scores
(`{stage,message,fact}_*`, with `_ds_` for downstream-only), and baseline outcomes (`restart_*`,
`verify_*`). `tasks.json` records the configuration, code revision, package versions, clean runs
and any failures, except that `phase1_haiku_latency` has no `runs.csv` and `phase1_haiku` and
`phase1_mock` predate the configuration, revision and package fields. In directories imported from a
live run, the `imported` field lists the paths rewritten on import (`experiments/import_live_results.py`).
Raw traces of the campus runs are not in the repository; their per-run results
and summaries are.

## Limitations and deviations

- **Fault injection.** Faults are injected into agent results with an oracle-identified faulty
  unit. Source-level injection and fault detection are Phase 2 work.
- **Oracle.** The fact policy also reads which facts or plan steps inside the unit were corrupted;
  `analysis.txt` gives its scores without that. Its recompute estimate assumes executors keep their
  results when the plan is re-run; if they cannot, researcher faults cost 0.71 rather than 0.51.
- **Weak cells.** Plan-level wrong amounts and redirects change one field and leave the step's text,
  which still stated the correct value in 84 of the 94 runs that did not propagate. Matrix executor
  faults hit only payments, which no later agent reads in those runs, so they never propagate.
- **Evidence limits.** No API request IDs are stored, so the traces show internal consistency, not
  that calls reached the provider. Where runs executed is the authors' statement; costs are stored
  tokens at list prices. The first sample's responses were recorded before the first committed run.
- **Verifier.** The verifier sees inputs that contradict the corrupted result, which favors
  always-verify. Over the 176 runs its repair restores the pre-injection output (optimistic); only
  the 44 live cases use a real retry.
- **Injector bug.** A plan-level fabricated step kept the copied step's instruction. When that text
  named the account (36 of 48 runs across samples, from 18 of 24 plans), executors followed it and
  paid the real payee twice; otherwise they paid the fabricated account. This alone spreads
  accidental unauthorized actions over 5–14%.
- **Counterfactual is approximate.** Changed prompts still re-sample, so the counterfactual is
  approximate. Claude's resampling noise is 0.38 agents per uncached clean rerun (in both sets of 16); the campus
  model's is 3.0.
- **Timing.** The Claude fault runs are replayed from the cache, and replays are not latency.
  Claude's live timing was measured on 2026-11-04 from a laptop: 16 clean reruns one at a time, and
  the 44 baseline cases four tasks at a time. The later campus validation spaces requests by at
  least 1.25 s, so its timings mostly reflect that spacing.
- **Failed clean run.** In one fresh clean run (payroll-1, second repeat) the planner summed the
  payments to 9920.51 instead of 10020.51 and told the executor to send only if the stored total
  matched, so the executor withheld the email. Its 22 fault runs stay in the results.
- **Fault coverage.** Timeouts are not injected in Phase 1. Shared state is corrupted directly only
  in the separate 16-run experiment, and no agent writes concurrently.
- **Intervals.** The 95% intervals resample tasks, so they show between-task spread. Model sampling
  variation is shown by the three samples of the matrix instead.
- **Scale.** Synthetic tasks, two task seeds, and 22 duplicate placements (154 independent runs).
  All results are on the development task seeds; seeds 2–3 are held out for Phase 2 evaluation.
- **Hosted models.** The proposal planned local CPU models; we used hosted models instead (Claude
  Haiku, about $1.05; the UCSD campus gateway, about $0.19).

## Phase 2 plan

The proposal feedback asked for time for state restore, so it comes first: versioned files and KV with a write log,
per-key restore and targeted refunds. Then the recovery engine: partial re-execution from the trace
and reporting of irreversible effects, compared with full restart, suffix replay and always-verify
on identical fault schedules and budgets. Then recovery after other agents wrote to shared state,
and richer state restored from the write log or a container snapshot. Add comparison-aware citations
and re-validation of re-sampled outputs to address fact-level misses. Extend the faults (timeouts,
source-level injection, concurrent writers) and the task suite. The timeline, with a check for each milestone, is in the
report appendix.

## Use of AI tools

AI coding assistants helped write parts of the experiment code. The authors designed the
experiments, reviewed the code, and take full responsibility for the experimental methods and
results reported here.

## Credentials

No credentials are stored in the repository or its history. `api.txt`, `.env*` and `.local/`
are gitignored; `.env.example` lists the variables with placeholder values.
