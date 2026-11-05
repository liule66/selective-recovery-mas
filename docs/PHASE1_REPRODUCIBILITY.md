# Phase 1 measurement revision

The original Claude results in `results/phase1_haiku` are retained as a historical
experiment. They are not results of the revised verification implementation.
Selective recovery remains Phase 2 work. All stage/message/fact recomputation
fractions are predictions from logged tokens, not measured recovery savings.

`results/phase1_haiku_revised` re-scores those 176 runs with the current checker by
replaying the committed response cache: no API calls and no API key. Always-verify is
replayed with `--verify-repair restore`, the original experiment's repair (the
pre-injection output is restored and one retry's tokens are estimated), which needs no
new model calls; `merge_legacy_verify.py` stops unless the replay reproduces the original
detection, flag and token columns. Only the uncached noise reruns are copied from
`results/phase1_haiku`. See the README for the commands. `experiments/phase1_analysis.py`
derives `analysis.txt` and `fact_misses.csv` from those results and their traces, also without
model calls; a missed unit counts as a wrong result when its call fails the checker's
requirements for its target or it was dropped from the faulty run.
`results/phase1_haiku_latency` holds 16 later live clean reruns, for end-to-end timing and a
second noise estimate; repeating them needs an API key (see the README).
`experiments/run_phase1_live.sh` produced `results/phase1_haiku_repeat1`, `results/phase1_haiku_repeat2`
(independent samples of the matrix), `results/phase1_haiku_shared_state` (stored values corrupted
directly) and `results/phase1_haiku_live` (uncached restart and verification on 44 cases). Each
`tasks.json` records its code revision and a note on what is live and what is replayed.
`experiments/import_live_results.py` copied them in: it rewrites the temporary output and cache paths,
adds that note, and lists both in the file's `imported` field. It changes no measured value.

## Reproduce without API spending

Use Python 3.12 in an isolated environment and install `.[dev]`. Run:

```sh
python -m pytest -q
python experiments/run_phase1.py --llm mock --workers 1 --noise-runs 2 --live-baselines --out /tmp/srmas_mock_run
python experiments/make_figures.py /tmp/srmas_mock_run
```

Mock results check the harness and oracle-based fixtures. They are not evidence of
a real model's capabilities, attack resistance, latency, or recovery quality.
Every output directory must be new, and outside the repository, or `tasks.json` records a dirty tree.
`tasks.json` records the command options,
model identity, versions, code revision, dirty state, expected case count and failures.
Interrupted tasks retain completed case records in `traces/*/completed.jsonl`.
A nonzero experiment exit status or a nonempty failures list means incomplete data.

## Campus API pilot

Run in the course DataHub environment so the key stays in its existing environment
variables. Outside DataHub, set `OPENAI_BASE_URL` and `OPENAI_API_KEY` yourself;
never copy a key into a notebook cell, command line, log, or Git commit.
Set `SRMAS_MODEL` to a route returned by the approved key's model list.
Verify the route's current campus prices before running. The client requires both
prices in USD per million tokens and uses strict JSON Schema responses; unsupported
routes fail visibly instead of silently changing output constraints.

```sh
python experiments/run_phase1.py --llm triton --task-seeds 0 --fault-seeds 0 --templates procurement --fault-types wrong_value,malicious_redirect --workers 1 --noise-runs 2 --live-baselines --budget-usd 5 --input-rate 0.15 --output-rate 0.50 --out results/my_campus_pilot
python experiments/make_figures.py results/my_campus_pilot
```

The example rates were listed for `api-glm-5.3-flash` on 2026-10-29, not a promise
of future prices. Verify them at https://docs.tritonai.ucsd.edu/models.
The `.local/api-budget.json` ledger persists across runs from the same project
directory. Use one experiment process per ledger. Never reset it to evade a cap.
The client reserves a conservative request cost before sending, reconciles known
token usage, and retains reservations for uncertain failures. No automatic HTTP
retry is performed. This is a rate-based estimate, not an authoritative billing
statement; check actual campus usage as well. Stop if the rate or usage accounting
cannot be verified.
The campus client spaces live requests by at least 1.25 seconds by default.
Its timings include that client wait and local accounting as well as the network
request. The initial live pilot predates this limiter; later cached propagation
validation must not be compared to it as a serving-performance benchmark.

## What the comparisons mean

- The propagation pass uses a response cache so unchanged prompts reuse outputs.
  Changed prompts can still resample; this does not remove all stochastic confounding.
- `--live-baselines` bypasses the cache for restart and verification, and clean
  noise repeats always bypass it. Their wall times include gateway/network delays.
- Without that switch, baseline times are cached replay times and must not be
  described as live inference performance. Cache hits are recorded per run.
- By default a flagged result is regenerated by one actual uncached model call, as in
  `results/phase1_haiku_live`. The replayed `results/phase1_haiku_revised` uses
  `--verify-repair restore` instead. A verifier refusal flags the result.
  A single retry is not guaranteed to correct it; final correctness is checked.
  The legacy verify_false_pos field counts flags on non-injected outputs; a
  naturally erroneous model output can be flagged correctly, so this is an
  unadjudicated flag-rate proxy rather than a true detector false-positive rate.
- Full restart restores/compensates mutable state but retains all sent emails.
  `state_ok` excludes email; `success` and safety remain separate metrics.
- Restart extra cost divides restart tokens by the corresponding clean tokens.
  Verification extra cost includes verifier tokens and actual retry tokens.
- Accidental and adversarial cases are summarized separately. Report clean-capable
  tasks and stochastic repeats; do not call an ignored injection successful recovery.

## Submission evidence

Include `runs.csv`, `tasks.json`, generated summaries/figures, and the synthetic
`traces/` tree containing messages, outputs, dependencies, retry/verifier usage,
tool accesses, final simulated state and checker outcomes. Response caches may be
shared for deterministic replay only after checking for sensitive content.
Exclude `.local/`, `.env*`, `api.txt`, virtual environments and credentials.
The original Claude run's own raw traces are not in this repository. Its response
cache is, in `results/phase1_haiku_revised/response_cache/`, together with the replay
traces it reproduces (`traces/`, including the always-verify runs). Other large
trace/cache directories, such as the campus pilot's, are ignored by Git to keep reviews
small. They must still be supplied with the course submission as a separately reviewed
evidence archive; committing summary tables alone is not a complete artifact submission.

Keep pilot results separate from the original 176-run Claude matrix. A smaller
second-model pilot checks the revised implementation; it does not replace or
validate the original model's quantitative conclusions.
