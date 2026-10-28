# Dependency-Aware Selective Recovery in Multi-Agent Workflows

CSE 291A (Fall 2026) course project. See [docs/proposal.txt](docs/proposal.txt).

When one agent's output in a multi-agent workflow turns out to be faulty (wrong, stale,
fabricated, or carrying an injected instruction), can we track dependencies between agent
results and recompute only the affected work, instead of restarting everything, while keeping
the task correct and safe?

## Setup

```bash
conda create -p D:/miniconda3/envs/srmas python=3.12
D:/miniconda3/envs/srmas/python.exe -m pip install --no-cache-dir -e .[dev]
```

Put the Anthropic API key in `ANTHROPIC_API_KEY` or in `api.txt` at the repo root (gitignored).

## Layout

| Path | Contents |
|---|---|
| `src/srmas/env/` | Simulated tools: file workspace, KV store (restorable), payment ledger (compensable), email outbox (irreversible) |
| `src/srmas/llm/` | LLM clients (Claude Haiku 5.5, deterministic mock) and a response cache |
| `src/srmas/agents/`, `src/srmas/workflow/` | Researcher → planner → executor workflow with fan-out |
| `src/srmas/trace/` | Provenance log and dependency graph at stage / message / fact granularity |
| `src/srmas/faults/` | Accidental and adversarial fault injection |
| `src/srmas/analysis/` | Fault propagation metrics and counterfactual replay |
| `src/srmas/baselines/` | Full restart and always-verify baselines |
| `tasks/` | Task suite with programmatic checkers |
| `experiments/` | Experiment drivers and configs |
| `report/` | Phase reports |

## Phase 1 results (Claude Haiku 5.5, 8 tasks × 22 faults)

Report: [report/phase1/main.pdf](report/phase1/main.pdf). Numbers: [results/phase1_haiku/summary.txt](results/phase1_haiku/summary.txt).

- Without recovery only 39% of fault runs pass; researcher faults change 2.3 downstream agent calls on average and leave a wrong (irreversible) email in 44% of runs.
- Dependency tracking: stage / message / fact granularity reach precision 0.24 / 0.44 / 0.86 and recall 1.00 / 1.00 / 0.96, recomputing 84% / 69% / 34% of a full run.
- Full restart repairs all restorable and compensable state but cannot recall wrong emails (24% of runs); always-verify catches every injected fault but costs 1.18× tokens on every run.

## Running

```bash
D:/miniconda3/envs/srmas/python.exe -m pytest
D:/miniconda3/envs/srmas/python.exe experiments/run_phase1.py --llm mock
D:/miniconda3/envs/srmas/python.exe experiments/run_phase1.py --llm haiku
```
