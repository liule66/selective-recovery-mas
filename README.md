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

## Running

```bash
D:/miniconda3/envs/srmas/python.exe -m pytest
D:/miniconda3/envs/srmas/python.exe experiments/run_phase1.py --llm mock
D:/miniconda3/envs/srmas/python.exe experiments/run_phase1.py --llm haiku
```
