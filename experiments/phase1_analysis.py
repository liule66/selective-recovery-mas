"""Phase 1 analyses beyond summary.txt, from one results directory and its traces.

Accidental and adversarial faults separately, affected stages, the shared-state path to email,
bootstrap intervals, what the fact-level misses did, and live timing from the campus pilot.
Makes no model calls. Writes analysis.txt and fact_misses.csv into the results directory.

    python experiments/phase1_analysis.py results/phase1_haiku_revised
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from srmas.analysis.misses import call_correct, missed_wrong_units  # noqa: E402
from srmas.tasks.base import Task, _kv_match  # noqa: E402
from srmas.tasks.templates import make_suite  # noqa: E402

GRAN = ["stage", "message", "fact"]
# Read off the traces of the Claude runs whose misses left a wrong result.
MISS_CAUSE = {"wrong_value": "comparative dependency", "stale_value": "comparative dependency",
              "fabricated": "fabricated entity", "injected_instruction": "re-sampling",
              "malicious_redirect": "re-sampling"}


def load_trace(out: Path, r: Any) -> dict[str, Any]:
    path = out / "traces" / r.task_id / f"{r.stage}.{r.ftype}.{r.fault_seed}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def by_class(d: pd.DataFrame) -> str:
    bad = d.faulty_emails_unexpected + d.faulty_emails_incorrect
    state = (d.faulty_missing_payments + d.faulty_extra_payments + d.faulty_kv_wrong + d.faulty_files_wrong) == 0
    outcomes = pd.DataFrame({
        "fault_class": d.fault_class, "pass": d.faulty_success, "state_ok": state,
        "unauthorized": d.faulty_unauthorized_action, "wrong_email": bad > 0,
        "propagated": d.propagated, "restart_unauthorized": d.restart_unauthorized,
        "restart_wrong_email": d.restart_unrecovered_emails > 0,
        "verify_unauthorized": d.verify_unauthorized, "verify_wrong_email": d.verify_bad_emails > 0,
    }).groupby("fault_class").mean()
    p = d[d.propagated]
    gran = pd.DataFrame([{
        "fault_class": cls, "policy": g, "runs": len(sub), "precision": sub[f"{g}_ds_precision"].mean(),
        "recall": sub[f"{g}_ds_recall"].mean(), "miss_runs": (sub[f"{g}_ds_under"] > 0).mean(),
        "recompute": sub[f"{g}_recompute_frac"].mean(),
    } for cls, sub in p.groupby("fault_class") for g in GRAN]).set_index(["fault_class", "policy"])
    return (f"Outcomes by fault class (all runs):\n{outcomes.T.round(3).to_string()}\n\n"
            f"Granularity by fault class (propagated runs, downstream units):\n{gran.round(3).to_string()}")


def shared_state(d: pd.DataFrame, tasks: dict[str, Task], out: Path) -> str:
    wrong = d[d.faulty_kv_wrong > 0]
    path = 0
    for r in wrong.itertuples():
        trace, task = load_trace(out, r), tasks[r.task_id]
        kv = trace["environment"]["kv"]
        bad_keys = {f"kv:{k}" for k, v in task.expected.kv.items() if not _kv_match(kv.get(k), v)}
        path += any(n["output"].get("tool") == "send_email" and not call_correct(task, n["output"])
                    and bad_keys & {res for res, _ in n["reads"]}
                    for n in trace["nodes"].values() if n["stage"] == "exec")
    return (f"Shared state: {len(wrong)} runs store a wrong KV value without recovery; in {path} of them "
            "an incorrect email was sent by an executor that read a wrong key.")


def bootstrap_ci(sub: pd.DataFrame, values: pd.Series, n: int = 10000, seed: int = 0) -> np.ndarray:
    """95% interval of a pooled mean, resampling whole tasks."""
    g = pd.DataFrame({"task": sub.task_id.values, "v": values.astype(float).values}).groupby("task").v.agg(["sum", "count"])
    picks = np.random.default_rng(seed).integers(0, len(g), (n, len(g)))
    return np.percentile(g["sum"].values[picks].sum(1) / g["count"].values[picks].sum(1), [2.5, 97.5])


def intervals(d: pd.DataFrame) -> str:
    p = d[d.propagated]
    rows = [("fact recall (propagated)", p, p.fact_ds_recall), ("fact miss runs (propagated)", p, p.fact_ds_under > 0),
            ("fact recompute (propagated)", p, p.fact_recompute_frac), ("stage recompute (propagated)", p, p.stage_recompute_frac),
            ("pass without recovery (all)", d, d.faulty_success), ("unauthorized (all)", d, d.faulty_unauthorized_action)]
    lines = ["95% intervals, resampling tasks (between-task spread only, not model sampling):"]
    for name, sub, values in rows:
        lo, hi = bootstrap_ci(sub, values)
        lines.append(f"  {name}: {values.astype(float).mean():.3f} [{lo:.3f}, {hi:.3f}]")
    return "\n".join(lines)


def fact_misses(d: pd.DataFrame, tasks: dict[str, Task], out: Path) -> str:
    rows = []
    for r in d[d.fact_missed_units.notna()].itertuples():
        missed = r.fact_missed_units.split(";")
        wrong = missed_wrong_units(tasks[r.task_id], load_trace(out, r)["nodes"], missed)
        rows.append({"task_id": r.task_id, "stage": r.stage, "ftype": r.ftype, "fault_seed": r.fault_seed,
                     "applied": r.applied, "missed_units": r.fact_missed_units, "wrong_units": ";".join(wrong),
                     "effect": "wrong result" if wrong else "no wrong result",
                     "cause": MISS_CAUSE[r.ftype] if wrong else ""})
    m = pd.DataFrame(rows)
    m.to_csv(out / "fact_misses.csv", index=False)
    n_prop = int(d.propagated.sum())
    n_wrong = int((m.effect == "wrong result").sum())
    return (f"Fact-level misses: {len(m)} runs of {n_prop} propagated; {n_wrong} ({n_wrong / n_prop:.1%}) "
            f"leave a wrong result, {len(m) - n_wrong} only change a signature (fact_misses.csv).\n"
            + m[m.effect == "wrong result"].groupby("cause").size().to_string())


def live_timing() -> str:
    pilot = ROOT / "results" / "phase1_triton_pilot"
    meta = json.loads((pilot / "tasks.json").read_text(encoding="utf-8"))
    runs = pd.read_csv(pilot / "runs.csv")
    task = meta["tasks"][0]
    repeats = [round(n["elapsed_s"], 1) for n in task["noise"]]
    return (f"Live timing, campus pilot (no request spacing; {task['task_id']}, {task['clean_units']} units, s):\n"
            f"  first clean run {task['clean_elapsed_s']:.1f}; uncached clean repeats {repeats}\n"
            f"  restart median {statistics.median(runs.restart_elapsed_s):.2f}, "
            f"always-verify median {statistics.median(runs.verify_elapsed_s):.2f} (n={len(runs)})\n"
            "phase1_triton_validation_serial spaces requests by at least 1.25 s, so its timings mostly "
            "reflect that spacing.\nThe Claude run recorded no per-run times; its replay timings are not latency.")


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "results/phase1_haiku_revised")
    d = pd.read_csv(out / "runs.csv")
    meta = json.loads((out / "tasks.json").read_text(encoding="utf-8"))
    tasks = {t.task_id: t for t in make_suite((0, 1))}
    noise = [n for t in meta["tasks"] for n in t.get("noise", [])]
    clean = (f"Clean baseline: {sum(t['clean_success'] for t in meta['tasks'])}/{len(meta['tasks'])} runs pass; "
             f"uncached reruns {sum(n['success'] for n in noise)}/{len(noise)}.")
    affected = d.groupby("stage")[["affected_research", "affected_plan", "affected_exec"]].mean()
    text = "\n\n".join([
        clean,
        "Mean affected units by injection stage:\n" + affected.reindex(["research", "plan", "exec"]).round(2).to_string(),
        by_class(d), shared_state(d, tasks, out), intervals(d), fact_misses(d, tasks, out), live_timing(),
    ]) + "\n"
    (out / "analysis.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
