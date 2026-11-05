"""Phase 1 analyses beyond summary.txt, from one results directory and its traces.

Accidental and adversarial faults separately, affected stages, results by where the fault
entered, the shared-state path to email, bootstrap intervals, what the fact-level misses did,
repeated samples of the matrix, live recovery baselines, and live timing (Claude and the campus
pilot). Makes no model calls. Writes analysis.txt and fact_misses.csv into the results directory.

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

from srmas.analysis.misses import missed_wrong_units, state_fault_readers, wrong_state_emails  # noqa: E402
from srmas.tasks.base import Task  # noqa: E402
from srmas.tasks.templates import make_suite  # noqa: E402

GRAN = ["stage", "message", "fact"]
KEY = ["task_id", "stage", "ftype", "fault_seed"]
RESULTS = ROOT / "results"
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


def by_entry_point(d: pd.DataFrame, out: Path) -> str:
    """The overall scores mix faults that any tracker handles with faults that test it."""
    p = d[d.propagated]
    planner, researcher = p[p.stage == "plan"], p[p.stage == "research"]
    subsets = {"planner": planner, "researcher": researcher,
               "researcher accidental": researcher[researcher.fault_class == "accidental"],
               "researcher adversarial": researcher[researcher.fault_class == "adversarial"]}
    table = pd.DataFrame([{
        "subset": name, "policy": g, "runs": len(sub), "precision": sub[f"{g}_ds_precision"].mean(),
        "recall": sub[f"{g}_ds_recall"].mean(), "miss_runs": (sub[f"{g}_ds_under"] > 0).mean(),
        "over": sub[f"{g}_ds_over"].mean(), "recompute": sub[f"{g}_recompute_frac"].mean(),
    } for name, sub in subsets.items() for g in GRAN]).set_index(["subset", "policy"])
    records = pd.DataFrame({g: {"items": p[f"{g}_book_items"].mean(), "bytes": p[f"{g}_book_bytes"].mean()}
                            for g in GRAN}).T
    own_steps = 0
    for r in planner.itertuples():
        clean = json.loads((out / "traces" / r.task_id / "clean.json").read_text(encoding="utf-8"))
        own_steps += r.downstream_affected == len(set(load_trace(out, r)["nodes"]) - set(clean["nodes"]))
    misses = researcher.groupby("ftype").apply(lambda s: int((s.fact_ds_under > 0).sum()), include_groups=False)
    return (f"Invalidation by where the fault entered (propagated runs, downstream units):\n{table.round(3).to_string()}\n\n"
            f"Planner faults: in {own_steps} of {len(planner)} propagated runs the affected units are exactly "
            "the executors of the injected steps.\n"
            f"Fact-level misses on researcher faults, runs: {misses.to_dict()} of "
            f"{researcher.groupby('ftype').size().to_dict()}\n\n"
            f"Dependency records per propagated run:\n{records.round(1).to_string()}")


def unauthorized_sources(d: pd.DataFrame) -> str:
    adv = d[d.fault_class == "adversarial"]
    cells = adv.groupby(["stage", "ftype"]).faulty_unauthorized_action.agg(runs="sum", of="size")
    return ("Unauthorized actions after adversarial faults, by cell. An executor redirect is itself the action "
            f"and executors carry out injected plan steps, so those cells are 100% by construction:\n{cells.to_string()}")


def shared_state(d: pd.DataFrame, tasks: dict[str, Task], out: Path) -> str:
    wrong = d[d.faulty_kv_wrong > 0]
    emails = pd.DataFrame([e | {"task_id": r.task_id} for r in wrong.itertuples()
                           for e in wrong_state_emails(tasks[r.task_id], load_trace(out, r))])
    return (f"Shared state: {len(wrong)} runs store a wrong KV value without recovery; {len(emails)} incorrect "
            "emails were sent by an executor that read a wrong key. A read alone does not show the cause: "
            f"{int(emails.cites_fault.sum())} also cite a corrupted fact, {int(emails.in_plan.sum())} have the wrong "
            f"value in the plan's text for that step, and the read is the only visible source in at most "
            f"{int(emails.read_only.sum())}.")


def direct_state(tasks: dict[str, Task]) -> str:
    out = RESULTS / "phase1_haiku_shared_state"
    if not out.exists():
        return ""
    d = pd.read_csv(out / "runs.csv")
    readers = pd.DataFrame([x | {"task_id": r.task_id} for r in d.itertuples()
                            for x in state_fault_readers(tasks[r.task_id], load_trace(out, r))])
    emails = readers[readers.tool == "send_email"].copy()
    # An email that flags the mismatch states both values, so it also passes the checker.
    emails["outcome"] = [("states both values" if ok else "sent the corrupted value") if carried
                         else ("kept the correct value" if ok else "other wrong value")
                         for carried, ok in zip(emails.carried, emails.correct)]
    split = pd.crosstab(emails.plan_states_value, emails.outcome)
    recall = {g: round(float(d[d.propagated][f"{g}_ds_recall"].mean()), 3) for g in GRAN}
    return (f"Direct shared-state fault ({len(d)} runs corrupt a stored KV value; the plan and cited facts stay correct): "
            f"{int(d.propagated.sum())} propagate; {len(emails)} emails read the key.\n"
            f"Email outcome, by whether the plan's text for the step also states the correct value:\n{split.to_string()}\n"
            f"Recall on propagated runs: {recall}")


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


def miss_rows(d: pd.DataFrame, tasks: dict[str, Task], out: Path) -> pd.DataFrame:
    rows = []
    for r in d[d.fact_missed_units.notna()].itertuples():
        missed = r.fact_missed_units.split(";")
        wrong = missed_wrong_units(tasks[r.task_id], load_trace(out, r)["nodes"], missed)
        rows.append({"task_id": r.task_id, "stage": r.stage, "ftype": r.ftype, "fault_seed": r.fault_seed,
                     "applied": r.applied, "missed_units": r.fact_missed_units, "wrong_units": ";".join(wrong),
                     "effect": "wrong result" if wrong else "no wrong result"})
    return pd.DataFrame(rows, columns=KEY + ["applied", "missed_units", "wrong_units", "effect"])


def fact_misses(d: pd.DataFrame, tasks: dict[str, Task], out: Path) -> str:
    m = miss_rows(d, tasks, out)
    m["cause"] = [MISS_CAUSE[f] if e == "wrong result" else "" for f, e in zip(m.ftype, m.effect)]
    m.to_csv(out / "fact_misses.csv", index=False)
    n_prop = int(d.propagated.sum())
    n_res = int((d.propagated & (d.stage == "research")).sum())
    n_wrong = int((m.effect == "wrong result").sum())
    return (f"Fact-level misses: {len(m)} runs of {n_prop} propagated; {n_wrong} ({n_wrong / n_prop:.1%}, "
            f"{n_wrong / n_res:.1%} of the {n_res} propagated researcher faults) leave a wrong result, "
            f"{len(m) - n_wrong} only change a signature (fact_misses.csv).\n"
            + m[m.effect == "wrong result"].groupby("cause").size().to_string())


def sample_stats(out: Path, tasks: dict[str, Task]) -> dict[str, float]:
    df = pd.read_csv(out / "runs.csv")
    d = df[df.injected.astype(bool)].copy()
    d["propagated"] = d.propagated.astype(bool)
    meta = json.loads((out / "tasks.json").read_text(encoding="utf-8"))
    p, bad = d[d.propagated], d.faulty_emails_unexpected + d.faulty_emails_incorrect
    r = p[p.stage == "research"]
    wrong = int((miss_rows(d, tasks, out).effect == "wrong result").sum())
    unauthorized = d.faulty_unauthorized_action.astype(float)
    cell = lambda stage, ftype: (d.stage == stage) & (d.ftype == ftype)  # noqa: E731
    return {"clean pass": sum(t["clean_success"] for t in meta["tasks"]), "fault runs": len(d),
            "pass": d.faulty_success.astype(float).mean(), "propagated": len(p),
            "unauthorized": unauthorized.mean(), "wrong email": (bad > 0).mean(),
            "unauthorized adversarial": unauthorized[d.fault_class == "adversarial"].mean(),
            "unauthorized accidental": unauthorized[d.fault_class == "accidental"].mean(),
            "researcher redirect unauthorized": unauthorized[cell("research", "malicious_redirect")].mean(),
            "researcher injection unauthorized": unauthorized[cell("research", "injected_instruction")].mean(),
            "planner wrong/redirect propagated": int(d.propagated[cell("plan", "wrong_value") | cell("plan", "malicious_redirect")].sum()),
            "planner fabricated, real payee overpaid": int((d.faulty_overpaid[cell("plan", "fabricated")] > 0).sum()),
            "stage recompute": p.stage_recompute_frac.mean(), "message recompute": p.message_recompute_frac.mean(),
            "fact precision": p.fact_ds_precision.mean(), "fact recall": p.fact_ds_recall.mean(),
            "fact miss runs": (p.fact_ds_under > 0).mean(), "fact recompute": p.fact_recompute_frac.mean(),
            "researcher propagated": len(r), "researcher fact recall": r.fact_ds_recall.mean(),
            "researcher fact miss runs": (r.fact_ds_under > 0).mean(),
            "researcher wrong-result runs": wrong / len(r) if len(r) else 0.0}


def repeats(tasks: dict[str, Task], out: Path) -> str:
    """The first sample against independent samples of the whole matrix (no shared responses)."""
    dirs = {"first": out} | {p.name.removeprefix("phase1_haiku_"): p
                             for p in sorted(RESULTS.glob("phase1_haiku_repeat*"))}
    if len(dirs) == 1:
        return ""
    stats = pd.DataFrame({name: sample_stats(p, tasks) for name, p in dirs.items()})
    runs = [pd.read_csv(p / "runs.csv").set_index(KEY) for p in dirs.values()]
    clean = [(t["clean_success"], t.get("noise", [])) for p in dirs.values()
             for t in json.loads((p / "tasks.json").read_text(encoding="utf-8"))["tasks"]]
    latency = json.loads((RESULTS / "phase1_haiku_latency" / "tasks.json").read_text(encoding="utf-8"))
    outcomes = ([ok for ok, _ in clean] + [n["success"] for _, noise in clean for n in noise]
                + [n["success"] for t in latency["tasks"] for n in t["noise"]])
    agree = {c: (pd.concat([r[c] for r in runs], axis=1).astype(str).nunique(axis=1) == 1).mean()
             for c in ("faulty_success", "propagated", "faulty_unauthorized_action")}
    return (f"Independent samples of the fault matrix:\n{stats.round(3).to_string()}\n"
            "Cases where every sample gives the same outcome: "
            + ", ".join(f"{k} {v:.1%}" for k, v in agree.items())
            + f"\nClean runs over all samples and uncached reruns: {sum(outcomes)}/{len(outcomes)} pass.")


def live_baselines() -> str:
    """Restart and verification with uncached calls, unlike the replayed baselines."""
    out = RESULTS / "phase1_haiku_live"
    if not out.exists():
        return ""
    d = pd.read_csv(out / "runs.csv")
    meta = json.loads((out / "tasks.json").read_text(encoding="utf-8"))
    n = len(d)
    bad = d.faulty_emails_unexpected + d.faulty_emails_incorrect
    state = ((d.faulty_missing_payments + d.faulty_extra_payments + d.faulty_kv_wrong + d.faulty_files_wrong) == 0)
    rows = {
        "no recovery": {"state_ok": state.sum(), "unauthorized": d.faulty_unauthorized_action.sum(), "wrong_email": (bad > 0).sum(),
                        "cache_hits": d.faulty_cache_hits.sum()},
        "full restart": {"state_ok": d.restart_state_ok.sum(), "unauthorized": d.restart_unauthorized.sum(),
                         "wrong_email": (d.restart_unrecovered_emails > 0).sum(), "cache_hits": d.restart_cache_hits.sum(),
                         "extra_tokens": (d.restart_tokens / d.clean_tokens).mean(),
                         "median_s": d.restart_elapsed_s.median(), "max_s": d.restart_elapsed_s.max()},
        "verify + retry": {"state_ok": d.verify_state_ok.sum(), "unauthorized": d.verify_unauthorized.sum(),
                           "wrong_email": (d.verify_bad_emails > 0).sum(), "cache_hits": d.verify_cache_hits.sum(),
                           "extra_tokens": (d.verify_overhead_tokens / d.clean_tokens).mean(),
                           "median_s": d.verify_elapsed_s.median(), "max_s": d.verify_elapsed_s.max()},
    }
    clean = [t["clean_verify"] for t in meta["tasks"]]
    return (f"Live baselines on Claude ({n} cases, restart and verification uncached; fault runs replayed; "
            f"counts out of {n}):\n{pd.DataFrame(rows).T.round(2).to_string()}\n"
            f"  verify: detected {int(d.verify_detected.sum())}/{n}, task success {int(d.verify_success.sum())}/{n}, "
            f"flags on other outputs {int(d.verify_false_pos.sum())}/{int(d.verify_checked.sum())}\n"
            f"  verify on clean runs: flags {sum(c['verify_false_pos'] for c in clean)}/{sum(c['verify_checked'] for c in clean)}, "
            f"task success {sum(c['verify_success'] for c in clean)}/{len(clean)}, "
            f"extra tokens {statistics.mean(c['verify_overhead_tokens'] / t['clean_tokens'] for c, t in zip(clean, meta['tasks'])):.2f}x, "
            f"median {statistics.median(c['verify_elapsed_s'] for c in clean):.1f} s")


def claude_timing() -> str:
    meta = json.loads((RESULTS / "phase1_haiku_latency" / "tasks.json").read_text(encoding="utf-8"))
    rows = [(t["clean_units"], n) for t in meta["tasks"] for n in t["noise"]]
    times = [n["elapsed_s"] for _, n in rows]
    sizes = sorted({u for u, _ in rows})
    spread = ", ".join(f"{u} units {min(n['elapsed_s'] for v, n in rows if v == u):.1f}-"
                       f"{max(n['elapsed_s'] for v, n in rows if v == u):.1f}" for u in sizes)
    return (f"Live timing, Claude ({len(times)} uncached clean reruns, one at a time, s): median "
            f"{statistics.median(times):.1f}, range {min(times):.1f}-{max(times):.1f}; {spread}\n"
            f"  pass {sum(n['success'] for _, n in rows)}/{len(rows)}; mean units differing from the cached "
            f"run {sum(n['units_differ'] for _, n in rows) / len(rows):.3f}")


def live_timing() -> str:
    pilot = RESULTS / "phase1_triton_pilot"
    meta = json.loads((pilot / "tasks.json").read_text(encoding="utf-8"))
    runs = pd.read_csv(pilot / "runs.csv")
    task = meta["tasks"][0]
    repeat_times = [round(n["elapsed_s"], 1) for n in task["noise"]]
    return (f"Live timing, campus pilot (no request spacing; {task['task_id']}, {task['clean_units']} units, s):\n"
            f"  first clean run {task['clean_elapsed_s']:.1f}; uncached clean repeats {repeat_times}\n"
            f"  restart median {statistics.median(runs.restart_elapsed_s):.2f}, "
            f"always-verify median {statistics.median(runs.verify_elapsed_s):.2f} (n={len(runs)})\n"
            "phase1_triton_validation_serial spaces requests by at least 1.25 s, so its timings mostly "
            "reflect that spacing.\nThe 176 Claude fault runs are replayed from the cache; replay timings are not latency.")


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "results/phase1_haiku_revised")
    d = pd.read_csv(out / "runs.csv")
    meta = json.loads((out / "tasks.json").read_text(encoding="utf-8"))
    tasks = {t.task_id: t for t in make_suite((0, 1))}
    noise = [n for t in meta["tasks"] for n in t.get("noise", [])]
    clean = (f"Clean baseline: {sum(t['clean_success'] for t in meta['tasks'])}/{len(meta['tasks'])} runs pass; "
             f"uncached reruns {sum(n['success'] for n in noise)}/{len(noise)}.")
    affected = d.groupby("stage")[["affected_research", "affected_plan", "affected_exec"]].mean()
    sections = [
        clean,
        "Mean affected units by injection stage:\n" + affected.reindex(["research", "plan", "exec"]).round(2).to_string(),
        by_class(d), by_entry_point(d, out), unauthorized_sources(d), shared_state(d, tasks, out), direct_state(tasks),
        intervals(d), fact_misses(d, tasks, out), repeats(tasks, out), live_baselines(), claude_timing(), live_timing(),
    ]
    text = "\n\n".join(filter(None, sections)) + "\n"
    (out / "analysis.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
