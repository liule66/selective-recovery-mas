"""Phase 1 experiments: fault propagation, counterfactual confirmation, dependency
tracking at three granularities, and the full-restart / always-verify baselines.

    python experiments/run_phase1.py --llm mock
    python experiments/run_phase1.py --llm haiku --workers 8 --noise-runs 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from srmas.analysis.counterfactual import actual_affected, affected_pairs, score  # noqa: E402
from srmas.analysis.propagation import propagation  # noqa: E402
from srmas.baselines.baselines import always_verify, full_restart  # noqa: E402
from srmas.faults.injector import ACCIDENTAL, fault_matrix  # noqa: E402
from srmas.llm.base import LLM, make_llm  # noqa: E402
from srmas.tasks.base import Task  # noqa: E402
from srmas.tasks.templates import make_suite  # noqa: E402
from srmas.trace.deps import GRANULARITIES, bookkeeping, predict  # noqa: E402
from srmas.workflow.runner import run_workflow  # noqa: E402


def run_task(task: Task, llm: LLM, fault_seeds: tuple[int, ...], raw_dir: Path,
             noise_runs: int, skip_verify: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tdir = raw_dir / task.task_id
    tdir.mkdir(parents=True, exist_ok=True)
    clean = run_workflow(task, llm)
    clean.trace.dump(str(tdir / "clean.json"))
    info: dict[str, Any] = {
        "task_id": task.task_id, "template": task.template,
        "clean_success": clean.check["success"], "clean_check": clean.check,
        "clean_tokens": clean.tokens, "clean_units": len(clean.trace.nodes),
    }
    if not skip_verify:
        info["clean_verify"] = always_verify(task, llm, None)

    # Sampling noise: clean reruns with the cache bypassed.
    noise = []
    for _ in range(noise_runs):
        rerun = run_workflow(task, llm, use_cache=False)
        noise.append({"units_differ": len(actual_affected(rerun.trace, clean.trace)),
                      "success": rerun.check["success"]})
    info["noise"] = noise

    rows = []
    for spec in fault_matrix(fault_seeds):
        row: dict[str, Any] = {
            "task_id": task.task_id, "template": task.template, "stage": spec.stage,
            "ftype": spec.ftype, "fault_seed": spec.seed,
            "fault_class": "accidental" if spec.ftype in ACCIDENTAL else "adversarial",
            "clean_success": clean.check["success"],
        }
        faulty = run_workflow(task, llm, spec)
        row["applied"] = faulty.trace.origin.get("applied", "")
        row["injected"] = bool(faulty.trace.origin)
        if not row["injected"]:
            rows.append(row)
            continue
        faulty.trace.dump(str(tdir / f"{spec.label}.json"))
        row |= {f"faulty_{k}": v for k, v in faulty.check.items()}
        row |= propagation(faulty, clean)

        full_tokens = clean.tokens
        for g in GRANULARITIES:
            pred = predict(faulty.trace, g)
            for k, v in score(pred, faulty.trace, clean.trace).items():
                row[f"{g}_{k}"] = v
            for k, v in bookkeeping(faulty.trace, g).items():
                row[f"{g}_book_{k}"] = v
            recompute = sum(faulty.trace.nodes[u].tokens for u in pred)
            row[f"{g}_recompute_frac"] = recompute / full_tokens if full_tokens else 0.0
        fact_pred = predict(faulty.trace, "fact")
        row["fact_missed_units"] = ";".join(
            f or c for f, c in affected_pairs(faulty.trace, clean.trace)
            if not (f in fact_pred if f else ("plan" in fact_pred or c in fact_pred)))
        row |= full_restart(task, llm, faulty, clean)
        if not skip_verify:
            row |= always_verify(task, llm, spec)
        rows.append(row)
    return rows, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="mock", choices=["mock", "haiku"])
    ap.add_argument("--task-seeds", default="0,1")
    ap.add_argument("--fault-seeds", default="0,1")
    ap.add_argument("--templates", default="", help="comma list; default all")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--noise-runs", type=int, default=0)
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    task_seeds = tuple(int(s) for s in args.task_seeds.split(","))
    fault_seeds = tuple(int(s) for s in args.fault_seeds.split(","))
    tasks = make_suite(task_seeds)
    if args.templates:
        keep = set(args.templates.split(","))
        tasks = [t for t in tasks if t.template in keep]

    out = Path(args.out) if args.out else ROOT / "results" / f"phase1_{args.llm}"
    raw = ROOT / "results" / "raw" / f"phase1_{args.llm}"
    out.mkdir(parents=True, exist_ok=True)
    tasks_dir = ROOT / "tasks"
    tasks_dir.mkdir(exist_ok=True)
    for t in tasks:
        (tasks_dir / f"{t.task_id}.json").write_text(json.dumps(t.to_json(), indent=1), encoding="utf-8")

    llm = make_llm(args.llm, cache_dir=ROOT / ".llm_cache" / args.llm)
    t0 = time.time()
    rows, infos = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_task, t, llm, fault_seeds, raw, args.noise_runs, args.skip_verify): t
                for t in tasks}
        for fut in as_completed(futs):
            t = futs[fut]
            try:
                r, info = fut.result()
            except Exception:
                print(f"[{t.task_id}] FAILED\n{traceback.format_exc()}", flush=True)
                continue
            rows += r
            infos.append(info)
            print(f"[{t.task_id}] done: {len(r)} fault runs, clean_success={info['clean_success']} "
                  f"({time.time() - t0:.0f}s, {llm.stats.as_dict()})", flush=True)

    df = pd.DataFrame(rows).sort_values(["task_id", "stage", "ftype", "fault_seed"])
    df.to_csv(out / "runs.csv", index=False)
    meta = {"llm": llm.name, "elapsed_s": round(time.time() - t0, 1), "llm_stats": llm.stats.as_dict(),
            "tasks": sorted(infos, key=lambda i: i["task_id"])}
    (out / "tasks.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    print(f"wrote {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
