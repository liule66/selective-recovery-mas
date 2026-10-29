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
import platform
import subprocess
from importlib.metadata import version, PackageNotFoundError
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


def source_identity() -> dict[str, Any]:
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
        return {"code_revision": revision, "working_tree_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        stamp = ROOT / "SOURCE_REVISION.json"
        return json.loads(stamp.read_text()) if stamp.exists() else {"code_revision": "unknown"}


def run_task(task: Task, llm: LLM, fault_seeds: tuple[int, ...], raw_dir: Path,
             noise_runs: int, skip_verify: bool, live_baselines: bool = False,
             fault_types: tuple[str, ...] = ()) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tdir = raw_dir / task.task_id
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / "active_case.json").write_text(json.dumps({"phase": "clean_setup"}), encoding="utf-8")
    clean = run_workflow(task, llm)
    clean.dump(tdir / "clean.json")
    info: dict[str, Any] = {
        "task_id": task.task_id, "template": task.template,
        "clean_success": clean.check["success"], "clean_check": clean.check,
        "clean_tokens": clean.tokens, "clean_units": len(clean.trace.nodes),
        "clean_elapsed_s": clean.elapsed_s,
        "clean_cache_hits": sum(n.cached for n in clean.trace.nodes.values()),
    }
    if not skip_verify:
        info["clean_verify"] = always_verify(task, llm, None, use_cache=not live_baselines,
                                              trace_path=tdir / "clean.verify.json")

    # Sampling noise: clean reruns with the cache bypassed.
    noise = []
    for i in range(noise_runs):
        rerun = run_workflow(task, llm, use_cache=False)
        rerun.dump(tdir / f"clean.repeat-{i}.json")
        noise.append({"units_differ": len(actual_affected(rerun.trace, clean.trace)),
                      "success": rerun.check["success"], "elapsed_s": rerun.elapsed_s,
                      "tokens": rerun.tokens})
    info["noise"] = noise

    rows = []
    for spec in fault_matrix(fault_seeds):
        if fault_types and spec.ftype not in fault_types:
            continue
        (tdir / "active_case.json").write_text(json.dumps({
            "phase": "fault_case", "stage": spec.stage, "ftype": spec.ftype,
            "fault_seed": spec.seed}), encoding="utf-8")
        row: dict[str, Any] = {
            "task_id": task.task_id, "template": task.template, "stage": spec.stage,
            "ftype": spec.ftype, "fault_seed": spec.seed,
            "fault_class": "accidental" if spec.ftype in ACCIDENTAL else "adversarial",
            "clean_success": clean.check["success"],
            "clean_tokens": clean.tokens, "status": "ok",
        }
        faulty = run_workflow(task, llm, spec)
        row["applied"] = faulty.trace.origin.get("applied", "")
        row["injected"] = bool(faulty.trace.origin)
        if not row["injected"]:
            rows.append(row)
            continue
        faulty.dump(tdir / f"{spec.label}.json")
        row["faulty_elapsed_s"] = faulty.elapsed_s
        row["faulty_cache_hits"] = sum(n.cached for n in faulty.trace.nodes.values())
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
        row |= full_restart(task, llm, faulty, clean, use_cache=not live_baselines,
                            trace_path=tdir / f"{spec.label}.restart.json")
        if not skip_verify:
            row |= always_verify(task, llm, spec, use_cache=not live_baselines,
                                 trace_path=tdir / f"{spec.label}.verify.json")
        rows.append(row)
        # Survives interruption and a later failure in this task.
        with (tdir / "completed.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"[{task.task_id}] {spec.label}: recorded", flush=True)
    return rows, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="mock", choices=["mock", "haiku", "triton"])
    ap.add_argument("--task-seeds", default="0,1")
    ap.add_argument("--fault-seeds", default="0,1")
    ap.add_argument("--templates", default="", help="comma list; default all")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--noise-runs", type=int, default=0)
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--live-baselines", action="store_true", help="uncached restart/verification measurements")
    ap.add_argument("--fault-types", default="", help="optional comma-separated pilot subset")
    ap.add_argument("--cache-dir", default="", help="use a fresh directory for independent experiments")
    ap.add_argument("--budget-usd", type=float, default=5.0)
    ap.add_argument("--input-rate", type=float, help="verified gateway USD per million input tokens")
    ap.add_argument("--output-rate", type=float, help="verified gateway USD per million output tokens")
    args = ap.parse_args()

    task_seeds = tuple(int(s) for s in args.task_seeds.split(","))
    fault_seeds = tuple(int(s) for s in args.fault_seeds.split(","))
    tasks = make_suite(task_seeds)
    if args.templates:
        keep = set(args.templates.split(","))
        tasks = [t for t in tasks if t.template in keep]
    if not tasks or args.workers < 1:
        ap.error("Choose at least one valid task and a positive worker count")
    fault_types = tuple(filter(None, args.fault_types.split(",")))
    if fault_types and not set(fault_types) <= {s.ftype for s in fault_matrix(fault_seeds)}:
        ap.error("Unknown fault type")

    out = Path(args.out) if args.out else ROOT / "results" / f"phase1_{args.llm}_{time.time_ns()}"
    if out.exists() and any(out.iterdir()):
        ap.error("Output directory is not empty; use a new --out to preserve prior evidence")
    raw = out / "traces"
    out.mkdir(parents=True, exist_ok=True)
    tasks_dir = out / "workload"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    for t in tasks:
        (tasks_dir / f"{t.task_id}.json").write_text(json.dumps(t.to_json(), indent=1), encoding="utf-8")

    kwargs = ({"budget_usd": args.budget_usd, "input_rate": args.input_rate,
               "output_rate": args.output_rate} if args.llm == "triton" else {})
    llm = make_llm(args.llm, cache_dir=Path(args.cache_dir) if args.cache_dir else out / "response_cache", **kwargs)
    t0 = time.time()
    rows, infos, failures = [], [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_task, t, llm, fault_seeds, raw, args.noise_runs, args.skip_verify,
                            args.live_baselines, fault_types): t
                for t in tasks}
        for fut in as_completed(futs):
            t = futs[fut]
            try:
                r, info = fut.result()
            except Exception as exc:
                failure = {"task_id": t.task_id, "error_type": type(exc).__name__, "status": "failed"}
                failure["http_status"] = getattr(exc, "http_status", None)
                active = raw / t.task_id / "active_case.json"
                if active.exists():
                    failure.update(json.loads(active.read_text(encoding="utf-8")))
                failures.append(failure)
                (raw / t.task_id / "failure.json").write_text(json.dumps(failure), encoding="utf-8")
                # Keep completed cases from this task; never silently exclude failures.
                checkpoint = raw / t.task_id / "completed.jsonl"
                if checkpoint.exists():
                    rows += [json.loads(line) for line in checkpoint.read_text(encoding="utf-8").splitlines()]
                print(f"[{t.task_id}] FAILED ({type(exc).__name__}); see manifest", flush=True)
                continue
            rows += r
            infos.append(info)
            print(f"[{t.task_id}] done: {len(r)} fault runs, clean_success={info['clean_success']} "
                  f"({time.time() - t0:.0f}s, {llm.stats.as_dict()})", flush=True)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["task_id", "stage", "ftype", "fault_seed"])
    df.to_csv(out / "runs.csv", index=False)
    meta = {"llm": llm.name, "elapsed_s": round(time.time() - t0, 1), "llm_stats": llm.stats.as_dict(),
            "tasks": sorted(infos, key=lambda i: i["task_id"]), "failures": failures,
            "config": vars(args), "python": platform.python_version(),
            **source_identity(),
            "expected_fault_runs": len(tasks) * sum(not fault_types or s.ftype in fault_types for s in fault_matrix(fault_seeds)),
            "measurement_mode": "live_baselines" if args.live_baselines else "cached_replay",
            "client_min_interval_s": getattr(llm, "min_interval_s", 0.0),
            "verification_repair": "one_uncached_model_retry", "recompute_cost": "token_estimate"}
    meta["packages"] = {}
    for package in ("pandas", "matplotlib", "jsonschema", "anthropic"):
        try:
            meta["packages"][package] = version(package)
        except PackageNotFoundError:
            pass
    (out / "tasks.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    print(f"wrote {len(df)} rows to {out}")
    if failures or len(df) != meta["expected_fault_runs"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
