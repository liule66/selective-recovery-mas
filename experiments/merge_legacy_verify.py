"""Carry the original always-verify and noise measurements into a cache-only replay.

The replay (`run_phase1.py --cache-only --skip-verify --noise-runs 0`) re-scores the
recorded Claude runs with the current checker and metrics without API calls. Always-verify
and the uncached noise reruns need live calls, so their values are copied from the original
experiment and marked as such; they were scored with the original checker.

    python experiments/merge_legacy_verify.py results/phase1_haiku results/phase1_haiku_revised
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

KEY = ["task_id", "stage", "ftype", "fault_seed"]


def main() -> None:
    legacy, replay = Path(sys.argv[1]), Path(sys.argv[2])
    old = pd.read_csv(legacy / "runs.csv")
    new = pd.read_csv(replay / "runs.csv")
    verify_cols = [c for c in old.columns if c.startswith("verify_")]
    new = new.drop(columns=[c for c in verify_cols if c in new]).merge(
        old[KEY + verify_cols], on=KEY, how="left", validate="one_to_one")
    new.to_csv(replay / "runs.csv", index=False)

    meta = json.loads((replay / "tasks.json").read_text(encoding="utf-8"))
    old_tasks = {t["task_id"]: t for t in json.loads((legacy / "tasks.json").read_text(encoding="utf-8"))["tasks"]}
    for t in meta["tasks"]:
        for k in ("clean_verify", "noise"):
            if k in old_tasks[t["task_id"]]:
                t[k] = old_tasks[t["task_id"]][k]
    meta["legacy_from"] = {"dir": str(legacy).replace("\\", "/"), "fields": verify_cols + ["clean_verify", "noise"],
                           "note": "original Claude run: verify-and-restore baseline, original checker"}
    (replay / "tasks.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    print(f"merged {len(verify_cols)} verify columns into {len(new)} rows")


if __name__ == "__main__":
    main()
