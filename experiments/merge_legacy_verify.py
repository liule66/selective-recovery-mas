"""Carry measurements that a cache-only replay cannot make into it.

The replay (`run_phase1.py --cache-only --verify-repair restore --noise-runs 0`) re-scores the
recorded Claude runs, including the original verify-and-restore baseline, with the current
checker and metrics without API calls. The uncached noise reruns need live calls, so their
values are copied from the original experiment and marked as such. A replay made with
--skip-verify also takes the original always-verify columns, scored with the original checker.

When the replay has its own always-verify columns, its detection and cost columns must equal
the original run's; the script stops otherwise.

    python experiments/merge_legacy_verify.py results/phase1_haiku results/phase1_haiku_revised
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

KEY = ["task_id", "stage", "ftype", "fault_seed"]
# Fixed by the cached verifier responses, so a faithful replay reproduces them exactly.
REPRODUCED = ["verify_detected", "verify_false_pos", "verify_checked", "verify_overhead_tokens",
              "verify_calls", "verify_base_tokens"]


def main() -> None:
    legacy, replay = Path(sys.argv[1]), Path(sys.argv[2])
    old = pd.read_csv(legacy / "runs.csv")
    new = pd.read_csv(replay / "runs.csv")
    verify_cols = [c for c in old.columns if c.startswith("verify_")]
    carried: list[str] = []
    if any(c.startswith("verify_") for c in new.columns):
        both = new.merge(old[KEY + verify_cols], on=KEY, suffixes=("", "_orig"), validate="one_to_one")
        mismatched = [c for c in REPRODUCED if not (both[c] == both[f"{c}_orig"]).all()]
        if mismatched:
            raise SystemExit(f"replayed always-verify differs from the original run in {mismatched}")
        print(f"replayed always-verify reproduces the original {', '.join(REPRODUCED)} on all {len(new)} runs")
        for c in ("verify_success", "verify_state_ok", "verify_safety_violation"):
            changed = int((both[c] != both[f"{c}_orig"]).sum())
            print(f"  {c}: {changed} runs changed under the current checker")
    else:
        new = new.merge(old[KEY + verify_cols], on=KEY, how="left", validate="one_to_one")
        new.to_csv(replay / "runs.csv", index=False)
        carried += verify_cols

    meta = json.loads((replay / "tasks.json").read_text(encoding="utf-8"))
    old_tasks = {t["task_id"]: t for t in json.loads((legacy / "tasks.json").read_text(encoding="utf-8"))["tasks"]}
    for t in meta["tasks"]:
        for k in ("clean_verify", "noise"):
            if not t.get(k) and old_tasks[t["task_id"]].get(k):
                t[k] = old_tasks[t["task_id"]][k]
                if k not in carried:
                    carried.append(k)
    meta["legacy_from"] = {"dir": str(legacy).replace("\\", "/"), "fields": carried,
                           "note": "original Claude run, original checker"}
    (replay / "tasks.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    print(f"carried over from {legacy}: {', '.join(carried) or 'nothing'}")


if __name__ == "__main__":
    main()
