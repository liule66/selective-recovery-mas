"""Copy the output of run_phase1_live.sh into results/. Measured values are not changed.

Rewrites two fields of each tasks.json, because the run's own output and cache paths were
temporary and outside the repository, and adds a note on what is live. Records both in the file's
`imported` field. Drops what replay regenerates: restart traces of the repeats, and cached
responses the committed cache already holds.

    python experiments/import_live_results.py /absolute/output/dir
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"
PRIMARY_CACHE = RESULTS / "phase1_haiku_revised" / "response_cache"
IMPORTED = {"rewritten": ["config.out", "config.cache_dir"], "added": ["note"],
            "reason": "the run's output and cache paths were temporary and outside the repository",
            "by": "experiments/import_live_results.py"}
REPEAT_NOTE = ("An independent sample of the fault matrix: a fresh response cache shares no responses with "
               "results/phase1_haiku_revised. Always-verify is skipped. Restart replays this sample's clean run, "
               "so its traces are not kept.")
STATE_NOTE = ("Stored KV values corrupted directly (state_value), on the recorded clean runs, so the fault leaves the "
              "plan and cited facts unchanged. response_cache holds only the responses new to this experiment; "
              "replaying needs them together with results/phase1_haiku_revised/response_cache.")
LIVE_NOTE = ("Fault runs are replayed from results/phase1_haiku_revised/response_cache. Clean-run verification, full "
             "restart and always-verify use uncached calls, and a flagged output gets one real retry. Four tasks ran "
             "at a time.")
STEPS = (("repeat1", "phase1_haiku_repeat1", REPEAT_NOTE), ("repeat2", "phase1_haiku_repeat2", REPEAT_NOTE),
         ("shared_state", "phase1_haiku_shared_state", STATE_NOTE), ("live", "phase1_haiku_live", LIVE_NOTE))


def copy_tree(src: Path, dst: Path, keep_restart: bool) -> None:
    for f in sorted(src.rglob("*")):
        if f.is_file() and f.name != "active_case.json" and (keep_restart or not f.name.endswith(".restart.json")):
            out = dst / f.relative_to(src)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, out)


def main() -> None:
    source = Path(sys.argv[1])
    primary = {p.name for p in PRIMARY_CACHE.iterdir()}
    for short, name, note in STEPS:
        src, dst = source / short, RESULTS / name
        if dst.exists():
            raise SystemExit(f"{dst} already exists; imported evidence is not overwritten")
        dst.mkdir()
        shutil.copy2(src / "runs.csv", dst / "runs.csv")
        copy_tree(src / "traces", dst / "traces", keep_restart=short == "live")
        cache = f"results/{name}/response_cache"
        if short.startswith("repeat"):
            copy_tree(src / "response_cache", dst / "response_cache", keep_restart=True)
        elif short == "shared_state":
            (dst / "response_cache").mkdir()
            for f in (source / "shared_state_cache").iterdir():
                if f.name not in primary:
                    shutil.copy2(f, dst / "response_cache" / f.name)
        else:
            new = [f for f in (source / "live_cache").iterdir() if f.name not in primary]
            if new:
                raise SystemExit(f"the live run wrote {len(new)} new cached responses; its fault runs were not replays")
            cache = "results/phase1_haiku_revised/response_cache"
        meta = json.loads((src / "tasks.json").read_text(encoding="utf-8"))
        meta["config"]["out"], meta["config"]["cache_dir"] = f"results/{name}", cache
        meta["note"], meta["imported"] = note, IMPORTED
        (dst / "tasks.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
        print(f"{name}: failures={meta['failures']}, dirty={meta['working_tree_dirty']}, stats={meta['llm_stats']}")


if __name__ == "__main__":
    main()
