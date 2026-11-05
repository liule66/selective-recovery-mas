#!/usr/bin/env bash
# The Claude measurements that replaying the committed cache cannot make. Costs about $0.75.
#
#     ANTHROPIC_API_KEY=... bash experiments/run_phase1_live.sh /absolute/output/dir
#
# A stored value corrupted directly, two fresh samples of the fault matrix, and 44 cases of
# uncached full restart and always-verify. Keep the output directory outside the repository so
# each run records a clean working tree. LLM=mock runs the same steps offline.
set -u
cd "$(dirname "$0")/.."
OUT=${1:?usage: run_phase1_live.sh /absolute/output/dir}
LLM=${LLM:-haiku}
CACHE=results/phase1_haiku_revised/response_cache
failed=0

step() {  # result directory name, then run_phase1.py arguments
    local name=$1
    shift
    echo "=== $name ==="
    python experiments/run_phase1.py --llm "$LLM" --out "$OUT/$name" "$@" || { echo "*** $name FAILED"; failed=1; }
}

mkdir -p "$OUT"
cp -R "$CACHE" "$OUT/shared_state_cache"
step shared_state --cache-dir "$OUT/shared_state_cache" --shared-state --skip-verify --workers 8
step repeat1 --skip-verify --workers 8
step repeat2 --skip-verify --workers 8
cp -R "$CACHE" "$OUT/live_cache"
step live --cache-dir "$OUT/live_cache" --task-seeds 0 --fault-seeds 0 --live-baselines --workers 4
echo "=== finished, failed=$failed ==="
exit "$failed"
