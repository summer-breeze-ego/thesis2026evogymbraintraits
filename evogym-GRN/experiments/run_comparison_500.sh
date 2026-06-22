#!/bin/bash
# Original-GRN vs TF6/7-GRN comparison: 10 runs x 500 generations per condition.
#
# - "tf67"      (use_grn_brain_traits=1): continuous, GRN-derived (TF6/TF7) phase+amplitude.
# - "original"  (use_grn_brain_traits=0): fixed/binary phase (0/pi) + fixed amplitude (0.4).
#
# Population params match the final_200 experiment (pop=30, offspring=30, cube_face_size=4).
#
# Within each condition, all 10 runs are launched in parallel, each pinned to
# --evogym_num_workers=1 (10 runs x 1 worker = 10 cores, matching this machine's core count).
# Benchmarked: per-run generation time stays ~unchanged vs. running alone with the
# default auto-worker pool, so 10 runs finish in ~the time of 1 (~10x throughput).
# The two conditions run one after another (10 runs, then the other 10).
#
# Safe to interrupt/resume: basic_EA.py checkpoints per generation and recovers
# automatically on re-run.
#
# Usage: ./experiments/run_comparison_500.sh [>> tmp_out/thesis/run_comparison_500.log 2>&1 &]

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="/opt/miniconda3/envs/evogym/bin/python"

study_name="thesis"
num_generations=500
population_size=30
offspring_size=30
cube_face_size=4

for cond_flag in "tf67:1" "original:0"; do
  cond="${cond_flag%%:*}"
  flag="${cond_flag##*:}"
  exp_name="cmp500_${cond}"
  logdir="tmp_out/${study_name}/${exp_name}"
  mkdir -p "$logdir"

  echo ">> [$(date '+%Y-%m-%d %H:%M:%S')] Launching condition=${cond} (use_grn_brain_traits=${flag}): 10 runs in parallel (1 worker each)"

  pids=()
  for run in $(seq 1 10); do
    logfile="${logdir}/run_${run}.log"
    "$PYTHON" -u algorithms/basic_EA.py \
      --study_name "$study_name" \
      --experiment_name "$exp_name" \
      --run "$run" \
      --population_size "$population_size" \
      --offspring_size "$offspring_size" \
      --num_generations "$num_generations" \
      --cube_face_size "$cube_face_size" \
      --use_grn_brain_traits "$flag" \
      --evogym_num_workers 1 \
      --run_simulation 1 \
      >> "$logfile" 2>&1 &
    pids+=($!)
  done

  fail=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      fail=1
    fi
  done

  if [ $fail -ne 0 ]; then
    echo "!! one or more runs in condition=${cond} exited non-zero -- check logs in ${logdir}"
  else
    echo "<< [$(date '+%Y-%m-%d %H:%M:%S')] Finished condition=${cond} (all 10 runs)"
  fi
done

echo "All 20 runs finished."
