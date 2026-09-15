#!/bin/bash
##
## Runs basalt_vio with --online-loop-closure enabled across a set of EuRoC
## sequences and captures the [LOOP-CLOSURE SUMMARY] console line plus the
## three per-run CSV dumps (corrected_trajectory.csv, gt_aligned_to_corrected.csv,
## raw_trajectory.csv) for each one.
##
## This is deliberately a separate script from run_evaluations.sh: that script
## drives the OFFLINE mapper/pose-graph/pure-BA comparison path (basalt_mapper),
## a different code path from the live --online-loop-closure flag in vio.cpp.
## Mixing the two would conflate two unrelated accuracy questions.
##
## Env overrides (all have sane defaults matching run_evaluations.sh's layout):
##   DATASET_PATH   - directory containing one subfolder per EuRoC sequence
##   DATA_DIR       - directory containing euroc_eucm_calib.json / euroc_config.json
##   DATASETS       - space-separated sequence names to run (default: full EuRoC set)
##   OUT_DIR        - where per-sequence results/logs/CSVs are written
##
## Usage: ./run_loop_closure_eval.sh
##        DATASETS="V1_01_easy" ./run_loop_closure_eval.sh   # just one sequence

set -e

DATASET_PATH="${DATASET_PATH:-/data/euroc}"
DATA_DIR="${DATA_DIR:-${HOME}/.local/etc/basalt}"
DATASETS=(${DATASETS:-MH_01_easy MH_02_easy MH_03_medium MH_04_difficult MH_05_difficult V1_01_easy V1_02_medium V1_03_difficult V2_01_easy V2_02_medium})
OUT_DIR="${OUT_DIR:-loop_closure_eval_results}"

mkdir -p "$OUT_DIR"

for d in "${DATASETS[@]}"; do
  if [ ! -d "$DATASET_PATH/$d" ]; then
    echo "[skip] $d -- not found under $DATASET_PATH"
    continue
  fi

  seq_dir="$OUT_DIR/$d"
  mkdir -p "$seq_dir"

  echo "=== $d ==="
  ( cd "$seq_dir" && \
    basalt_vio --dataset-path "$DATASET_PATH/$d" \
               --cam-calib "${DATA_DIR}/euroc_eucm_calib.json" \
               --dataset-type euroc --show-gui 0 \
               --config-path "${DATA_DIR}/euroc_config.json" \
               --online-loop-closure true \
               --result-path result.json \
               --save-trajectory tum \
    2>&1 | tee "log.txt" )
done

echo ""
echo "Done. Parse results with: ./gen_loop_closure_results.py $OUT_DIR"
