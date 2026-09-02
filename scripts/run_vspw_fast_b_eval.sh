#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${VSPW_ROOT:-/home/lin/datasets/VSPW_480p}"
HOST_CHECKPOINT="${VSPW_HOST_CHECKPOINT:?Set VSPW_HOST_CHECKPOINT to best_vspw_host.pt}"
FAST_B_CHECKPOINT="${FAST_B_CHECKPOINT:?Set FAST_B_CHECKPOINT to best_vspw_fast_b.pt}"
DYNAMICS_CHECKPOINT="${FAST_B_DYNAMICS_CHECKPOINT:-/home/lin/predify/experiments/kitti_step_role_separated_predictor_3370f78/best_role_separated_predictor.pt}"
OUTPUT_DIR="${FAST_B_EVAL_OUTPUT_DIR:-/home/lin/predify/experiments/vspw_fast_b/full_val}"

exec /home/lin/anaconda3/envs/predifyproject/bin/python -m predify2021.mce_scores.evaluate_vspw_fast_b \
  --data-root "$DATA_ROOT" \
  --host-checkpoint "$HOST_CHECKPOINT" \
  --fast-b-checkpoint "$FAST_B_CHECKPOINT" \
  --dynamics-checkpoint "$DYNAMICS_CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --num-workers "${FAST_B_NUM_WORKERS:-8}" \
  --temporal-mode both
