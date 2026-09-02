#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${VSPW_ROOT:-/home/lin/datasets/VSPW_480p}"
HOST_CHECKPOINT="${VSPW_HOST_CHECKPOINT:-/home/lin/predify/experiments/vspw_static_deeplabv3plus_r50_v1/best_vspw_static_deeplabv3plus_r50.pt}"
DYNAMICS_CHECKPOINT="${FAST_B_DYNAMICS_CHECKPOINT:-/home/lin/predify/experiments/kitti_step_role_separated_predictor_3370f78/best_role_separated_predictor.pt}"
OUTPUT_DIR="${FAST_B_OUTPUT_DIR:-/home/lin/predify/experiments/vspw_fast_b}"

exec /home/lin/anaconda3/envs/predifyproject/bin/python -m predify2021.mce_scores.train_vspw_fast_b \
  --data-root "$DATA_ROOT" \
  --host-checkpoint "$HOST_CHECKPOINT" \
  --dynamics-checkpoint "$DYNAMICS_CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --clip-length "${FAST_B_CLIP_LENGTH:-16}" \
  --bptt "${FAST_B_BPTT:-16}" \
  --num-workers "${FAST_B_NUM_WORKERS:-8}" \
  --epochs "${FAST_B_EPOCHS:-15}" \
  --patience "${FAST_B_PATIENCE:-3}"
