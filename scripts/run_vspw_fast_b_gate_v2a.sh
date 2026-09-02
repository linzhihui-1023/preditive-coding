#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${VSPW_ROOT:-/home/lin/datasets/VSPW_480p}"
HOST_CHECKPOINT="${VSPW_HOST_CHECKPOINT:?Set VSPW_HOST_CHECKPOINT to best_vspw_host.pt}"
DYNAMICS_CHECKPOINT="${FAST_B_DYNAMICS_CHECKPOINT:-/home/lin/predify/experiments/kitti_step_role_separated_predictor_3370f78/best_role_separated_predictor.pt}"

# Gate V2-A is intentionally a short CUDA forward/backward/metric smoke, not formal training.
exec /home/lin/anaconda3/envs/predifyproject/bin/python -m predify2021.mce_scores.validate_vspw_fast_b \
  --data-root "$DATA_ROOT" \
  --host-checkpoint "$HOST_CHECKPOINT" \
  --dynamics-checkpoint "$DYNAMICS_CHECKPOINT" \
  --output-dir "${FAST_B_GATE_OUTPUT_DIR:-/home/lin/predify/experiments/vspw_fast_b/gate_v2a}" \
  --num-workers "${FAST_B_NUM_WORKERS:-4}" \
  --max-videos "${FAST_B_GATE_VIDEOS:-4}" \
  --temporal-mode both
