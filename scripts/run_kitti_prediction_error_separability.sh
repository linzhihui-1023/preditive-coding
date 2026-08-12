#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PREDIFY_PYTHON_BIN:-/home/lin/anaconda3/envs/predifyproject/bin/python}"
GIT_REVISION="$(git -C "$REPO_ROOT" rev-parse HEAD)"
SHORT_REVISION="${GIT_REVISION:0:7}"
OUTPUT_ROOT="${PREDIFY_ERROR_SEPARABILITY_OUTPUT_ROOT:-/tmp/predify-storage/experiments/prediction_error_separability_${SHORT_REVISION}}"
CHECKPOINT="${PREDIFY_ERROR_SEPARABILITY_CHECKPOINT:-/tmp/predify-storage/experiments/seed0_future_feature_matrix_3ffbff0/temporal_error_seed0_best_student.pt}"

if [[ -n "$(git -C "$REPO_ROOT" status --porcelain)" ]]; then
    echo "Refusing a formal separability run from a dirty worktree." >&2
    exit 2
fi
if [[ -e "$OUTPUT_ROOT" || -e "${OUTPUT_ROOT}.log" ]]; then
    echo "Refusing to overwrite separability output: $OUTPUT_ROOT" >&2
    exit 3
fi

env -i \
    HOME=/home/lin \
    LANG=C.UTF-8 \
    TZ=Asia/Shanghai \
    PATH=/home/lin/anaconda3/envs/predifyproject/bin:/usr/local/bin:/usr/bin:/bin \
    PYTHONHASHSEED=0 \
    PYTHONUNBUFFERED=1 \
    PYTHONPYCACHEPREFIX=/tmp/predify-pycache \
    PYTHONPATH="$REPO_ROOT:/home/lin/predify" \
    TORCH_HOME=/home/lin/predify/.torch \
    CUDA_VISIBLE_DEVICES=0 \
    OMP_NUM_THREADS=24 \
    MKL_NUM_THREADS=24 \
    PREDIFY_GIT_REVISION="$GIT_REVISION" \
    PREDIFY_CHECKPOINT_GIT_REVISION=3ffbff0156dc9435fe3b060f4c9999703729ea8a \
    PREDIFY_ERROR_SEPARABILITY_CHECKPOINT="$CHECKPOINT" \
    PREDIFY_ERROR_SEPARABILITY_OUTPUT_DIR="$OUTPUT_ROOT" \
    PREDIFY_KITTI_ROOT=/home/lin/predify/kitti_raw \
    PREDIFY_KITTI_CAMERA=image_02 \
    PREDIFY_ERROR_SEPARABILITY_DRIVES=2011_09_26/2011_09_26_drive_0005_sync,2011_09_26/2011_09_26_drive_0011_sync \
    PREDIFY_FIXED_TS_S=0.1035 \
    PREDIFY_FIXED_TS_TOL_S=0.001 \
    PREDIFY_ERROR_BASELINE_FRAMES=40 \
    PREDIFY_ERROR_SHIFT_FRAMES=80 \
    PREDIFY_ERROR_RECOVERY_FRAMES=30 \
    PREDIFY_ERROR_NORM_EMA_ALPHA=0.207 \
    PREDIFY_ERROR_ROLLING_WINDOW=8 \
    PREDIFY_ERROR_BLUR_KERNEL_SIZE=11 \
    PREDIFY_ERROR_BLUR_SIGMA=3.0 \
    PREDIFY_ERROR_IID_NOISE_STD=0.08 \
    "$PYTHON_BIN" -u -m predify2021.mce_scores.evaluate_kitti_prediction_error_separability \
    2>&1 | tee "${OUTPUT_ROOT}.log"
