#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PREDIFY_PYTHON_BIN:-/home/lin/anaconda3/envs/predifyproject/bin/python}"
GIT_REVISION="$(git -C "$REPO_ROOT" rev-parse HEAD)"
MODEL_REVISION="c9fec52"
OUTPUT_ROOT="${PREDIFY_RECURRENT_ERROR_TEST_OUTPUT_ROOT:-$REPO_ROOT/results/real_frame_recurrent_error_frozen_test_${MODEL_REVISION}}"
LOG_PATH="${PREDIFY_RECURRENT_ERROR_TEST_LOG_PATH:-/tmp/predify-storage/experiments/real_frame_recurrent_error_frozen_test_${MODEL_REVISION}.log}"
TRAIN_OUTPUT="/tmp/predify-storage/experiments/real_frame_recurrent_error_train_${MODEL_REVISION}"
PCODER_WEIGHTS="${PREDIFY_PCODER_WEIGHTS:-/home/lin/predify/weights_pvgg16_imagenet}"

if [[ "$(git -C "$REPO_ROOT" branch --show-current)" != "predify-selective-adaptation-v2" ]]; then
    echo "Frozen Test must run from predify-selective-adaptation-v2." >&2
    exit 2
fi
if [[ "$REPO_ROOT" != "/home/lin/predify2021_selective_adaptation" ]]; then
    echo "Frozen Test must use the isolated active worktree." >&2
    exit 3
fi
if [[ -n "$(git -C "$REPO_ROOT" status --porcelain)" ]]; then
    echo "Refusing Frozen Test from a dirty worktree." >&2
    exit 4
fi
if [[ -e "$OUTPUT_ROOT" || -e "$LOG_PATH" ]]; then
    echo "Refusing to overwrite Frozen Test artifacts." >&2
    exit 5
fi

mkdir -p "$(dirname "$LOG_PATH")"
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
    PREDIFY_RECURRENT_ERROR_TEST_OUTPUT_DIR="$OUTPUT_ROOT" \
    PREDIFY_RECURRENT_ERROR_TRAIN_OUTPUT_DIR="$TRAIN_OUTPUT" \
    PREDIFY_PCODER_WEIGHTS="$PCODER_WEIGHTS" \
    PREDIFY_KITTI_ROOT=/home/lin/predify/kitti_raw \
    PREDIFY_KITTI_CAMERA=image_02 \
    PREDIFY_RECURRENT_ERROR_TEST_DRIVES=2011_09_26/2011_09_26_drive_0051_sync,2011_09_26/2011_09_26_drive_0056_sync \
    PREDIFY_FIXED_TS_S=0.1035 \
    PREDIFY_FIXED_TS_TOL_S=0.001 \
    PREDIFY_SEED=0 \
    "$PYTHON_BIN" -u -m predify2021.mce_scores.evaluate_kitti_real_frame_recurrent_error_frozen_test \
    2>&1 | tee "$LOG_PATH"

printf 'evaluator_revision=%s\nmodel_revision=c9fec52\ncheckpoint_epoch=1\ncheckpoint_sha256=26c5333da95a6b754af6036f9d16a5dd394673400e6f9456c0ebc0b27e714cf4\nfrozen_test_drives=0051,0056\nconditions=current_stateful,learned_recurrent_error,learned_recurrent_error_zeroed\ntraining=false\ntuning=false\n' \
    "$GIT_REVISION" > "$OUTPUT_ROOT/manifest.txt"
