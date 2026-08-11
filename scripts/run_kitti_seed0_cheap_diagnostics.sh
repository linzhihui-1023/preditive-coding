#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PREDIFY_PYTHON_BIN:-/home/lin/anaconda3/envs/predifyproject/bin/python}"
GIT_REVISION="$(git -C "$REPO_ROOT" rev-parse HEAD)"
SHORT_REVISION="${GIT_REVISION:0:7}"
MATRIX_ROOT="/home/lin/predify/experiments/seed0_frozen_matrix_6c446d9"
OUTPUT_ROOT="${PREDIFY_DIAGNOSTIC_OUTPUT_ROOT:-/home/lin/predify/experiments/seed0_cheap_diagnostics_${SHORT_REVISION}}"

if [[ -e "$OUTPUT_ROOT" ]]; then
    echo "Refusing to overwrite existing diagnostic artifacts at $OUTPUT_ROOT." >&2
    exit 3
fi
mkdir -p "$OUTPUT_ROOT"

COMMON_ENV=(
    "HOME=/home/lin"
    "LANG=C.UTF-8"
    "TZ=Asia/Shanghai"
    "PATH=/home/lin/anaconda3/envs/predifyproject/bin:/usr/local/bin:/usr/bin:/bin"
    "PYTHONHASHSEED=0"
    "PYTHONUNBUFFERED=1"
    "PYTHONPYCACHEPREFIX=/tmp/predify-pycache"
    "PYTHONPATH=$REPO_ROOT:/home/lin/predify"
    "TORCH_HOME=/home/lin/predify/.torch"
    "OMP_NUM_THREADS=24"
    "MKL_NUM_THREADS=24"
    "PREDIFY_GIT_REVISION=$GIT_REVISION"
    "PREDIFY_KITTI_ROOT=/home/lin/predify/kitti_raw"
    "PREDIFY_TRAIN_DRIVES=2011_09_26/2011_09_26_drive_0005_sync"
    "PREDIFY_VAL_DRIVES=2011_09_26/2011_09_26_drive_0011_sync"
    "PREDIFY_FORMAL_SPLIT=1"
    "PREDIFY_KITTI_CAMERA=image_02"
    "PREDIFY_MAX_TRAIN_PAIRS=0"
    "PREDIFY_MAX_VAL_PAIRS=0"
    "PREDIFY_BATCHSIZE=1"
    "PREDIFY_NUM_WORKERS=0"
    "PREDIFY_EPOCHS=10"
    "PREDIFY_LR=1e-4"
    "PREDIFY_WEIGHT_DECAY=0"
    "PREDIFY_PRETRAINED=1"
    "PREDIFY_FREEZE_BACKBONE=1"
    "PREDIFY_FIXED_TS_S=0.1035"
    "PREDIFY_FIXED_TS_TOL_S=0.001"
    "PREDIFY_SHUFFLE_TRAIN_PAIRS=0"
    "PREDIFY_SHUFFLE_VAL_PAIRS=0"
    "PREDIFY_SHUFFLE_SEED=0"
    "PREDIFY_SEED=0"
)

printf 'git_revision=%s\nmatrix_root=%s\noutput_root=%s\n' \
    "$GIT_REVISION" "$MATRIX_ROOT" "$OUTPUT_ROOT" > "$OUTPUT_ROOT/manifest.txt"

env -i \
    "${COMMON_ENV[@]}" \
    "PREDIFY_MATRIX_ROOT=$MATRIX_ROOT" \
    "PREDIFY_GROUPS=A,B,C,D,E" \
    "PREDIFY_OUTPUT_PATH=$OUTPUT_ROOT/seed0_motion_diagnostics.json" \
    "$PYTHON_BIN" -u -m predify2021.mce_scores.evaluate_kitti_motion_diagnostics \
    2>&1 | tee "$OUTPUT_ROOT/seed0_motion_diagnostics.log"

env -i \
    "${COMMON_ENV[@]}" \
    "PREDIFY_OUTPUT_PATH=$OUTPUT_ROOT/static_vgg_mlp_seed0.p" \
    "PREDIFY_SAVE_MODEL_PATH=$OUTPUT_ROOT/static_vgg_mlp_seed0_final.pt" \
    "PREDIFY_SAVE_BEST_MODEL_PATH=$OUTPUT_ROOT/static_vgg_mlp_seed0_best.pt" \
    "$PYTHON_BIN" -u -m predify2021.mce_scores.train_kitti_vgg_motion_baseline \
    2>&1 | tee "$OUTPUT_ROOT/static_vgg_mlp_seed0.log"
