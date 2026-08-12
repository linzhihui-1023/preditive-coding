#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PREDIFY_PYTHON_BIN:-/home/lin/anaconda3/envs/predifyproject/bin/python}"
GIT_REVISION="$(git -C "$REPO_ROOT" rev-parse HEAD)"
SHORT_REVISION="${GIT_REVISION:0:7}"
OUTPUT_ROOT="${PREDIFY_GATE3_OUTPUT_ROOT:-/tmp/predify-storage/experiments/gate3_persistence_task_degradation_${SHORT_REVISION}}"
GATE2_DIR="${PREDIFY_GATE2_RESULT_DIR:-$REPO_ROOT/results/matched_blur_persistence_99b7e21}"
DETECTOR_CHECKPOINT="${PREDIFY_GATE3_DETECTOR_CHECKPOINT:-/tmp/predify-storage/experiments/seed0_future_feature_matrix_3ffbff0/temporal_error_seed0_best_student.pt}"
MOTION_CHECKPOINT="${PREDIFY_GATE3_MOTION_CHECKPOINT:-/home/lin/predify/experiments/seed0_frozen_matrix_6c446d9/A_seed0_best_student.pt}"

if [[ "$(git -C "$REPO_ROOT" branch --show-current)" != "predify-selective-adaptation-v2" ]]; then
    echo "Gate 3 must run from predify-selective-adaptation-v2." >&2
    exit 2
fi
if [[ "$REPO_ROOT" != "/home/lin/predify2021_selective_adaptation" ]]; then
    echo "Gate 3 must run from the isolated selective-adaptation worktree." >&2
    exit 3
fi
if [[ -n "$(git -C "$REPO_ROOT" status --porcelain)" ]]; then
    echo "Refusing a formal Gate 3 run from a dirty worktree." >&2
    exit 4
fi
if [[ -e "$OUTPUT_ROOT" || -e "${OUTPUT_ROOT}.log" ]]; then
    echo "Refusing to overwrite Gate 3 output: $OUTPUT_ROOT" >&2
    exit 5
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
    PREDIFY_GATE3_OUTPUT_DIR="$OUTPUT_ROOT" \
    PREDIFY_GATE2_RESULT_DIR="$GATE2_DIR" \
    PREDIFY_GATE2_GIT_REVISION=99b7e210f7922c747a08d57b9340162f359527ad \
    PREDIFY_GATE2_SUMMARY_SHA256=c9ea7936135be84926cb8c9342f8a67b57faed163bd0592fa7fe518a282afd37 \
    PREDIFY_GATE2_PER_FRAME_SHA256=27cacb0efee858799b4b70a630d1609c131aad34d7a760cf9b8bbb3457b0ebeb \
    PREDIFY_GATE3_DETECTOR_CHECKPOINT="$DETECTOR_CHECKPOINT" \
    PREDIFY_DETECTOR_CHECKPOINT_GIT_REVISION=3ffbff0156dc9435fe3b060f4c9999703729ea8a \
    PREDIFY_GATE3_MOTION_CHECKPOINT="$MOTION_CHECKPOINT" \
    PREDIFY_MOTION_CHECKPOINT_GIT_REVISION=6c446d901a581323208a335489a7c065f1946adc \
    PREDIFY_KITTI_ROOT=/home/lin/predify/kitti_raw \
    PREDIFY_KITTI_CAMERA=image_02 \
    PREDIFY_FIXED_TS_S=0.1035 \
    PREDIFY_FIXED_TS_TOL_S=0.001 \
    PREDIFY_ERROR_NORM_EMA_ALPHA=0.207 \
    PREDIFY_ERROR_ROLLING_WINDOW=8 \
    "$PYTHON_BIN" -u -m predify2021.mce_scores.evaluate_kitti_gate3_persistence_task_degradation \
    2>&1 | tee "${OUTPUT_ROOT}.log"
