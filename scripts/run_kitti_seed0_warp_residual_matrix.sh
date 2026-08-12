#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PREDIFY_PYTHON_BIN:-/home/lin/anaconda3/envs/predifyproject/bin/python}"
GIT_REVISION="$(git -C "$REPO_ROOT" rev-parse HEAD)"
SHORT_REVISION="${GIT_REVISION:0:7}"
OUTPUT_ROOT="${PREDIFY_WARP_MATRIX_OUTPUT_ROOT:-/tmp/predify-storage/experiments/seed0_warp_residual_matrix_${SHORT_REVISION}}"

if [[ -n "$(git -C "$REPO_ROOT" status --porcelain)" ]]; then
    echo "Refusing a formal warp-residual run from a dirty worktree." >&2
    exit 2
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
    echo "Refusing to overwrite warp-residual output: $OUTPUT_ROOT" >&2
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
    "CUDA_VISIBLE_DEVICES=0"
    "OMP_NUM_THREADS=24"
    "MKL_NUM_THREADS=24"
    "PREDIFY_GIT_REVISION=$GIT_REVISION"
    "PREDIFY_TASK=future_feature"
    "PREDIFY_KITTI_ROOT=/home/lin/predify/kitti_raw"
    "PREDIFY_TRAIN_DRIVES=2011_09_26/2011_09_26_drive_0005_sync"
    "PREDIFY_VAL_DRIVES=2011_09_26/2011_09_26_drive_0011_sync"
    "PREDIFY_FORMAL_SPLIT=1"
    "PREDIFY_KITTI_CAMERA=image_02"
    "PREDIFY_MAX_PAIRS=0"
    "PREDIFY_MAX_TRAIN_PAIRS=0"
    "PREDIFY_MAX_VAL_PAIRS=0"
    "PREDIFY_BATCHSIZE=1"
    "PREDIFY_NUM_WORKERS=0"
    "PREDIFY_EPOCHS=10"
    "PREDIFY_LR=1e-4"
    "PREDIFY_WEIGHT_DECAY=0"
    "PREDIFY_EMA_DECAY=0.99"
    "PREDIFY_TRAIN_FRACTION=0.8"
    "PREDIFY_VAL_FRACTION=0.2"
    "PREDIFY_TARGET_FLOW_MODE=recursive"
    "PREDIFY_TOP_TARGET_SOURCE=student_self"
    "PREDIFY_TEMPORAL_TARGET_MODE=next_top"
    "PREDIFY_TASK_ALIGNED_TARGET="
    "PREDIFY_TEMPORAL_HORIZONS=1"
    "PREDIFY_PRETRAINED=1"
    "PREDIFY_TRAIN_BACKBONE=0"
    "PREDIFY_PCODER_WEIGHTS=/home/lin/predify/weights_pvgg16_imagenet"
    "PREDIFY_TOP_VARIANCE_WEIGHT=0"
    "PREDIFY_TOP_VARIANCE_TARGET=0.01"
    "PREDIFY_TOP_VARIANCE_EPS=1e-6"
    "PREDIFY_TOP_VARIANCE_WINDOW=16"
    "PREDIFY_TEMPORAL_PREDICTION_WEIGHT=1.0"
    "PREDIFY_FEATURE_PREDICTION_WEIGHT=1.0"
    "PREDIFY_LOCAL_RECONSTRUCTION_WEIGHT=1.0"
    "PREDIFY_FEATURE_METRIC_EPS=1e-8"
    "PREDIFY_LAYER_LOSS_WEIGHTS=0.1,0.1,0.2,0.2,1.0"
    "PREDIFY_FIXED_TS_S=0.1035"
    "PREDIFY_FIXED_TS_TOL_S=0.001"
    "PREDIFY_DYNAMIC_ERROR=1"
    "PREDIFY_ERROR_STATE_MODE=ema"
    "PREDIFY_LOCAL_LOSS_ERROR_SOURCE=instant"
    "PREDIFY_ERROR_TS=0.1035"
    "PREDIFY_ERROR_TAU=0.5"
    "PREDIFY_ERROR_GAIN=1.0"
    "PREDIFY_TEMPORAL_ERROR_TS=0.1035"
    "PREDIFY_TEMPORAL_ERROR_TAU=0.5"
    "PREDIFY_TEMPORAL_ERROR_GAIN=1.0"
    "PREDIFY_STREAM_MODE=1"
    "PREDIFY_RESET_EACH_FRAME=0"
    "PREDIFY_SHUFFLE_TRAIN_PAIRS=0"
    "PREDIFY_SHUFFLE_VAL_PAIRS=0"
    "PREDIFY_SHUFFLE_SEED=0"
    "PREDIFY_SEED=0"
    "PREDIFY_CURRENT_TOP_DUPLICATE=0"
    "PREDIFY_CURRENT_TEACHER_CONTEXT=0"
    "PREDIFY_SAME_DRIVE_SPLIT=0"
    "PREDIFY_SAME_DRIVE_GAP_FRAMES=20"
    "PREDIFY_SAVE_FINAL_CHECKPOINTS=0"
    "PREDIFY_FUTURE_FEATURE_PREDICTOR_KERNEL_SIZE=1"
    "PREDIFY_FUTURE_MOTION_RADIUS=1"
    "PREDIFY_FUTURE_MOTION_PATCH_SIZE=3"
)

run_group() {
    local label="$1"
    local history_mode="$2"
    local prediction_form="$3"
    local prefix="$OUTPUT_ROOT/${label}_seed0"
    printf 'Starting %s at %s\n' "$label" "$(date --iso-8601=seconds)"
    env -i \
        "${COMMON_ENV[@]}" \
        "PREDIFY_FEATURE_HISTORY_MODE=$history_mode" \
        "PREDIFY_FUTURE_FEATURE_PREDICTION_FORM=$prediction_form" \
        "PREDIFY_OUTPUT_PATH=${prefix}.p" \
        "PREDIFY_SAVE_BEST_STUDENT_PATH=${prefix}_best_student.pt" \
        "$PYTHON_BIN" -u -m predify2021.mce_scores.train_kitti_targetflow_adjacent_pairs \
        2>&1 | tee "${prefix}.log"
}

printf 'git_revision=%s\noutput_root=%s\ngroups=copy_current historical_warp warp_residual\nmotion_radius=1\nmotion_patch_size=3\n' \
    "$GIT_REVISION" "$OUTPUT_ROOT" > "$OUTPUT_ROOT/manifest.txt"

run_group copy_current copy_current current_residual
run_group historical_warp none historical_warp
run_group warp_residual none historical_warp_residual

CHECKPOINTS="copy_current=$OUTPUT_ROOT/copy_current_seed0_best_student.pt,historical_warp=$OUTPUT_ROOT/historical_warp_seed0_best_student.pt,warp_residual=$OUTPUT_ROOT/warp_residual_seed0_best_student.pt"
env -i \
    "${COMMON_ENV[@]}" \
    "PREDIFY_WARP_MATRIX_CHECKPOINTS=$CHECKPOINTS" \
    "PREDIFY_WARP_MATRIX_EVAL_OUTPUT_DIR=$OUTPUT_ROOT/evaluation" \
    "$PYTHON_BIN" -u -m predify2021.mce_scores.evaluate_kitti_warp_residual_matrix \
    2>&1 | tee "$OUTPUT_ROOT/evaluation.log"
