#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PREDIFY_PYTHON_BIN:-/home/lin/anaconda3/envs/predifyproject/bin/python}"
GIT_REVISION="$(git -C "$REPO_ROOT" rev-parse HEAD)"
SHORT_REVISION="${GIT_REVISION:0:7}"
OUTPUT_ROOT="${PREDIFY_CONTROLLED_OUTPUT_ROOT:-/tmp/predify-storage/experiments/seed0_same_drive_controlled_corruption_${SHORT_REVISION}}"

if [[ -n "$(git -C "$REPO_ROOT" status --porcelain)" ]]; then
    echo "Refusing a formal controlled-corruption run from a dirty worktree." >&2
    exit 2
fi
if [[ -e "$OUTPUT_ROOT/manifest.txt" ]]; then
    echo "Refusing to overwrite existing controlled-corruption run: $OUTPUT_ROOT" >&2
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
    "PREDIFY_TASK=future_feature"
    "PREDIFY_FUTURE_FEATURE_PREDICTOR_KERNEL_SIZE=1"
    "PREDIFY_KITTI_ROOT=/home/lin/predify/kitti_raw"
    "PREDIFY_KITTI_DRIVE=2011_09_26/2011_09_26_drive_0005_sync"
    "PREDIFY_TRAIN_DRIVES="
    "PREDIFY_VAL_DRIVES="
    "PREDIFY_FORMAL_SPLIT=0"
    "PREDIFY_SAME_DRIVE_SPLIT=1"
    "PREDIFY_SAME_DRIVE_GAP_FRAMES=20"
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
    "PREDIFY_TRAIN_FRACTION=0.6"
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
    "PREDIFY_SAVE_FINAL_CHECKPOINTS=0"
)

run_training_group() {
    local label="$1"
    local history_mode="$2"
    local prefix="$OUTPUT_ROOT/${label}_seed0"
    if [[ -e "${prefix}.p" || -e "${prefix}.log" || -e "${prefix}_best_student.pt" ]]; then
        echo "Refusing to overwrite artifacts at $prefix." >&2
        return 4
    fi
    printf 'Starting same-drive group %s at %s\n' "$label" "$(date --iso-8601=seconds)"
    env -i \
        "${COMMON_ENV[@]}" \
        "PREDIFY_FEATURE_HISTORY_MODE=$history_mode" \
        "PREDIFY_OUTPUT_PATH=${prefix}.p" \
        "PREDIFY_SAVE_BEST_STUDENT_PATH=${prefix}_best_student.pt" \
        "$PYTHON_BIN" -u -m predify2021.mce_scores.train_kitti_targetflow_adjacent_pairs \
        2>&1 | tee "${prefix}.log"
}

printf 'git_revision=%s\noutput_root=%s\ndrive=%s\ngroups=copy_current current_only temporal_error\ntrain_fraction=0.6\nminimum_gap_frames=20\nval_fraction=0.2\n' \
    "$GIT_REVISION" \
    "$OUTPUT_ROOT" \
    "2011_09_26/2011_09_26_drive_0005_sync" > "$OUTPUT_ROOT/manifest.txt"

run_training_group copy_current copy_current
run_training_group current_only none
run_training_group temporal_error temporal_error

CONTROLLED_DIR="$OUTPUT_ROOT/evaluation"
CHECKPOINTS="copy_current=$OUTPUT_ROOT/copy_current_seed0_best_student.pt,current_only=$OUTPUT_ROOT/current_only_seed0_best_student.pt,temporal_error=$OUTPUT_ROOT/temporal_error_seed0_best_student.pt"
printf 'Starting controlled corruption evaluation at %s\n' "$(date --iso-8601=seconds)"
env -i \
    "${COMMON_ENV[@]}" \
    "PREDIFY_CONTROLLED_CHECKPOINTS=$CHECKPOINTS" \
    "PREDIFY_CONTROLLED_OUTPUT_DIR=$CONTROLLED_DIR" \
    "PREDIFY_CORRUPTION_BASELINE_FRAMES=6" \
    "PREDIFY_CORRUPTION_STEP_FRAMES=3" \
    "PREDIFY_CORRUPTION_RAMP_FRAMES=5" \
    "PREDIFY_CORRUPTION_PERSISTENT_FRAMES=7" \
    "PREDIFY_CORRUPTION_RECOVERY_FRAMES=9" \
    "PREDIFY_CORRUPTION_STEP_LEVEL=0.5" \
    "PREDIFY_CORRUPTION_BIAS_RGB=0.15,-0.08,0.05" \
    "PREDIFY_CORRUPTION_NOISE_STD=0.03" \
    "PREDIFY_CORRUPTION_SEED=0" \
    "PREDIFY_RECOVERY_THRESHOLD_FRACTION=0.1" \
    "PREDIFY_RECOVERY_CONSECUTIVE_FRAMES=3" \
    "$PYTHON_BIN" -u -m predify2021.mce_scores.evaluate_kitti_same_drive_controlled_corruption \
    2>&1 | tee "$OUTPUT_ROOT/evaluation.log"
