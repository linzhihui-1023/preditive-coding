#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PREDIFY_PYTHON_BIN:-/home/lin/anaconda3/envs/predifyproject/bin/python}"
GIT_REVISION="$(git -C "$REPO_ROOT" rev-parse HEAD)"
SHORT_REVISION="${GIT_REVISION:0:7}"
OUTPUT_ROOT="${PREDIFY_MATRIX_OUTPUT_ROOT:-/tmp/predify-storage/experiments/seed0_future_feature_matrix_${SHORT_REVISION}}"

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

run_group() {
    local group="$1"
    local history_mode

    case "$group" in
        copy_current) history_mode=copy_current ;;
        current_only) history_mode=none ;;
        latest|instant) history_mode=latest ;;
        two_tap|lag1) history_mode=two_tap ;;
        recursive) history_mode=recursive ;;
        *)
            echo "Unknown group '$group'." >&2
            return 2
            ;;
    esac

    local prefix="$OUTPUT_ROOT/${group}_seed0"
    if [[ -e "${prefix}.p" || -e "${prefix}.log" ]]; then
        echo "Refusing to overwrite artifacts at $prefix." >&2
        return 3
    fi

    printf 'Starting group %s at %s\n' "$group" "$(date --iso-8601=seconds)"
    env -i \
        "${COMMON_ENV[@]}" \
        "PREDIFY_FEATURE_HISTORY_MODE=$history_mode" \
        "PREDIFY_OUTPUT_PATH=${prefix}.p" \
        "PREDIFY_SAVE_BEST_STUDENT_PATH=${prefix}_best_student.pt" \
        "$PYTHON_BIN" -u -m predify2021.mce_scores.train_kitti_targetflow_adjacent_pairs \
        2>&1 | tee "${prefix}.log"
}

groups=("$@")
if [[ ${#groups[@]} -eq 0 ]]; then
    groups=(copy_current current_only latest two_tap recursive)
fi

printf 'git_revision=%s\noutput_root=%s\ngroups=%s\n' \
    "$GIT_REVISION" "$OUTPUT_ROOT" "${groups[*]}" > "$OUTPUT_ROOT/manifest.txt"

for group in "${groups[@]}"; do
    run_group "$group"
done
