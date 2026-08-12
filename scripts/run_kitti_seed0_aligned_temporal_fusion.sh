#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PREDIFY_PYTHON_BIN:-/home/lin/anaconda3/envs/predifyproject/bin/python}"
GIT_REVISION="$(git -C "$REPO_ROOT" rev-parse HEAD)"
SHORT_REVISION="${GIT_REVISION:0:7}"
OUTPUT_ROOT="${PREDIFY_ALIGNED_FUSION_OUTPUT_ROOT:-/tmp/predify-storage/experiments/seed0_aligned_temporal_fusion_${SHORT_REVISION}}"

if [[ "$(git -C "$REPO_ROOT" branch --show-current)" != "predify-selective-adaptation-v2" ]]; then
    echo "Aligned Temporal Fusion must run from predify-selective-adaptation-v2." >&2
    exit 2
fi
if [[ "$REPO_ROOT" != "/home/lin/predify2021_selective_adaptation" ]]; then
    echo "Aligned Temporal Fusion must use the isolated selective-adaptation worktree." >&2
    exit 3
fi
if [[ -n "$(git -C "$REPO_ROOT" status --porcelain)" ]]; then
    echo "Refusing a formal Aligned Temporal Fusion run from a dirty worktree." >&2
    exit 4
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
    echo "Refusing to overwrite Aligned Temporal Fusion output: $OUTPUT_ROOT" >&2
    exit 5
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
    "PREDIFY_TRAIN_FEEDBACK_DECODERS=0"
    "PREDIFY_PCODER_WEIGHTS=/home/lin/predify/weights_pvgg16_imagenet"
    "PREDIFY_TOP_VARIANCE_WEIGHT=0"
    "PREDIFY_TEMPORAL_PREDICTION_WEIGHT=1.0"
    "PREDIFY_FEATURE_PREDICTION_WEIGHT=1.0"
    "PREDIFY_LOCAL_RECONSTRUCTION_WEIGHT=0"
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
    "PREDIFY_FEATURE_HISTORY_MODE=none"
    "PREDIFY_TEMPORAL_FUSION_MODE=aligned_two_frame_residual"
    "PREDIFY_FUTURE_FEATURE_PREDICTOR_KERNEL_SIZE=1"
    "PREDIFY_FUTURE_FEATURE_PREDICTION_FORM=current_residual"
    "PREDIFY_FUTURE_MOTION_RADIUS=1"
    "PREDIFY_FUTURE_MOTION_PATCH_SIZE=3"
)

PREFIX="$OUTPUT_ROOT/aligned_temporal_fusion_seed0"
printf 'git_revision=%s\noutput_root=%s\ncondition=aligned_temporal_fusion\ntrain_drive=0005\nval_drive=0011\ncopy_mse_gate=0.06008010\ncopy_cosine_gate=0.91990469\ncopy_normalized_error_gate=0.37576611\n' \
    "$GIT_REVISION" "$OUTPUT_ROOT" > "$OUTPUT_ROOT/manifest.txt"

env -i \
    "${COMMON_ENV[@]}" \
    "PREDIFY_OUTPUT_PATH=${PREFIX}.p" \
    "PREDIFY_SAVE_BEST_STUDENT_PATH=${PREFIX}_best_student.pt" \
    "$PYTHON_BIN" -u -m predify2021.mce_scores.train_kitti_targetflow_adjacent_pairs \
    2>&1 | tee "${PREFIX}.log"

env -i \
    "${COMMON_ENV[@]}" \
    "PREDIFY_ALIGNED_FUSION_CHECKPOINT=${PREFIX}_best_student.pt" \
    "PREDIFY_ALIGNED_FUSION_EVAL_OUTPUT_DIR=$OUTPUT_ROOT/evaluation" \
    "$PYTHON_BIN" -u -m predify2021.mce_scores.evaluate_kitti_aligned_temporal_fusion \
    2>&1 | tee "$OUTPUT_ROOT/evaluation.log"

"$PYTHON_BIN" - "$PREFIX" <<'PY'
import json
import pickle
import sys
from pathlib import Path


prefix = Path(sys.argv[1])
with prefix.with_suffix(".p").open("rb") as handle:
    history = pickle.load(handle)
with prefix.with_name("training_history.json").open("w") as handle:
    json.dump(history, handle, indent=2, sort_keys=True)
PY
