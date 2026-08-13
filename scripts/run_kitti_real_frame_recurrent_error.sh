#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PREDIFY_PYTHON_BIN:-/home/lin/anaconda3/envs/predifyproject/bin/python}"
GIT_REVISION="$(git -C "$REPO_ROOT" rev-parse HEAD)"
SHORT_REVISION="${GIT_REVISION:0:7}"
TRAIN_OUTPUT="${PREDIFY_RECURRENT_ERROR_TRAIN_OUTPUT_ROOT:-/tmp/predify-storage/experiments/real_frame_error_memory_train_${SHORT_REVISION}}"
EVAL_OUTPUT="${PREDIFY_RECURRENT_ERROR_EVAL_OUTPUT_ROOT:-$REPO_ROOT/results/real_frame_error_memory_${SHORT_REVISION}}"
LOG_ROOT="${PREDIFY_RECURRENT_ERROR_LOG_ROOT:-/tmp/predify-storage/experiments}"
PCODER_WEIGHTS="${PREDIFY_PCODER_WEIGHTS:-/home/lin/predify/weights_pvgg16_imagenet}"

if [[ "$(git -C "$REPO_ROOT" branch --show-current)" != "predify-selective-adaptation-v2" ]]; then
    echo "Recurrent-error experiment must run from predify-selective-adaptation-v2." >&2
    exit 2
fi
if [[ "$REPO_ROOT" != "/home/lin/predify2021_selective_adaptation" ]]; then
    echo "Recurrent-error experiment must use the isolated active worktree." >&2
    exit 3
fi
if [[ -n "$(git -C "$REPO_ROOT" status --porcelain)" ]]; then
    echo "Refusing a formal run from a dirty worktree." >&2
    exit 4
fi
if [[ -e "$TRAIN_OUTPUT" || -e "$EVAL_OUTPUT" ]]; then
    echo "Refusing to overwrite recurrent-error artifacts." >&2
    exit 5
fi

mkdir -p "$LOG_ROOT"
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
    "PREDIFY_KITTI_ROOT=/home/lin/predify/kitti_raw"
    "PREDIFY_KITTI_CAMERA=image_02"
    "PREDIFY_PCODER_WEIGHTS=$PCODER_WEIGHTS"
    "PREDIFY_FIXED_TS_S=0.1035"
    "PREDIFY_FIXED_TS_TOL_S=0.001"
    "PREDIFY_SEED=0"
)

env -i \
    "${COMMON_ENV[@]}" \
    "PREDIFY_EPOCHS=5" \
    "PREDIFY_LR=1e-4" \
    "PREDIFY_RECURRENT_ERROR_TRAIN_OUTPUT_DIR=$TRAIN_OUTPUT" \
    "$PYTHON_BIN" -u -m predify2021.mce_scores.train_kitti_real_frame_recurrent_error \
    2>&1 | tee "$LOG_ROOT/real_frame_recurrent_error_train_${SHORT_REVISION}.log"

env -i \
    "${COMMON_ENV[@]}" \
    "PREDIFY_RECURRENT_ERROR_TRAIN_OUTPUT_DIR=$TRAIN_OUTPUT" \
    "PREDIFY_RECURRENT_ERROR_EVAL_OUTPUT_DIR=$EVAL_OUTPUT" \
    "PREDIFY_RECURRENT_ERROR_VAL_DRIVES=2011_09_26/2011_09_26_drive_0011_sync,2011_09_26/2011_09_26_drive_0039_sync" \
    "$PYTHON_BIN" -u -m predify2021.mce_scores.evaluate_kitti_real_frame_recurrent_error \
    2>&1 | tee "$LOG_ROOT/real_frame_recurrent_error_eval_${SHORT_REVISION}.log"

printf 'git_revision=%s\ntrain_drives=0005,0013,0014,0036\nval_drives=0011,0039\nfrozen_test_read=false\ncore_conditions=temporal_only,instant_error,error_memory\ncorruptions=gaussian_blur,brightness_overexposure\nmatched_transition_capacity=true\ndedicated_error_encoder=true\ntemporal_predictor_trained=false\ntop_down_feedback=true\nobservation_input=true\ninstant_error=e_t=F_t-Fhat_t\ndynamic_error=epsilon_t=0.207e_t+0.793epsilon_t_minus_1\ntraining_target=F_(t+1)\nepochs=5\nlr=1e-4\n' \
    "$GIT_REVISION" > "$EVAL_OUTPUT/manifest.txt"
