#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PREDIFY_PYTHON_BIN:-/home/lin/anaconda3/envs/predifyproject/bin/python}"
GIT_REVISION="$(git -C "$REPO_ROOT" rev-parse HEAD)"
SHORT_REVISION="${GIT_REVISION:0:7}"
OUTPUT_ROOT="${PREDIFY_CROSS_CORRUPTION_OUTPUT_ROOT:-$REPO_ROOT/results/real_frame_recurrent_error_cross_corruption_${SHORT_REVISION}}"
LOG_PATH="${PREDIFY_CROSS_CORRUPTION_LOG_PATH:-/tmp/predify-storage/experiments/real_frame_recurrent_error_cross_corruption_${SHORT_REVISION}.log}"
TRAIN_OUTPUT="/tmp/predify-storage/experiments/real_frame_recurrent_error_train_c9fec52"
PCODER_WEIGHTS="${PREDIFY_PCODER_WEIGHTS:-/home/lin/predify/weights_pvgg16_imagenet}"

if [[ "$(git -C "$REPO_ROOT" branch --show-current)" != "predify-selective-adaptation-v2" ]]; then
    echo "Cross-corruption validation must run from predify-selective-adaptation-v2." >&2
    exit 2
fi
if [[ "$REPO_ROOT" != "/home/lin/predify2021_selective_adaptation" ]]; then
    echo "Cross-corruption validation must use the isolated active worktree." >&2
    exit 3
fi
if [[ -n "$(git -C "$REPO_ROOT" status --porcelain)" ]]; then
    echo "Refusing cross-corruption validation from a dirty worktree." >&2
    exit 4
fi
if [[ -e "$OUTPUT_ROOT" || -e "$LOG_PATH" ]]; then
    echo "Refusing to overwrite cross-corruption artifacts." >&2
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
    PREDIFY_CROSS_CORRUPTION_OUTPUT_DIR="$OUTPUT_ROOT" \
    PREDIFY_RECURRENT_ERROR_TRAIN_OUTPUT_DIR="$TRAIN_OUTPUT" \
    PREDIFY_PCODER_WEIGHTS="$PCODER_WEIGHTS" \
    PREDIFY_KITTI_ROOT=/home/lin/predify/kitti_raw \
    PREDIFY_KITTI_CAMERA=image_02 \
    PREDIFY_CROSS_CORRUPTION_DRIVES=2011_09_26/2011_09_26_drive_0011_sync,2011_09_26/2011_09_26_drive_0039_sync \
    PREDIFY_FIXED_TS_S=0.1035 \
    PREDIFY_FIXED_TS_TOL_S=0.001 \
    PREDIFY_SEED=0 \
    "$PYTHON_BIN" -u -m predify2021.mce_scores.evaluate_kitti_real_frame_recurrent_error_cross_corruption \
    2>&1 | tee "$LOG_PATH"

printf 'evaluator_revision=%s\nmodel_revision=c9fec52\ncheckpoint_epoch=1\ncheckpoint_sha256=26c5333da95a6b754af6036f9d16a5dd394673400e6f9456c0ebc0b27e714cf4\nval_drives=0011,0039\nfrozen_test_read=false\ncorruptions=gaussian_noise_std0.08,brightness_shift_rgb+0.15\nconditions=current_stateful,learned_recurrent_error,learned_recurrent_error_zeroed\ntraining=false\ntuning=false\ncheckpoint_selection=false\n' \
    "$GIT_REVISION" > "$OUTPUT_ROOT/manifest.txt"
