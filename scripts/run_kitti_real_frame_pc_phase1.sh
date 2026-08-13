#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PREDIFY_PYTHON_BIN:-/home/lin/anaconda3/envs/predifyproject/bin/python}"
GIT_REVISION="$(git -C "$REPO_ROOT" rev-parse HEAD)"
SHORT_REVISION="${GIT_REVISION:0:7}"
OUTPUT_ROOT="${PREDIFY_REAL_FRAME_PC_PHASE1_OUTPUT_ROOT:-$REPO_ROOT/results/real_frame_pc_phase1_${SHORT_REVISION}}"
LOG_PATH="${PREDIFY_REAL_FRAME_PC_PHASE1_LOG_PATH:-/tmp/predify-storage/experiments/real_frame_pc_phase1_${SHORT_REVISION}.log}"
MANIFEST_PATH="${PREDIFY_REAL_FRAME_PC_PHASE1_MANIFEST_PATH:-/tmp/predify-storage/experiments/real_frame_pc_phase1_${SHORT_REVISION}.manifest.txt}"
PCODER_WEIGHTS="${PREDIFY_PCODER_WEIGHTS:-/home/lin/predify/weights_pvgg16_imagenet}"

if [[ "$(git -C "$REPO_ROOT" branch --show-current)" != "predify-selective-adaptation-v2" ]]; then
    echo "Phase 1 must run from predify-selective-adaptation-v2." >&2
    exit 2
fi
if [[ "$REPO_ROOT" != "/home/lin/predify2021_selective_adaptation" ]]; then
    echo "Phase 1 must run from the isolated active worktree." >&2
    exit 3
fi
if [[ -n "$(git -C "$REPO_ROOT" status --porcelain)" ]]; then
    echo "Refusing a formal Phase-1 run from a dirty worktree." >&2
    exit 4
fi
if [[ -e "$OUTPUT_ROOT" || -e "$LOG_PATH" || -e "$MANIFEST_PATH" ]]; then
    echo "Refusing to overwrite Phase-1 output." >&2
    exit 5
fi
if [[ ! -d "$PCODER_WEIGHTS" ]]; then
    echo "Missing pretrained PCoder weights: $PCODER_WEIGHTS" >&2
    exit 6
fi

mkdir -p "$(dirname "$LOG_PATH")"
printf 'git_revision=%s\nexperiment=real_frame_predictive_coding_phase1\ndrives=0011,0039\nfrozen_test_read=false\nconditions=feedforward,pc_no_error,pc_dynamic_error\ntrajectory=clean40,persistent_gaussian_blur80,clean_recovery40\nnetwork_training=false\nonline_learning=false\nfuture_predictor=false\n' \
    "$GIT_REVISION" > "$MANIFEST_PATH"

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
    PREDIFY_REAL_FRAME_PC_PHASE1_OUTPUT_DIR="$OUTPUT_ROOT" \
    PREDIFY_PCODER_WEIGHTS="$PCODER_WEIGHTS" \
    PREDIFY_KITTI_ROOT=/home/lin/predify/kitti_raw \
    PREDIFY_KITTI_CAMERA=image_02 \
    PREDIFY_REAL_FRAME_PC_PHASE1_DRIVES=2011_09_26/2011_09_26_drive_0011_sync,2011_09_26/2011_09_26_drive_0039_sync \
    PREDIFY_FIXED_TS_S=0.1035 \
    PREDIFY_FIXED_TS_TOL_S=0.001 \
    PREDIFY_SEED=0 \
    "$PYTHON_BIN" -u -m predify2021.mce_scores.evaluate_kitti_real_frame_pc_phase1 \
    2>&1 | tee "$LOG_PATH"

mv "$MANIFEST_PATH" "$OUTPUT_ROOT/manifest.txt"
