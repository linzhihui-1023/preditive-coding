#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PREDIFY_PYTHON_BIN:-/home/lin/anaconda3/envs/predifyproject/bin/python}"
GIT_REVISION="$(git -C "$REPO_ROOT" rev-parse HEAD)"
SHORT_REVISION="${GIT_REVISION:0:7}"
OUTPUT_ROOT="${PREDIFY_LOCAL_MOTION_OUTPUT_ROOT:-/tmp/predify-storage/experiments/vgg_local_motion_${SHORT_REVISION}}"

if [[ -n "$(git -C "$REPO_ROOT" status --porcelain)" ]]; then
    echo "Refusing a formal local-motion run from a dirty worktree." >&2
    exit 2
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
    echo "Refusing to overwrite local-motion output: $OUTPUT_ROOT" >&2
    exit 3
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
    OMP_NUM_THREADS=24 \
    MKL_NUM_THREADS=24 \
    PREDIFY_GIT_REVISION="$GIT_REVISION" \
    PREDIFY_KITTI_ROOT=/home/lin/predify/kitti_raw \
    PREDIFY_TRAIN_DRIVES=2011_09_26/2011_09_26_drive_0005_sync \
    PREDIFY_VAL_DRIVES=2011_09_26/2011_09_26_drive_0011_sync \
    PREDIFY_KITTI_CAMERA=image_02 \
    PREDIFY_FIXED_TS_S=0.1035 \
    PREDIFY_FIXED_TS_TOL_S=0.001 \
    PREDIFY_LOCAL_MOTION_BATCH_SIZE=4 \
    PREDIFY_LOCAL_MOTION_NUM_WORKERS=0 \
    PREDIFY_LOCAL_MOTION_OUTPUT_DIR="$OUTPUT_ROOT" \
    "$PYTHON_BIN" -u -m predify2021.mce_scores.diagnose_kitti_local_motion
