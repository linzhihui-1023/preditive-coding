#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PREDIFY_PYTHON_BIN:-/home/lin/anaconda3/envs/predifyproject/bin/python}"
GIT_REVISION="$(git -C "$REPO_ROOT" rev-parse HEAD)"
SHORT_REVISION="${GIT_REVISION:0:7}"
OUTPUT_ROOT="${PREDIFY_STAGE4_ERROR_STATE_OUTPUT_ROOT:-/tmp/predify-storage/experiments/stage4_dynamic_error_state_phase1_${SHORT_REVISION}}"
CHECKPOINT="${PREDIFY_STAGE4_ERROR_STATE_CHECKPOINT:-/tmp/predify-storage/experiments/seed0_stage4_multidrive_c887e94/stage4_multidrive_seed0_best_student.pt}"

if [[ "$(git -C "$REPO_ROOT" branch --show-current)" != "predify-selective-adaptation-v2" ]]; then
    echo "Stage-4 error-state work must run from predify-selective-adaptation-v2." >&2
    exit 2
fi
if [[ "$REPO_ROOT" != "/home/lin/predify2021_selective_adaptation" ]]; then
    echo "Stage-4 error-state work must use the isolated worktree." >&2
    exit 3
fi
if [[ -n "$(git -C "$REPO_ROOT" status --porcelain)" ]]; then
    echo "Refusing a formal error-state diagnostic from a dirty worktree." >&2
    exit 4
fi
if [[ -e "$OUTPUT_ROOT" || -e "${OUTPUT_ROOT}.log" ]]; then
    echo "Refusing to overwrite error-state output: $OUTPUT_ROOT" >&2
    exit 5
fi

printf 'evaluation_git_revision=%s\ncheckpoint_git_revision=c887e94edda86885fd0bedfaaf25e56c957719bb\ncheckpoint=%s\nmodel=stage4_aligned_temporal_difference\nnetwork_update=false\npredictor_uses_dynamic_error_state=false\nfrozen_test_read=false\ndrives=0011,0039\ncorruptions=gaussian_blur,iid_gaussian_noise,rgb_bias_domain_proxy\nconditions=persistent,shuffled\nonly_condition_difference=temporal_order_of_same_severity_multiset\nerror=e_t=Fhat_t_stage4-F_t_stage4\ndynamic_state=epsilon_t\n' \
    "$GIT_REVISION" "$CHECKPOINT" > "${OUTPUT_ROOT}.manifest.txt"

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
    PREDIFY_CHECKPOINT_GIT_REVISION=c887e94edda86885fd0bedfaaf25e56c957719bb \
    PREDIFY_STAGE4_ERROR_STATE_CHECKPOINT="$CHECKPOINT" \
    PREDIFY_STAGE4_ERROR_STATE_OUTPUT_DIR="$OUTPUT_ROOT" \
    PREDIFY_KITTI_ROOT=/home/lin/predify/kitti_raw \
    PREDIFY_KITTI_CAMERA=image_02 \
    PREDIFY_STAGE4_ERROR_STATE_DRIVES=2011_09_26/2011_09_26_drive_0011_sync,2011_09_26/2011_09_26_drive_0039_sync \
    PREDIFY_FIXED_TS_S=0.1035 \
    PREDIFY_FIXED_TS_TOL_S=0.001 \
    PREDIFY_ERROR_BASELINE_FRAMES=40 \
    PREDIFY_ERROR_RECOVERY_FRAMES=40 \
    PREDIFY_ERROR_SEVERITY_LEVELS=0.25,0.5,0.75,1.0 \
    PREDIFY_ERROR_FRAMES_PER_LEVEL=20 \
    PREDIFY_ERROR_REPLICATES=4 \
    PREDIFY_ERROR_SHUFFLE_SEED=20260813 \
    PREDIFY_ERROR_WINDOW_SIZE=8 \
    "$PYTHON_BIN" -u -m predify2021.mce_scores.evaluate_kitti_stage4_dynamic_error_state \
    2>&1 | tee "${OUTPUT_ROOT}.log"

mv "${OUTPUT_ROOT}.manifest.txt" "$OUTPUT_ROOT/manifest.txt"
