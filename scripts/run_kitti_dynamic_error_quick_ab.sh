#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PREDIFY_PYTHON_BIN:-/home/lin/anaconda3/envs/predifyproject/bin/python}"
SHORT_REVISION="$(git -C "$REPO_ROOT" rev-parse --short HEAD)"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${PREDIFY_DYNAMIC_QUICK_OUTPUT_ROOT:-/tmp/predify-storage/experiments/dynamic_error_quick_ab_${SHORT_REVISION}_${RUN_STAMP}}"

if [[ -e "$OUTPUT_ROOT" ]]; then
    echo "Refusing to overwrite quick A/B output: $OUTPUT_ROOT" >&2
    exit 2
fi

mkdir -p "$OUTPUT_ROOT"

run_case() {
    local case_name="$1"
    local dynamics_enabled="$2"
    local output_dir="$OUTPUT_ROOT/$case_name"

    env \
        PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
        PREDIFY_ENABLE_DYNAMICS_ERROR="$dynamics_enabled" \
        PREDIFY_ENABLE_TEMPORAL_PREDICTION=1 \
        PREDIFY_EPOCHS="${PREDIFY_QUICK_EPOCHS:-1}" \
        PREDIFY_EARLY_STOPPING_PATIENCE="${PREDIFY_QUICK_PATIENCE:-1}" \
        PREDIFY_TRAIN_SEQUENCE_LIMIT="${PREDIFY_QUICK_TRAIN_SEQUENCES:-1}" \
        PREDIFY_VAL_SEQUENCE_LIMIT="${PREDIFY_QUICK_VAL_SEQUENCES:-1}" \
        PREDIFY_FRAMES_PER_SEQUENCE_LIMIT="${PREDIFY_QUICK_FRAMES_PER_SEQUENCE:-64}" \
        PREDIFY_LOADER_WORKERS="${PREDIFY_QUICK_LOADER_WORKERS:-2}" \
        PREDIFY_DYNAMIC_ERROR_SCALE_Z1="${PREDIFY_DYNAMIC_ERROR_SCALE_Z1:-1.0}" \
        PREDIFY_DYNAMIC_ERROR_SCALE_Z4="${PREDIFY_DYNAMIC_ERROR_SCALE_Z4:-1.0}" \
        PREDIFY_SEMANTIC_TEMPORAL_ERROR_OUTPUT_DIR="$output_dir" \
        "$PYTHON_BIN" -u -m \
        predify2021.mce_scores.train_kitti_step_semantic_temporal_error_correction \
        2>&1 | tee "$OUTPUT_ROOT/${case_name}.log"
}

run_case temporal_head_baseline 0
run_case dynamic_aware 1

"$PYTHON_BIN" - "$OUTPUT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
baseline = json.loads((root / "temporal_head_baseline" / "summary.json").read_text())
dynamics = json.loads((root / "dynamic_aware" / "summary.json").read_text())

def metrics(summary):
    return summary["final_validation"]

base = metrics(baseline)
dyn = metrics(dynamics)
comparison = {
    "baseline": base,
    "dynamic_aware": dyn,
    "dynamic_minus_baseline": {
        key: dyn[key] - base[key]
        for key in ("miou", "mvc8", "mvc16", "temporal_prediction_loss")
        if key in base and key in dyn
    },
    "quick_screen_only": True,
    "go_screen": bool(
        dyn.get("mvc8", float("-inf")) > base.get("mvc8", float("inf"))
        and dyn.get("mvc16", float("-inf")) > base.get("mvc16", float("inf"))
        and dyn.get("temporal_prediction_loss", float("inf"))
        < base.get("temporal_prediction_loss", float("-inf"))
        and dyn.get("miou", float("-inf")) >= base.get("miou", float("inf")) - 0.002
        and dyn.get("temporal_prediction_loss", float("inf"))
        < dyn.get("persistence_temporal_loss", float("-inf"))
    ),
}
(root / "comparison.json").write_text(
    json.dumps(comparison, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(comparison, indent=2, sort_keys=True))
PY

echo "Quick A/B results: $OUTPUT_ROOT/comparison.json"
