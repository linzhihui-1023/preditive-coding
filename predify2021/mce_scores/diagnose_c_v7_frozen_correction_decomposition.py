"""Frozen C-V7 correction-chain decomposition on KITTI-STEP Full9.

This is an inference-only diagnostic.  The trained CE-only C-V7 E1 stack is
frozen and the same causal frames are evaluated with four logit paths:

* ``z_cur``: frozen C-V3 current logits;
* ``z_raw``: ``z_cur + DeltaZ_raw``;
* ``z_tanh``: ``z_cur + tanh(DeltaZ_raw)``;
* ``z_final``: the deployed C-V7 ``z_cur + gate*tanh(DeltaZ_raw)``.

No weights are changed and no probe is trained.  The metrics are restricted to
the same rescue/protection masks used by the formal decodability diagnostic.
"""

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import (
    sequence_groups,
)
from predify2021.mce_scores import diagnose_c_v7_semantic_decodability_probes as base
from predify2021.mce_scores import (
    diagnose_c_v7_semantic_decodability_probes_formal as formal,
)


OUTPUT_DEFAULT = (
    "/home/lin/predify/experiments/"
    "kitti_step_c_v7_frozen_correction_decomposition"
)
RESULT_DEFAULT = "results/kitti_step_c_v7_frozen_correction_decomposition"
HIDDEN_PROBE_REFERENCE = (
    "results/kitti_step_c_v7_semantic_decodability_probes_formal/summary.json"
)

PATHS = ("z_cur", "z_raw", "z_tanh", "z_final")


def _logit_paths(frame):
    current = frame["c_v3_logits"]
    full_size = tuple(current.shape[-2:])
    valid = frame["any_valid_full"].to(current.dtype)
    raw_low = frame["delta_z_raw_low"]
    gate_low = frame["gate_low"]
    raw = F.interpolate(raw_low, size=full_size, mode="bilinear", align_corners=False)
    bounded = F.interpolate(
        torch.tanh(raw_low), size=full_size, mode="bilinear", align_corners=False
    )
    gate = F.interpolate(gate_low, size=full_size, mode="bilinear", align_corners=False)
    raw = raw * valid
    bounded = bounded * valid
    gated = gate * bounded
    return {
        "z_cur": current,
        "z_raw": current + raw,
        "z_tanh": current + bounded,
        "z_final": current + gated,
    }


def _new_diag():
    return {
        "rescue_pixels": 0,
        "rescue_recovered": 0,
        "rescue_margin_gain_sum": 0.0,
        "current_correct_pixels": 0,
        "current_correct_damaged": 0,
    }


@torch.inference_mode()
def _evaluate(
    model,
    observer,
    residual,
    correction,
    mask_predictor,
    refiner,
    corrector,
    dynamics,
    groups,
):
    confusion = {
        name: torch.zeros(
            (base.c_v5.NUM_CLASSES, base.c_v5.NUM_CLASSES), dtype=torch.int64
        )
        for name in PATHS
    }
    diag = {name: _new_diag() for name in PATHS}
    correction_frames = 0

    for sequence in base.c_v5.FULL9:
        samples = groups[sequence]
        # The first two frames follow the existing evaluator: all paths equal
        # the frozen Host/C-V3 output before a correction path exists.
        for host_logits, gt_cpu in base._first_two_host_predictions(model, samples):
            pred = host_logits.argmax(1)[0].cpu()
            for name in PATHS:
                base.c_v5.update_confusion_matrix(confusion[name], pred, gt_cpu)

        for frame in base._frozen_sequence_frames(
            model,
            observer,
            residual,
            correction,
            mask_predictor,
            refiner,
            corrector,
            dynamics,
            samples,
        ):
            correction_frames += 1
            gt_cpu = frame["gt_cpu"]
            gt = gt_cpu.to(frame["c_v3_logits"].device).long()
            masks = frame["masks"]
            rescue = masks["rescue"]
            current_correct = masks["current_correct"]
            current_margin = base._true_class_margin(frame["c_v3_logits"], gt_cpu)
            candidates = _logit_paths(frame)

            for name, logits in candidates.items():
                pred = logits.argmax(1)[0]
                base.c_v5.update_confusion_matrix(
                    confusion[name], pred.cpu(), gt_cpu
                )
                row = diag[name]
                n_rescue = int(rescue.sum().item())
                n_correct = int(current_correct.sum().item())
                row["rescue_pixels"] += n_rescue
                row["current_correct_pixels"] += n_correct
                if n_rescue:
                    margin_gain = (
                        base._true_class_margin(logits, gt_cpu) - current_margin
                    )
                    row["rescue_recovered"] += int(
                        (rescue & (pred == gt)).sum().item()
                    )
                    row["rescue_margin_gain_sum"] += float(
                        margin_gain[rescue].sum().item()
                    )
                if n_correct:
                    row["current_correct_damaged"] += int(
                        (current_correct & (pred != gt)).sum().item()
                    )

    metrics = {}
    for name in PATHS:
        row = diag[name]
        rescue_den = max(row["rescue_pixels"], 1)
        correct_den = max(row["current_correct_pixels"], 1)
        metrics[name] = {
            "mIoU": float(
                torch.nanmean(base.c_v5.compute_iou(confusion[name])).item()
            ),
            "rescue_pixels": row["rescue_pixels"],
            "rescue_recovery_rate": row["rescue_recovered"] / rescue_den,
            "rescue_margin_gain": row["rescue_margin_gain_sum"] / rescue_den,
            "current_correct_pixels": row["current_correct_pixels"],
            "current_correct_damage_rate": (
                row["current_correct_damaged"] / correct_den
            ),
        }
    return {
        "correction_path_frames": correction_frames,
        "metrics": metrics,
    }


def _load_hidden_probe_reference(path):
    reference = Path(path)
    if not reference.exists():
        return {"available": False, "path": str(reference)}
    payload = json.loads(reference.read_text())
    metrics = payload.get("history", [{}])[-1].get("full9", {}).get("metrics", {})
    row = metrics.get("hidden_32")
    if row is None:
        return {"available": False, "path": str(reference)}
    return {
        "available": True,
        "path": str(reference),
        "source": "existing formal frozen hidden_32 linear probe",
        "metrics": {
            "mIoU": row.get("mIoU"),
            "rescue_recovery_rate": row.get("rescue_recovery_rate"),
            "rescue_margin_gain": row.get("rescue_margin_gain"),
            "current_correct_damage_rate": row.get("current_correct_damage_rate"),
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/home/lin/predify/kitti_step")
    parser.add_argument(
        "--fast-b-checkpoint", default=base.c_v5.FAST_B_CHECKPOINT_DEFAULT
    )
    parser.add_argument(
        "--observer-checkpoint", default=base.c_v5.OBSERVER_CHECKPOINT_DEFAULT
    )
    parser.add_argument(
        "--residual-checkpoint", default=base.c_v5.RESIDUAL_CHECKPOINT_DEFAULT
    )
    parser.add_argument("--base-checkpoint", default=base.c_v5.BASE_CHECKPOINT_DEFAULT)
    parser.add_argument("--c-v3-checkpoint", default=base.c_v5.C_V3_CHECKPOINT_DEFAULT)
    parser.add_argument("--c-v7-checkpoint", default=base.C_V7_CE_CHECKPOINT_DEFAULT)
    parser.add_argument("--output", default=OUTPUT_DEFAULT)
    parser.add_argument("--result-output", default=RESULT_DEFAULT)
    parser.add_argument("--hidden-probe-reference", default=HIDDEN_PROBE_REFERENCE)
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    (
        model,
        observer,
        residual,
        correction,
        mask_predictor,
        refiner,
        corrector,
        dynamics,
        c_v7_payload,
    ) = base._load_frozen_stack(args)
    val = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "val")
    all_groups = sequence_groups(val)
    missing = [sequence for sequence in base.c_v5.FULL9 if sequence not in all_groups]
    if missing:
        raise RuntimeError(f"missing Full9 validation sequences: {missing}")
    groups = {sequence: all_groups[sequence] for sequence in base.c_v5.FULL9}

    result = _evaluate(
        model,
        observer,
        residual,
        correction,
        mask_predictor,
        refiner,
        corrector,
        dynamics,
        groups,
    )
    result["checkpoint"] = {
        "path": args.c_v7_checkpoint,
        "epoch": c_v7_payload.get("epoch"),
        "experiment": c_v7_payload.get("experiment"),
        "g_max": corrector.g_max,
    }
    result["hidden_32_linear_probe"] = _load_hidden_probe_reference(
        args.hidden_probe_reference
    )
    result["definition"] = {
        "z_cur": "frozen C-V3 current logits",
        "z_raw": "z_cur + upsample(DeltaZ_raw)",
        "z_tanh": "z_cur + upsample(tanh(DeltaZ_raw))",
        "z_final": "z_cur + upsample(gate) * upsample(tanh(DeltaZ_raw))",
        "validity": "same full-resolution any-history-valid mask as deployed C-V7",
        "training": False,
        "weights_modified": False,
    }
    output = Path(args.output)
    result_output = Path(args.result_output)
    output.mkdir(parents=True, exist_ok=True)
    result_output.mkdir(parents=True, exist_ok=True)
    (result_output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
