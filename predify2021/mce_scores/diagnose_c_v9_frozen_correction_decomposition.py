"""Frozen C-V9 correction-chain decomposition on KITTI-STEP Full9.

Inference-only counterpart of the C-V8 decomposition.  The C-V9 checkpoint is
frozen and evaluated through ``Z_cur``, raw Proposal, tanh-bounded Proposal,
and deployed gated correction paths.  No weights are trained or modified.
"""

import argparse
import json
from pathlib import Path

import torch

from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import (
    sequence_groups,
)
from predify2021.mce_scores import (
    diagnose_c_v7_semantic_decodability_probes as base,
)
from predify2021.mce_scores import (
    diagnose_c_v8_frozen_correction_decomposition as c_v8_diag,
)
from predify2021.mce_scores import (
    train_kitti_step_task_space_prior_c_v5_multiframe_candidate_memory as c_v5,
)
from predify2021.mce_scores import (
    train_kitti_step_task_space_prior_c_v9_proposal_supervision as c_v9,
)
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_direct_error_proposal_corrector import (
    DirectErrorProposalCorrector,
)
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_semantic_hysteresis import (
    EulerDynamicsError,
)


CHECKPOINT_DEFAULT = str(Path(c_v9.OUTPUT_DEFAULT) / "best.pt")
RESULT_DEFAULT = "results/kitti_step_c_v9_frozen_correction_decomposition"
C_V7_REFERENCE_DEFAULT = (
    "results/kitti_step_c_v7_frozen_correction_decomposition/summary.json"
)
PATHS = c_v8_diag.PATHS
RECONSTRUCTION_ATOL = 1e-6
METRIC_ATOL = 1e-8


def _load_c_v9_corrector(checkpoint_path, device):
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"C-V9 checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu")
    if "corrector_state_dict" not in payload:
        raise KeyError(f"checkpoint lacks corrector_state_dict: {path}")
    experiment = str(payload.get("experiment", ""))
    if experiment and "c_v9" not in experiment.lower():
        raise RuntimeError(
            f"checkpoint is not identified as C-V9: experiment={experiment!r}"
        )
    architecture = dict(payload.get("architecture") or {})
    corrector = DirectErrorProposalCorrector(
        num_classes=c_v5.NUM_CLASSES,
        history_length=int(architecture.get("history_length", c_v5.HISTORY_LENGTH)),
        hidden_channels=c_v5.CONTROLLER_HIDDEN_CHANNELS,
        g_max=float(architecture.get("g_max", c_v9.G_MAX)),
        gate_bias=float(architecture.get("gate_bias_init", c_v9.GATE_BIAS)),
    ).to(device)
    corrector.load_state_dict(payload["corrector_state_dict"], strict=True)
    corrector.requires_grad_(False)
    corrector.eval()
    return corrector, payload


def _load_stack(args):
    device = torch.device("cuda")
    c_v5._validate_bounded_motion_checkpoint(args.residual_checkpoint)
    model = c_v5.load_fast_b_model(args.fast_b_checkpoint).cuda().eval()
    model.requires_grad_(False)
    observer, _ = c_v5._load_frozen_observer(args.observer_checkpoint)
    residual, _ = c_v5._load_frozen_residual(args.residual_checkpoint, observer)
    correction, mask_predictor, _ = c_v5._load_frozen_e1_base(
        args.base_checkpoint, observer
    )
    refiner, _ = c_v5._load_frozen_cv3_refiner(args.c_v3_checkpoint)
    corrector, payload = _load_c_v9_corrector(args.c_v9_checkpoint, device)
    dynamics_config = dict(payload.get("dynamics") or {})
    dynamics = EulerDynamicsError(
        tau_e=float(dynamics_config.get("tau_e", c_v5.DYNAMICS_TAU_E)),
        k_e=float(dynamics_config.get("k_e", c_v5.DYNAMICS_K_E)),
        dt=float(dynamics_config.get("dt", c_v5.DYNAMICS_DT)),
    )
    for module in (observer, residual, correction, mask_predictor, refiner, corrector):
        if isinstance(module, torch.nn.Module):
            module.eval()
            module.requires_grad_(False)
    return (
        model, observer, residual, correction, mask_predictor, refiner,
        corrector, dynamics, payload,
    )


def _load_c_v7_reference(path):
    reference = Path(path)
    if not reference.is_file():
        return {"available": False, "path": str(reference)}
    payload = json.loads(reference.read_text())
    metrics = payload.get("metrics") or {}
    return {
        "available": True,
        "path": str(reference),
        "metrics": {name: metrics[name] for name in PATHS if name in metrics},
    }


def _compare(current, reference):
    if not reference.get("available"):
        return {"available": False}
    result = {"available": True, "paths": {}}
    for name in PATHS:
        if name not in reference["metrics"]:
            continue
        old = reference["metrics"][name]
        row = current[name]
        result["paths"][name] = {
            "delta_rescue_recovery": row["rescue_recovery_rate"] - old["rescue_recovery_rate"],
            "delta_rescue_margin_gain": row["rescue_margin_gain"] - old["rescue_margin_gain"],
            "delta_current_correct_damage": row["current_correct_damage_rate"] - old["current_correct_damage_rate"],
            "delta_mIoU": row["mIoU"] - old["mIoU"],
        }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/home/lin/predify/kitti_step")
    parser.add_argument("--fast-b-checkpoint", default=c_v5.FAST_B_CHECKPOINT_DEFAULT)
    parser.add_argument("--observer-checkpoint", default=c_v5.OBSERVER_CHECKPOINT_DEFAULT)
    parser.add_argument("--residual-checkpoint", default=c_v5.RESIDUAL_CHECKPOINT_DEFAULT)
    parser.add_argument("--base-checkpoint", default=c_v5.BASE_CHECKPOINT_DEFAULT)
    parser.add_argument("--c-v3-checkpoint", default=c_v5.C_V3_CHECKPOINT_DEFAULT)
    parser.add_argument("--c-v9-checkpoint", default=CHECKPOINT_DEFAULT)
    parser.add_argument("--result-output", default=RESULT_DEFAULT)
    parser.add_argument("--c-v7-reference", default=C_V7_REFERENCE_DEFAULT)
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    stack = _load_stack(args)
    model, observer, residual, correction, mask_predictor, refiner, corrector, dynamics, payload = stack
    val = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "val")
    all_groups = sequence_groups(val)
    missing = [sequence for sequence in c_v5.FULL9 if sequence not in all_groups]
    if missing:
        raise RuntimeError(f"missing Full9 validation sequences: {missing}")
    groups = {sequence: all_groups[sequence] for sequence in c_v5.FULL9}

    evaluated = c_v8_diag._evaluate(
        model, observer, residual, correction, mask_predictor, refiner,
        corrector, dynamics, groups,
    )
    measured = evaluated["metrics"]
    checkpoint_metric = payload.get("metrics", {}).get("c_v9")
    if not isinstance(checkpoint_metric, dict) or "mIoU" not in checkpoint_metric:
        raise RuntimeError("C-V9 checkpoint lacks metrics.c_v9.mIoU")
    diff = measured["z_final"]["mIoU"] - float(checkpoint_metric["mIoU"])
    if abs(diff) > METRIC_ATOL:
        raise RuntimeError(
            f"C-V9 z_final mIoU mismatch: measured={measured['z_final']['mIoU']} "
            f"checkpoint={checkpoint_metric['mIoU']} diff={diff}"
        )

    result = dict(evaluated)
    result.update({
        "checkpoint": {
            "path": args.c_v9_checkpoint,
            "epoch": payload.get("epoch"),
            "experiment": payload.get("experiment"),
            "g_max": corrector.g_max,
        },
        "checkpoint_metric_contract": {
            "available": True,
            "checkpoint_mIoU": float(checkpoint_metric["mIoU"]),
            "measured_z_final_mIoU": measured["z_final"]["mIoU"],
            "difference": diff,
            "atol": METRIC_ATOL,
        },
        "definition": {
            "z_cur": "frozen C-V3 current logits",
            "z_raw": "z_cur + upsample(DeltaZ_proposal_raw)",
            "z_tanh": "z_cur + upsample(tanh(DeltaZ_proposal_raw))",
            "z_final": "z_cur + upsample(gate) * upsample(tanh(DeltaZ_proposal_raw))",
            "validity": "same full-resolution any-history-valid mask as deployed C-V9",
            "rescue_mask": "same formal C-V7 mask",
            "training": False,
            "weights_modified": False,
        },
    })
    reference = _load_c_v7_reference(args.c_v7_reference)
    result["c_v7_reference"] = reference
    result["delta_vs_c_v7_decomposition"] = _compare(measured, reference)
    output = Path(args.result_output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
