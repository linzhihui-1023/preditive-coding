"""Frozen C-V8 correction-chain decomposition on KITTI-STEP Full9.

中文：C-V8 冻结修正链分解诊断。

This is inference-only.  The trained C-V8 checkpoint is frozen and the same
causal frames are evaluated through four logit paths:

  z_cur   = frozen C-V3 current logits
  z_raw   = z_cur + upsample(DeltaZ_proposal_raw)
  z_tanh  = z_cur + upsample(tanh(DeltaZ_proposal_raw))
  z_final = z_cur + upsample(gate) * upsample(tanh(DeltaZ_proposal_raw))

No weights are changed and no probe is trained.  Rescue / current-correct masks
reuse the formal C-V7 diagnostic definitions, so C-V8 is directly comparable
with the existing C-V7 frozen decomposition.
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
    train_kitti_step_task_space_prior_c_v5_multiframe_candidate_memory as c_v5,
)
from predify2021.mce_scores import (
    train_kitti_step_task_space_prior_c_v7_error_residual_correction as c_v7,
)
from predify2021.mce_scores import (
    train_kitti_step_task_space_prior_c_v8_direct_error_proposal_gate as c_v8,
)
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_direct_error_proposal_corrector import (
    DirectErrorProposalCorrector,
)
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_semantic_hysteresis import (
    EulerDynamicsError,
)


C_V8_CHECKPOINT_DEFAULT = str(Path(c_v8.OUTPUT_DEFAULT) / "best.pt")
RESULT_DEFAULT = "results/kitti_step_c_v8_frozen_correction_decomposition"
C_V7_REFERENCE_DEFAULT = (
    "results/kitti_step_c_v7_frozen_correction_decomposition/summary.json"
)
PATHS = ("z_cur", "z_raw", "z_tanh", "z_final")


def _load_c_v8_corrector(checkpoint_path, device):
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"C-V8 checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu")
    if "corrector_state_dict" not in payload:
        raise KeyError(f"checkpoint lacks corrector_state_dict: {path}")
    experiment = str(payload.get("experiment", ""))
    if experiment and "c_v8" not in experiment.lower():
        raise RuntimeError(
            f"checkpoint is not identified as C-V8: experiment={experiment!r}"
        )

    architecture = dict(payload.get("architecture") or {})
    corrector = DirectErrorProposalCorrector(
        num_classes=c_v5.NUM_CLASSES,
        history_length=int(architecture.get("history_length", c_v5.HISTORY_LENGTH)),
        hidden_channels=c_v5.CONTROLLER_HIDDEN_CHANNELS,
        g_max=float(architecture.get("g_max", c_v8.G_MAX)),
        gate_bias=float(architecture.get("gate_bias_init", c_v8.GATE_BIAS)),
    ).to(device)
    corrector.load_state_dict(payload["corrector_state_dict"], strict=True)
    corrector.requires_grad_(False)
    corrector.eval()
    return corrector, payload


def _build_dynamics(payload):
    config = dict(payload.get("dynamics") or {})
    return EulerDynamicsError(
        tau_e=float(config.get("tau_e", c_v5.DYNAMICS_TAU_E)),
        k_e=float(config.get("k_e", c_v5.DYNAMICS_K_E)),
        dt=float(config.get("dt", c_v5.DYNAMICS_DT)),
    )


def _load_frozen_stack(args):
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
    corrector, payload = _load_c_v8_corrector(args.c_v8_checkpoint, device)
    dynamics = _build_dynamics(payload)

    for module in (observer, residual, correction, mask_predictor, refiner, corrector):
        if isinstance(module, torch.nn.Module):
            module.eval()
            module.requires_grad_(False)
    return (
        model,
        observer,
        residual,
        correction,
        mask_predictor,
        refiner,
        corrector,
        dynamics,
        payload,
    )


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
    gate = F.interpolate(
        gate_low, size=full_size, mode="bilinear", align_corners=False
    )

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
        name: torch.zeros((c_v5.NUM_CLASSES, c_v5.NUM_CLASSES), dtype=torch.int64)
        for name in PATHS
    }
    diag = {name: _new_diag() for name in PATHS}
    correction_frames = 0

    for sequence in c_v5.FULL9:
        samples = groups[sequence]
        # Match the existing Full9 evaluator: the first two frames use Host.
        for host_logits, gt_cpu in base._first_two_host_predictions(model, samples):
            pred = host_logits.argmax(1)[0].cpu()
            for name in PATHS:
                c_v5.update_confusion_matrix(confusion[name], pred, gt_cpu)

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
            rescue = frame["masks"]["rescue"]
            current_correct = frame["masks"]["current_correct"]
            current_margin = base._true_class_margin(frame["c_v3_logits"], gt_cpu)
            candidates = _logit_paths(frame)

            n_rescue = int(rescue.sum().item())
            n_correct = int(current_correct.sum().item())
            for name, logits in candidates.items():
                pred = logits.argmax(1)[0]
                c_v5.update_confusion_matrix(confusion[name], pred.cpu(), gt_cpu)
                row = diag[name]
                row["rescue_pixels"] += n_rescue
                row["current_correct_pixels"] += n_correct

                if n_rescue:
                    margin_gain = base._true_class_margin(logits, gt_cpu) - current_margin
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
            "mIoU": float(torch.nanmean(c_v5.compute_iou(confusion[name])).item()),
            "rescue_pixels": row["rescue_pixels"],
            "rescue_recovery_rate": row["rescue_recovered"] / rescue_den,
            "rescue_margin_gain": row["rescue_margin_gain_sum"] / rescue_den,
            "current_correct_pixels": row["current_correct_pixels"],
            "current_correct_damage_rate": row["current_correct_damaged"] / correct_den,
        }
    return {"correction_path_frames": correction_frames, "metrics": metrics}


def _load_c_v7_reference(path):
    reference = Path(path)
    if not reference.is_file():
        return {"available": False, "path": str(reference)}
    payload = json.loads(reference.read_text())
    metrics = payload.get("metrics") or {}
    return {
        "available": True,
        "path": str(reference),
        "metrics": {
            name: metrics.get(name)
            for name in PATHS
            if isinstance(metrics.get(name), dict)
        },
    }


def _compare_to_c_v7(current_metrics, reference):
    if not reference.get("available"):
        return {"available": False}
    result = {"available": True, "paths": {}}
    ref_metrics = reference.get("metrics") or {}
    for name in PATHS:
        if name not in ref_metrics:
            continue
        current = current_metrics[name]
        old = ref_metrics[name]
        result["paths"][name] = {
            "delta_rescue_recovery": (
                current["rescue_recovery_rate"] - old["rescue_recovery_rate"]
            ),
            "delta_rescue_margin_gain": (
                current["rescue_margin_gain"] - old["rescue_margin_gain"]
            ),
            "delta_current_correct_damage": (
                current["current_correct_damage_rate"]
                - old["current_correct_damage_rate"]
            ),
            "delta_mIoU": current["mIoU"] - old["mIoU"],
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
    parser.add_argument("--c-v8-checkpoint", default=C_V8_CHECKPOINT_DEFAULT)
    parser.add_argument("--result-output", default=RESULT_DEFAULT)
    parser.add_argument("--c-v7-reference", default=C_V7_REFERENCE_DEFAULT)
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
        payload,
    ) = _load_frozen_stack(args)

    val = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "val")
    all_groups = sequence_groups(val)
    missing = [sequence for sequence in c_v5.FULL9 if sequence not in all_groups]
    if missing:
        raise RuntimeError(f"missing Full9 validation sequences: {missing}")
    groups = {sequence: all_groups[sequence] for sequence in c_v5.FULL9}

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
        "path": args.c_v8_checkpoint,
        "epoch": payload.get("epoch"),
        "experiment": payload.get("experiment"),
        "g_max": corrector.g_max,
    }
    result["definition"] = {
        "z_cur": "frozen C-V3 current logits",
        "z_raw": "z_cur + upsample(DeltaZ_proposal_raw)",
        "z_tanh": "z_cur + upsample(tanh(DeltaZ_proposal_raw))",
        "z_final": "z_cur + upsample(gate) * upsample(tanh(DeltaZ_proposal_raw))",
        "validity": "same full-resolution any-history-valid mask as deployed C-V8",
        "rescue_mask": "same formal mask: C-V3 wrong AND any valid aligned history predicts GT",
        "training": False,
        "weights_modified": False,
    }
    reference = _load_c_v7_reference(args.c_v7_reference)
    result["c_v7_reference"] = reference
    result["delta_vs_c_v7_decomposition"] = _compare_to_c_v7(
        result["metrics"], reference
    )

    result_output = Path(args.result_output)
    result_output.mkdir(parents=True, exist_ok=True)
    (result_output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
