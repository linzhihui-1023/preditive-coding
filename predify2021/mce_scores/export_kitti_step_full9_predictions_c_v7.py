"""Export KITTI-STEP Full9 Host and C-V7 predictions.

中文：导出 KITTI-STEP Full9（完整 9 序列）的 Host（宿主）与 C-V7 预测 PNG，
供 PC / VEC8 / VEC16（感知一致性 / 8帧与16帧视频评估一致性）重复评测。

Constraints / 约束：
- no training（不训练）；
- no RAFT construction（不构造 RAFT）；
- no Oracle / GT-assisted decision（模型决策不使用真实标签）；
- C-V7 follows History -> Prediction -> Prediction Error -> Residual Correction；
- first two frames follow the existing evaluator and fall back to Host；
- saved predictions are read back after inference for mIoU/mVC sanity checking.
"""

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import sequence_groups
from predify2021.mce_scores import export_kitti_step_full9_predictions as export_base
from predify2021.mce_scores import (
    train_kitti_step_task_space_prior_c_v5_multiframe_candidate_memory as c_v5,
)
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v7_error_residual_correction import (
    GATE_BIAS,
    G_MAX,
    _correction_evidence,
)
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_multihypothesis_error_residual_corrector import (
    MultiHypothesisErrorResidualCorrector,
)
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_semantic_hysteresis import (
    EulerDynamicsError,
)


DEFAULT_OUTPUT_ROOT = "/home/lin/predify/predictions/kitti_step_full9_c_v7"
DEFAULT_MODEL_CHECKPOINT = (
    "/home/lin/predify/experiments/"
    "kitti_step_task_space_prior_c_v7_error_residual_correction/best.pt"
)
NAMES = ("host", "current_model")


def _load_corrector(checkpoint_path, device):
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"C-V7 checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu")
    if "corrector_state_dict" not in payload:
        raise KeyError(f"checkpoint lacks corrector_state_dict: {path}")
    experiment = str(payload.get("experiment", ""))
    if experiment and "c_v7" not in experiment.lower():
        raise RuntimeError(
            f"checkpoint is not identified as C-V7: experiment={experiment!r}"
        )

    architecture = dict(payload.get("architecture") or {})
    corrector = MultiHypothesisErrorResidualCorrector(
        num_classes=c_v5.NUM_CLASSES,
        history_length=int(architecture.get("history_length", c_v5.HISTORY_LENGTH)),
        hidden_channels=c_v5.CONTROLLER_HIDDEN_CHANNELS,
        g_max=float(architecture.get("g_max", G_MAX)),
        gate_bias=float(architecture.get("gate_bias_init", GATE_BIAS)),
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


def _reference_metrics(payload):
    metrics = payload.get("metrics") or {}
    host = metrics.get("host")
    model = metrics.get("c_v7")
    if not isinstance(host, dict) or not isinstance(model, dict):
        return None
    return {"host": host, "current_model": model}


@torch.inference_mode()
def _export_full9(
    model,
    observer,
    residual,
    correction,
    mask_predictor,
    refiner,
    corrector,
    dynamics,
    groups,
    output_root,
    overwrite,
):
    output_root = Path(output_root)
    counts = {name: 0 for name in NAMES}
    per_sequence = {}

    for sequence in c_v5.FULL9:
        samples = groups[sequence]
        previous = None
        pending_motion = motion_hidden = None
        transport_hidden = semantic_hidden = mask_hidden = None
        semantic_state_low = None
        memory_state = None
        error_hidden = None
        dynamics_state = None
        raw_history = []
        motion_history = []
        seq_counts = {name: 0 for name in NAMES}

        for sample in samples:
            image, host_logits, host_low, current_c1, output_size = c_v5._host_observation(
                model,
                sample,
            )
            host_pred = host_logits.argmax(1)

            if previous is None:
                model_pred = host_pred
                semantic_state_low = torch.zeros_like(host_low)
                raw_history = [host_logits.detach()]
                previous = (image, host_low.detach(), current_c1.detach())

            elif pending_motion is None:
                _, previous_low, previous_c1 = previous
                observed = c_v5._observe_motion(
                    observer,
                    previous_low,
                    previous_c1,
                    host_low,
                    current_c1,
                )
                motion_error = F.softmax(host_low, dim=1) - F.softmax(previous_low, dim=1)
                pending_motion, _, motion_hidden = residual.predict_next(
                    observed,
                    motion_error,
                    None,
                )
                model_pred = host_pred
                raw_history = [host_logits.detach()]
                previous = (image, host_low.detach(), current_c1.detach())

            else:
                _, previous_low, previous_c1 = previous
                prior_low, _ = c_v5.warp_low_logits(previous_low, pending_motion)
                e1 = c_v5._frozen_e1_step(
                    correction,
                    mask_predictor,
                    current_c1,
                    host_low,
                    prior_low,
                    pending_motion,
                    semantic_state_low,
                    transport_hidden,
                    semantic_hidden,
                    mask_hidden,
                )
                transport_hidden = e1["transport_hidden"]
                semantic_hidden = e1["semantic_hidden"]
                mask_hidden = e1["mask_hidden"]
                semantic_state_low = e1["semantic_state_low"]

                memory_row, memory_state, _, c_v3_logits = c_v5._frozen_cv3_step(
                    refiner,
                    current_c1,
                    host_low,
                    prior_low,
                    e1,
                    pending_motion,
                    memory_state,
                    output_size,
                    host_logits,
                )
                candidate_rows = c_v5._build_history_candidates(
                    raw_history,
                    motion_history,
                    pending_motion,
                    corrector.history_length,
                )
                evidence = _correction_evidence(
                    corrector,
                    dynamics,
                    c_v3_logits,
                    candidate_rows,
                    pending_motion,
                    e1["transportability_low"],
                    memory_row["memory_reliability"],
                    error_hidden,
                    dynamics_state,
                )
                error_hidden = evidence["row"]["hidden"]
                dynamics_state = evidence["dynamics_state"]
                model_pred = evidence["final_logits_full"].argmax(1)

                observed = c_v5._observe_motion(
                    observer,
                    previous_low,
                    previous_c1,
                    host_low,
                    current_c1,
                )
                motion_error = F.softmax(host_low, dim=1) - F.softmax(prior_low, dim=1)
                next_motion, _, next_motion_hidden = residual.predict_next(
                    observed,
                    motion_error,
                    motion_hidden,
                )

                raw_history.insert(0, c_v3_logits.detach())
                raw_history = raw_history[: corrector.history_length]
                motion_history.insert(0, pending_motion.detach())
                motion_history = motion_history[: max(corrector.history_length - 1, 0)]
                previous = (image, host_low.detach(), current_c1.detach())
                pending_motion = next_motion.detach()
                motion_hidden = next_motion_hidden.detach()

            predictions = {
                "host": host_pred.squeeze(0).detach().cpu(),
                "current_model": model_pred.squeeze(0).detach().cpu(),
            }
            frame_hw = tuple(predictions["host"].shape)
            if tuple(predictions["current_model"].shape) != frame_hw:
                raise RuntimeError("Host/current-model output shape mismatch")

            for name, prediction in predictions.items():
                export_base._save_mask(
                    output_root / name / str(sequence) / f"{sample['frame_id']}.png",
                    prediction,
                    frame_hw,
                    overwrite,
                )
                counts[name] += 1
                seq_counts[name] += 1

        per_sequence[str(sequence)] = {
            "frames": len(samples),
            "exported": seq_counts,
        }
        print(json.dumps({"sequence": str(sequence), "exported": seq_counts}), flush=True)

    return counts, per_sequence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/home/lin/predify/kitti_step")
    parser.add_argument("--model-checkpoint", default=DEFAULT_MODEL_CHECKPOINT)
    parser.add_argument("--fast-b-checkpoint", default=c_v5.FAST_B_CHECKPOINT_DEFAULT)
    parser.add_argument("--observer-checkpoint", default=c_v5.OBSERVER_CHECKPOINT_DEFAULT)
    parser.add_argument("--residual-checkpoint", default=c_v5.RESIDUAL_CHECKPOINT_DEFAULT)
    parser.add_argument("--base-checkpoint", default=c_v5.BASE_CHECKPOINT_DEFAULT)
    parser.add_argument("--c-v3-checkpoint", default=c_v5.C_V3_CHECKPOINT_DEFAULT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--metric-tolerance", type=float, default=1e-10)
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    export_base._preflight_output(args.output_root, args.overwrite)
    c_v5._validate_bounded_motion_checkpoint(args.residual_checkpoint)

    device = torch.device("cuda")
    model = c_v5.load_fast_b_model(args.fast_b_checkpoint).cuda().eval()
    model.requires_grad_(False)
    observer, _ = c_v5._load_frozen_observer(args.observer_checkpoint)
    residual, _ = c_v5._load_frozen_residual(args.residual_checkpoint, observer)
    correction, mask_predictor, _ = c_v5._load_frozen_e1_base(
        args.base_checkpoint,
        observer,
    )
    refiner, _ = c_v5._load_frozen_cv3_refiner(args.c_v3_checkpoint)
    corrector, payload = _load_corrector(args.model_checkpoint, device)
    dynamics = _build_dynamics(payload)

    val = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "val")
    all_groups = sequence_groups(val)
    missing = [sequence for sequence in c_v5.FULL9 if sequence not in all_groups]
    if missing:
        raise RuntimeError(f"Missing Full9 validation sequences: {missing}")
    groups = {sequence: all_groups[sequence] for sequence in c_v5.FULL9}

    counts, per_sequence = _export_full9(
        model,
        observer,
        residual,
        correction,
        mask_predictor,
        refiner,
        corrector,
        dynamics,
        groups,
        args.output_root,
        args.overwrite,
    )
    verified = export_base._verify_saved_masks(groups, args.output_root)
    reference = _reference_metrics(payload)
    comparison = export_base._compare_reference(
        verified["metrics"],
        reference,
        args.metric_tolerance,
    )
    if comparison["available"] and not comparison["passed"]:
        raise RuntimeError(
            "exported PNG metrics do not match C-V7 checkpoint metrics: "
            f"{comparison['diff']}"
        )

    manifest = {
        "experiment": "C-V7 Full9 Prediction Export",
        "model_checkpoint": args.model_checkpoint,
        "model_epoch": payload.get("epoch"),
        "output_root": args.output_root,
        "counts": counts,
        "per_sequence": per_sequence,
        "verified": verified,
        "checkpoint_metric_comparison": comparison,
        "architecture": {
            "history_length": corrector.history_length,
            "causal_path": "History -> Prediction -> Prediction Error -> Residual Correction",
            "raw_history_enters_correction_head": False,
            "error_hidden_motion_aligned": True,
            "error_hidden_reliability_gated": True,
            "g_max": corrector.g_max,
            "raft_inference": False,
            "gt_inference_decision": False,
        },
    }
    manifest_path = Path(args.output_root) / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
