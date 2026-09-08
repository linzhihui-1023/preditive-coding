"""Export KITTI-STEP Full9 Host and C-V7 predictions.

中文：导出 KITTI-STEP Full9（完整 9 序列）的 Host（宿主）与 C-V7 逐帧预测，
供 PC（感知一致性）、VEC8/16（8/16 帧视频评估一致性）等 prediction-only
（仅预测结果）指标重复使用。

约束：
- no training（不训练）；
- no RAFT / optical-flow teacher in model inference（模型推理不使用 RAFT/光流教师）；
- no GT-assisted decision（模型决策不读取 GT/真实标签）；
- C-V7 uses Prediction Error -> residual correction（预测误差 -> 残差修正），
  never hard Current/History replacement（绝不做当前/历史硬替换）；
- first two frames follow the validated initialization and fall back to Host
  （前两帧沿用已验证初始化并回退 Host）；
- GT is read only after export for saved-PNG verification（GT 只在导出完成后用于校验）。
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from predify2021.datasets.kitti_step import (
    KITTISTEPSegmentationDataset,
    semantic_mask_from_panoptic_png,
)
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import (
    sequence_groups,
)
from predify2021.mce_scores.evaluate_kitti_step_static_baseline import (
    compute_iou,
    update_confusion_matrix,
)
from predify2021.mce_scores.train_kitti_step_frozen_error_utility_probe import (
    load_fast_b_model,
)
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v2_stage1b_residual import (
    _load_frozen_observer,
)
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v2_stage1b_role_vs_mask import (
    _load_frozen_residual,
)
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v2_stage1b_role_vs_mask_r2 import (
    _validate_bounded_motion_checkpoint,
)
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v2_stage1b_utility_from_e1_r2 import (
    _load_frozen_e1_base,
)
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v3_temporal_semantic_prediction import (
    _load_frozen_cv3_refiner,
)
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v7_multihypothesis_error_residual_correction import (
    ERROR_HIDDEN_CHANNELS,
    CORRECTION_CHANNELS,
    GATE_MAX,
    _build_c_v7_evidence,
)
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_multihypothesis_error_residual_corrector import (
    MultiHypothesisErrorResidualCorrector,
)
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_semantic_hysteresis import (
    EulerDynamicsError,
)
from predify2021.mce_scores import (
    train_kitti_step_task_space_prior_c_v5_multiframe_candidate_memory as c_v5,
)


NUM_CLASSES = 19
EXPECTED_FULL9_FRAMES = 2981
NAMES = ("host", "current_model")
DEFAULT_OUTPUT_ROOT = "/home/lin/predify/predictions/kitti_step_full9_c_v7"
DEFAULT_MODEL_CHECKPOINT = (
    "/home/lin/predify/experiments/"
    "kitti_step_task_space_prior_c_v7_multihypothesis_error_residual_correction/best.pt"
)
METRIC_KEYS = ("mIoU", "mVC8", "mVC16")


def _load_corrector(checkpoint_path, device):
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"C-V7 checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu")
    if "corrector_state_dict" not in payload:
        raise KeyError(f"checkpoint lacks corrector_state_dict: {path}")

    architecture = dict(payload.get("architecture") or {})
    history_length = int(architecture.get("history_length", c_v5.HISTORY_LENGTH))
    gate_max = float(architecture.get("gate_max", GATE_MAX))
    corrector = MultiHypothesisErrorResidualCorrector(
        num_classes=NUM_CLASSES,
        history_length=history_length,
        hidden_channels=ERROR_HIDDEN_CHANNELS,
        correction_channels=CORRECTION_CHANNELS,
        gate_max=gate_max,
    ).to(device)
    corrector.load_state_dict(payload["corrector_state_dict"], strict=True)
    corrector.requires_grad_(False)
    corrector.eval()

    experiment = str(payload.get("experiment", ""))
    if experiment and "c_v7" not in experiment.lower():
        raise RuntimeError(
            "model checkpoint is not identified as C-V7: "
            f"experiment={experiment!r} path={path}"
        )
    return corrector, payload


def _build_dynamics(payload):
    config = dict(payload.get("dynamics") or {})
    return EulerDynamicsError(
        tau_e=float(config.get("tau_e", c_v5.DYNAMICS_TAU_E)),
        k_e=float(config.get("k_e", c_v5.DYNAMICS_K_E)),
        dt=float(config.get("dt", c_v5.DYNAMICS_DT)),
    )


def _preflight_output(output_root, overwrite):
    output_root = Path(output_root)
    if overwrite:
        return
    for name in NAMES:
        root = output_root / name
        if root.exists():
            first = next(root.rglob("*.png"), None)
            if first is not None:
                raise FileExistsError(
                    "prediction PNGs already exist; pass --overwrite to replace them. "
                    f"Example: {first}"
                )


def _save_mask(path, prediction, expected_hw, overwrite):
    prediction = torch.as_tensor(prediction).detach().cpu().long()
    if prediction.ndim == 3 and prediction.shape[0] == 1:
        prediction = prediction[0]
    if prediction.ndim != 2:
        raise ValueError(f"prediction must be HxW, got {tuple(prediction.shape)}")
    if tuple(prediction.shape) != tuple(expected_hw):
        raise ValueError(
            f"prediction size mismatch: pred={tuple(prediction.shape)} expected={tuple(expected_hw)}"
        )
    invalid = (prediction < 0) | (prediction >= NUM_CLASSES)
    if bool(invalid.any()):
        raise ValueError(
            f"prediction contains invalid class ids: {torch.unique(prediction[invalid])[:16].tolist()}"
        )

    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"prediction already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(prediction.numpy().astype(np.uint8, copy=False), mode="L").save(path)


def _load_saved_mask(path, expected_hw):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"exported prediction missing: {path}")
    with Image.open(path) as image:
        array = np.array(image, copy=True)
    if array.ndim != 2:
        raise ValueError(f"exported PNG must be single-channel HxW: {path}")
    prediction = torch.from_numpy(array.astype(np.int64, copy=False))
    if tuple(prediction.shape) != tuple(expected_hw):
        raise ValueError(
            f"exported PNG/GT size mismatch: pred={tuple(prediction.shape)} "
            f"gt={tuple(expected_hw)} path={path}"
        )
    invalid = (prediction < 0) | (prediction >= NUM_CLASSES)
    if bool(invalid.any()):
        raise ValueError(f"exported PNG contains invalid class ids: {path}")
    return prediction


def _metric_row(confusion, vc_sum, vc_count, name):
    return {
        "mIoU": float(torch.nanmean(compute_iou(confusion[name])).item()),
        "mVC8": vc_sum[name][8] / vc_count[name][8] if vc_count[name][8] else float("nan"),
        "mVC16": vc_sum[name][16] / vc_count[name][16] if vc_count[name][16] else float("nan"),
    }


def _reference_metrics(payload):
    metrics = payload.get("metrics") or {}
    if not isinstance(metrics, dict):
        return None
    host = metrics.get("host")
    model = metrics.get("c_v7")
    if not isinstance(host, dict) or not isinstance(model, dict):
        return None
    return {"host": host, "current_model": model}


def _compare_reference(export_metrics, reference, tolerance):
    if reference is None:
        return {"available": False, "passed": None, "tolerance": float(tolerance), "diff": {}}
    diff = {}
    passed = True
    for name in NAMES:
        diff[name] = {}
        for key in METRIC_KEYS:
            if key not in reference[name]:
                continue
            delta = float(export_metrics[name][key] - reference[name][key])
            diff[name][key] = delta
            if abs(delta) > float(tolerance):
                passed = False
    return {
        "available": True,
        "passed": bool(passed),
        "tolerance": float(tolerance),
        "diff": diff,
    }


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
    counts = {name: 0 for name in NAMES}
    per_sequence = {}
    output_root = Path(output_root)

    for sequence in c_v5.FULL9:
        samples = groups[sequence]
        previous = None
        pending_motion = None
        motion_hidden = None
        transport_hidden = semantic_hidden = mask_hidden = None
        semantic_state_low = None
        memory_state = None
        error_state = None
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
                error = F.softmax(host_low, dim=1) - F.softmax(previous_low, dim=1)
                pending_motion, _, motion_hidden = residual.predict_next(
                    observed,
                    error,
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
                evidence = _build_c_v7_evidence(
                    corrector,
                    dynamics,
                    c_v3_logits,
                    candidate_rows,
                    pending_motion,
                    e1["transportability_low"],
                    memory_row["memory_reliability"],
                    error_state,
                    dynamics_state,
                )
                error_state = evidence["row"]["error_state"]
                dynamics_state = evidence["dynamics_state"]
                model_pred = evidence["final_logits"].argmax(1)

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
                raise RuntimeError(
                    "Host/current-model output shape mismatch: "
                    f"host={frame_hw} model={tuple(predictions['current_model'].shape)}"
                )
            for name, prediction in predictions.items():
                _save_mask(
                    output_root / name / str(sequence) / f"{sample['frame_id']}.png",
                    prediction,
                    frame_hw,
                    overwrite,
                )
                counts[name] += 1
                seq_counts[name] += 1

        per_sequence[str(sequence)] = {"frames": len(samples), "exported": seq_counts}
        print(json.dumps({"sequence": str(sequence), **per_sequence[str(sequence)]}), flush=True)

    return counts, per_sequence


def _verify_saved_masks(groups, output_root):
    """Read saved PNGs and recompute mIoU/mVC8/mVC16（回读并重算指标）."""
    output_root = Path(output_root)
    confusion = {
        name: torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
        for name in NAMES
    }
    vc_sum = {name: {8: 0.0, 16: 0.0} for name in NAMES}
    vc_count = {name: {8: 0, 16: 0} for name in NAMES}
    read_counts = {name: 0 for name in NAMES}

    for sequence in c_v5.FULL9:
        seq_vc = {name: c_v5.VideoConsistency() for name in NAMES}
        for sample in groups[sequence]:
            gt = semantic_mask_from_panoptic_png(sample["mask_path"])
            for name in NAMES:
                prediction = _load_saved_mask(
                    output_root / name / str(sequence) / f"{sample['frame_id']}.png",
                    gt.shape,
                )
                update_confusion_matrix(confusion[name], prediction, gt)
                seq_vc[name].update(gt, prediction)
                read_counts[name] += 1
        for name in NAMES:
            stats = seq_vc[name].stats()
            for length in (8, 16):
                vc_sum[name][length] += stats[length]["sum"]
                vc_count[name][length] += stats[length]["count"]

    disk_counts = {
        name: sum(1 for _ in (output_root / name).rglob("*.png")) for name in NAMES
    }
    for name in NAMES:
        if read_counts[name] != EXPECTED_FULL9_FRAMES:
            raise RuntimeError(
                f"saved PNG read-count mismatch for {name}: "
                f"{read_counts[name]} != {EXPECTED_FULL9_FRAMES}"
            )
        if disk_counts[name] != EXPECTED_FULL9_FRAMES:
            raise RuntimeError(
                f"saved PNG disk-count mismatch for {name}: "
                f"{disk_counts[name]} != {EXPECTED_FULL9_FRAMES}"
            )

    metrics = {
        name: _metric_row(confusion, vc_sum, vc_count, name) for name in NAMES
    }
    return metrics, read_counts, disk_counts


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
    parser.add_argument("--metric-tolerance", type=float, default=1e-12)
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    _preflight_output(args.output_root, args.overwrite)
    _validate_bounded_motion_checkpoint(args.residual_checkpoint)

    model = load_fast_b_model(args.fast_b_checkpoint).cuda().eval()
    model.requires_grad_(False)
    observer, _ = _load_frozen_observer(args.observer_checkpoint)
    residual, _ = _load_frozen_residual(args.residual_checkpoint, observer)
    correction, mask_predictor, _ = _load_frozen_e1_base(args.base_checkpoint, observer)
    refiner, _ = _load_frozen_cv3_refiner(args.c_v3_checkpoint)
    corrector, payload = _load_corrector(args.model_checkpoint, "cuda")
    dynamics = _build_dynamics(payload)

    for module in (observer, residual, correction, mask_predictor, refiner):
        if hasattr(module, "requires_grad_"):
            module.requires_grad_(False)
        if hasattr(module, "eval"):
            module.eval()

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
    for name in NAMES:
        if counts[name] != EXPECTED_FULL9_FRAMES:
            raise RuntimeError(
                f"export frame-count mismatch for {name}: {counts[name]} != {EXPECTED_FULL9_FRAMES}"
            )

    export_metrics, read_counts, disk_counts = _verify_saved_masks(groups, args.output_root)
    reference = _reference_metrics(payload)
    reference_check = _compare_reference(
        export_metrics,
        reference,
        args.metric_tolerance,
    )
    if reference_check["available"] and not reference_check["passed"]:
        raise RuntimeError(
            "C-V7 exported prediction metrics do not match checkpoint reference: "
            f"{reference_check['diff']}"
        )

    summary = {
        "experiment": "export_kitti_step_full9_predictions_c_v7",
        "model_checkpoint": str(args.model_checkpoint),
        "output_root": str(args.output_root),
        "counts": counts,
        "read_counts": read_counts,
        "disk_counts": disk_counts,
        "per_sequence": per_sequence,
        "metrics": export_metrics,
        "reference_check": reference_check,
        "inference_constraints": {
            "raft_in_model_inference": False,
            "gt_in_model_decision": False,
            "hard_candidate_selection": False,
            "prediction_error_residual_correction": True,
            "raw_history_to_correction_path": False,
        },
    }
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
