"""Export KITTI-STEP Full9 Host and current C-V6H predictions.

中文：只做一次模型推理，导出 Host 与当前 C-V6H 的逐帧语义预测 PNG，供
PC / VEC8 / VEC16 等 prediction-only 指标重复使用。

Important constraints / 约束：
- no training;
- no RAFT construction or optical-flow teacher evaluation;
- no Oracle / GT-assisted model decision;
- C-V6H inference follows the reviewed formal selector path;
- first two frames preserve the existing Full9 evaluator behaviour and fall
  back to Host while temporal state is initialized;
- exported prediction PNGs are uint8 HxW class-index masks with ids 0..18;
- GT is read only after prediction for export sanity checks (mIoU/mVC8/mVC16).
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
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v5_multiframe_candidate_memory import (
    FULL9,
    HISTORY_LENGTH,
    CONTROLLER_HIDDEN_CHANNELS,
    DYNAMICS_TAU_E,
    DYNAMICS_K_E,
    DYNAMICS_DT,
    FAST_B_CHECKPOINT_DEFAULT,
    OBSERVER_CHECKPOINT_DEFAULT,
    RESIDUAL_CHECKPOINT_DEFAULT,
    BASE_CHECKPOINT_DEFAULT,
    C_V3_CHECKPOINT_DEFAULT,
    VideoConsistency,
    compute_iou,
    update_confusion_matrix,
    _host_observation,
    _initialize_motion,
    _observe_motion,
    _frozen_e1_step,
    _frozen_cv3_step,
    _build_history_candidates,
    _candidate_predictions,
    _apply_selection,
    _load_frozen_observer,
    _load_frozen_residual,
    _validate_bounded_motion_checkpoint,
    _load_frozen_e1_base,
    _load_frozen_cv3_refiner,
    load_fast_b_model,
    warp_low_logits,
)
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v6h_hierarchical_error_selector_main import (
    _selector_evidence_no_mask_bleed,
)
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_hierarchical_error_selector import (
    HierarchicalMultiHypothesisErrorSelector,
)
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_semantic_hysteresis import (
    EulerDynamicsError,
)


NUM_CLASSES = 19
EXPECTED_FULL9_FRAMES = 2981
DEFAULT_OUTPUT_ROOT = "/home/lin/predify/predictions/kitti_step_full9"
DEFAULT_MODEL_CHECKPOINT = (
    "/home/lin/predify/experiments/"
    "kitti_step_task_space_prior_c_v6h_hierarchical_error_selector/best.pt"
)
METRIC_KEYS = ("mIoU", "mVC8", "mVC16")


def _load_selector(checkpoint_path, device):
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"C-V6H checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu")
    if "selector_state_dict" not in payload:
        raise KeyError(f"checkpoint lacks selector_state_dict: {path}")

    selector = HierarchicalMultiHypothesisErrorSelector(
        num_classes=NUM_CLASSES,
        history_length=HISTORY_LENGTH,
        hidden_channels=CONTROLLER_HIDDEN_CHANNELS,
    ).to(device)
    selector.load_state_dict(payload["selector_state_dict"], strict=True)
    selector.requires_grad_(False)
    selector.eval()

    experiment = str(payload.get("experiment", ""))
    if experiment and "c_v6h" not in experiment.lower():
        raise RuntimeError(
            "model checkpoint is not identified as C-V6H: "
            f"experiment={experiment!r} path={path}"
        )
    return selector, payload


def _build_dynamics(payload):
    config = dict(payload.get("dynamics") or {})
    return EulerDynamicsError(
        tau_e=float(config.get("tau_e", DYNAMICS_TAU_E)),
        k_e=float(config.get("k_e", DYNAMICS_K_E)),
        dt=float(config.get("dt", DYNAMICS_DT)),
    )


def _save_mask(path, prediction, expected_hw, overwrite):
    prediction = torch.as_tensor(prediction).detach().cpu().long()
    if prediction.ndim == 3 and prediction.shape[0] == 1:
        prediction = prediction[0]
    if prediction.ndim != 2:
        raise ValueError(f"prediction must be HxW, got {tuple(prediction.shape)}")
    if tuple(prediction.shape) != tuple(expected_hw):
        raise ValueError(
            f"prediction/GT size mismatch: pred={tuple(prediction.shape)} "
            f"gt={tuple(expected_hw)}"
        )
    invalid = (prediction < 0) | (prediction >= NUM_CLASSES)
    if bool(invalid.any()):
        values = torch.unique(prediction[invalid])[:16].tolist()
        raise ValueError(f"prediction contains invalid class ids: {values}")

    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"prediction already exists: {path}; pass --overwrite to replace"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    array = prediction.numpy().astype(np.uint8, copy=False)
    Image.fromarray(array, mode="L").save(path)


def _metric_row(confusion, vc_sum, vc_count, name):
    return {
        "mIoU": float(torch.nanmean(compute_iou(confusion[name])).item()),
        "mVC8": (
            vc_sum[name][8] / vc_count[name][8]
            if vc_count[name][8]
            else float("nan")
        ),
        "mVC16": (
            vc_sum[name][16] / vc_count[name][16]
            if vc_count[name][16]
            else float("nan")
        ),
    }


def _reference_metrics(payload):
    metrics = payload.get("metrics") or {}
    if not isinstance(metrics, dict):
        return None
    host = metrics.get("host")
    model = metrics.get("c_v6") or metrics.get("c_v6h")
    if not isinstance(host, dict) or not isinstance(model, dict):
        return None
    return {"host": host, "current_model": model}


def _compare_reference(export_metrics, reference, tolerance):
    if reference is None:
        return {
            "available": False,
            "passed": None,
            "tolerance": float(tolerance),
            "diff": {},
        }
    diff = {}
    passed = True
    for name in ("host", "current_model"):
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
    selector,
    dynamics,
    groups,
    output_root,
    overwrite,
):
    names = ("host", "current_model")
    confusion = {
        name: torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
        for name in names
    }
    vc_sum = {name: {8: 0.0, 16: 0.0} for name in names}
    vc_count = {name: {8: 0, 16: 0} for name in names}
    counts = {name: 0 for name in names}
    per_sequence = {}

    output_root = Path(output_root)

    for sequence in FULL9:
        samples = groups[sequence]
        previous = None
        pending_motion = None
        motion_hidden = None
        transport_hidden = semantic_hidden = mask_hidden = None
        semantic_state_low = None
        memory_state = None
        selector_hidden = None
        dynamics_state = None
        raw_history = []
        motion_history = []
        seq_vc = {name: VideoConsistency() for name in names}
        seq_counts = {name: 0 for name in names}

        for sample in samples:
            image, host_logits, host_low, current_c1, output_size = _host_observation(
                model,
                sample,
            )
            host_pred = host_logits.argmax(1)

            if previous is None:
                # Existing formal Full9 behaviour: first frame has no temporal
                # state, so the current model falls back exactly to Host.
                model_pred = host_pred
                semantic_state_low = torch.zeros_like(host_low)
                raw_history = [host_logits.detach()]
                previous = (image, host_low.detach(), current_c1.detach())

            elif pending_motion is None:
                # Second frame is used to initialize the learned motion state.
                # It also falls back exactly to Host in the existing evaluator.
                previous_image, previous_low, previous_c1 = previous
                observed = _observe_motion(
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
                previous_image, previous_low, previous_c1 = previous
                prior_low, _ = warp_low_logits(previous_low, pending_motion)
                e1 = _frozen_e1_step(
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

                memory_row, memory_state, _, c_v3_logits = _frozen_cv3_step(
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
                candidate_rows = _build_history_candidates(
                    raw_history,
                    motion_history,
                    pending_motion,
                    selector.history_length,
                )
                evidence = _selector_evidence_no_mask_bleed(
                    selector,
                    dynamics,
                    c_v3_logits,
                    candidate_rows,
                    pending_motion,
                    e1["transportability_low"],
                    memory_row["memory_reliability"],
                    selector_hidden,
                    dynamics_state,
                )
                selector_hidden = evidence["row"]["hidden"]
                dynamics_state = evidence["dynamics_state"]

                current_pred, history_preds = _candidate_predictions(
                    c_v3_logits,
                    candidate_rows,
                )
                hard_selection = evidence["selector_logits_full"].argmax(1)[0]
                model_pred = _apply_selection(
                    current_pred,
                    history_preds,
                    hard_selection,
                )

                observed = _observe_motion(
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
                raw_history = raw_history[: selector.history_length]
                motion_history.insert(0, pending_motion.detach())
                motion_history = motion_history[: max(selector.history_length - 1, 0)]
                previous = (image, host_low.detach(), current_c1.detach())
                pending_motion = next_motion.detach()
                motion_hidden = next_motion_hidden.detach()

            # GT is deliberately outside the model-decision path and is read
            # only to verify that exported masks reproduce normal metrics.
            gt_cpu = semantic_mask_from_panoptic_png(sample["mask_path"])
            predictions = {
                "host": host_pred.squeeze(0).detach().cpu(),
                "current_model": model_pred.squeeze(0).detach().cpu(),
            }

            for name, prediction in predictions.items():
                _save_mask(
                    output_root / name / str(sequence) / f"{sample['frame_id']}.png",
                    prediction,
                    gt_cpu.shape,
                    overwrite,
                )
                counts[name] += 1
                seq_counts[name] += 1
                update_confusion_matrix(confusion[name], prediction, gt_cpu)
                seq_vc[name].update(gt_cpu, prediction)

        for name in names:
            stats = seq_vc[name].stats()
            for length in (8, 16):
                vc_sum[name][length] += stats[length]["sum"]
                vc_count[name][length] += stats[length]["count"]

        per_sequence[str(sequence)] = {
            "frames": len(samples),
            "exported": seq_counts,
        }
        print(
            json.dumps(
                {
                    "sequence": str(sequence),
                    "frames": len(samples),
                    "exported": seq_counts,
                }
            ),
            flush=True,
        )

    metrics = {
        name: _metric_row(confusion, vc_sum, vc_count, name)
        for name in names
    }
    return metrics, counts, per_sequence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/home/lin/predify/kitti_step")
    parser.add_argument("--model-checkpoint", default=DEFAULT_MODEL_CHECKPOINT)
    parser.add_argument("--fast-b-checkpoint", default=FAST_B_CHECKPOINT_DEFAULT)
    parser.add_argument("--observer-checkpoint", default=OBSERVER_CHECKPOINT_DEFAULT)
    parser.add_argument("--residual-checkpoint", default=RESIDUAL_CHECKPOINT_DEFAULT)
    parser.add_argument("--base-checkpoint", default=BASE_CHECKPOINT_DEFAULT)
    parser.add_argument("--c-v3-checkpoint", default=C_V3_CHECKPOINT_DEFAULT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--metric-tolerance", type=float, default=1.0e-6)
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")

    _validate_bounded_motion_checkpoint(args.residual_checkpoint)
    model = load_fast_b_model(args.fast_b_checkpoint).to(device).eval()
    model.requires_grad_(False)
    observer, _ = _load_frozen_observer(args.observer_checkpoint)
    residual, _ = _load_frozen_residual(args.residual_checkpoint, observer)
    correction, mask_predictor, _ = _load_frozen_e1_base(
        args.base_checkpoint,
        observer,
    )
    refiner, _ = _load_frozen_cv3_refiner(args.c_v3_checkpoint)
    selector, selector_payload = _load_selector(args.model_checkpoint, device)
    dynamics = _build_dynamics(selector_payload)

    val = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "val")
    all_groups = sequence_groups(val)
    missing = [sequence for sequence in FULL9 if sequence not in all_groups]
    if missing:
        raise RuntimeError(f"missing KITTI-STEP Full9 sequences: {missing}")
    groups = {sequence: all_groups[sequence] for sequence in FULL9}

    total_frames = sum(len(groups[sequence]) for sequence in FULL9)
    if total_frames != EXPECTED_FULL9_FRAMES:
        raise RuntimeError(
            "KITTI-STEP Full9 frame-count mismatch: "
            f"expected={EXPECTED_FULL9_FRAMES} actual={total_frames}"
        )

    metrics, counts, per_sequence = _export_full9(
        model,
        observer,
        residual,
        correction,
        mask_predictor,
        refiner,
        selector,
        dynamics,
        groups,
        args.output_root,
        args.overwrite,
    )

    for name, count in counts.items():
        if count != EXPECTED_FULL9_FRAMES:
            raise RuntimeError(
                f"export count mismatch for {name}: "
                f"expected={EXPECTED_FULL9_FRAMES} actual={count}"
            )

    reference = _reference_metrics(selector_payload)
    reference_check = _compare_reference(
        metrics,
        reference,
        args.metric_tolerance,
    )

    result = {
        "experiment": "KITTI-STEP Full9 Host + C-V6H prediction export",
        "full9": list(FULL9),
        "expected_frames_per_model": EXPECTED_FULL9_FRAMES,
        "exported_frames": counts,
        "output_root": str(Path(args.output_root)),
        "host_prediction_root": str(Path(args.output_root) / "host"),
        "model_prediction_root": str(Path(args.output_root) / "current_model"),
        "metrics_recomputed_from_export_run": metrics,
        "reference_metrics_from_model_checkpoint": reference,
        "reference_metric_check": reference_check,
        "checkpoints": {
            "current_model": args.model_checkpoint,
            "fast_b": args.fast_b_checkpoint,
            "observer": args.observer_checkpoint,
            "residual": args.residual_checkpoint,
            "e1_base": args.base_checkpoint,
            "c_v3_base": args.c_v3_checkpoint,
            "current_model_epoch": selector_payload.get("epoch"),
            "current_model_experiment": selector_payload.get("experiment"),
        },
        "inference_protocol": {
            "raft": False,
            "oracle": False,
            "gt_used_for_model_decision": False,
            "gt_used_only_for_export_sanity_metrics": True,
            "first_frame_output": "Host fallback",
            "second_frame_output": "Host fallback while motion state initializes",
            "third_frame_onward": (
                "frozen Host/E1/C-V3 + K=4 non-autoregressive history + "
                "Prediction Error/Dynamics Error + reviewed C-V6H hierarchical selector"
            ),
            "prediction_format": "uint8 single-channel PNG class ids 0..18",
        },
        "per_sequence": per_sequence,
    }

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = output_root / "manifest.json"
    with manifest.open("w") as handle:
        json.dump(result, handle, indent=2)

    print(json.dumps(result, indent=2), flush=True)
    if reference_check["available"] and not reference_check["passed"]:
        raise RuntimeError(
            "exported inference metrics do not match the metrics stored in the "
            f"model checkpoint within tolerance={args.metric_tolerance}; "
            f"see {manifest}"
        )
    print(f"wrote export manifest: {manifest}", flush=True)
    return result


if __name__ == "__main__":
    main()
