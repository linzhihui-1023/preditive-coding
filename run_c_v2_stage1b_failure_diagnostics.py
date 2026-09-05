#!/usr/bin/env python3
"""Run the complete C-V2 Stage-1B failure diagnostics in one entry point.

This script is intentionally diagnostic, not a new model stage. It runs:

A) Observer Flow Calibration Diagnostic (no training)
   - scale sweep for Current Observer Warp and strict-causal Lagged Observer Warp
   - Persistence, current RAFT same-path reference, and Lagged RAFT reference
   - mIoU / mTC / mVC8 / mVC16
   - repair-only Oracle delta mTC vs Host for every scale/path
   - Observer-vs-RAFT cosine, EPE, component-L1, flow magnitude
   - RAFT teacher correlation-radius coverage for r={4,6,8,12,16}

B) RAFT-History Residual Diagnostic (3 epochs by default)
   - bypasses the learned Observer during training/evaluation
   - feeds strictly causal historical RAFT motion F_t to the existing
     MotionResidualPredictor and predicts F_{t+1}=F_t+DeltaF_{t+1}
   - compares Lagged RAFT vs RAFT-History + Residual at every epoch
   - records mIoU / mTC / mVC8 / mVC16, repair-only Oracle, and residual amplitude

No D0, semantic-residual Stage-2, selector, or final fusion is trained here.
RAFT is allowed in B only because B is an explicit diagnostic upper-quality-input test;
it is not an inference/deployment path.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch.nn import functional as F


SEED = 0
DEFAULT_ROOT = "/home/lin/predify/kitti_step"
DEFAULT_OBSERVER_CHECKPOINT = (
    "/home/lin/predify/experiments/"
    "kitti_step_task_space_prior_c_v2_stage1b_observer/best.pt"
)
DEFAULT_RESULT_ROOT = "results/kitti_step_c_v2_stage1b_failure_diagnostics"
DEFAULT_RAFT_RESIDUAL_OUTPUT = (
    "/home/lin/predify/experiments/"
    "kitti_step_c_v2_raft_history_residual_diagnostic"
)
DEFAULT_SCALES = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)
DEFAULT_RADII = (4, 6, 8, 12, 16)


def _load_project():
    """Lazy project imports so --self-test/--help work outside the repository."""
    try:
        from predify2021.datasets.kitti_step import (
            KITTISTEPSegmentationDataset,
            semantic_mask_from_panoptic_png,
        )
        from predify2021.mce_scores.evaluate_kitti_step_clean_fast_iss_vss import (
            VideoConsistency,
        )
        from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import (
            sequence_groups,
        )
        from predify2021.mce_scores.evaluate_kitti_step_static_baseline import (
            compute_iou,
            update_confusion_matrix,
        )
        from predify2021.mce_scores.train_kitti_step_frozen_error_utility_probe import (
            FAST_B_CHECKPOINT_DEFAULT,
            load_fast_b_model,
        )
        from predify2021.mce_scores.train_kitti_step_task_space_prior_c_only import (
            FULL9,
            IGNORE_LABEL,
            NUM_CLASSES,
            _add_counts,
            _new_counts,
            _rates,
            _upsample_prior,
        )
        from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v2_stage1b_observer import (
            _host_observation,
        )
        from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v2_stage1b_residual import (
            _load_frozen_observer,
        )
        from predify2021.mce_scores.train_kitti_step_temporal_joint import (
            FrozenRAFT,
            flow_grid,
        )
        from predify2021.model_factory.deeplabv3plus_resnet50.task_space_motion_transport import (
            MotionResidualPredictor,
            downsample_backward_flow,
            normalized_flow_distillation_loss,
            normalized_motion_residual_l2,
            teacher_reachable_mask,
            warp_low_logits,
        )
    except Exception as exc:
        raise RuntimeError(
            "Project imports failed. Run this file from the repository root with the "
            "same Python environment used by the existing Stage-1B scripts."
        ) from exc

    return SimpleNamespace(**locals())


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _scale_tag(scale: float) -> str:
    return f"{scale:g}".replace(".", "p").replace("-", "m")


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if abs(b) > 1e-12 else float("nan")


def _isfinite(x: float) -> bool:
    return math.isfinite(float(x))


def _miou_from_confusion(confusion: torch.Tensor) -> float:
    c = confusion.to(dtype=torch.float64)
    inter = torch.diag(c)
    union = c.sum(1) + c.sum(0) - inter
    valid = union > 0
    if not bool(valid.any()):
        return float("nan")
    return float((inter[valid] / union[valid]).mean().item())


@torch.inference_mode()
def _batched_pair_mtc(
    previous_predictions: Mapping[str, torch.Tensor],
    current_predictions: Mapping[str, torch.Tensor],
    flow: torch.Tensor,
    flow_grid_fn,
    num_classes: int,
    chunk_size: int = 8,
) -> Dict[str, float]:
    """Compute the repository mTC definition for many paths with one flow grid.

    This matches _pair_mtc semantically but batches grid_sample and keeps the
    confusion calculation on GPU to reduce CPU transfer overhead during the
    8-scale sweep.
    """
    names = list(current_predictions)
    if not names:
        return {}
    sample = current_predictions[names[0]]
    height, width = sample.shape[-2:]
    grid, valid = flow_grid_fn(flow, height, width)
    keep = valid.squeeze(0)
    if not bool(keep.any()):
        return {name: float("nan") for name in names}

    result: Dict[str, float] = {}
    for start in range(0, len(names), max(int(chunk_size), 1)):
        chunk = names[start : start + max(int(chunk_size), 1)]
        previous = torch.stack(
            [previous_predictions[name].squeeze(0) for name in chunk], dim=0
        ).unsqueeze(1).float()
        current = torch.stack(
            [current_predictions[name].squeeze(0) for name in chunk], dim=0
        ).to(torch.int64)
        warped = F.grid_sample(
            previous,
            grid.expand(len(chunk), -1, -1, -1),
            mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(1).to(torch.int64)
        for i, name in enumerate(chunk):
            a = warped[i][keep]
            b = current[i][keep]
            if not a.numel():
                result[name] = float("nan")
                continue
            confusion = torch.bincount(
                num_classes * a + b,
                minlength=num_classes * num_classes,
            ).reshape(num_classes, num_classes)
            result[name] = _miou_from_confusion(confusion)
    return result


def _metric_struct(names: Sequence[str], P):
    return {
        "confusion": {
            n: torch.zeros((P.NUM_CLASSES, P.NUM_CLASSES), dtype=torch.int64)
            for n in names
        },
        "mtc_sum": {n: 0.0 for n in names},
        "mtc_count": {n: 0 for n in names},
        "vc_sum": {n: {8: 0.0, 16: 0.0} for n in names},
        "vc_count": {n: {8: 0, 16: 0} for n in names},
    }


def _new_sequence_metric_struct(names: Sequence[str], P):
    return {
        "confusion": {
            n: torch.zeros((P.NUM_CLASSES, P.NUM_CLASSES), dtype=torch.int64)
            for n in names
        },
        "mtc_sum": {n: 0.0 for n in names},
        "mtc_count": {n: 0 for n in names},
        "vc": {n: P.VideoConsistency() for n in names},
    }


def _update_segmentation_metrics(
    global_bank,
    seq_bank,
    predictions: Mapping[str, torch.Tensor],
    gt_cpu: torch.Tensor,
    P,
) -> None:
    # Transfer each semantic map once. uint8 transfer is substantially cheaper
    # than copying int64 maps for the 8-scale sweep; values are only 0..18.
    for name, prediction in predictions.items():
        pc = prediction.squeeze(0).to(torch.uint8).cpu().to(torch.int64)
        P.update_confusion_matrix(global_bank["confusion"][name], pc, gt_cpu)
        P.update_confusion_matrix(seq_bank["confusion"][name], pc, gt_cpu)
        seq_bank["vc"][name].update(gt_cpu, pc)


def _update_mtc_metrics(global_bank, seq_bank, mtc_scores: Mapping[str, float]) -> None:
    for name, score in mtc_scores.items():
        if _isfinite(score):
            global_bank["mtc_sum"][name] += float(score)
            global_bank["mtc_count"][name] += 1
            seq_bank["mtc_sum"][name] += float(score)
            seq_bank["mtc_count"][name] += 1


def _finalize_sequence_metrics(seq_bank, names: Sequence[str], P):
    result = {}
    for name in names:
        values = seq_bank["vc"][name].values()
        result[name] = {
            "mIoU": float(torch.nanmean(P.compute_iou(seq_bank["confusion"][name])).item()),
            "mTC": _safe_div(seq_bank["mtc_sum"][name], seq_bank["mtc_count"][name]),
            "mVC8": float(values[8]),
            "mVC16": float(values[16]),
            "valid_frame_pairs": int(seq_bank["mtc_count"][name]),
        }
    return result


def _close_sequence(global_bank, seq_bank, names: Sequence[str]) -> None:
    for name in names:
        stats = seq_bank["vc"][name].stats()
        for length in (8, 16):
            global_bank["vc_sum"][name][length] += float(stats[length]["sum"])
            global_bank["vc_count"][name][length] += int(stats[length]["count"])


def _finalize_global_metrics(global_bank, names: Sequence[str], P):
    return {
        name: {
            "mIoU": float(torch.nanmean(P.compute_iou(global_bank["confusion"][name])).item()),
            "mTC": _safe_div(global_bank["mtc_sum"][name], global_bank["mtc_count"][name]),
            "mVC8": _safe_div(global_bank["vc_sum"][name][8], global_bank["vc_count"][name][8]),
            "mVC16": _safe_div(global_bank["vc_sum"][name][16], global_bank["vc_count"][name][16]),
            "valid_frame_pairs": int(global_bank["mtc_count"][name]),
        }
        for name in names
    }


def _make_repair_only_oracle(host_pred, prior_pred, gt, ignore_label: int):
    valid = gt != ignore_label
    host_correct = host_pred.squeeze(0) == gt
    prior_correct = prior_pred.squeeze(0) == gt
    recoverable = valid & ~host_correct & prior_correct
    oracle = host_pred.clone()
    oracle[recoverable.unsqueeze(0)] = prior_pred[recoverable.unsqueeze(0)]
    return oracle, host_correct, prior_correct, valid


def _new_flow_scale_accumulator(scales: Sequence[float]):
    return {
        str(float(s)): {
            "component_values": 0,
            "vector_values": 0,
            "pred_abs": 0.0,
            "teacher_abs": 0.0,
            "l1": 0.0,
            "epe": 0.0,
            "cosine_sum": 0.0,
            "cosine_count": 0,
        }
        for s in scales
    }


def _accumulate_flow_scale_diagnostics(
    accumulator,
    scales: Sequence[float],
    observed_motion: torch.Tensor,
    teacher_low: torch.Tensor,
    reachable: torch.Tensor,
) -> None:
    vector_count = int(reachable.sum().item())
    if vector_count <= 0:
        return
    component_mask = reachable.unsqueeze(1).expand_as(observed_motion)
    teacher_norm = torch.linalg.vector_norm(teacher_low, dim=1)
    observed_norm = torch.linalg.vector_norm(observed_motion, dim=1)
    cosine_valid = reachable & (teacher_norm > 1e-6) & (observed_norm > 1e-6)
    dot = (teacher_low * observed_motion).sum(dim=1)
    cosine = dot / (teacher_norm * observed_norm).clamp_min(1e-12)

    for scale in scales:
        row = accumulator[str(float(scale))]
        predicted = observed_motion * float(scale)
        error = predicted - teacher_low
        row["component_values"] += int(component_mask.sum().item())
        row["vector_values"] += vector_count
        row["pred_abs"] += float(predicted[component_mask].abs().sum().item())
        row["teacher_abs"] += float(teacher_low[component_mask].abs().sum().item())
        row["l1"] += float(error[component_mask].abs().sum().item())
        row["epe"] += float(torch.linalg.vector_norm(error, dim=1)[reachable].sum().item())
        count = int(cosine_valid.sum().item())
        if count:
            row["cosine_sum"] += float(cosine[cosine_valid].sum().item())
            row["cosine_count"] += count


def _finalize_flow_scale_diagnostics(accumulator):
    out = {}
    for key, row in accumulator.items():
        c = max(int(row["component_values"]), 1)
        v = max(int(row["vector_values"]), 1)
        cc = max(int(row["cosine_count"]), 1)
        out[key] = {
            "scale": float(key),
            "predicted_abs_mean_low_pixels_reachable": row["pred_abs"] / c,
            "teacher_abs_mean_low_pixels_reachable": row["teacher_abs"] / c,
            "component_l1_mean_low_pixels_reachable": row["l1"] / c,
            "epe_mean_low_pixels_reachable": row["epe"] / v,
            "cosine_similarity_mean_nonzero_reachable": row["cosine_sum"] / cc,
            "cosine_valid_vectors": int(row["cosine_count"]),
            "reachable_vectors": int(row["vector_values"]),
        }
    return out


def _observer_calibration_diagnostic(
    model,
    observer,
    val_groups,
    raft,
    scales: Sequence[float],
    radii: Sequence[int],
    mtc_chunk_size: int,
    P,
):
    scales = tuple(float(s) for s in scales)
    radii = tuple(int(r) for r in radii)
    ref_names = (
        "host",
        "semantic_persistence",
        "raft_current_same_path",
        "raft_lagged",
    )
    prior_names: List[str] = []
    oracle_names: List[str] = []
    for s in scales:
        tag = _scale_tag(s)
        prior_names.extend((f"observer_current_s{tag}", f"observer_lagged_s{tag}"))
        oracle_names.extend((f"oracle_current_s{tag}", f"oracle_lagged_s{tag}"))
    metric_names = list(ref_names) + prior_names
    mtc_names = metric_names + oracle_names

    bank = _metric_struct(metric_names, P)
    oracle_mtc_sum = {name: 0.0 for name in oracle_names}
    oracle_mtc_count = {name: 0 for name in oracle_names}
    complement = {name: P._new_counts() for name in prior_names}
    scale_diag = _new_flow_scale_accumulator(scales)
    radius_diag = {
        "spatial_valid_pixels": 0,
        "reachable_max_displacement_pixels": 0,
        "within": {str(r): 0 for r in radii},
        "pairs": 0,
    }
    per_sequence = {}
    observer.eval()

    for sequence in P.FULL9:
        samples = val_groups[sequence]
        seq_bank = _new_sequence_metric_struct(metric_names, P)
        seq_oracle_mtc_sum = {name: 0.0 for name in oracle_names}
        seq_oracle_mtc_count = {name: 0 for name in oracle_names}
        previous = None
        previous_observed_motion = None
        previous_teacher_low = None
        previous_predictions: Dict[str, torch.Tensor] = {}
        previous_mtc_predictions: Dict[str, torch.Tensor] = {}

        for sample in samples:
            image, host_logits, host_low, c1, output_size = P._host_observation(model, sample)
            host_pred = host_logits.argmax(1)
            gt_cpu = P.semantic_mask_from_panoptic_png(sample["mask_path"])
            gt = gt_cpu.cuda(non_blocking=True)
            teacher_full = None

            predictions: Dict[str, torch.Tensor] = {"host": host_pred}
            mtc_predictions: Dict[str, torch.Tensor] = {"host": host_pred}

            if previous is None:
                predictions["semantic_persistence"] = host_pred
                predictions["raft_current_same_path"] = host_pred
                predictions["raft_lagged"] = host_pred
                for s in scales:
                    tag = _scale_tag(s)
                    current_name = f"observer_current_s{tag}"
                    lagged_name = f"observer_lagged_s{tag}"
                    predictions[current_name] = host_pred
                    predictions[lagged_name] = host_pred
                    mtc_predictions[f"oracle_current_s{tag}"] = host_pred
                    mtc_predictions[f"oracle_lagged_s{tag}"] = host_pred
                mtc_predictions.update(predictions)
            else:
                previous_image, previous_low, previous_c1 = previous
                persistence = P._upsample_prior(previous_low, output_size).argmax(1)
                predictions["semantic_persistence"] = persistence

                observed_motion = observer(
                    previous_c1,
                    c1,
                    F.softmax(previous_low, dim=1),
                    F.softmax(host_low, dim=1),
                )
                teacher_full = raft.current_to_previous(image, previous_image)
                teacher_low = P.downsample_backward_flow(
                    teacher_full, tuple(host_low.shape[-2:])
                )

                current_raft_low, _ = P.warp_low_logits(previous_low, teacher_low)
                predictions["raft_current_same_path"] = P._upsample_prior(
                    current_raft_low, output_size
                ).argmax(1)
                if previous_teacher_low is None:
                    lagged_raft_low = previous_low
                else:
                    lagged_raft_low, _ = P.warp_low_logits(previous_low, previous_teacher_low)
                predictions["raft_lagged"] = P._upsample_prior(
                    lagged_raft_low, output_size
                ).argmax(1)

                reachable, spatial_valid = P.teacher_reachable_mask(
                    teacher_low, observer.max_displacement_low
                )
                radius_diag["pairs"] += 1
                radius_diag["spatial_valid_pixels"] += int(spatial_valid.sum().item())
                radius_diag["reachable_max_displacement_pixels"] += int(reachable.sum().item())
                for radius in radii:
                    within = (
                        spatial_valid
                        & (teacher_low[:, 0].abs() <= radius)
                        & (teacher_low[:, 1].abs() <= radius)
                    )
                    radius_diag["within"][str(radius)] += int(within.sum().item())
                _accumulate_flow_scale_diagnostics(
                    scale_diag, scales, observed_motion, teacher_low, reachable
                )

                for s in scales:
                    tag = _scale_tag(s)
                    current_name = f"observer_current_s{tag}"
                    lagged_name = f"observer_lagged_s{tag}"
                    current_warped, _ = P.warp_low_logits(
                        previous_low, observed_motion * s
                    )
                    current_pred = P._upsample_prior(current_warped, output_size).argmax(1)
                    if previous_observed_motion is None:
                        lagged_warped = previous_low
                    else:
                        lagged_warped, _ = P.warp_low_logits(
                            previous_low, previous_observed_motion * s
                        )
                    lagged_pred = P._upsample_prior(lagged_warped, output_size).argmax(1)
                    predictions[current_name] = current_pred
                    predictions[lagged_name] = lagged_pred

                    current_oracle, hc, pc, valid = _make_repair_only_oracle(
                        host_pred, current_pred, gt, P.IGNORE_LABEL
                    )
                    P._add_counts(complement[current_name], hc, pc, valid)
                    lagged_oracle, hc, pc, valid = _make_repair_only_oracle(
                        host_pred, lagged_pred, gt, P.IGNORE_LABEL
                    )
                    P._add_counts(complement[lagged_name], hc, pc, valid)
                    mtc_predictions[f"oracle_current_s{tag}"] = current_oracle
                    mtc_predictions[f"oracle_lagged_s{tag}"] = lagged_oracle

                mtc_predictions.update(predictions)

            _update_segmentation_metrics(bank, seq_bank, predictions, gt_cpu, P)

            if previous is not None:
                scores = _batched_pair_mtc(
                    previous_mtc_predictions,
                    mtc_predictions,
                    teacher_full,
                    P.flow_grid,
                    P.NUM_CLASSES,
                    mtc_chunk_size,
                )
                _update_mtc_metrics(
                    bank,
                    seq_bank,
                    {name: scores[name] for name in metric_names},
                )
                for name in oracle_names:
                    score = scores[name]
                    if _isfinite(score):
                        oracle_mtc_sum[name] += score
                        oracle_mtc_count[name] += 1
                        seq_oracle_mtc_sum[name] += score
                        seq_oracle_mtc_count[name] += 1

            previous = (image, host_low.detach(), c1.detach())
            if previous is not None and teacher_full is not None:
                previous_observed_motion = observed_motion.detach()
                previous_teacher_low = teacher_low.detach()
            previous_predictions = {k: v.detach() for k, v in predictions.items()}
            previous_mtc_predictions = {k: v.detach() for k, v in mtc_predictions.items()}

        _close_sequence(bank, seq_bank, metric_names)
        seq_metrics = _finalize_sequence_metrics(seq_bank, metric_names, P)
        host_mtc = seq_metrics["host"]["mTC"]
        per_sequence[sequence] = {
            "metrics": seq_metrics,
            "repair_only_oracle_delta_mTC_vs_host": {
                name: _safe_div(seq_oracle_mtc_sum[name], seq_oracle_mtc_count[name]) - host_mtc
                for name in oracle_names
            },
        }

    metrics = _finalize_global_metrics(bank, metric_names, P)
    host_mtc = metrics["host"]["mTC"]
    oracle_mtc = {
        name: _safe_div(oracle_mtc_sum[name], oracle_mtc_count[name])
        for name in oracle_names
    }
    oracle_delta = {name: value - host_mtc for name, value in oracle_mtc.items()}
    flow_diag = _finalize_flow_scale_diagnostics(scale_diag)
    spatial_valid = max(radius_diag["spatial_valid_pixels"], 1)
    radius_coverage = {
        str(r): radius_diag["within"][str(r)] / spatial_valid for r in radii
    }
    reachable_fraction = radius_diag["reachable_max_displacement_pixels"] / spatial_valid

    persistence = metrics["semantic_persistence"]
    gate_passing_lagged = []
    scale_rows = []
    for s in scales:
        tag = _scale_tag(s)
        current_name = f"observer_current_s{tag}"
        lagged_name = f"observer_lagged_s{tag}"
        current = metrics[current_name]
        lagged = metrics[lagged_name]
        lagged_gate = (
            lagged["mIoU"] - persistence["mIoU"] >= 0.02
            and lagged["mTC"] > persistence["mTC"]
        )
        if lagged_gate:
            gate_passing_lagged.append(s)
        scale_rows.append(
            {
                "scale": s,
                "current_name": current_name,
                "lagged_name": lagged_name,
                "current": current,
                "lagged": lagged,
                "current_delta_vs_persistence": {
                    k: current[k] - persistence[k]
                    for k in ("mIoU", "mTC", "mVC8", "mVC16")
                },
                "lagged_delta_vs_persistence": {
                    k: lagged[k] - persistence[k]
                    for k in ("mIoU", "mTC", "mVC8", "mVC16")
                },
                "current_repair_only_oracle_delta_mTC_vs_host": oracle_delta[
                    f"oracle_current_s{tag}"
                ],
                "lagged_repair_only_oracle_delta_mTC_vs_host": oracle_delta[
                    f"oracle_lagged_s{tag}"
                ],
                "flow": flow_diag[str(float(s))],
                "lagged_stage1b_motion_gate": bool(lagged_gate),
            }
        )

    def _argmax_path(field: str, mode: str):
        key = "current" if mode == "current" else "lagged"
        row = max(scale_rows, key=lambda r: r[key][field])
        return {"scale": row["scale"], field: row[key][field]}

    best_epe = min(scale_rows, key=lambda r: r["flow"]["epe_mean_low_pixels_reachable"])
    best_l1 = min(scale_rows, key=lambda r: r["flow"]["component_l1_mean_low_pixels_reachable"])
    s1 = next(r for r in scale_rows if abs(r["scale"] - 1.0) < 1e-9)
    best_lagged_miou = max(scale_rows, key=lambda r: r["lagged"]["mIoU"])
    best_lagged_mtc = max(scale_rows, key=lambda r: r["lagged"]["mTC"])

    return {
        "experiment": "Observer Flow Calibration Diagnostic",
        "no_training": True,
        "scales": list(scales),
        "radius_coverage_diagnostic_only_no_observer_retraining": {
            "pairs": radius_diag["pairs"],
            "teacher_reachable_fraction_at_observer_max_displacement": reachable_fraction,
            "coverage_fraction_of_spatial_valid": radius_coverage,
        },
        "reference_metrics": {name: metrics[name] for name in ref_names},
        "scale_results": scale_rows,
        "repair_only_oracle_mTC": oracle_mtc,
        "complementarity": {name: P._rates(counts) for name, counts in complement.items()},
        "best_scales": {
            "current_by_mIoU": _argmax_path("mIoU", "current"),
            "current_by_mTC": _argmax_path("mTC", "current"),
            "lagged_by_mIoU": _argmax_path("mIoU", "lagged"),
            "lagged_by_mTC": _argmax_path("mTC", "lagged"),
            "flow_by_EPE": {
                "scale": best_epe["scale"],
                "EPE": best_epe["flow"]["epe_mean_low_pixels_reachable"],
            },
            "flow_by_component_L1": {
                "scale": best_l1["scale"],
                "L1": best_l1["flow"]["component_l1_mean_low_pixels_reachable"],
            },
        },
        "hypothesis_tests": {
            "lagged_gate_passing_scales": gate_passing_lagged,
            "scale_rescues_formal_lagged_gate": bool(gate_passing_lagged),
            "s1_lagged_mIoU": s1["lagged"]["mIoU"],
            "s1_lagged_mTC": s1["lagged"]["mTC"],
            "best_lagged_mIoU_gain_vs_s1": best_lagged_miou["lagged"]["mIoU"] - s1["lagged"]["mIoU"],
            "best_lagged_mTC_gain_vs_s1": best_lagged_mtc["lagged"]["mTC"] - s1["lagged"]["mTC"],
            "interpretation_rule": (
                "If a scaled Lagged Observer path passes the existing +2 pp mIoU and "
                "mTC>Persistence gate, amplitude calibration is sufficient to rescue this "
                "candidate. If no scale passes, a pure scalar-amplitude explanation is "
                "weakened; direction/correspondence/local-matching quality still requires "
                "separate structural testing. Radius coverage alone is not causal evidence."
            ),
        },
        "per_sequence": per_sequence,
    }


def _raft_history_train_sequence(
    model,
    residual,
    raft,
    samples,
    optimizer,
    args,
    P,
):
    if len(samples) < 3:
        return {
            "frames": 0,
            "windows": 0,
            "semantic_loss": 0.0,
            "flow_loss": 0.0,
            "delta_loss": 0.0,
            "total_loss": 0.0,
        }
    max_future_steps = len(samples) - 2
    steps = min(max_future_steps, args.max_train_steps) if args.max_train_steps else max_future_steps

    frame0 = P._host_observation(model, samples[0])
    frame1 = P._host_observation(model, samples[1])
    image0, _, low0, _, _ = frame0
    image1, _, low1, _, _ = frame1
    with torch.no_grad():
        teacher1_full = raft.current_to_previous(image1, image0)
        teacher1 = P.downsample_backward_flow(teacher1_full, tuple(low1.shape[-2:]))
    error1 = F.softmax(low1.detach(), dim=1) - F.softmax(low0.detach(), dim=1)
    pending_motion, pending_delta, hidden = residual.predict_next(
        teacher1.detach(), error1.detach(), None
    )

    previous_image, previous_low = image1, low1.detach()
    semantic_losses = []
    flow_losses = []
    delta_losses = []
    sums = {
        "frames": 0,
        "windows": 0,
        "semantic_loss": 0.0,
        "flow_loss": 0.0,
        "delta_loss": 0.0,
        "total_loss": 0.0,
    }

    for step_index in range(steps):
        frame_index = step_index + 2
        current_image, _, current_low, _, output_size = P._host_observation(
            model, samples[frame_index]
        )
        warped_low, _ = P.warp_low_logits(previous_low, pending_motion)
        target = P.semantic_mask_from_panoptic_png(samples[frame_index]["mask_path"]).cuda(
            non_blocking=True
        )
        semantic_loss = F.cross_entropy(
            P._upsample_prior(warped_low, output_size),
            target.unsqueeze(0),
            ignore_index=P.IGNORE_LABEL,
        )
        with torch.no_grad():
            teacher_full = raft.current_to_previous(current_image, previous_image)
            teacher_low = P.downsample_backward_flow(
                teacher_full, tuple(current_low.shape[-2:])
            )
        flow_loss = P.normalized_flow_distillation_loss(
            pending_motion, teacher_low, args.max_observed_low
        )
        delta_loss = P.normalized_motion_residual_l2(
            pending_delta, residual.max_residual_displacement_low
        )
        if not all(
            torch.isfinite(x).item() for x in (semantic_loss, flow_loss, delta_loss)
        ):
            raise FloatingPointError("Non-finite RAFT-History Residual loss")
        semantic_losses.append(semantic_loss)
        flow_losses.append(flow_loss)
        delta_losses.append(delta_loss)
        sums["frames"] += 1

        error_t = F.softmax(current_low.detach(), dim=1) - F.softmax(warped_low, dim=1)
        previous_hidden = hidden
        next_motion, next_delta, next_hidden = residual.predict_next(
            teacher_low.detach(), error_t, previous_hidden
        )

        boundary = len(semantic_losses) == args.tbptt_steps or step_index == steps - 1
        if boundary:
            sem = torch.stack(semantic_losses).mean()
            flo = torch.stack(flow_losses).mean()
            delt = torch.stack(delta_losses).mean()
            total = (
                args.lambda_semantic * sem
                + args.lambda_flow * flo
                + args.lambda_delta * delt
            )
            if not torch.isfinite(total):
                raise FloatingPointError("Non-finite RAFT-History Residual total loss")
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            sums["windows"] += 1
            sums["semantic_loss"] += float(sem.detach().item())
            sums["flow_loss"] += float(flo.detach().item())
            sums["delta_loss"] += float(delt.detach().item())
            sums["total_loss"] += float(total.detach().item())

            # Recompute at the TBPTT boundary using a detached recurrent state,
            # matching the existing Stage-1B-2 implementation.
            pending_motion, pending_delta, hidden = residual.predict_next(
                teacher_low.detach(),
                error_t.detach(),
                previous_hidden.detach() if previous_hidden is not None else None,
            )
            semantic_losses, flow_losses, delta_losses = [], [], []
        else:
            pending_motion, pending_delta, hidden = next_motion, next_delta, next_hidden

        previous_image = current_image
        previous_low = current_low.detach()

    windows = max(sums["windows"], 1)
    for key in ("semantic_loss", "flow_loss", "delta_loss", "total_loss"):
        sums[key] /= windows
    return sums


def _raft_history_train_epoch(model, residual, raft, train_groups, optimizer, args, P):
    residual.train()
    total = {
        "frames": 0,
        "windows": 0,
        "semantic_loss": 0.0,
        "flow_loss": 0.0,
        "delta_loss": 0.0,
        "total_loss": 0.0,
    }
    for samples in train_groups.values():
        row = _raft_history_train_sequence(
            model, residual, raft, samples, optimizer, args, P
        )
        total["frames"] += row["frames"]
        for key in ("semantic_loss", "flow_loss", "delta_loss", "total_loss"):
            total[key] += row[key] * row["windows"]
        total["windows"] += row["windows"]
    windows = max(total["windows"], 1)
    for key in ("semantic_loss", "flow_loss", "delta_loss", "total_loss"):
        total[key] /= windows
    return total


@torch.inference_mode()
def _evaluate_raft_history_residual(model, residual, val_groups, raft, P):
    names = (
        "host",
        "semantic_persistence",
        "lagged_raft",
        "raft_history_residual",
        "lagged_raft_repair_only_oracle",
        "raft_history_residual_repair_only_oracle",
    )
    bank = _metric_struct(names, P)
    lagged_counts = P._new_counts()
    residual_counts = P._new_counts()
    diag = {
        "predicted_pairs": 0,
        "component_values": 0,
        "delta_abs": 0.0,
        "history_abs": 0.0,
        "predicted_abs": 0.0,
        "teacher_abs": 0.0,
        "predicted_teacher_l1": 0.0,
        "predicted_teacher_epe": 0.0,
    }
    per_sequence = {}
    residual.eval()

    for sequence in P.FULL9:
        seq_bank = _new_sequence_metric_struct(names, P)
        previous = None
        previous_teacher_low = None
        pending_motion = pending_delta = hidden = None
        previous_predictions = {}

        for sample in val_groups[sequence]:
            image, host_logits, host_low, _, output_size = P._host_observation(model, sample)
            host_pred = host_logits.argmax(1)
            gt_cpu = P.semantic_mask_from_panoptic_png(sample["mask_path"])
            gt = gt_cpu.cuda(non_blocking=True)
            teacher_full = None

            if previous is None:
                persistence = lagged_pred = residual_pred = host_pred
                lagged_oracle = residual_oracle = host_pred
                residual_warped_low = host_low
            else:
                previous_image, previous_low = previous
                persistence = P._upsample_prior(previous_low, output_size).argmax(1)
                teacher_full = raft.current_to_previous(image, previous_image)
                teacher_low = P.downsample_backward_flow(
                    teacher_full, tuple(host_low.shape[-2:])
                )

                if previous_teacher_low is None:
                    lagged_warped_low = previous_low
                else:
                    lagged_warped_low, _ = P.warp_low_logits(
                        previous_low, previous_teacher_low
                    )
                lagged_pred = P._upsample_prior(lagged_warped_low, output_size).argmax(1)

                if pending_motion is None:
                    residual_warped_low = previous_low
                else:
                    residual_warped_low, _ = P.warp_low_logits(previous_low, pending_motion)
                residual_pred = P._upsample_prior(residual_warped_low, output_size).argmax(1)

                lagged_oracle, hc, pc, valid = _make_repair_only_oracle(
                    host_pred, lagged_pred, gt, P.IGNORE_LABEL
                )
                P._add_counts(lagged_counts, hc, pc, valid)
                residual_oracle, hc, pc, valid = _make_repair_only_oracle(
                    host_pred, residual_pred, gt, P.IGNORE_LABEL
                )
                P._add_counts(residual_counts, hc, pc, valid)

                error_t = F.softmax(host_low, dim=1) - F.softmax(
                    residual_warped_low, dim=1
                )
                next_motion, next_delta, next_hidden = residual.predict_next(
                    teacher_low, error_t, hidden
                )

                if pending_motion is not None and previous_teacher_low is not None:
                    component_values = pending_motion.numel()
                    diag["predicted_pairs"] += 1
                    diag["component_values"] += component_values
                    diag["delta_abs"] += float(pending_delta.abs().sum().item())
                    diag["history_abs"] += float(previous_teacher_low.abs().sum().item())
                    diag["predicted_abs"] += float(pending_motion.abs().sum().item())
                    diag["teacher_abs"] += float(teacher_low.abs().sum().item())
                    error = pending_motion - teacher_low
                    diag["predicted_teacher_l1"] += float(error.abs().sum().item())
                    diag["predicted_teacher_epe"] += float(
                        torch.linalg.vector_norm(error, dim=1).sum().item()
                    )

                previous_teacher_low = teacher_low.detach()
                pending_motion = next_motion.detach()
                pending_delta = next_delta.detach()
                hidden = next_hidden.detach()

            predictions = {
                "host": host_pred,
                "semantic_persistence": persistence,
                "lagged_raft": lagged_pred,
                "raft_history_residual": residual_pred,
                "lagged_raft_repair_only_oracle": lagged_oracle,
                "raft_history_residual_repair_only_oracle": residual_oracle,
            }
            _update_segmentation_metrics(bank, seq_bank, predictions, gt_cpu, P)

            if previous is not None:
                scores = _batched_pair_mtc(
                    previous_predictions,
                    predictions,
                    teacher_full,
                    P.flow_grid,
                    P.NUM_CLASSES,
                    chunk_size=6,
                )
                _update_mtc_metrics(bank, seq_bank, scores)

            previous = (image, host_low.detach())
            previous_predictions = {k: v.detach() for k, v in predictions.items()}

        _close_sequence(bank, seq_bank, names)
        per_sequence[sequence] = _finalize_sequence_metrics(seq_bank, names, P)

    metrics = _finalize_global_metrics(bank, names, P)
    delta = {
        key: metrics["raft_history_residual"][key] - metrics["lagged_raft"][key]
        for key in ("mIoU", "mTC", "mVC8", "mVC16")
    }
    values = max(diag["component_values"], 1)
    vectors = max(diag["component_values"] // 2, 1)
    history_abs = diag["history_abs"] / values
    delta_abs = diag["delta_abs"] / values
    host_mtc = metrics["host"]["mTC"]
    return {
        "metrics": metrics,
        "delta_residual_vs_lagged_raft": delta,
        "lagged_complementarity_causal_frames_only": P._rates(lagged_counts),
        "residual_complementarity_causal_frames_only": P._rates(residual_counts),
        "repair_only_oracle_delta_mTC_vs_host": {
            "lagged_raft": metrics["lagged_raft_repair_only_oracle"]["mTC"] - host_mtc,
            "raft_history_residual": metrics[
                "raft_history_residual_repair_only_oracle"
            ]["mTC"]
            - host_mtc,
        },
        "residual_diagnostics": {
            "predicted_pairs": diag["predicted_pairs"],
            "delta_abs_mean_low_pixels": delta_abs,
            "history_raft_abs_mean_low_pixels": history_abs,
            "predicted_motion_abs_mean_low_pixels": diag["predicted_abs"] / values,
            "teacher_future_motion_abs_mean_low_pixels": diag["teacher_abs"] / values,
            "predicted_teacher_l1_mean_low_pixels": diag["predicted_teacher_l1"] / values,
            "predicted_teacher_epe_mean_low_pixels": diag["predicted_teacher_epe"] / vectors,
            "delta_to_history_amplitude_ratio": delta_abs / max(history_abs, 1e-12),
        },
        "per_sequence": per_sequence,
    }


def _raft_history_decision(metrics: Mapping) -> Mapping:
    delta = metrics["delta_residual_vs_lagged_raft"]
    d_iou = float(delta["mIoU"])
    d_mtc = float(delta["mTC"])
    pareto = d_iou >= 0.0 and d_mtc >= 0.0 and (d_iou > 0.0 or d_mtc > 0.0)
    reproduces_tradeoff = d_iou > 0.0 and d_mtc < 0.0
    if reproduces_tradeoff:
        decision = "reject_current_residual_predictor_design"
        reason = (
            "With high-quality causal RAFT history, Residual still raises mIoU while "
            "lowering mTC. The Observer is therefore not sufficient to explain the "
            "Stage-1B-2 tradeoff."
        )
    elif d_mtc > 0.0:
        decision = "observer_motion_quality_is_a_supported_failure_contributor"
        reason = (
            "With high-quality causal RAFT history, Residual improves mTC. This supports "
            "Observer motion quality as a contributor to the original Stage-1B-2 failure; "
            "it does not prove it is the only cause."
        )
    else:
        decision = "inconclusive_or_residual_not_useful"
        reason = (
            "High-quality history did not produce a positive mTC increment, but the exact "
            "mIoU/mTC pattern does not match the pre-specified rejection case. Review the "
            "per-epoch and per-sequence results before changing structure."
        )
    return {
        "decision": decision,
        "reason": reason,
        "delta_mIoU": d_iou,
        "delta_mTC": d_mtc,
        "residual_pareto_improves_lagged_raft": bool(pareto),
        "reproduces_mIoU_up_mTC_down_tradeoff": bool(reproduces_tradeoff),
    }


def _run_raft_history_residual(
    model,
    train_groups,
    val_groups,
    raft,
    args,
    result_dir: Path,
    P,
):
    residual = P.MotionResidualPredictor(
        num_classes=P.NUM_CLASSES,
        hidden_channels=args.residual_hidden_channels,
        max_observed_displacement_low=args.max_observed_low,
        max_residual_displacement_low=args.max_residual_low,
    ).cuda()
    optimizer = torch.optim.AdamW(
        residual.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Exact E0 check with real causal RAFT history.
    sanity_samples = next(iter(val_groups.values()))
    if len(sanity_samples) < 2:
        raise RuntimeError("Need at least two validation frames for zero-step check")
    frame0 = P._host_observation(model, sanity_samples[0])
    frame1 = P._host_observation(model, sanity_samples[1])
    with torch.no_grad():
        teacher_full = raft.current_to_previous(frame1[0], frame0[0])
        teacher_low = P.downsample_backward_flow(
            teacher_full, tuple(frame1[2].shape[-2:])
        )
        error = F.softmax(frame1[2], dim=1) - F.softmax(frame0[2], dim=1)
        predicted, delta, _ = residual.predict_next(teacher_low, error, None)
    zero_step = {
        "delta_motion_max_abs": float(delta.abs().max().item()),
        "predicted_equals_history_raft_max_abs": float(
            (predicted - teacher_low).abs().max().item()
        ),
    }
    if zero_step["delta_motion_max_abs"] != 0.0 or zero_step[
        "predicted_equals_history_raft_max_abs"
    ] != 0.0:
        raise RuntimeError(f"RAFT-History zero-step check failed: {zero_step}")

    ckpt_dir = Path(args.raft_residual_output)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    history = []

    for epoch in range(1, args.epochs + 1):
        train_stats = _raft_history_train_epoch(
            model, residual, raft, train_groups, optimizer, args, P
        )
        metrics = _evaluate_raft_history_residual(
            model, residual, val_groups, raft, P
        )
        decision = _raft_history_decision(metrics)
        row = {
            "epoch": epoch,
            "train": train_stats,
            "metrics": metrics,
            "diagnostic_decision_if_stopped_here": decision,
        }
        history.append(row)
        payload = {
            "experiment": "c_v2_raft_history_residual_diagnostic",
            "diagnostic_only": True,
            "not_an_inference_path": True,
            "epoch": epoch,
            "residual_state_dict": residual.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "architecture": {
                "input_motion": "strictly causal RAFT F_t = F_{t->t-1}",
                "semantic_error": "Host probability - residual-prior probability",
                "prediction": "F_hat_{t+1} = F_t^RAFT + DeltaF_{t+1}",
                "hidden_channels": args.residual_hidden_channels,
                "max_observed_low": args.max_observed_low,
                "max_residual_low": args.max_residual_low,
                "delta_zero_initialized": True,
            },
            "row": row,
        }
        torch.save(payload, ckpt_dir / f"epoch_{epoch:03d}.pt")
        (result_dir / f"epoch_{epoch:03d}.json").write_text(
            json.dumps(row, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "phase": "raft_history_residual",
                    "epoch": epoch,
                    "delta_mIoU": decision["delta_mIoU"],
                    "delta_mTC": decision["delta_mTC"],
                    "decision": decision["decision"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    final = history[-1]
    summary = {
        "experiment": "RAFT-History Residual Diagnostic",
        "diagnostic_only": True,
        "not_an_inference_path": True,
        "training": {
            "epochs": args.epochs,
            "tbptt_steps": args.tbptt_steps,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "lambda_semantic": args.lambda_semantic,
            "lambda_flow": args.lambda_flow,
            "lambda_delta": args.lambda_delta,
            "max_observed_low": args.max_observed_low,
            "max_residual_low": args.max_residual_low,
            "max_train_steps": args.max_train_steps,
        },
        "zero_step": zero_step,
        "history": history,
        "fixed_final_epoch_decision": final["diagnostic_decision_if_stopped_here"],
        "decision_uses_fixed_final_epoch_not_best_checkpoint": True,
    }
    (result_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _pct(x: float) -> str:
    return "nan" if not _isfinite(x) else f"{100.0 * x:.4f}%"


def _pp(x: float) -> str:
    return "nan" if not _isfinite(x) else f"{100.0 * x:+.4f} pp"


def _render_markdown(summary: Mapping) -> str:
    lines = [
        "# C-V2 Stage-1B Failure Diagnostics",
        "",
        "This report is diagnostic only. It does not open D0 or Semantic Residual Stage-2.",
        "",
    ]
    cal = summary.get("observer_calibration")
    if cal:
        lines.extend(
            [
                "## A. Observer Flow Calibration Diagnostic",
                "",
                "| Scale | Current mIoU | Current mTC | Lagged mIoU | Lagged mTC | Lagged mVC8 | Lagged mVC16 | Lagged Oracle ΔmTC vs Host | Flow EPE | Flow L1 | Cosine | Lagged Gate |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
            ]
        )
        for row in cal["scale_results"]:
            lines.append(
                "| {scale:g} | {ciou} | {cmtc} | {liou} | {lmtc} | {mvc8} | {mvc16} | {oracle} | {epe:.4f} | {l1:.4f} | {cos:.4f} | {gate} |".format(
                    scale=row["scale"],
                    ciou=_pct(row["current"]["mIoU"]),
                    cmtc=_pct(row["current"]["mTC"]),
                    liou=_pct(row["lagged"]["mIoU"]),
                    lmtc=_pct(row["lagged"]["mTC"]),
                    mvc8=_pct(row["lagged"]["mVC8"]),
                    mvc16=_pct(row["lagged"]["mVC16"]),
                    oracle=_pp(row["lagged_repair_only_oracle_delta_mTC_vs_host"]),
                    epe=row["flow"]["epe_mean_low_pixels_reachable"],
                    l1=row["flow"]["component_l1_mean_low_pixels_reachable"],
                    cos=row["flow"]["cosine_similarity_mean_nonzero_reachable"],
                    gate="GO" if row["lagged_stage1b_motion_gate"] else "NO-GO",
                )
            )
        lines.extend(["", "### Correlation-radius coverage", "", "| Radius | RAFT teacher coverage |", "|---:|---:|"])
        for radius, fraction in cal[
            "radius_coverage_diagnostic_only_no_observer_retraining"
        ]["coverage_fraction_of_spatial_valid"].items():
            lines.append(f"| {radius} | {_pct(fraction)} |")
        lines.extend(
            [
                "",
                f"Lagged gate-passing scales: `{cal['hypothesis_tests']['lagged_gate_passing_scales']}`",
                f"Scale rescues formal Lagged gate: `{cal['hypothesis_tests']['scale_rescues_formal_lagged_gate']}`",
                "",
            ]
        )

    rh = summary.get("raft_history_residual")
    if rh:
        lines.extend(
            [
                "## B. RAFT-History Residual Diagnostic",
                "",
                "| Epoch | Lagged RAFT mIoU | Residual mIoU | ΔmIoU | Lagged RAFT mTC | Residual mTC | ΔmTC | Δ/History amplitude | Decision |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in rh["history"]:
            m = row["metrics"]
            base = m["metrics"]["lagged_raft"]
            res = m["metrics"]["raft_history_residual"]
            d = m["delta_residual_vs_lagged_raft"]
            amp = m["residual_diagnostics"]["delta_to_history_amplitude_ratio"]
            decision = row["diagnostic_decision_if_stopped_here"]["decision"]
            lines.append(
                f"| {row['epoch']} | {_pct(base['mIoU'])} | {_pct(res['mIoU'])} | {_pp(d['mIoU'])} | {_pct(base['mTC'])} | {_pct(res['mTC'])} | {_pp(d['mTC'])} | {amp:.4f} | `{decision}` |"
            )
        final = rh["fixed_final_epoch_decision"]
        lines.extend(
            [
                "",
                f"Fixed final-epoch decision: `{final['decision']}`",
                "",
                final["reason"],
                "",
            ]
        )

    lines.extend(
        [
            "## Boundary",
            "",
            "- No D0.",
            "- No Semantic Residual Stage-2.",
            "- Repair-only Oracle is diagnostic and never replaces the formal gate.",
            "- Radius coverage is a coverage diagnostic; it is not evidence that radius alone caused failure.",
            "- RAFT-History Residual is a diagnostic with privileged motion input, not an inference model.",
            "",
        ]
    )
    return "\n".join(lines)


def _validate_args(args) -> None:
    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0")
    if args.tbptt_steps <= 0:
        raise ValueError("--tbptt-steps must be > 0")
    if args.max_observed_low <= 0 or args.max_residual_low <= 0:
        raise ValueError("motion bounds must be > 0")
    if args.mtc_chunk_size <= 0:
        raise ValueError("--mtc-chunk-size must be > 0")
    if not args.scales or any(s <= 0 for s in args.scales):
        raise ValueError("all --scales values must be > 0")
    if not args.radii or any(r < 0 for r in args.radii):
        raise ValueError("all --radii values must be >= 0")
    if len(set(args.scales)) != len(args.scales):
        raise ValueError("--scales contains duplicates")
    if len(set(args.radii)) != len(args.radii):
        raise ValueError("--radii contains duplicates")


def _self_test() -> None:
    # Pure logic tests that do not require project files, data, CUDA, or checkpoints.
    assert _scale_tag(0.5) == "0p5"
    assert _scale_tag(1.0) == "1"
    assert abs(_miou_from_confusion(torch.eye(3, dtype=torch.int64)) - 1.0) < 1e-12

    case = {
        "delta_residual_vs_lagged_raft": {
            "mIoU": 0.01,
            "mTC": -0.005,
            "mVC8": 0.0,
            "mVC16": 0.0,
        }
    }
    decision = _raft_history_decision(case)
    assert decision["decision"] == "reject_current_residual_predictor_design"

    case["delta_residual_vs_lagged_raft"]["mTC"] = 0.003
    decision = _raft_history_decision(case)
    assert decision["decision"] == "observer_motion_quality_is_a_supported_failure_contributor"
    print(json.dumps({"self_test": "PASS"}, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("all", "calibration", "raft-history-residual"), default="all")
    p.add_argument("--root", default=DEFAULT_ROOT)
    p.add_argument("--fast-b-checkpoint", default=None)
    p.add_argument("--observer-checkpoint", default=DEFAULT_OBSERVER_CHECKPOINT)
    p.add_argument("--result-root", default=DEFAULT_RESULT_ROOT)
    p.add_argument("--raft-residual-output", default=DEFAULT_RAFT_RESIDUAL_OUTPUT)
    p.add_argument("--scales", nargs="+", type=float, default=list(DEFAULT_SCALES))
    p.add_argument("--radii", nargs="+", type=int, default=list(DEFAULT_RADII))
    p.add_argument("--mtc-chunk-size", type=int, default=8)

    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--tbptt-steps", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--lambda-semantic", type=float, default=1.0)
    p.add_argument("--lambda-flow", type=float, default=10.0)
    p.add_argument("--lambda-delta", type=float, default=1e-2)
    p.add_argument("--residual-hidden-channels", type=int, default=64)
    p.add_argument("--max-observed-low", type=float, default=32.0)
    p.add_argument("--max-residual-low", type=float, default=16.0)
    p.add_argument("--max-train-steps", type=int, default=0)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--self-test", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.self_test:
        _self_test()
        return
    _validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the real KITTI-STEP diagnostics")

    P = _load_project()
    _seed_everything(args.seed)
    fast_b_checkpoint = args.fast_b_checkpoint or P.FAST_B_CHECKPOINT_DEFAULT
    result_root = Path(args.result_root)
    result_root.mkdir(parents=True, exist_ok=True)

    print(
        json.dumps(
            {
                "phase": "setup",
                "mode": args.mode,
                "root": args.root,
                "fast_b_checkpoint": fast_b_checkpoint,
                "observer_checkpoint": args.observer_checkpoint,
                "result_root": str(result_root),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    model = P.load_fast_b_model(fast_b_checkpoint).cuda().eval()
    model.requires_grad_(False)
    raft = P.FrozenRAFT()

    val = P.KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "val")
    all_val = P.sequence_groups(val)
    missing = [s for s in P.FULL9 if s not in all_val]
    if missing:
        raise RuntimeError(f"Missing Full9 validation sequences: {missing}")
    val_groups = {s: all_val[s] for s in P.FULL9}

    overall = {
        "experiment": "C-V2 Stage-1B Failure Diagnostics",
        "mode": args.mode,
        "seed": args.seed,
        "full9": list(P.FULL9),
        "boundaries": {
            "D0": False,
            "semantic_residual_stage2": False,
            "repair_only_oracle_is_formal_gate": False,
            "raft_history_residual_is_inference_path": False,
        },
    }

    if args.mode in ("all", "calibration"):
        print(json.dumps({"phase": "observer_calibration", "status": "START"}), flush=True)
        observer, observer_payload = P._load_frozen_observer(args.observer_checkpoint)
        calibration = _observer_calibration_diagnostic(
            model=model,
            observer=observer,
            val_groups=val_groups,
            raft=raft,
            scales=args.scales,
            radii=args.radii,
            mtc_chunk_size=args.mtc_chunk_size,
            P=P,
        )
        calibration["observer_checkpoint"] = args.observer_checkpoint
        calibration["observer_architecture"] = observer_payload["architecture"]
        calibration["observer_stage1b1_checks"] = observer_payload["row"]["stage1b1_checks"]
        overall["observer_calibration"] = calibration
        cal_dir = result_root / "observer_calibration"
        cal_dir.mkdir(parents=True, exist_ok=True)
        (cal_dir / "summary.json").write_text(
            json.dumps(calibration, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "phase": "observer_calibration",
                    "status": "DONE",
                    "gate_passing_scales": calibration["hypothesis_tests"][
                        "lagged_gate_passing_scales"
                    ],
                    "best_flow_epe_scale": calibration["best_scales"]["flow_by_EPE"]["scale"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if args.mode in ("all", "raft-history-residual"):
        print(json.dumps({"phase": "raft_history_residual", "status": "START"}), flush=True)
        train = P.KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "train")
        train_groups = P.sequence_groups(train)
        residual_summary = _run_raft_history_residual(
            model=model,
            train_groups=train_groups,
            val_groups=val_groups,
            raft=raft,
            args=args,
            result_dir=result_root / "raft_history_residual",
            P=P,
        )
        overall["raft_history_residual"] = residual_summary
        print(
            json.dumps(
                {
                    "phase": "raft_history_residual",
                    "status": "DONE",
                    "final_decision": residual_summary["fixed_final_epoch_decision"]["decision"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    (result_root / "summary.json").write_text(
        json.dumps(overall, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (result_root / "summary.md").write_text(
        _render_markdown(overall) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "DONE",
                "summary_json": str(result_root / "summary.json"),
                "summary_markdown": str(result_root / "summary.md"),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
