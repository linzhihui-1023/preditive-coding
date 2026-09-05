"""One-shot Full9 transportability oracle diagnostic for C-V2 Stage 1B.

Purpose
-------
Answer one question only: does temporal transport behave materially differently
between pixels that can be explained by previous-frame transport and pixels that
cannot?

This is a diagnostic, not a training protocol. It does not modify any weights.
RAFT and ground-truth labels are used only to build an oracle/proxy partition:

    transportable:
        current GT is valid, the RAFT current->previous sample is in bounds and
        lands on a valid previous GT pixel, and warped previous GT == current GT.

    non_transportable:
        every other valid current-GT pixel.

The second region therefore includes true disocclusion/newly visible semantics,
but also RAFT correspondence errors. It must not be reported as a pure physical
occlusion mask.

Only four predictions are compared:
    - Host
    - Observer-Lagged prior
    - Stage 1B-2 Residual prior
    - RAFT-warp prior (geometric reference)

Only region fraction and region-wise mIoU are reported. No extra threshold sweep,
class-wise analysis, boundary metric, or mTC analysis is performed here.
"""

import argparse
import json
from pathlib import Path

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
    FAST_B_CHECKPOINT_DEFAULT,
    load_fast_b_model,
)
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_only import (
    FULL9,
    IGNORE_LABEL,
    NUM_CLASSES,
    _upsample_prior,
)
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v2_stage1b_observer import (
    _host_observation,
)
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v2_stage1b_residual import (
    OBSERVER_CHECKPOINT_DEFAULT,
    RESIDUAL_HIDDEN_CHANNELS,
    _load_frozen_observer,
    _observe_motion,
)
from predify2021.mce_scores.train_kitti_step_temporal_joint import FrozenRAFT
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_motion_transport import (
    MotionResidualPredictor,
    downsample_backward_flow,
    warp_low_logits,
)


RESIDUAL_CHECKPOINT_DEFAULT = (
    "/home/lin/predify/experiments/"
    "kitti_step_task_space_prior_c_v2_stage1b_residual/best.pt"
)
RESULT_DEFAULT = "results/kitti_step_task_space_prior_c_v2_transportability_oracle.json"
PREDICTION_NAMES = (
    "host",
    "observer_lagged",
    "observer_residual",
    "raft_prior",
)
REGION_NAMES = ("transportable", "non_transportable")


def _load_frozen_residual(path, observer):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("experiment") != "c_v2_stage1b2_residual_motion":
        raise RuntimeError("Residual checkpoint is not a Stage 1B-2 checkpoint")
    architecture = payload.get("architecture", {})
    max_residual_low = float(architecture.get("max_residual_low", 16.0))
    residual = MotionResidualPredictor(
        num_classes=NUM_CLASSES,
        hidden_channels=RESIDUAL_HIDDEN_CHANNELS,
        max_observed_displacement_low=observer.max_displacement_low,
        max_residual_displacement_low=max_residual_low,
    ).cuda()
    residual.load_state_dict(payload["residual_state_dict"], strict=True)
    residual.requires_grad_(False).eval()
    return residual, payload


def _warp_label_nearest(previous_gt, backward_flow):
    """Sample previous labels with current->previous full-resolution flow."""
    if previous_gt.ndim != 2:
        raise ValueError("previous_gt must be HxW")
    if backward_flow.shape[0] != 1 or backward_flow.shape[1] != 2:
        raise ValueError("backward_flow must be 1x2xHxW")
    height, width = previous_gt.shape
    if tuple(backward_flow.shape[-2:]) != (height, width):
        raise ValueError(
            f"GT/flow size mismatch: GT={(height, width)}, flow={tuple(backward_flow.shape[-2:])}"
        )

    dtype = backward_flow.dtype
    device = backward_flow.device
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    source_x = x.unsqueeze(0) + backward_flow[:, 0]
    source_y = y.unsqueeze(0) + backward_flow[:, 1]
    in_bounds = (
        (source_x >= 0)
        & (source_x <= width - 1)
        & (source_y >= 0)
        & (source_y <= height - 1)
    )
    grid = torch.stack(
        (
            2.0 * source_x / max(width - 1, 1) - 1.0,
            2.0 * source_y / max(height - 1, 1) - 1.0,
        ),
        dim=-1,
    )
    sampled = F.grid_sample(
        previous_gt.to(device=device, dtype=torch.float32)[None, None],
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
    )[0, 0].long()
    source_valid = in_bounds.squeeze(0) & (sampled != IGNORE_LABEL)
    return sampled, source_valid


def _transportability_partition(previous_gt, current_gt, teacher_full):
    warped_previous_gt, source_valid = _warp_label_nearest(previous_gt, teacher_full)
    current_gt_gpu = current_gt.cuda(non_blocking=True)
    current_valid = current_gt_gpu != IGNORE_LABEL
    transportable = current_valid & source_valid & (warped_previous_gt == current_gt_gpu)
    non_transportable = current_valid & ~transportable

    if torch.any(transportable & non_transportable):
        raise RuntimeError("Transportability regions overlap")
    if not torch.equal(transportable | non_transportable, current_valid):
        raise RuntimeError("Transportability regions do not partition valid pixels")
    return transportable, non_transportable


def _masked_target(gt_cpu, region_gpu):
    target = gt_cpu.clone()
    target[~region_gpu.cpu()] = IGNORE_LABEL
    return target


def _miou(confusion):
    value = torch.nanmean(compute_iou(confusion))
    return float(value.item())


@torch.inference_mode()
def evaluate(model, observer, residual, groups, raft):
    confusion = {
        region: {
            name: torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
            for name in PREDICTION_NAMES
        }
        for region in REGION_NAMES
    }
    region_pixels = {region: 0 for region in REGION_NAMES}
    total_valid_pixels = 0
    evaluated_pairs = 0

    observer.eval()
    residual.eval()

    for sequence in FULL9:
        previous = None
        previous_gt = None
        previous_observed_motion = None
        pending_motion = None
        hidden = None

        for sample in groups[sequence]:
            image, host_logits, host_low, c1, output_size = _host_observation(model, sample)
            host_pred = host_logits.argmax(1)
            gt_cpu = semantic_mask_from_panoptic_png(sample["mask_path"])

            if previous is None:
                previous = (image, host_low.detach(), c1.detach())
                previous_gt = gt_cpu
                continue

            previous_image, previous_low, previous_c1 = previous
            teacher_full = raft.current_to_previous(image, previous_image)
            teacher_low = downsample_backward_flow(
                teacher_full, tuple(host_low.shape[-2:])
            )

            if previous_observed_motion is None:
                lagged_warped_low = previous_low
            else:
                lagged_warped_low, _ = warp_low_logits(
                    previous_low, previous_observed_motion
                )

            if pending_motion is None:
                residual_warped_low = previous_low
            else:
                residual_warped_low, _ = warp_low_logits(previous_low, pending_motion)

            raft_warped_low, _ = warp_low_logits(previous_low, teacher_low)

            predictions = {
                "host": host_pred,
                "observer_lagged": _upsample_prior(lagged_warped_low, output_size).argmax(1),
                "observer_residual": _upsample_prior(residual_warped_low, output_size).argmax(1),
                "raft_prior": _upsample_prior(raft_warped_low, output_size).argmax(1),
            }

            transportable, non_transportable = _transportability_partition(
                previous_gt, gt_cpu, teacher_full
            )
            regions = {
                "transportable": transportable,
                "non_transportable": non_transportable,
            }
            valid_count = int((gt_cpu != IGNORE_LABEL).sum().item())
            total_valid_pixels += valid_count
            for region_name, region_mask in regions.items():
                count = int(region_mask.sum().item())
                region_pixels[region_name] += count
                target = _masked_target(gt_cpu, region_mask)
                for name, prediction in predictions.items():
                    update_confusion_matrix(
                        confusion[region_name][name],
                        prediction.squeeze(0).cpu(),
                        target,
                    )

            current_observed_motion = _observe_motion(
                observer, previous_low, previous_c1, host_low, c1
            )
            error_t = F.softmax(host_low, dim=1) - F.softmax(residual_warped_low, dim=1)
            next_motion, _, next_hidden = residual.predict_next(
                current_observed_motion, error_t, hidden
            )
            previous_observed_motion = current_observed_motion.detach()
            pending_motion = next_motion.detach()
            hidden = next_hidden.detach()

            previous = (image, host_low.detach(), c1.detach())
            previous_gt = gt_cpu
            evaluated_pairs += 1

    metrics = {
        region: {
            name: {"mIoU": _miou(confusion[region][name])}
            for name in PREDICTION_NAMES
        }
        for region in REGION_NAMES
    }

    for region in REGION_NAMES:
        host_miou = metrics[region]["host"]["mIoU"]
        for name in PREDICTION_NAMES:
            metrics[region][name]["delta_mIoU_vs_host"] = (
                metrics[region][name]["mIoU"] - host_miou
            )

    lagged_gap = (
        metrics["transportable"]["observer_lagged"]["delta_mIoU_vs_host"]
        - metrics["non_transportable"]["observer_lagged"]["delta_mIoU_vs_host"]
    )
    residual_gap = (
        metrics["transportable"]["observer_residual"]["delta_mIoU_vs_host"]
        - metrics["non_transportable"]["observer_residual"]["delta_mIoU_vs_host"]
    )
    raft_gap = (
        metrics["transportable"]["raft_prior"]["delta_mIoU_vs_host"]
        - metrics["non_transportable"]["raft_prior"]["delta_mIoU_vs_host"]
    )

    return {
        "definition": {
            "transportable": (
                "valid current GT; RAFT current->previous source is in bounds and valid; "
                "nearest-warped previous GT equals current GT"
            ),
            "non_transportable": (
                "all remaining valid current-GT pixels; includes true disocclusion/new semantics "
                "and RAFT correspondence errors"
            ),
        },
        "evaluated_frame_pairs": evaluated_pairs,
        "valid_pixels": total_valid_pixels,
        "region_support": {
            region: {
                "pixels": region_pixels[region],
                "fraction": region_pixels[region] / max(total_valid_pixels, 1),
            }
            for region in REGION_NAMES
        },
        "region_metrics": metrics,
        "effect_summary": {
            "observer_lagged_transport_minus_nontransport_gain": lagged_gap,
            "observer_residual_transport_minus_nontransport_gain": residual_gap,
            "raft_prior_transport_minus_nontransport_gain_reference_only": raft_gap,
            "decision_signal": (
                "Use the Observer/Residual regional gaps as the decision signal. RAFT is part of "
                "the oracle partition and is reference-only, not independent evidence. Only a "
                "large regional separation is actionable; do not chase sub-percentage differences."
            ),
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/home/lin/predify/kitti_step")
    parser.add_argument("--fast-b-checkpoint", default=FAST_B_CHECKPOINT_DEFAULT)
    parser.add_argument("--observer-checkpoint", default=OBSERVER_CHECKPOINT_DEFAULT)
    parser.add_argument("--residual-checkpoint", default=RESIDUAL_CHECKPOINT_DEFAULT)
    parser.add_argument("--output", default=RESULT_DEFAULT)
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    model = load_fast_b_model(args.fast_b_checkpoint).cuda().eval()
    model.requires_grad_(False)
    observer, _ = _load_frozen_observer(args.observer_checkpoint)
    residual, residual_payload = _load_frozen_residual(args.residual_checkpoint, observer)
    raft = FrozenRAFT()

    val = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "val")
    all_groups = sequence_groups(val)
    missing = [sequence for sequence in FULL9 if sequence not in all_groups]
    if missing:
        raise RuntimeError(f"Missing Full9 validation sequences: {missing}")
    groups = {sequence: all_groups[sequence] for sequence in FULL9}

    result = evaluate(model, observer, residual, groups, raft)
    result["experiment"] = "C-V2 Full9 Transportability Oracle Diagnostic"
    result["weights_modified"] = False
    result["residual_checkpoint"] = args.residual_checkpoint
    result["residual_checkpoint_epoch"] = residual_payload.get("epoch")
    result["scope"] = (
        "One-shot diagnostic only. No threshold sweep, no new training, and no automatic model change."
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
