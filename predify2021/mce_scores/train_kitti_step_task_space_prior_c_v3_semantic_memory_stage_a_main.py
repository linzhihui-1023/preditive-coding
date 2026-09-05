"""Safe entrypoint for C-V3 Stage A semantic-memory training.

This wrapper applies two corrections to the base Stage-A implementation:

1) Invalid-history semantic evidence is handled by the updated
   ``task_space_semantic_memory`` module: invalid motion sources contribute
   neither prediction error nor transportability to the new Memory/Readout.

2) Temporal-loss scaling is calibrated only after one segmentation-only TBPTT
   warm-up window. The next valid window measures gradients through the now-open
   full refinement path and fixes

       lambda_temporal = 0.5 * ||grad L_seg|| / ||grad L_temp||.

There is no lambda sweep. The warm-up exists only to move the zero-initialized
DeltaR head away from zero before gradient-scale calibration.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

from predify2021.mce_scores import (
    train_kitti_step_task_space_prior_c_v3_semantic_memory_stage_a as base,
)


TEMPORAL_GRAD_RATIO = 0.5


def _train_sequence_warmup_then_calibrate(
    model,
    observer,
    residual,
    correction,
    mask_predictor,
    refiner,
    raft,
    samples,
    optimizer,
    tbptt_steps,
    temporal_scale_state,
):
    if len(samples) < 3:
        return None

    temporal_scale_state.setdefault("warmup_done", False)
    temporal_scale_state.setdefault("warmup_windows", 0)
    temporal_scale_state.setdefault("calibration_ratio", TEMPORAL_GRAD_RATIO)
    temporal_scale_state.setdefault("calibration_window", None)
    temporal_scale_state.setdefault("global_window_count", 0)

    frame0 = base._host_observation(model, samples[0])
    frame1 = base._host_observation(model, samples[1])
    pending_motion, motion_hidden = base._initialize_motion(
        observer, residual, frame0, frame1
    )
    previous_image, previous_host_logits, previous_low, previous_c1, _ = frame1
    previous_gt = base.semantic_mask_from_panoptic_png(samples[1]["mask_path"])
    previous_output_logits = previous_host_logits.detach()

    transport_hidden = semantic_hidden = mask_hidden = None
    semantic_state_low = torch.zeros_like(previous_low)
    memory_state = None
    buffered_seg = []
    buffered_temp = []
    totals = {
        "frames": 0,
        "windows": 0,
        "segmentation_ce": 0.0,
        "temporal_l1": 0.0,
        "total": 0.0,
        "temporal_weight_mean": 0.0,
        "temporal_valid_fraction": 0.0,
        "previous_true_confidence_mean": 0.0,
        "memory_abs": 0.0,
        "warped_memory_abs": 0.0,
        "memory_reliability_mean": 0.0,
        "agreement_mean": 0.0,
        "delta_refinement_abs": 0.0,
        "semantic_state_abs": 0.0,
        "transportability_mean": 0.0,
    }

    trainable = [
        parameter
        for parameter in refiner.parameters()
        if parameter.requires_grad
    ]

    for frame_index in range(2, len(samples)):
        current_image, host_logits, host_low, current_c1, output_size = (
            base._host_observation(model, samples[frame_index])
        )
        current_gt = base.semantic_mask_from_panoptic_png(
            samples[frame_index]["mask_path"]
        )

        with torch.no_grad():
            prior_low, _ = base.warp_low_logits(previous_low, pending_motion)
            e1 = base._frozen_e1_step(
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

        memory_row = refiner(
            current_c1.detach(),
            host_low.detach(),
            prior_low.detach(),
            e1["transportability_low"].detach(),
            pending_motion.detach(),
            semantic_state_low.detach(),
            memory_state,
        )
        memory_state = memory_row["memory"]
        c_v3_delta_low = (
            e1["e1_delta_low"].detach()
            + memory_row["delta_refinement"]
        )
        current_output_logits = (
            host_logits.detach()
            + base._upsample_prior(c_v3_delta_low, output_size)
        )
        target_gpu = current_gt.to(
            current_output_logits.device,
            non_blocking=True,
        ).unsqueeze(0)
        segmentation_ce = F.cross_entropy(
            current_output_logits,
            target_gpu,
            ignore_index=base.IGNORE_LABEL,
        )

        with torch.no_grad():
            teacher_full = raft.current_to_previous(
                current_image,
                previous_image,
            )
        temporal_l1, temporal_diag = base._strict_temporal_l1(
            current_output_logits,
            previous_output_logits,
            current_gt,
            previous_gt,
            teacher_full,
            e1["transportability_low"],
        )

        if not torch.isfinite(segmentation_ce) or not torch.isfinite(temporal_l1):
            raise FloatingPointError("Non-finite C-V3 Stage-A loss")
        buffered_seg.append(segmentation_ce)
        buffered_temp.append(temporal_l1)

        with torch.no_grad():
            observed_motion = base._observe_motion(
                observer,
                previous_low,
                previous_c1,
                host_low,
                current_c1,
            )
            prediction_error = F.softmax(host_low, dim=1) - F.softmax(
                prior_low,
                dim=1,
            )
            next_motion, _, next_motion_hidden = residual.predict_next(
                observed_motion,
                prediction_error,
                motion_hidden,
            )

        totals["frames"] += 1
        totals["segmentation_ce"] += float(segmentation_ce.detach().item())
        totals["temporal_l1"] += float(temporal_l1.detach().item())
        for key in (
            "temporal_weight_mean",
            "temporal_valid_fraction",
            "previous_true_confidence_mean",
        ):
            totals[key] += temporal_diag[key]
        totals["memory_abs"] += float(
            memory_state.abs().mean().detach().item()
        )
        totals["warped_memory_abs"] += float(
            memory_row["warped_memory"].abs().mean().detach().item()
        )
        totals["memory_reliability_mean"] += float(
            memory_row["memory_reliability"].mean().detach().item()
        )
        totals["agreement_mean"] += float(
            memory_row["agreement"].mean().detach().item()
        )
        totals["delta_refinement_abs"] += float(
            memory_row["delta_refinement"].abs().mean().detach().item()
        )
        totals["semantic_state_abs"] += float(
            semantic_state_low.abs().mean().item()
        )
        totals["transportability_mean"] += float(
            e1["transportability_low"].mean().item()
        )

        boundary = (
            len(buffered_seg) == tbptt_steps
            or frame_index == len(samples) - 1
        )
        if boundary:
            window_seg = torch.stack(buffered_seg).mean()
            window_temp = torch.stack(buffered_temp).mean()
            temporal_scale_state["global_window_count"] += 1

            # Window 1 is deliberately segmentation-only. With the refinement
            # head zero-initialized, this first update opens the gradient path to
            # Memory/pre/dilated branches for the later calibration window.
            is_warmup = not temporal_scale_state["warmup_done"]
            if is_warmup:
                lambda_temporal = 0.0
                window_loss = window_seg
            else:
                if (
                    temporal_scale_state["value"] is None
                    and float(window_temp.detach().item()) > 0.0
                ):
                    g_seg = base._gradient_norm(window_seg, trainable)
                    g_temp = base._gradient_norm(window_temp, trainable)
                    if g_seg > base.GRAD_EPS and g_temp > base.GRAD_EPS:
                        temporal_scale_state["value"] = (
                            TEMPORAL_GRAD_RATIO * g_seg / g_temp
                        )
                        temporal_scale_state["seg_grad_norm"] = g_seg
                        temporal_scale_state["temp_grad_norm"] = g_temp
                        temporal_scale_state["calibration_window"] = int(
                            temporal_scale_state["global_window_count"]
                        )

                lambda_temporal = temporal_scale_state["value"]
                if lambda_temporal is None:
                    lambda_temporal = 0.0
                window_loss = (
                    window_seg
                    + float(lambda_temporal) * window_temp
                )

            optimizer.zero_grad(set_to_none=True)
            window_loss.backward()
            optimizer.step()

            if is_warmup:
                temporal_scale_state["warmup_done"] = True
                temporal_scale_state["warmup_windows"] += 1

            totals["windows"] += 1
            totals["total"] += float(window_loss.detach().item())
            buffered_seg = []
            buffered_temp = []
            memory_state = memory_state.detach()

        previous_image = current_image
        previous_output_logits = current_output_logits.detach()
        previous_gt = current_gt
        previous_low = host_low.detach()
        previous_c1 = current_c1.detach()
        pending_motion = next_motion.detach()
        motion_hidden = next_motion_hidden.detach()

    frames = max(totals["frames"], 1)
    windows = max(totals["windows"], 1)
    for key in (
        "segmentation_ce",
        "temporal_l1",
        "temporal_weight_mean",
        "temporal_valid_fraction",
        "previous_true_confidence_mean",
        "memory_abs",
        "warped_memory_abs",
        "memory_reliability_mean",
        "agreement_mean",
        "delta_refinement_abs",
        "semantic_state_abs",
        "transportability_mean",
    ):
        totals[key] /= frames
    totals["total"] /= windows
    totals["lambda_temporal"] = (
        float(temporal_scale_state["value"])
        if temporal_scale_state["value"] is not None
        else 0.0
    )
    totals["temporal_warmup_windows"] = int(
        temporal_scale_state["warmup_windows"]
    )
    totals["temporal_calibration_window"] = (
        int(temporal_scale_state["calibration_window"])
        if temporal_scale_state["calibration_window"] is not None
        else -1
    )
    return totals


def _resolve_output_paths(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", default=base.OUTPUT_DEFAULT)
    parser.add_argument("--result-output", default=base.RESULT_DEFAULT)
    known, _ = parser.parse_known_args(argv)
    return Path(known.output), Path(known.result_output)


def _patch_saved_metadata(output, result_output):
    contract_text = (
        "one segmentation-only TBPTT warm-up window, then the next valid "
        "window fixes lambda_temporal = 0.5 * ||grad L_seg|| / "
        "||grad L_temp||; no lambda sweep"
    )

    summary_path = result_output / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        summary["training"]["temporal_warmup_windows"] = 1
        summary["training"]["temporal_grad_ratio"] = TEMPORAL_GRAD_RATIO
        summary["training"]["temporal_scale_policy"] = contract_text
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )

    for checkpoint_path in list(output.glob("epoch_*.pt")) + [output / "best.pt"]:
        if not checkpoint_path.exists():
            continue
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        payload.setdefault("architecture", {})[
            "invalid_history_semantics"
        ] = "invalid warp zeros Memory, prediction error, and T evidence"
        contract = payload.setdefault("training_contract", {})
        contract["temporal_scale"] = contract_text
        contract["temporal_grad_ratio"] = TEMPORAL_GRAD_RATIO
        contract["segmentation_only_warmup_windows"] = 1
        torch.save(payload, checkpoint_path)


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    output, result_output = _resolve_output_paths(args)

    base._train_sequence = _train_sequence_warmup_then_calibrate
    base.main(args)
    _patch_saved_metadata(output, result_output)


if __name__ == "__main__":
    main()
