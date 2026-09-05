"""C-V3 Stage A: Stable Memory, Correct Output.

This entrypoint keeps the validated C-V3 architecture fixed and changes only
supervision responsibilities:

- no output-level temporal loss;
- 64-D semantic Memory receives training-only RAFT-aligned cosine supervision;
- frozen E1-wrong pixels receive Repair CE;
- frozen E1-right pixels receive a Do-No-Harm margin-preservation loss.

The first TBPTT window is task-only so the zero-initialized DeltaR head can open.
The next valid window fixes

    lambda_mem = 0.5 * ||grad L_task|| / ||grad L_mem||

once, with no sweep. Motion, T, Q_mem, Memory size, Readout structure and E1 are
unchanged.
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
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v2_stage1b_role_vs_mask import (
    _role_masks,
)
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_motion_transport import (
    downsample_backward_flow,
    low_flow_grid,
)


MEMORY_GRAD_RATIO = 0.5
OUTPUT_DEFAULT = (
    "/home/lin/predify/experiments/"
    "kitti_step_task_space_prior_c_v3_memory_supervision_stage_a"
)
RESULT_DEFAULT = (
    "results/kitti_step_task_space_prior_c_v3_memory_supervision_stage_a"
)


def _masked_mean(values, mask, zero_source):
    if not bool(mask.any()):
        return zero_source.sum() * 0.0
    return values[mask].mean()


def _repair_preserve_losses(e1_logits, output_logits, target_cpu):
    """Repair E1 errors and preserve the true-class margin where E1 is right."""
    target = target_cpu.to(output_logits.device, non_blocking=True).unsqueeze(0)
    valid = target != base.IGNORE_LABEL
    safe_target = target.clamp(0, base.NUM_CLASSES - 1)

    with torch.no_grad():
        e1_prediction = e1_logits.argmax(dim=1)
        e1_right = valid & (e1_prediction == target)
        e1_wrong = valid & ~e1_right

    per_pixel_ce = F.cross_entropy(
        output_logits,
        target,
        ignore_index=base.IGNORE_LABEL,
        reduction="none",
    )
    repair_ce = _masked_mean(per_pixel_ce, e1_wrong, output_logits)

    def true_margin(logits):
        true_logit = logits.gather(1, safe_target.unsqueeze(1)).squeeze(1)
        class_index = torch.arange(
            base.NUM_CLASSES,
            device=logits.device,
        ).view(1, base.NUM_CLASSES, 1, 1)
        true_mask = class_index == safe_target.unsqueeze(1)
        strongest_other = logits.masked_fill(true_mask, float("-inf")).amax(dim=1)
        return true_logit - strongest_other

    with torch.no_grad():
        e1_margin = true_margin(e1_logits.detach())
    output_margin = true_margin(output_logits)
    preserve_penalty = F.relu(e1_margin - output_margin)
    preserve_margin = _masked_mean(
        preserve_penalty,
        e1_right,
        output_logits,
    )

    valid_count = int(valid.sum().item())
    wrong_count = int(e1_wrong.sum().item())
    right_count = int(e1_right.sum().item())
    return repair_ce, preserve_margin, {
        "valid_pixels": valid_count,
        "e1_wrong_pixels": wrong_count,
        "e1_right_pixels": right_count,
        "repair_fraction": wrong_count / max(valid_count, 1),
        "preserve_fraction": right_count / max(valid_count, 1),
    }


def _memory_cosine_loss(
    previous_memory,
    current_seed,
    previous_gt,
    current_gt,
    teacher_full,
    transportability_low,
):
    """Align historical Memory to detached current seed on strict valid semantics."""
    if previous_memory is None:
        zero = current_seed.detach().sum() * 0.0
        return zero, {
            "memory_valid_weight_mean": 0.0,
            "memory_transportable_density_mean": 0.0,
            "memory_cosine_similarity": 0.0,
        }

    low_size = tuple(current_seed.shape[-2:])
    teacher_low = downsample_backward_flow(teacher_full, low_size)
    grid_low, flow_valid_low = low_flow_grid(teacher_low)
    warped_previous_memory = F.grid_sample(
        previous_memory.float(),
        grid_low,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    warped_previous_memory = warped_previous_memory * flow_valid_low.unsqueeze(1).to(
        warped_previous_memory.dtype
    )

    with torch.no_grad():
        _, transportable_full, _ = _role_masks(
            previous_gt,
            current_gt,
            teacher_full,
        )
        transportable_density = F.interpolate(
            transportable_full.float().unsqueeze(0).unsqueeze(0),
            size=low_size,
            mode="area",
        )
        weight = (
            transportable_density
            * transportability_low.detach().float()
            * flow_valid_low.unsqueeze(1).float()
        ).detach()

    seed_target = current_seed.detach().float()
    cosine = F.cosine_similarity(
        warped_previous_memory,
        seed_target,
        dim=1,
        eps=1e-6,
    )
    per_pixel = 1.0 - cosine
    weight_2d = weight[:, 0]
    weight_sum = weight_2d.sum()
    if not bool(weight_sum > 0):
        return previous_memory.sum() * 0.0, {
            "memory_valid_weight_mean": 0.0,
            "memory_transportable_density_mean": float(
                transportable_density.mean().item()
            ),
            "memory_cosine_similarity": 0.0,
        }

    loss = (per_pixel * weight_2d).sum() / weight_sum.clamp_min(1e-6)
    weighted_cosine = (cosine.detach() * weight_2d).sum() / weight_sum.clamp_min(1e-6)
    return loss, {
        "memory_valid_weight_mean": float(weight.mean().item()),
        "memory_transportable_density_mean": float(
            transportable_density.mean().item()
        ),
        "memory_cosine_similarity": float(weighted_cosine.item()),
    }


def _train_sequence_stable_memory(
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
    scale_state,
):
    if len(samples) < 3:
        return None

    scale_state.setdefault("warmup_done", False)
    scale_state.setdefault("warmup_windows", 0)
    scale_state.setdefault("calibration_ratio", MEMORY_GRAD_RATIO)
    scale_state.setdefault("calibration_window", None)
    scale_state.setdefault("global_window_count", 0)

    frame0 = base._host_observation(model, samples[0])
    frame1 = base._host_observation(model, samples[1])
    pending_motion, motion_hidden = base._initialize_motion(
        observer, residual, frame0, frame1
    )
    previous_image, _, previous_low, previous_c1, _ = frame1
    previous_gt = base.semantic_mask_from_panoptic_png(samples[1]["mask_path"])

    transport_hidden = semantic_hidden = mask_hidden = None
    semantic_state_low = torch.zeros_like(previous_low)
    memory_state = None
    buffered_task = []
    buffered_memory = []

    totals = {
        "frames": 0,
        "windows": 0,
        "repair_ce": 0.0,
        "preserve_margin": 0.0,
        "task_loss": 0.0,
        "memory_cosine": 0.0,
        "total": 0.0,
        "repair_fraction": 0.0,
        "preserve_fraction": 0.0,
        "memory_valid_weight_mean": 0.0,
        "memory_transportable_density_mean": 0.0,
        "memory_cosine_similarity": 0.0,
        "memory_abs": 0.0,
        "warped_memory_abs": 0.0,
        "memory_reliability_mean": 0.0,
        "agreement_mean": 0.0,
        "delta_refinement_abs": 0.0,
        "semantic_state_abs": 0.0,
        "transportability_mean": 0.0,
    }

    task_parameters = [
        parameter for parameter in refiner.parameters() if parameter.requires_grad
    ]
    memory_parameters = [
        parameter for parameter in refiner.memory.parameters() if parameter.requires_grad
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

        previous_memory_for_supervision = memory_state
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

        e1_delta_low = e1["e1_delta_low"].detach()
        e1_logits = host_logits.detach() + base._upsample_prior(
            e1_delta_low,
            output_size,
        )
        output_logits = e1_logits + base._upsample_prior(
            memory_row["delta_refinement"],
            output_size,
        )

        repair_ce, preserve_margin, role_diag = _repair_preserve_losses(
            e1_logits,
            output_logits,
            current_gt,
        )
        task_loss = repair_ce + preserve_margin

        if previous_memory_for_supervision is None:
            memory_loss = memory_row["current_seed"].detach().sum() * 0.0
            memory_diag = {
                "memory_valid_weight_mean": 0.0,
                "memory_transportable_density_mean": 0.0,
                "memory_cosine_similarity": 0.0,
            }
        else:
            with torch.no_grad():
                teacher_full = raft.current_to_previous(
                    current_image,
                    previous_image,
                )
            memory_loss, memory_diag = _memory_cosine_loss(
                previous_memory_for_supervision,
                memory_row["current_seed"],
                previous_gt,
                current_gt,
                teacher_full,
                e1["transportability_low"],
            )

        if not torch.isfinite(task_loss) or not torch.isfinite(memory_loss):
            raise FloatingPointError("Non-finite C-V3 Stable-Memory loss")
        buffered_task.append(task_loss)
        buffered_memory.append(memory_loss)

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
        totals["repair_ce"] += float(repair_ce.detach().item())
        totals["preserve_margin"] += float(preserve_margin.detach().item())
        totals["task_loss"] += float(task_loss.detach().item())
        totals["memory_cosine"] += float(memory_loss.detach().item())
        totals["repair_fraction"] += role_diag["repair_fraction"]
        totals["preserve_fraction"] += role_diag["preserve_fraction"]
        for key in (
            "memory_valid_weight_mean",
            "memory_transportable_density_mean",
            "memory_cosine_similarity",
        ):
            totals[key] += memory_diag[key]
        totals["memory_abs"] += float(memory_state.abs().mean().detach().item())
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
            len(buffered_task) == tbptt_steps
            or frame_index == len(samples) - 1
        )
        if boundary:
            window_task = torch.stack(buffered_task).mean()
            window_memory = torch.stack(buffered_memory).mean()
            scale_state["global_window_count"] += 1

            is_warmup = not scale_state["warmup_done"]
            if is_warmup:
                lambda_mem = 0.0
                window_loss = window_task
            else:
                if (
                    scale_state["value"] is None
                    and window_memory.requires_grad
                    and float(window_memory.detach().item()) > 0.0
                ):
                    g_task = base._gradient_norm(window_task, task_parameters)
                    g_mem = base._gradient_norm(window_memory, memory_parameters)
                    if g_task > base.GRAD_EPS and g_mem > base.GRAD_EPS:
                        scale_state["value"] = MEMORY_GRAD_RATIO * g_task / g_mem
                        scale_state["task_grad_norm"] = g_task
                        scale_state["mem_grad_norm"] = g_mem
                        scale_state["calibration_window"] = int(
                            scale_state["global_window_count"]
                        )

                lambda_mem = scale_state["value"]
                if lambda_mem is None:
                    lambda_mem = 0.0
                window_loss = window_task + float(lambda_mem) * window_memory

            optimizer.zero_grad(set_to_none=True)
            window_loss.backward()
            optimizer.step()

            if is_warmup:
                scale_state["warmup_done"] = True
                scale_state["warmup_windows"] += 1

            totals["windows"] += 1
            totals["total"] += float(window_loss.detach().item())
            buffered_task = []
            buffered_memory = []
            memory_state = memory_state.detach()

        previous_image = current_image
        previous_gt = current_gt
        previous_low = host_low.detach()
        previous_c1 = current_c1.detach()
        pending_motion = next_motion.detach()
        motion_hidden = next_motion_hidden.detach()

    frames = max(totals["frames"], 1)
    windows = max(totals["windows"], 1)
    for key in (
        "repair_ce",
        "preserve_margin",
        "task_loss",
        "memory_cosine",
        "repair_fraction",
        "preserve_fraction",
        "memory_valid_weight_mean",
        "memory_transportable_density_mean",
        "memory_cosine_similarity",
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
    totals["lambda_mem"] = (
        float(scale_state["value"])
        if scale_state["value"] is not None
        else 0.0
    )
    totals["memory_warmup_windows"] = int(scale_state["warmup_windows"])
    totals["memory_calibration_window"] = (
        int(scale_state["calibration_window"])
        if scale_state["calibration_window"] is not None
        else -1
    )
    return totals


def _rewrite_result_metadata(result_output):
    summary_path = result_output / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        summary["experiment"] = "C-V3 Stage A Stable Memory, Correct Output"
        summary["purpose"] = (
            "Test whether moving temporal supervision from final segmentation "
            "output to 64-D semantic Memory recovers semantic accuracy while "
            "retaining temporal gains."
        )
        summary["training"].update({
            "supervision": "Repair CE + Preserve Margin + lambda_mem * Memory Cosine",
            "output_temporal_loss": False,
            "full_image_ce": False,
            "memory_grad_ratio": MEMORY_GRAD_RATIO,
            "memory_scale_policy": (
                "one task-only TBPTT warm-up window, then next valid window fixes "
                "lambda_mem = 0.5 * ||grad L_task|| / ||grad L_mem||; no sweep"
            ),
        })
        for row in summary.get("history", []):
            if "temporal_scale_calibration" in row:
                row["memory_scale_calibration"] = row.pop(
                    "temporal_scale_calibration"
                )
        best = summary.get("best", {})
        delta_e1 = best.get("delta_vs_e1_base", {})
        summary["supervision_role_check"] = {
            "both_positive_vs_e1": bool(
                delta_e1.get("mIoU", float("-inf")) > 0.0
                and delta_e1.get("mTC", float("-inf")) > 0.0
            ),
            "criterion": "best delta vs E1 Base: mIoU > 0 and mTC > 0",
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    for epoch_path in result_output.glob("epoch_*.json"):
        row = json.loads(epoch_path.read_text())
        if "temporal_scale_calibration" in row:
            row["memory_scale_calibration"] = row.pop("temporal_scale_calibration")
        epoch_path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")


def _patch_checkpoint_metadata(output):
    contract = (
        "Repair CE on frozen-E1-wrong pixels + Preserve Margin on frozen-E1-right "
        "pixels + lambda_mem*RAFT-aligned Memory cosine; no output temporal loss"
    )
    for checkpoint_path in list(output.glob("epoch_*.pt")) + [output / "best.pt"]:
        if not checkpoint_path.exists():
            continue
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        payload["experiment"] = "c_v3_memory_supervision_stage_a"
        payload.setdefault("architecture", {})[
            "supervision_role"
        ] = "Stable Memory, Correct Output"
        training_contract = payload.setdefault("training_contract", {})
        training_contract["loss"] = contract
        training_contract["output_temporal_loss"] = False
        training_contract["full_image_ce"] = False
        training_contract["memory_grad_ratio"] = MEMORY_GRAD_RATIO
        torch.save(payload, checkpoint_path)


def _resolve_output_paths(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", default=OUTPUT_DEFAULT)
    parser.add_argument("--result-output", default=RESULT_DEFAULT)
    known, _ = parser.parse_known_args(argv)
    return Path(known.output), Path(known.result_output)


def _inject_default_paths(argv):
    args = list(argv)
    if "--output" not in args:
        args.extend(["--output", OUTPUT_DEFAULT])
    if "--result-output" not in args:
        args.extend(["--result-output", RESULT_DEFAULT])
    return args


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    args = _inject_default_paths(args)
    output, result_output = _resolve_output_paths(args)

    base._train_sequence = _train_sequence_stable_memory
    base.main(args)
    _rewrite_result_metadata(result_output)
    _patch_checkpoint_metadata(output)

    summary_path = result_output / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        print(
            json.dumps(
                {
                    "stable_memory_correct_output": summary.get(
                        "supervision_role_check", {}
                    ),
                    "best": summary.get("best"),
                    "result": str(summary_path),
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
