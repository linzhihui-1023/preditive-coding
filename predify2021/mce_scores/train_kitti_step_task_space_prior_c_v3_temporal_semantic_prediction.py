"""C-V3 temporal semantic prediction and selector-oracle evaluation.

This stage does not train a Selector. It asks one architecture question first:
does the frozen best C-V3 memory contain a useful independent temporal semantic
hypothesis before the current-frame memory update?

Causal temporal candidate:
    H_pred_t = Warp(H_{t-1}, M_hat_t)
    L_temp_t = TemporalSemanticDecoder(H_pred_t)

The decoder never sees current C1, Host logits, prediction error, T, Q_mem,
semantic correction state, or updated H_t. The frozen C-V3 base still updates
its own memory from current evidence after H_pred_t has been formed.

Training uses RAFT only to define a strict semantic-stability supervision mask.
The decoder itself receives the inference-time predicted motion state, not RAFT
features or RAFT-warped memory.

Evaluation reports Host, E1 Base, frozen best C-V3 Base, Temporal Prediction and
a training-infeasible correctness Selector Oracle. The oracle chooses Temporal
only when C-V3 Base is wrong and Temporal is correct. Its mIoU/mTC therefore
measures the upper bound of the proposed correctness-based selector family,
not the performance of a learned selector.
"""

import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch.nn import functional as F

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
    _pair_mtc,
    _upsample_prior,
)
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v2_stage1b_observer import (
    _host_observation,
)
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v2_stage1b_residual import (
    OBSERVER_CHECKPOINT_DEFAULT,
    _load_frozen_observer,
    _observe_motion,
)
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v2_stage1b_role_vs_mask import (
    _initialize_motion,
    _load_frozen_residual,
    _role_masks,
)
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v2_stage1b_role_vs_mask_r2 import (
    RESIDUAL_CHECKPOINT_DEFAULT,
    _validate_bounded_motion_checkpoint,
)
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v2_stage1b_utility_from_e1_r2 import (
    BASE_CHECKPOINT_DEFAULT,
    _load_frozen_e1_base,
)
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v3_semantic_memory_stage_a import (
    _frozen_e1_step,
)
from predify2021.mce_scores.train_kitti_step_temporal_joint import FrozenRAFT
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_motion_transport import (
    warp_low_logits,
)
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_semantic_memory import (
    MotionGatedSemanticMemoryRefiner,
)
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_temporal_semantic_prediction import (
    TemporalSemanticDecoder,
)


SEED = 0
EPOCHS = 3
TBPTT_STEPS = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2

C_V3_CHECKPOINT_DEFAULT = (
    "/home/lin/predify/experiments/"
    "kitti_step_task_space_prior_c_v3_memory_supervision_stage_a/best.pt"
)
OUTPUT_DEFAULT = (
    "/home/lin/predify/experiments/"
    "kitti_step_task_space_prior_c_v3_temporal_semantic_prediction"
)
RESULT_DEFAULT = (
    "results/kitti_step_task_space_prior_c_v3_temporal_semantic_prediction"
)
CANDIDATES = (
    "host",
    "e1_base",
    "c_v3_base",
    "temporal",
    "selector_oracle",
)


def _load_frozen_cv3_refiner(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("refiner_state_dict")
    if state is None:
        raise RuntimeError(f"Missing refiner_state_dict in {path}")
    refiner = MotionGatedSemanticMemoryRefiner(
        c1_channels=256,
        num_classes=NUM_CLASSES,
    ).cuda()
    refiner.load_state_dict(state)
    refiner.eval()
    refiner.requires_grad_(False)
    return refiner, payload


def _masked_cross_entropy(logits, target_cpu, mask_gpu):
    target = target_cpu.to(logits.device, non_blocking=True).unsqueeze(0)
    per_pixel = F.cross_entropy(
        logits,
        target,
        ignore_index=IGNORE_LABEL,
        reduction="none",
    )[0]
    if not bool(mask_gpu.any()):
        return logits.sum() * 0.0
    return per_pixel[mask_gpu].mean()


def _history_valid_full(memory_row, output_size):
    return (
        F.interpolate(
            memory_row["history_valid"].float(),
            size=output_size,
            mode="nearest",
        )[0, 0]
        > 0.5
    )


def _frozen_cv3_step(
    refiner,
    current_c1,
    host_low,
    prior_low,
    e1,
    pending_motion,
    memory_state,
    output_size,
    host_logits,
):
    with torch.no_grad():
        memory_row = refiner(
            current_c1,
            host_low,
            prior_low,
            e1["transportability_low"],
            pending_motion,
            e1["semantic_state_low"],
            memory_state,
        )
        next_memory = memory_row["memory"].detach()
        e1_delta_low = e1["e1_delta_low"]
        c_v3_delta_low = e1_delta_low + memory_row["delta_refinement"]
        e1_logits = host_logits + _upsample_prior(e1_delta_low, output_size)
        c_v3_logits = host_logits + _upsample_prior(c_v3_delta_low, output_size)
    return memory_row, next_memory, e1_logits, c_v3_logits


def _train_sequence(
    model,
    observer,
    residual,
    correction,
    mask_predictor,
    refiner,
    decoder,
    raft,
    samples,
    optimizer,
    tbptt_steps,
):
    if len(samples) < 4:
        return None

    frame0 = _host_observation(model, samples[0])
    frame1 = _host_observation(model, samples[1])
    pending_motion, motion_hidden = _initialize_motion(
        observer,
        residual,
        frame0,
        frame1,
    )
    previous_image, _, previous_low, previous_c1, _ = frame1
    previous_gt = semantic_mask_from_panoptic_png(samples[1]["mask_path"])

    transport_hidden = semantic_hidden = mask_hidden = None
    semantic_state_low = torch.zeros_like(previous_low)
    memory_state = None

    buffered_losses = []
    totals = {
        "frames": 0,
        "supervised_frames": 0,
        "windows": 0,
        "temporal_ce": 0.0,
        "strict_supervision_fraction": 0.0,
        "predicted_history_valid_fraction": 0.0,
        "temporal_logit_abs": 0.0,
    }

    for frame_index in range(2, len(samples)):
        current_image, host_logits, host_low, current_c1, output_size = (
            _host_observation(model, samples[frame_index])
        )
        current_gt = semantic_mask_from_panoptic_png(
            samples[frame_index]["mask_path"]
        )

        with torch.no_grad():
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

            previous_memory_exists = memory_state is not None
            memory_row, memory_state, _, _ = _frozen_cv3_step(
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

        totals["frames"] += 1
        history_valid_full = _history_valid_full(memory_row, output_size)
        totals["predicted_history_valid_fraction"] += float(
            history_valid_full.float().mean().item()
        )

        if previous_memory_exists:
            temporal_low = decoder(memory_row["warped_memory"].detach())
            temporal_logits = _upsample_prior(temporal_low, output_size)

            with torch.no_grad():
                teacher_full = raft.current_to_previous(
                    current_image,
                    previous_image,
                )
                _, transportable, _ = _role_masks(
                    previous_gt,
                    current_gt,
                    teacher_full,
                )
                strict_mask = transportable & history_valid_full

            loss = _masked_cross_entropy(
                temporal_logits,
                current_gt,
                strict_mask,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite temporal semantic decoder loss"
                )

            buffered_losses.append(loss)
            totals["supervised_frames"] += 1
            totals["temporal_ce"] += float(loss.detach().item())
            totals["strict_supervision_fraction"] += float(
                strict_mask.float().mean().item()
            )
            totals["temporal_logit_abs"] += float(
                temporal_logits.abs().mean().detach().item()
            )

        with torch.no_grad():
            observed_motion = _observe_motion(
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

        boundary = (
            len(buffered_losses) >= tbptt_steps
            or frame_index == len(samples) - 1
        )
        if boundary and buffered_losses:
            window_loss = torch.stack(buffered_losses).mean()
            optimizer.zero_grad(set_to_none=True)
            window_loss.backward()
            optimizer.step()

            totals["windows"] += 1
            buffered_losses = []

        previous_image = current_image
        previous_gt = current_gt
        previous_low = host_low.detach()
        previous_c1 = current_c1.detach()
        pending_motion = next_motion.detach()
        motion_hidden = next_motion_hidden.detach()

    supervised = max(totals["supervised_frames"], 1)
    for key in (
        "temporal_ce",
        "strict_supervision_fraction",
        "temporal_logit_abs",
    ):
        totals[key] /= supervised
    totals["predicted_history_valid_fraction"] /= max(totals["frames"], 1)
    return totals


def _train_epoch(
    model,
    observer,
    residual,
    correction,
    mask_predictor,
    refiner,
    decoder,
    raft,
    groups,
    optimizer,
    tbptt_steps,
):
    decoder.train()
    refiner.eval()
    correction.eval()
    mask_predictor.eval()

    rows = []
    for samples in groups.values():
        row = _train_sequence(
            model,
            observer,
            residual,
            correction,
            mask_predictor,
            refiner,
            decoder,
            raft,
            samples,
            optimizer,
            tbptt_steps,
        )
        if row is not None and row["supervised_frames"] > 0:
            rows.append(row)

    if not rows:
        raise RuntimeError("No valid temporal-decoder training sequences")

    keys = rows[0].keys()
    return {
        key: sum(row[key] for row in rows) / len(rows)
        for key in keys
    }


def _new_complementarity_counts():
    return {
        "valid_pixels": 0,
        "same_prediction": 0,
        "conflict_pixels": 0,
        "base_correct_temporal_correct": 0,
        "base_correct_temporal_wrong": 0,
        "base_wrong_temporal_correct": 0,
        "base_wrong_temporal_wrong": 0,
    }


def _update_complementarity_counts(
    counts,
    base_pred,
    temporal_pred,
    gt_cpu,
):
    base = base_pred.squeeze(0).cpu()
    temporal = temporal_pred.squeeze(0).cpu()
    valid = gt_cpu != IGNORE_LABEL
    base_correct = base == gt_cpu
    temporal_correct = temporal == gt_cpu
    same = base == temporal

    counts["valid_pixels"] += int(valid.sum().item())
    counts["same_prediction"] += int((valid & same).sum().item())
    counts["conflict_pixels"] += int((valid & ~same).sum().item())
    counts["base_correct_temporal_correct"] += int(
        (valid & base_correct & temporal_correct).sum().item()
    )
    counts["base_correct_temporal_wrong"] += int(
        (valid & base_correct & ~temporal_correct).sum().item()
    )
    counts["base_wrong_temporal_correct"] += int(
        (valid & ~base_correct & temporal_correct).sum().item()
    )
    counts["base_wrong_temporal_wrong"] += int(
        (valid & ~base_correct & ~temporal_correct).sum().item()
    )


def _complementarity_rates(counts):
    valid = max(counts["valid_pixels"], 1)
    conflict = max(counts["conflict_pixels"], 1)
    base_wrong = (
        counts["base_wrong_temporal_correct"]
        + counts["base_wrong_temporal_wrong"]
    )
    base_correct = (
        counts["base_correct_temporal_correct"]
        + counts["base_correct_temporal_wrong"]
    )
    return {
        **counts,
        "same_prediction_fraction": counts["same_prediction"] / valid,
        "conflict_fraction": counts["conflict_pixels"] / valid,
        "base_wrong_temporal_correct_fraction_of_valid": (
            counts["base_wrong_temporal_correct"] / valid
        ),
        "base_correct_temporal_wrong_fraction_of_valid": (
            counts["base_correct_temporal_wrong"] / valid
        ),
        "recoverable_rate_given_base_wrong": (
            counts["base_wrong_temporal_correct"] / max(base_wrong, 1)
        ),
        "damage_rate_given_base_correct": (
            counts["base_correct_temporal_wrong"] / max(base_correct, 1)
        ),
        "temporal_win_rate_on_conflicts": (
            counts["base_wrong_temporal_correct"] / conflict
        ),
        "base_win_rate_on_conflicts": (
            counts["base_correct_temporal_wrong"] / conflict
        ),
    }


@torch.inference_mode()
def _evaluate(
    model,
    observer,
    residual,
    correction,
    mask_predictor,
    refiner,
    decoder,
    groups,
    raft,
):
    decoder.eval()
    refiner.eval()
    correction.eval()
    mask_predictor.eval()

    confusion = {
        name: torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
        for name in CANDIDATES
    }
    mtc_sum = {name: 0.0 for name in CANDIDATES}
    mtc_count = {name: 0 for name in CANDIDATES}
    vc_sum = {
        name: {8: 0.0, 16: 0.0}
        for name in CANDIDATES
    }
    vc_count = {
        name: {8: 0, 16: 0}
        for name in CANDIDATES
    }
    complementarity = _new_complementarity_counts()
    diagnostics = {
        "frames_with_temporal_prediction": 0,
        "predicted_history_valid_fraction": 0.0,
        "temporal_logit_abs": 0.0,
        "temporal_fallback_fraction": 0.0,
    }

    for sequence in FULL9:
        previous = None
        pending_motion = motion_hidden = None
        transport_hidden = semantic_hidden = mask_hidden = None
        semantic_state_low = None
        memory_state = None
        previous_predictions = {}
        seq_vc = {
            name: VideoConsistency()
            for name in CANDIDATES
        }

        for sample in groups[sequence]:
            image, host_logits, host_low, current_c1, output_size = (
                _host_observation(model, sample)
            )
            host_pred = host_logits.argmax(1)
            gt_cpu = semantic_mask_from_panoptic_png(sample["mask_path"])
            previous_image_for_mtc = previous[0] if previous is not None else None

            if previous is None:
                e1_pred = c_v3_pred = temporal_pred = selector_oracle_pred = host_pred
                semantic_state_low = torch.zeros_like(host_low)
                previous = (
                    image,
                    host_low.detach(),
                    current_c1.detach(),
                )
            elif pending_motion is None:
                previous_image, previous_low, previous_c1 = previous
                observed = _observe_motion(
                    observer,
                    previous_low,
                    previous_c1,
                    host_low,
                    current_c1,
                )
                error = F.softmax(host_low, dim=1) - F.softmax(
                    previous_low,
                    dim=1,
                )
                pending_motion, _, motion_hidden = residual.predict_next(
                    observed,
                    error,
                    None,
                )
                e1_pred = c_v3_pred = temporal_pred = selector_oracle_pred = host_pred
                previous = (
                    image,
                    host_low.detach(),
                    current_c1.detach(),
                )
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

                previous_memory_exists = memory_state is not None
                memory_row, memory_state, e1_logits, c_v3_logits = (
                    _frozen_cv3_step(
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
                )
                e1_pred = e1_logits.argmax(1)
                c_v3_pred = c_v3_logits.argmax(1)

                if previous_memory_exists:
                    temporal_low = decoder(memory_row["warped_memory"])
                    temporal_logits = _upsample_prior(
                        temporal_low,
                        output_size,
                    )
                    raw_temporal_pred = temporal_logits.argmax(1)
                    history_valid_full = _history_valid_full(
                        memory_row,
                        output_size,
                    )
                    temporal_pred = c_v3_pred.clone()
                    temporal_pred[0][history_valid_full] = raw_temporal_pred[0][history_valid_full]

                    selector_oracle_pred = c_v3_pred.clone()
                    gt_gpu = gt_cpu.to(
                        c_v3_pred.device,
                        non_blocking=True,
                    )
                    valid = gt_gpu != IGNORE_LABEL
                    base_wrong = c_v3_pred[0] != gt_gpu
                    temporal_correct = temporal_pred[0] == gt_gpu
                    choose_temporal = valid & base_wrong & temporal_correct
                    selector_oracle_pred[0][choose_temporal] = temporal_pred[0][choose_temporal]

                    _update_complementarity_counts(
                        complementarity,
                        c_v3_pred,
                        temporal_pred,
                        gt_cpu,
                    )
                    diagnostics["frames_with_temporal_prediction"] += 1
                    diagnostics["predicted_history_valid_fraction"] += float(
                        history_valid_full.float().mean().item()
                    )
                    diagnostics["temporal_logit_abs"] += float(
                        temporal_logits.abs().mean().item()
                    )
                    diagnostics["temporal_fallback_fraction"] += float(
                        (~history_valid_full).float().mean().item()
                    )
                else:
                    temporal_pred = c_v3_pred
                    selector_oracle_pred = c_v3_pred

                observed = _observe_motion(
                    observer,
                    previous_low,
                    previous_c1,
                    host_low,
                    current_c1,
                )
                error = F.softmax(host_low, dim=1) - F.softmax(
                    prior_low,
                    dim=1,
                )
                pending_motion, _, motion_hidden = residual.predict_next(
                    observed,
                    error,
                    motion_hidden,
                )
                previous = (
                    image,
                    host_low.detach(),
                    current_c1.detach(),
                )

            predictions = {
                "host": host_pred,
                "e1_base": e1_pred,
                "c_v3_base": c_v3_pred,
                "temporal": temporal_pred,
                "selector_oracle": selector_oracle_pred,
            }

            for name, prediction in predictions.items():
                pred_cpu = prediction.squeeze(0).cpu()
                update_confusion_matrix(
                    confusion[name],
                    pred_cpu,
                    gt_cpu,
                )
                seq_vc[name].update(gt_cpu, pred_cpu)

            if previous_image_for_mtc is not None:
                teacher_full = raft.current_to_previous(
                    image,
                    previous_image_for_mtc,
                )
                for name, prediction in predictions.items():
                    score = _pair_mtc(
                        previous_predictions[name],
                        prediction,
                        teacher_full,
                    )
                    if math.isfinite(score):
                        mtc_sum[name] += score
                        mtc_count[name] += 1

            previous_predictions = {
                name: prediction.detach()
                for name, prediction in predictions.items()
            }

        for name in CANDIDATES:
            stats = seq_vc[name].stats()
            for length in (8, 16):
                vc_sum[name][length] += stats[length]["sum"]
                vc_count[name][length] += stats[length]["count"]

    metrics = {
        name: {
            "mIoU": float(
                torch.nanmean(compute_iou(confusion[name])).item()
            ),
            "mTC": (
                mtc_sum[name] / mtc_count[name]
                if mtc_count[name]
                else float("nan")
            ),
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
        for name in CANDIDATES
    }

    frames = max(
        diagnostics["frames_with_temporal_prediction"],
        1,
    )
    for key in (
        "predicted_history_valid_fraction",
        "temporal_logit_abs",
        "temporal_fallback_fraction",
    ):
        diagnostics[key] /= frames

    return (
        metrics,
        diagnostics,
        _complementarity_rates(complementarity),
    )


def _selection_key(metrics):
    oracle = metrics["selector_oracle"]
    return (
        oracle["mTC"],
        oracle["mIoU"],
        metrics["temporal"]["mIoU"],
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="/home/lin/predify/kitti_step",
    )
    parser.add_argument(
        "--fast-b-checkpoint",
        default=FAST_B_CHECKPOINT_DEFAULT,
    )
    parser.add_argument(
        "--observer-checkpoint",
        default=OBSERVER_CHECKPOINT_DEFAULT,
    )
    parser.add_argument(
        "--residual-checkpoint",
        default=RESIDUAL_CHECKPOINT_DEFAULT,
    )
    parser.add_argument(
        "--base-checkpoint",
        default=BASE_CHECKPOINT_DEFAULT,
    )
    parser.add_argument(
        "--c-v3-checkpoint",
        default=C_V3_CHECKPOINT_DEFAULT,
    )
    parser.add_argument("--output", default=OUTPUT_DEFAULT)
    parser.add_argument(
        "--result-output",
        default=RESULT_DEFAULT,
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument(
        "--tbptt-steps",
        type=int,
        default=TBPTT_STEPS,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=WEIGHT_DECAY,
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.epochs <= 0 or args.tbptt_steps <= 0:
        raise ValueError("epochs and tbptt-steps must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    _validate_bounded_motion_checkpoint(
        args.residual_checkpoint
    )
    model = load_fast_b_model(
        args.fast_b_checkpoint
    ).cuda().eval()
    model.requires_grad_(False)

    observer, _ = _load_frozen_observer(
        args.observer_checkpoint
    )
    residual, residual_payload = _load_frozen_residual(
        args.residual_checkpoint,
        observer,
    )
    correction, mask_predictor, base_payload = (
        _load_frozen_e1_base(
            args.base_checkpoint,
            observer,
        )
    )
    refiner, cv3_payload = _load_frozen_cv3_refiner(
        args.c_v3_checkpoint
    )
    decoder = TemporalSemanticDecoder(
        memory_channels=64,
        num_classes=NUM_CLASSES,
    ).cuda()

    if any(
        parameter.requires_grad
        for parameter in refiner.parameters()
    ):
        raise RuntimeError(
            "C-V3 memory/refiner must remain frozen"
        )
    if any(
        parameter.requires_grad
        for parameter in correction.parameters()
    ):
        raise RuntimeError(
            "E1 correction must remain frozen"
        )
    if any(
        parameter.requires_grad
        for parameter in mask_predictor.parameters()
    ):
        raise RuntimeError(
            "E1 transportability mask must remain frozen"
        )

    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    raft = FrozenRAFT()

    train = KITTISTEPSegmentationDataset.from_kitti_step_root(
        Path(args.root),
        "train",
    )
    val = KITTISTEPSegmentationDataset.from_kitti_step_root(
        Path(args.root),
        "val",
    )
    train_groups = sequence_groups(train)
    all_val_groups = sequence_groups(val)
    missing = [
        sequence
        for sequence in FULL9
        if sequence not in all_val_groups
    ]
    if missing:
        raise RuntimeError(
            f"Missing Full9 validation sequences: {missing}"
        )
    val_groups = {
        sequence: all_val_groups[sequence]
        for sequence in FULL9
    }

    output = Path(args.output)
    result_output = Path(args.result_output)
    output.mkdir(parents=True, exist_ok=True)
    result_output.mkdir(parents=True, exist_ok=True)

    history = []
    best = None

    for epoch in range(1, args.epochs + 1):
        train_stats = _train_epoch(
            model,
            observer,
            residual,
            correction,
            mask_predictor,
            refiner,
            decoder,
            raft,
            train_groups,
            optimizer,
            args.tbptt_steps,
        )
        metrics, diagnostics, complementarity = _evaluate(
            model,
            observer,
            residual,
            correction,
            mask_predictor,
            refiner,
            decoder,
            val_groups,
            raft,
        )

        delta_vs_host = {
            name: {
                metric: metrics[name][metric]
                - metrics["host"][metric]
                for metric in (
                    "mIoU",
                    "mTC",
                    "mVC8",
                    "mVC16",
                )
            }
            for name in (
                "e1_base",
                "c_v3_base",
                "temporal",
                "selector_oracle",
            )
        }
        delta_vs_cv3 = {
            name: {
                metric: metrics[name][metric]
                - metrics["c_v3_base"][metric]
                for metric in (
                    "mIoU",
                    "mTC",
                    "mVC8",
                    "mVC16",
                )
            }
            for name in (
                "temporal",
                "selector_oracle",
            )
        }

        row = {
            "epoch": epoch,
            "train": train_stats,
            "metrics": metrics,
            "diagnostics": diagnostics,
            "complementarity": complementarity,
            "delta_vs_host": delta_vs_host,
            "delta_vs_c_v3_base": delta_vs_cv3,
        }
        history.append(row)

        payload = {
            "experiment": (
                "c_v3_temporal_semantic_prediction"
            ),
            "epoch": epoch,
            "architecture": {
                "frozen_base": (
                    "best C-V3 Stable Memory, Correct Output"
                ),
                "predictive_state": (
                    "raw motion-warped H_{t-1}; "
                    "no current semantic gating"
                ),
                "temporal_decoder": "64->64->32->19",
                "current_inputs_to_decoder": [],
                "selector_trained": False,
            },
            "training_contract": {
                "only_trainable": "TemporalSemanticDecoder",
                "loss": (
                    "CE on RAFT-valid, GT-semantic-stable, "
                    "predicted-history-valid pixels"
                ),
                "raft_in_decoder": False,
                "raft_in_inference": False,
                "tbptt_steps": args.tbptt_steps,
            },
            "frozen": [
                "Host",
                "Motion Observer",
                "bounded r=2 Task-Alignment Residual",
                "E1 correction",
                "E1 Transportability Mask",
                "C-V3 Memory",
                "C-V3 Adaptive Readout",
            ],
            "base_checkpoint": args.base_checkpoint,
            "base_checkpoint_epoch": int(
                base_payload.get("epoch", -1)
            ),
            "c_v3_checkpoint": args.c_v3_checkpoint,
            "c_v3_checkpoint_epoch": int(
                cv3_payload.get("epoch", -1)
            ),
            "observer_checkpoint": args.observer_checkpoint,
            "residual_checkpoint": args.residual_checkpoint,
            "residual_checkpoint_epoch": residual_payload.get(
                "epoch"
            ),
            "decoder_state_dict": decoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train": train_stats,
            "metrics": metrics,
            "diagnostics": diagnostics,
            "complementarity": complementarity,
        }
        torch.save(
            payload,
            output / f"epoch_{epoch:03d}.pt",
        )
        (
            result_output / f"epoch_{epoch:03d}.json"
        ).write_text(
            json.dumps(
                row,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

        key = _selection_key(metrics)
        if best is None or key > best["selection_key"]:
            best = {
                "epoch": epoch,
                "selection_key": key,
                "metrics": metrics,
                "delta_vs_host": delta_vs_host,
                "delta_vs_c_v3_base": delta_vs_cv3,
                "complementarity": complementarity,
            }
            torch.save(payload, output / "best.pt")

        print(json.dumps(row, sort_keys=True), flush=True)

    target_reachable = bool(
        best["delta_vs_host"]["selector_oracle"]["mIoU"]
        >= 0.010
        and best["delta_vs_host"]["selector_oracle"]["mTC"]
        >= 0.040
    )

    summary = {
        "experiment": (
            "C-V3 Temporal Semantic Prediction"
        ),
        "purpose": (
            "Decide whether the frozen C-V3 historical memory "
            "contains an independent semantic hypothesis with "
            "enough Base/Temporal complementarity to justify "
            "building a Stateful Selector."
        ),
        "c_v3_checkpoint": args.c_v3_checkpoint,
        "c_v3_checkpoint_epoch": int(
            cv3_payload.get("epoch", -1)
        ),
        "training": {
            "only_trainable": "TemporalSemanticDecoder",
            "epochs": args.epochs,
            "tbptt_steps": args.tbptt_steps,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "decoder_current_frame_inputs": False,
            "memory_frozen": True,
            "motion_frozen": True,
            "T_frozen": True,
            "Q_mem_not_used_by_decoder": True,
        },
        "history": history,
        "best": best,
        "selector_oracle_gate": {
            "target_reachable": target_reachable,
            "criterion": (
                "best correctness Selector Oracle Full9 "
                "delta_vs_host mIoU >= +1.0 pp AND "
                "mTC >= +4.0 pp"
            ),
            "if_false": (
                "Do not build or tune a learned Selector; "
                "the candidate pair does not expose enough "
                "upper-bound headroom for the stated target."
            ),
            "if_true": (
                "Proceed to Stateful Conflict Selector using "
                "prediction error and causal reliability evidence."
            ),
        },
    }
    (result_output / "summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print(
        json.dumps(
            {
                "best": best,
                "selector_target_reachable": target_reachable,
                "checkpoint": str(output / "best.pt"),
                "result": str(
                    result_output / "summary.json"
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
