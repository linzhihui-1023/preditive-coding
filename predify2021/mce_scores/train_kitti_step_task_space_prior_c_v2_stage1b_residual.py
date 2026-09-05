"""C-V2 Stage 1B-2: residual future-motion prediction on KITTI-STEP.

The Stage 1B-1 Motion Observer is loaded and frozen. Only the recurrent residual
motion predictor is trainable. At time t it consumes the already observed
M_t = F_{t->t-1} and task-space prediction error e_t, then predicts
M_hat_{t+1} = M_t + Delta M_{t+1}. RAFT is supervision/metric support only and
never enters the inference graph.
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
from predify2021.mce_scores.evaluate_kitti_step_clean_fast_iss_vss import VideoConsistency
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import sequence_groups
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
    _pair_mtc,
    _rates,
    _upsample_prior,
)
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v2_stage1b_observer import (
    C1_CHANNELS,
    CORRELATION_RADIUS,
    HIDDEN_CHANNELS as OBSERVER_HIDDEN_CHANNELS,
    MAX_DISPLACEMENT_LOW,
    PROJECTED_CHANNELS,
    _host_observation,
)
from predify2021.mce_scores.train_kitti_step_temporal_joint import FrozenRAFT
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_motion_transport import (
    LocalCorrelationMotionObserver,
    MotionResidualPredictor,
    downsample_backward_flow,
    normalized_flow_distillation_loss,
    normalized_motion_residual_l2,
    warp_low_logits,
)

SEED = 0
EPOCHS = 3
TBPTT_STEPS = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
LAMBDA_SEMANTIC = 1.0
LAMBDA_FLOW = 10.0
LAMBDA_DELTA = 1e-2
RESIDUAL_HIDDEN_CHANNELS = 64
MAX_RESIDUAL_LOW = 16.0
OBSERVER_CHECKPOINT_DEFAULT = (
    "/home/lin/predify/experiments/"
    "kitti_step_task_space_prior_c_v2_stage1b_observer/best.pt"
)
OUTPUT_DEFAULT = "/home/lin/predify/experiments/kitti_step_task_space_prior_c_v2_stage1b_residual"
RESULT_DEFAULT = "results/kitti_step_task_space_prior_c_v2_stage1b_residual"


def _load_frozen_observer(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("experiment") != "c_v2_stage1b1_motion_observer":
        raise RuntimeError("Observer checkpoint is not a Stage 1B-1 checkpoint")
    row = payload.get("row", {})
    checks = row.get("stage1b1_checks", {})
    if not checks.get("observer_minimum_go", False):
        raise RuntimeError("Stage 1B-1 Observer did not pass observer_minimum_go")
    architecture = payload.get("architecture", {})
    observer = LocalCorrelationMotionObserver(
        c1_channels=int(architecture.get("c1_channels", C1_CHANNELS)),
        num_classes=NUM_CLASSES,
        projected_channels=int(architecture.get("projected_channels", PROJECTED_CHANNELS)),
        hidden_channels=int(architecture.get("hidden_channels", OBSERVER_HIDDEN_CHANNELS)),
        correlation_radius=int(architecture.get("correlation_radius", CORRELATION_RADIUS)),
        max_displacement_low=float(architecture.get("max_displacement_low", MAX_DISPLACEMENT_LOW)),
    ).cuda()
    observer.load_state_dict(payload["observer_state_dict"], strict=True)
    observer.requires_grad_(False).eval()
    return observer, payload


def _observe_motion(observer, previous_low, previous_c1, current_low, current_c1):
    with torch.no_grad():
        return observer(
            previous_c1,
            current_c1,
            F.softmax(previous_low, dim=1),
            F.softmax(current_low, dim=1),
        )


def _zero_step_check(residual, observed_motion, probability_error):
    with torch.no_grad():
        predicted, delta, _ = residual.predict_next(observed_motion, probability_error, None)
    row = {
        "delta_motion_max_abs": float(delta.abs().max().item()),
        "predicted_equals_observed_max_abs": float((predicted - observed_motion).abs().max().item()),
    }
    if row["delta_motion_max_abs"] != 0.0 or row["predicted_equals_observed_max_abs"] != 0.0:
        raise RuntimeError(f"Stage 1B-2 zero-step check failed: {row}")
    return row


def _train_sequence(model, observer, residual, raft, samples, optimizer, args):
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

    frame0 = _host_observation(model, samples[0])
    frame1 = _host_observation(model, samples[1])
    _, _, low0, c10, _ = frame0
    _, _, low1, c11, _ = frame1
    observed_motion_1 = _observe_motion(observer, low0, c10, low1, c11)
    persistence_1 = low0.detach()
    error_1 = F.softmax(low1.detach(), dim=1) - F.softmax(persistence_1, dim=1)
    pending_motion, pending_delta, hidden = residual.predict_next(
        observed_motion_1.detach(), error_1.detach(), None
    )

    previous_image, _, previous_low, previous_c1, _ = frame1
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
        current_image, _, current_low, current_c1, output_size = _host_observation(
            model, samples[frame_index]
        )
        warped_low, _ = warp_low_logits(previous_low.detach(), pending_motion)
        target = semantic_mask_from_panoptic_png(samples[frame_index]["mask_path"]).cuda(non_blocking=True)
        semantic_loss = F.cross_entropy(
            _upsample_prior(warped_low, output_size),
            target.unsqueeze(0),
            ignore_index=IGNORE_LABEL,
        )
        with torch.no_grad():
            teacher_full = raft.current_to_previous(current_image, previous_image)
            teacher_low = downsample_backward_flow(
                teacher_full, tuple(current_low.shape[-2:])
            )
        flow_loss = normalized_flow_distillation_loss(
            pending_motion, teacher_low, observer.max_displacement_low
        )
        delta_loss = normalized_motion_residual_l2(
            pending_delta, residual.max_residual_displacement_low
        )
        if not all(torch.isfinite(x).item() for x in (semantic_loss, flow_loss, delta_loss)):
            raise FloatingPointError("Non-finite Stage 1B-2 loss")
        semantic_losses.append(semantic_loss)
        flow_losses.append(flow_loss)
        delta_losses.append(delta_loss)
        sums["frames"] += 1

        observed_motion_t = _observe_motion(
            observer, previous_low, previous_c1, current_low, current_c1
        )
        error_t = F.softmax(current_low.detach(), dim=1) - F.softmax(warped_low, dim=1)
        previous_hidden = hidden
        next_motion, next_delta, next_hidden = residual.predict_next(
            observed_motion_t.detach(), error_t, previous_hidden
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
                raise FloatingPointError("Non-finite Stage 1B-2 total loss")
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            sums["windows"] += 1
            sums["semantic_loss"] += float(sem.detach().item())
            sums["flow_loss"] += float(flo.detach().item())
            sums["delta_loss"] += float(delt.detach().item())
            sums["total_loss"] += float(total.detach().item())

            pending_motion, pending_delta, hidden = residual.predict_next(
                observed_motion_t.detach(),
                error_t.detach(),
                previous_hidden.detach() if previous_hidden is not None else None,
            )
            semantic_losses, flow_losses, delta_losses = [], [], []
        else:
            pending_motion, pending_delta, hidden = next_motion, next_delta, next_hidden

        previous_image = current_image
        previous_low = current_low.detach()
        previous_c1 = current_c1.detach()

    windows = max(sums["windows"], 1)
    for key in ("semantic_loss", "flow_loss", "delta_loss", "total_loss"):
        sums[key] /= windows
    return sums


def _train_epoch(model, observer, residual, raft, groups, optimizer, args):
    residual.train()
    observer.eval()
    total = {
        "frames": 0,
        "windows": 0,
        "semantic_loss": 0.0,
        "flow_loss": 0.0,
        "delta_loss": 0.0,
        "total_loss": 0.0,
    }
    for samples in groups.values():
        row = _train_sequence(model, observer, residual, raft, samples, optimizer, args)
        total["frames"] += row["frames"]
        for key in ("semantic_loss", "flow_loss", "delta_loss", "total_loss"):
            total[key] += row[key] * row["windows"]
        total["windows"] += row["windows"]
    windows = max(total["windows"], 1)
    for key in ("semantic_loss", "flow_loss", "delta_loss", "total_loss"):
        total[key] /= windows
    return total


@torch.inference_mode()
def _evaluate(model, observer, residual, groups, raft):
    names = (
        "host",
        "semantic_persistence",
        "observer_lagged",
        "observer_residual",
        "lagged_repair_only_oracle",
        "residual_repair_only_oracle",
    )
    confusion = {n: torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64) for n in names}
    mtc_sum = {n: 0.0 for n in names}
    mtc_count = {n: 0 for n in names}
    vc_sum = {n: {8: 0.0, 16: 0.0} for n in names}
    vc_count = {n: {8: 0, 16: 0} for n in names}
    lagged_counts = _new_counts()
    residual_counts = _new_counts()
    diag = {
        "predicted_pairs": 0,
        "delta_values": 0,
        "delta_abs": 0.0,
        "observed_abs": 0.0,
        "predicted_abs": 0.0,
        "teacher_abs": 0.0,
        "predicted_teacher_l1": 0.0,
    }
    residual.eval()
    observer.eval()

    for sequence in FULL9:
        seq_conf = {n: torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64) for n in names}
        seq_mtc_sum = {n: 0.0 for n in names}
        seq_mtc_count = {n: 0 for n in names}
        seq_vc = {n: VideoConsistency() for n in names}
        previous = None
        previous_observed_motion = None
        pending_motion = pending_delta = hidden = None
        previous_predictions = {}

        for index, sample in enumerate(groups[sequence]):
            image, host_logits, host_low, c1, output_size = _host_observation(model, sample)
            host_pred = host_logits.argmax(1)
            gt_cpu = semantic_mask_from_panoptic_png(sample["mask_path"])
            gt = gt_cpu.cuda(non_blocking=True)
            teacher_full = None

            if previous is None:
                persistence = lagged_pred = residual_pred = host_pred
                lagged_oracle = residual_oracle = host_pred
                residual_warped_low = host_low
            else:
                previous_image, previous_low, previous_c1 = previous
                persistence = _upsample_prior(previous_low, output_size).argmax(1)
                teacher_full = raft.current_to_previous(image, previous_image)

                if previous_observed_motion is None:
                    lagged_warped_low = previous_low
                else:
                    lagged_warped_low, _ = warp_low_logits(
                        previous_low, previous_observed_motion
                    )
                lagged_pred = _upsample_prior(lagged_warped_low, output_size).argmax(1)

                if pending_motion is None:
                    residual_warped_low = previous_low
                else:
                    residual_warped_low, _ = warp_low_logits(previous_low, pending_motion)
                residual_pred = _upsample_prior(residual_warped_low, output_size).argmax(1)

                valid = gt != IGNORE_LABEL
                host_correct = host_pred.squeeze(0) == gt
                lagged_correct = lagged_pred.squeeze(0) == gt
                residual_correct = residual_pred.squeeze(0) == gt
                lagged_recoverable = _add_counts(lagged_counts, host_correct, lagged_correct, valid)
                residual_recoverable = _add_counts(residual_counts, host_correct, residual_correct, valid)
                lagged_oracle = host_pred.clone()
                residual_oracle = host_pred.clone()
                lagged_oracle[lagged_recoverable.unsqueeze(0)] = lagged_pred[lagged_recoverable.unsqueeze(0)]
                residual_oracle[residual_recoverable.unsqueeze(0)] = residual_pred[residual_recoverable.unsqueeze(0)]

                current_observed_motion = _observe_motion(
                    observer, previous_low, previous_c1, host_low, c1
                )
                error_t = F.softmax(host_low, dim=1) - F.softmax(residual_warped_low, dim=1)
                next_motion, next_delta, next_hidden = residual.predict_next(
                    current_observed_motion, error_t, hidden
                )

                if pending_motion is not None:
                    teacher_low = downsample_backward_flow(
                        teacher_full, tuple(host_low.shape[-2:])
                    )
                    values = pending_motion.numel()
                    diag["predicted_pairs"] += 1
                    diag["delta_values"] += values
                    diag["delta_abs"] += float(pending_delta.abs().sum().item())
                    diag["observed_abs"] += float(previous_observed_motion.abs().sum().item())
                    diag["predicted_abs"] += float(pending_motion.abs().sum().item())
                    diag["teacher_abs"] += float(teacher_low.abs().sum().item())
                    diag["predicted_teacher_l1"] += float((pending_motion - teacher_low).abs().sum().item())

                previous_observed_motion = current_observed_motion.detach()
                pending_motion = next_motion.detach()
                pending_delta = next_delta.detach()
                hidden = next_hidden.detach()

            predictions = {
                "host": host_pred,
                "semantic_persistence": persistence,
                "observer_lagged": lagged_pred,
                "observer_residual": residual_pred,
                "lagged_repair_only_oracle": lagged_oracle,
                "residual_repair_only_oracle": residual_oracle,
            }
            for name, prediction in predictions.items():
                pc = prediction.squeeze(0).cpu()
                update_confusion_matrix(confusion[name], pc, gt_cpu)
                update_confusion_matrix(seq_conf[name], pc, gt_cpu)
                seq_vc[name].update(gt_cpu, pc)

            if previous is not None:
                for name, prediction in predictions.items():
                    score = _pair_mtc(previous_predictions[name], prediction, teacher_full)
                    if math.isfinite(score):
                        mtc_sum[name] += score
                        mtc_count[name] += 1
                        seq_mtc_sum[name] += score
                        seq_mtc_count[name] += 1

            previous = (image, host_low.detach(), c1.detach())
            previous_predictions = {n: p.detach() for n, p in predictions.items()}

        for name in names:
            stats = seq_vc[name].stats()
            for length in (8, 16):
                vc_sum[name][length] += stats[length]["sum"]
                vc_count[name][length] += stats[length]["count"]

    metrics = {
        name: {
            "mIoU": float(torch.nanmean(compute_iou(confusion[name])).item()),
            "mTC": mtc_sum[name] / mtc_count[name] if mtc_count[name] else float("nan"),
            "mVC8": vc_sum[name][8] / vc_count[name][8] if vc_count[name][8] else float("nan"),
            "mVC16": vc_sum[name][16] / vc_count[name][16] if vc_count[name][16] else float("nan"),
            "valid_frame_pairs": mtc_count[name],
        }
        for name in names
    }
    delta_full_vs_lagged = {
        key: metrics["observer_residual"][key] - metrics["observer_lagged"][key]
        for key in ("mIoU", "mTC", "mVC8", "mVC16")
    }
    delta_vs_persistence = {
        name: {
            key: metrics[name][key] - metrics["semantic_persistence"][key]
            for key in ("mIoU", "mTC", "mVC8", "mVC16")
        }
        for name in ("observer_lagged", "observer_residual")
    }
    delta_vs_host = {
        name: {
            key: metrics[name][key] - metrics["host"][key]
            for key in ("mIoU", "mTC", "mVC8", "mVC16")
        }
        for name in names if name != "host"
    }
    values = max(diag["delta_values"], 1)
    observed_abs_mean = diag["observed_abs"] / values
    delta_abs_mean = diag["delta_abs"] / values
    return {
        "metrics": metrics,
        "delta_full_vs_deltaM_zero_lagged": delta_full_vs_lagged,
        "delta_vs_persistence": delta_vs_persistence,
        "delta_vs_host": delta_vs_host,
        "lagged_complementarity_causal_frames_only": _rates(lagged_counts),
        "residual_complementarity_causal_frames_only": _rates(residual_counts),
        "residual_diagnostics": {
            "predicted_pairs": diag["predicted_pairs"],
            "delta_abs_mean_low_pixels": delta_abs_mean,
            "observed_motion_abs_mean_low_pixels": observed_abs_mean,
            "predicted_motion_abs_mean_low_pixels": diag["predicted_abs"] / values,
            "teacher_motion_abs_mean_low_pixels": diag["teacher_abs"] / values,
            "predicted_teacher_l1_mean_low_pixels": diag["predicted_teacher_l1"] / values,
            "delta_to_observed_amplitude_ratio": delta_abs_mean / max(observed_abs_mean, 1e-12),
        },
    }


def _candidate_check(metrics, candidate, oracle_name):
    persistence = metrics["metrics"]["semantic_persistence"]
    row = metrics["metrics"][candidate]
    oracle_delta_mtc = metrics["metrics"][oracle_name]["mTC"] - metrics["metrics"]["host"]["mTC"]
    gain = row["mIoU"] - persistence["mIoU"]
    return {
        "delta_mIoU_vs_persistence_ge_2pp": bool(gain >= 0.02),
        "mTC_gt_persistence": bool(row["mTC"] > persistence["mTC"]),
        "repair_oracle_delta_mTC_ge_0p4pp": bool(oracle_delta_mtc >= 0.004),
        "candidate_minimum_go": bool(
            gain >= 0.02
            and row["mTC"] > persistence["mTC"]
            and oracle_delta_mtc >= 0.004
        ),
    }


def _checks(metrics):
    lagged = _candidate_check(metrics, "observer_lagged", "lagged_repair_only_oracle")
    residual = _candidate_check(metrics, "observer_residual", "residual_repair_only_oracle")
    delta = metrics["delta_full_vs_deltaM_zero_lagged"]
    residual_pareto = (
        delta["mIoU"] >= 0.0
        and delta["mTC"] >= 0.0
        and (delta["mIoU"] > 0.0 or delta["mTC"] > 0.0)
    )
    lagged_pareto = (
        delta["mIoU"] <= 0.0
        and delta["mTC"] <= 0.0
        and (delta["mIoU"] < 0.0 or delta["mTC"] < 0.0)
    )
    if residual_pareto:
        preference = "observer_residual"
    elif lagged_pareto:
        preference = "observer_lagged"
    else:
        preference = "tradeoff_requires_metric_review"
    return {
        "observer_lagged": lagged,
        "observer_residual": residual,
        "stage1b_candidate_exists": bool(lagged["candidate_minimum_go"] or residual["candidate_minimum_go"]),
        "residual_increment_mIoU": float(delta["mIoU"]),
        "residual_increment_mTC": float(delta["mTC"]),
        "residual_pareto_improves_lagged": bool(residual_pareto),
        "lagged_pareto_dominates_residual": bool(lagged_pareto),
        "motion_candidate_preference": preference,
        "mvc16_is_gate": False,
    }


def _selection_key(row):
    checks = row["stage1b2_checks"]
    residual = row["metrics"]["metrics"]["observer_residual"]
    return (
        int(checks["stage1b_candidate_exists"]),
        int(checks["residual_pareto_improves_lagged"]),
        residual["mIoU"],
        residual["mTC"],
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/home/lin/predify/kitti_step")
    parser.add_argument("--fast-b-checkpoint", default=FAST_B_CHECKPOINT_DEFAULT)
    parser.add_argument("--observer-checkpoint", default=OBSERVER_CHECKPOINT_DEFAULT)
    parser.add_argument("--output", default=OUTPUT_DEFAULT)
    parser.add_argument("--result-output", default=RESULT_DEFAULT)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--tbptt-steps", type=int, default=TBPTT_STEPS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--lambda-semantic", type=float, default=LAMBDA_SEMANTIC)
    parser.add_argument("--lambda-flow", type=float, default=LAMBDA_FLOW)
    parser.add_argument("--lambda-delta", type=float, default=LAMBDA_DELTA)
    parser.add_argument("--max-residual-low", type=float, default=MAX_RESIDUAL_LOW)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.epochs <= 0 or args.tbptt_steps <= 0 or args.max_residual_low <= 0:
        raise ValueError("Invalid Stage 1B-2 arguments")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model = load_fast_b_model(args.fast_b_checkpoint).cuda().eval()
    model.requires_grad_(False)
    observer, observer_payload = _load_frozen_observer(args.observer_checkpoint)
    residual = MotionResidualPredictor(
        num_classes=NUM_CLASSES,
        hidden_channels=RESIDUAL_HIDDEN_CHANNELS,
        max_observed_displacement_low=observer.max_displacement_low,
        max_residual_displacement_low=args.max_residual_low,
    ).cuda()
    optimizer = torch.optim.AdamW(residual.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    raft = FrozenRAFT()

    train = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "train")
    val = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "val")
    train_groups = sequence_groups(train)
    all_val = sequence_groups(val)
    missing = [s for s in FULL9 if s not in all_val]
    if missing:
        raise RuntimeError(f"Missing Full9 validation sequences: {missing}")
    val_groups = {s: all_val[s] for s in FULL9}

    sanity_samples = next(iter(val_groups.values()))
    if len(sanity_samples) < 2:
        raise RuntimeError("Need at least two frames for Stage 1B-2 zero-step check")
    frame0 = _host_observation(model, sanity_samples[0])
    frame1 = _host_observation(model, sanity_samples[1])
    observed = _observe_motion(observer, frame0[2], frame0[3], frame1[2], frame1[3])
    persistence_error = F.softmax(frame1[2], dim=1) - F.softmax(frame0[2], dim=1)
    zero_step = _zero_step_check(residual, observed, persistence_error)

    output = Path(args.output)
    result_output = Path(args.result_output)
    output.mkdir(parents=True, exist_ok=True)
    result_output.mkdir(parents=True, exist_ok=True)
    history = []
    best = None

    for epoch in range(1, args.epochs + 1):
        train_stats = _train_epoch(model, observer, residual, raft, train_groups, optimizer, args)
        metrics = _evaluate(model, observer, residual, val_groups, raft)
        checks = _checks(metrics)
        row = {
            "epoch": epoch,
            "train": train_stats,
            "metrics": metrics,
            "stage1b2_checks": checks,
        }
        history.append(row)
        payload = {
            "experiment": "c_v2_stage1b2_residual_motion",
            "epoch": epoch,
            "observer_checkpoint": args.observer_checkpoint,
            "observer_frozen": True,
            "residual_state_dict": residual.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "architecture": {
                "state_update_input": "observed M_t + task-space prediction error e_t",
                "prediction": "M_hat_{t+1} = M_t + Delta M_{t+1}",
                "max_residual_low": args.max_residual_low,
                "delta_zero_initialized": True,
                "raft_at_inference": False,
            },
            "causal_contract": {
                "M_t": "observed from frame pair (t-1,t) after frame t arrives",
                "H_t": "updated from M_t and e_t",
                "M_hat_t_plus_1": "predicted only after H_t update and used to warp Host logits at t into t+1",
                "e_t_plus_1": "computed only after Host frame t+1 arrives and then updates H_{t+1}",
            },
            "row": row,
        }
        torch.save(payload, output / f"epoch_{epoch:03d}.pt")
        if best is None or _selection_key(row) > best["selection_key"]:
            best = {
                "epoch": epoch,
                "selection_key": _selection_key(row),
                "metrics": metrics,
                "stage1b2_checks": checks,
            }
            torch.save(payload, output / "best.pt")
        (result_output / f"epoch_{epoch:03d}.json").write_text(
            json.dumps(row, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(row, sort_keys=True), flush=True)

    summary = {
        "experiment": "C-V2 Stage 1B-2 Residual Motion Prediction",
        "training": {
            "epochs": args.epochs,
            "tbptt_steps": args.tbptt_steps,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "lambda_semantic": args.lambda_semantic,
            "lambda_flow": args.lambda_flow,
            "lambda_delta": args.lambda_delta,
            "max_residual_low": args.max_residual_low,
        },
        "observer": {
            "checkpoint": args.observer_checkpoint,
            "frozen": True,
            "stage1b1_checks": observer_payload["row"]["stage1b1_checks"],
        },
        "frozen": ["Host", "Adapter", "Writeback", "Decoder", "Motion Observer", "all pre-existing temporal modules"],
        "raft_role": "future-flow training target and mTC metric support only; absent from inference model",
        "training_inference_contract": "Residual predictor always consumes Observer motion, never RAFT motion, during both training and inference.",
        "zero_step": zero_step,
        "history": history,
        "best": best,
        "decision": "A positive residual is not mandatory. If Observer+Lagged passes the candidate gate and DeltaM adds no value, retain DeltaM=0. Do not open semantic-residual Stage 2 until Stage 1B yields a motion candidate and its repair-only Oracle delta mTC is >= +0.4 pp. mVC16 is recorded but is not a gate.",
    }
    (result_output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "best_epoch": best["epoch"] if best else None,
        "stage1b_candidate_exists": best["stage1b2_checks"]["stage1b_candidate_exists"] if best else False,
        "motion_candidate_preference": best["stage1b2_checks"]["motion_candidate_preference"] if best else None,
        "checkpoint": str(output / "best.pt"),
        "result": str(result_output / "summary.json"),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
