"""C-V2 Stage 1B-1: explicit Motion Observer training on KITTI-STEP.

Only the new pairwise observer is trainable. The frozen Host supplies C1 features
and task probabilities. A local correlation cost volume performs explicit
correspondence search. RAFT is a frozen training teacher and mTC metric support;
it is absent from the saved inference model.
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
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import (
    load_image,
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
    _pair_mtc,
    _rates,
    _upsample_prior,
)
from predify2021.mce_scores.train_kitti_step_temporal_joint import FrozenRAFT
from predify2021.model_factory.deeplabv3plus_resnet50 import HostFeature
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_motion_transport import (
    LocalCorrelationMotionObserver,
    downsample_backward_flow,
    normalized_flow_distillation_loss,
    teacher_reachable_mask,
    warp_low_logits,
)

SEED = 0
EPOCHS = 3
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
LAMBDA_WARP = 1.0
LAMBDA_FLOW = 10.0
C1_CHANNELS = 256
PROJECTED_CHANNELS = 32
HIDDEN_CHANNELS = 64
CORRELATION_RADIUS = 4
MAX_DISPLACEMENT_LOW = 32.0
OUTPUT_DEFAULT = "/home/lin/predify/experiments/kitti_step_task_space_prior_c_v2_stage1b_observer"
RESULT_DEFAULT = "results/kitti_step_task_space_prior_c_v2_stage1b_observer"


def _host_observation(model, sample):
    image = load_image(sample)
    with torch.no_grad():
        raw = model.extract_backbone_features(image)
        output_size = tuple(image.shape[-2:])
        logits = model.decode_from_host_feature(
            HostFeature(raw.c4, raw.c1, output_size)
        )
        low_size = tuple(raw.c1.shape[-2:])
        logits_low = F.interpolate(
            logits, size=low_size, mode="bilinear", align_corners=False
        )
    return image, logits, logits_low, raw.c1.detach(), output_size


def _zero_step_check(model, observer, sample_a, sample_b):
    _, _, low_a, c1_a, _ = _host_observation(model, sample_a)
    _, _, low_b, c1_b, _ = _host_observation(model, sample_b)
    with torch.no_grad():
        flow = observer(
            c1_a,
            c1_b,
            F.softmax(low_a, dim=1),
            F.softmax(low_b, dim=1),
        )
        warped, _ = warp_low_logits(low_a, flow)
    row = {
        "observer_flow_max_abs": float(flow.abs().max().item()),
        "zero_flow_warp_logits_max_abs": float((warped - low_a).abs().max().item()),
        "zero_flow_argmax_equals_persistence": bool(
            torch.equal(warped.argmax(1), low_a.argmax(1))
        ),
    }
    if row["observer_flow_max_abs"] != 0.0 or not row["zero_flow_argmax_equals_persistence"]:
        raise RuntimeError(f"Stage 1B-1 zero-step check failed: {row}")
    return row


def _train_pair(model, observer, raft, previous, current, optimizer, args):
    previous_image, _, previous_low, previous_c1, _ = _host_observation(model, previous)
    current_image, _, current_low, current_c1, output_size = _host_observation(model, current)
    previous_probability = F.softmax(previous_low.detach(), dim=1)
    current_probability = F.softmax(current_low.detach(), dim=1)

    observed_motion = observer(
        previous_c1,
        current_c1,
        previous_probability,
        current_probability,
    )
    warped_low, _ = warp_low_logits(previous_low.detach(), observed_motion)
    target = semantic_mask_from_panoptic_png(current["mask_path"]).cuda(non_blocking=True)
    warp_loss = F.cross_entropy(
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
        observed_motion, teacher_low, observer.max_displacement_low
    )
    total = args.lambda_warp * warp_loss + args.lambda_flow * flow_loss
    if not all(torch.isfinite(x).item() for x in (warp_loss, flow_loss, total)):
        raise FloatingPointError("Non-finite Stage 1B-1 loss")
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    optimizer.step()
    return {
        "warp_loss": float(warp_loss.detach().item()),
        "flow_loss": float(flow_loss.detach().item()),
        "total_loss": float(total.detach().item()),
    }


def _train_epoch(model, observer, raft, groups, optimizer, args):
    observer.train()
    sums = {"pairs": 0, "warp_loss": 0.0, "flow_loss": 0.0, "total_loss": 0.0}
    for samples in groups.values():
        limit = min(len(samples) - 1, args.max_train_pairs) if args.max_train_pairs else len(samples) - 1
        for index in range(1, limit + 1):
            row = _train_pair(model, observer, raft, samples[index - 1], samples[index], optimizer, args)
            sums["pairs"] += 1
            for key in ("warp_loss", "flow_loss", "total_loss"):
                sums[key] += row[key]
    denom = max(sums["pairs"], 1)
    for key in ("warp_loss", "flow_loss", "total_loss"):
        sums[key] /= denom
    return sums


@torch.inference_mode()
def _evaluate(model, observer, groups, raft):
    names = (
        "host",
        "semantic_persistence",
        "observer_warp",
        "raft_low_same_path_reference",
        "observer_repair_only_oracle",
    )
    confusion = {n: torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64) for n in names}
    mtc_sum = {n: 0.0 for n in names}
    mtc_count = {n: 0 for n in names}
    vc_sum = {n: {8: 0.0, 16: 0.0} for n in names}
    vc_count = {n: {8: 0, 16: 0} for n in names}
    counts = _new_counts()
    flow_diag = {
        "pairs": 0,
        "reachable_values": 0,
        "pred_abs": 0.0,
        "teacher_abs": 0.0,
        "l1": 0.0,
        "spatial_valid_pixels": 0,
        "reachable_pixels": 0,
        "within_correlation_radius_pixels": 0,
    }
    per_sequence = {}
    observer.eval()

    for sequence in FULL9:
        samples = groups[sequence]
        seq_conf = {n: torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64) for n in names}
        seq_mtc_sum = {n: 0.0 for n in names}
        seq_mtc_count = {n: 0 for n in names}
        seq_vc = {n: VideoConsistency() for n in names}
        seq_counts = _new_counts()
        previous = None
        previous_predictions = {}

        for sample in samples:
            image, host_logits, host_low, c1, output_size = _host_observation(model, sample)
            host_pred = host_logits.argmax(1)
            gt_cpu = semantic_mask_from_panoptic_png(sample["mask_path"])
            gt = gt_cpu.cuda(non_blocking=True)
            teacher_full = None

            if previous is None:
                persistence = observer_pred = raft_pred = oracle_pred = host_pred
            else:
                previous_image, previous_low, previous_c1 = previous
                persistence = _upsample_prior(previous_low, output_size).argmax(1)
                observed_motion = observer(
                    previous_c1,
                    c1,
                    F.softmax(previous_low, dim=1),
                    F.softmax(host_low, dim=1),
                )
                observer_warped, _ = warp_low_logits(previous_low, observed_motion)
                observer_pred = _upsample_prior(observer_warped, output_size).argmax(1)

                teacher_full = raft.current_to_previous(image, previous_image)
                teacher_low = downsample_backward_flow(
                    teacher_full, tuple(host_low.shape[-2:])
                )
                raft_warped, _ = warp_low_logits(previous_low, teacher_low)
                raft_pred = _upsample_prior(raft_warped, output_size).argmax(1)

                valid = gt != IGNORE_LABEL
                host_correct = host_pred.squeeze(0) == gt
                observer_correct = observer_pred.squeeze(0) == gt
                recoverable = _add_counts(counts, host_correct, observer_correct, valid)
                _add_counts(seq_counts, host_correct, observer_correct, valid)
                oracle_pred = host_pred.clone()
                oracle_pred[recoverable.unsqueeze(0)] = observer_pred[recoverable.unsqueeze(0)]

                reachable, spatial_valid = teacher_reachable_mask(
                    teacher_low, observer.max_displacement_low
                )
                within_correlation_radius = (
                    spatial_valid
                    & (teacher_low[:, 0].abs() <= observer.correlation_radius)
                    & (teacher_low[:, 1].abs() <= observer.correlation_radius)
                )
                mask = reachable.unsqueeze(1).expand_as(observed_motion)
                values = int(mask.sum().item())
                if values:
                    flow_diag["reachable_values"] += values
                    flow_diag["pred_abs"] += float(observed_motion[mask].abs().sum().item())
                    flow_diag["teacher_abs"] += float(teacher_low[mask].abs().sum().item())
                    flow_diag["l1"] += float((observed_motion[mask] - teacher_low[mask]).abs().sum().item())
                flow_diag["spatial_valid_pixels"] += int(spatial_valid.sum().item())
                flow_diag["reachable_pixels"] += int(reachable.sum().item())
                flow_diag["within_correlation_radius_pixels"] += int(
                    within_correlation_radius.sum().item()
                )
                flow_diag["pairs"] += 1

            predictions = {
                "host": host_pred,
                "semantic_persistence": persistence,
                "observer_warp": observer_pred,
                "raft_low_same_path_reference": raft_pred,
                "observer_repair_only_oracle": oracle_pred,
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
        per_sequence[sequence] = {
            "metrics": {
                name: {
                    "mIoU": float(torch.nanmean(compute_iou(seq_conf[name])).item()),
                    "mTC": seq_mtc_sum[name] / seq_mtc_count[name] if seq_mtc_count[name] else float("nan"),
                    "mVC8": seq_vc[name].values()[8],
                    "mVC16": seq_vc[name].values()[16],
                    "valid_frame_pairs": seq_mtc_count[name],
                }
                for name in names
            },
            "complementarity": _rates(seq_counts),
        }

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
    delta_vs_persistence = {
        name: {
            key: metrics[name][key] - metrics["semantic_persistence"][key]
            for key in ("mIoU", "mTC", "mVC8", "mVC16")
        }
        for name in ("observer_warp", "raft_low_same_path_reference")
    }
    delta_vs_host = {
        name: {
            key: metrics[name][key] - metrics["host"][key]
            for key in ("mIoU", "mTC", "mVC8", "mVC16")
        }
        for name in names if name != "host"
    }
    values = max(flow_diag["reachable_values"], 1)
    spatial_valid = max(flow_diag["spatial_valid_pixels"], 1)
    return {
        "metrics": metrics,
        "delta_vs_persistence": delta_vs_persistence,
        "delta_vs_host": delta_vs_host,
        "complementarity_causal_frames_only": _rates(counts),
        "flow_diagnostics": {
            "pairs": flow_diag["pairs"],
            "predicted_abs_mean_low_pixels_reachable": flow_diag["pred_abs"] / values,
            "teacher_abs_mean_low_pixels_reachable": flow_diag["teacher_abs"] / values,
            "l1_mean_low_pixels_reachable": flow_diag["l1"] / values,
            "teacher_reachable_fraction_of_spatial_valid": flow_diag["reachable_pixels"] / spatial_valid,
            "teacher_within_correlation_radius_fraction": (
                flow_diag["within_correlation_radius_pixels"] / spatial_valid
            ),
            "correlation_radius_low_pixels": observer.correlation_radius,
        },
        "per_sequence": per_sequence,
    }


def _checks(metrics):
    observer = metrics["metrics"]["observer_warp"]
    persistence = metrics["metrics"]["semantic_persistence"]
    raft = metrics["metrics"]["raft_low_same_path_reference"]
    gain = observer["mIoU"] - persistence["mIoU"]
    raft_gain = raft["mIoU"] - persistence["mIoU"]
    recovered = gain / raft_gain if raft_gain > 1e-12 else float("nan")
    return {
        "observer_delta_mIoU_vs_persistence_ge_2pp": bool(gain >= 0.02),
        "observer_mTC_gt_persistence": bool(observer["mTC"] > persistence["mTC"]),
        "observer_minimum_go": bool(gain >= 0.02 and observer["mTC"] > persistence["mTC"]),
        "observer_fraction_of_raft_same_path_mIoU_gain": float(recovered),
        "observer_recovers_half_raft_gain_diagnostic": bool(math.isfinite(recovered) and recovered >= 0.5),
    }


def _selection_key(row):
    checks = row["stage1b1_checks"]
    metrics = row["metrics"]["metrics"]["observer_warp"]
    return (int(checks["observer_minimum_go"]), metrics["mIoU"], metrics["mTC"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/home/lin/predify/kitti_step")
    parser.add_argument("--fast-b-checkpoint", default=FAST_B_CHECKPOINT_DEFAULT)
    parser.add_argument("--output", default=OUTPUT_DEFAULT)
    parser.add_argument("--result-output", default=RESULT_DEFAULT)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--lambda-warp", type=float, default=LAMBDA_WARP)
    parser.add_argument("--lambda-flow", type=float, default=LAMBDA_FLOW)
    parser.add_argument("--correlation-radius", type=int, default=CORRELATION_RADIUS)
    parser.add_argument("--max-displacement-low", type=float, default=MAX_DISPLACEMENT_LOW)
    parser.add_argument("--max-train-pairs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.epochs <= 0 or args.correlation_radius < 0 or args.max_displacement_low <= 0:
        raise ValueError("Invalid Stage 1B-1 arguments")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model = load_fast_b_model(args.fast_b_checkpoint).cuda().eval()
    model.requires_grad_(False)
    observer = LocalCorrelationMotionObserver(
        c1_channels=C1_CHANNELS,
        num_classes=NUM_CLASSES,
        projected_channels=PROJECTED_CHANNELS,
        hidden_channels=HIDDEN_CHANNELS,
        correlation_radius=args.correlation_radius,
        max_displacement_low=args.max_displacement_low,
    ).cuda()
    optimizer = torch.optim.AdamW(observer.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    raft = FrozenRAFT()

    train = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "train")
    val = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "val")
    train_groups = sequence_groups(train)
    all_val = sequence_groups(val)
    missing = [s for s in FULL9 if s not in all_val]
    if missing:
        raise RuntimeError(f"Missing Full9 validation sequences: {missing}")
    val_groups = {s: all_val[s] for s in FULL9}
    first_samples = next(iter(val_groups.values()))
    if len(first_samples) < 2:
        raise RuntimeError("Need at least two validation frames for zero-step check")
    zero_step = _zero_step_check(model, observer, first_samples[0], first_samples[1])

    output = Path(args.output)
    result_output = Path(args.result_output)
    output.mkdir(parents=True, exist_ok=True)
    result_output.mkdir(parents=True, exist_ok=True)
    history = []
    best = None

    for epoch in range(1, args.epochs + 1):
        train_stats = _train_epoch(model, observer, raft, train_groups, optimizer, args)
        metrics = _evaluate(model, observer, val_groups, raft)
        checks = _checks(metrics)
        row = {
            "epoch": epoch,
            "train": train_stats,
            "metrics": metrics,
            "stage1b1_checks": checks,
        }
        history.append(row)
        payload = {
            "experiment": "c_v2_stage1b1_motion_observer",
            "epoch": epoch,
            "observer_state_dict": observer.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "architecture": {
                "input": "pairwise frozen Host C1 features + pairwise Host task probabilities",
                "matching": "local correlation cost volume",
                "c1_channels": C1_CHANNELS,
                "num_classes": NUM_CLASSES,
                "projected_channels": PROJECTED_CHANNELS,
                "hidden_channels": HIDDEN_CHANNELS,
                "correlation_radius": args.correlation_radius,
                "max_displacement_low": args.max_displacement_low,
                "prediction": "observed backward motion M_t = F_{t->t-1}",
                "raft_at_inference": False,
            },
            "row": row,
        }
        torch.save(payload, output / f"epoch_{epoch:03d}.pt")
        if best is None or _selection_key(row) > best["selection_key"]:
            best = {
                "epoch": epoch,
                "selection_key": _selection_key(row),
                "metrics": metrics,
                "stage1b1_checks": checks,
            }
            torch.save(payload, output / "best.pt")
        (result_output / f"epoch_{epoch:03d}.json").write_text(
            json.dumps(row, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(row, sort_keys=True), flush=True)

    summary = {
        "experiment": "C-V2 Stage 1B-1 Explicit Motion Observer",
        "training": {
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "lambda_warp": args.lambda_warp,
            "lambda_flow": args.lambda_flow,
            "correlation_radius": args.correlation_radius,
            "max_displacement_low": args.max_displacement_low,
            "max_train_pairs_per_sequence": args.max_train_pairs,
        },
        "frozen": ["Host", "Adapter", "Writeback", "Decoder", "all existing temporal modules"],
        "raft_role": "training teacher and mTC metric support only; absent from inference model",
        "zero_step": zero_step,
        "history": history,
        "best": best,
        "decision": "Proceed to Stage 1B-2 only if best.stage1b1_checks.observer_minimum_go is true. mVC16 is recorded but is not a Stage-1B gate.",
    }
    (result_output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "best_epoch": best["epoch"] if best else None,
        "observer_minimum_go": best["stage1b1_checks"]["observer_minimum_go"] if best else False,
        "checkpoint": str(output / "best.pt"),
        "result": str(result_output / "summary.json"),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
