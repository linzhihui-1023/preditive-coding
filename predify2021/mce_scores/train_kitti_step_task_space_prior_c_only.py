"""C-only task-space temporal-prior training on KITTI-STEP.

The frozen DeepLabV3+ Host remains the semantic anchor. A lightweight causal
ConvGRU predicts the next frame's low-resolution semantic logits directly in
task space. Prediction error is defined in probability space and is used only
to update the temporal state for the next prediction. There is no fusion,
repair gate, feature correction, or writeback in this experiment.
"""

import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch import nn
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
from predify2021.mce_scores.train_kitti_step_temporal_joint import FrozenRAFT, flow_grid
from predify2021.model_factory.deeplabv3plus_resnet50 import HostFeature
from predify2021.model_factory.deeplabv3plus_resnet50.semantic_recurrent_predictor import ConvGRUCell


SEED = 0
NUM_CLASSES = 19
IGNORE_LABEL = 255
HIDDEN_CHANNELS = 64
TBPTT_STEPS = 8
EPOCHS = 3
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
FULL9 = ("0002", "0006", "0007", "0008", "0010", "0013", "0014", "0016", "0018")
OUTPUT_DEFAULT = "/home/lin/predify/experiments/kitti_step_task_space_prior_c_only"
RESULT_DEFAULT = "results/kitti_step_task_space_prior_c_only"


class TaskSpaceTemporalPrior(nn.Module):
    """Causal semantic-logit predictor with an exact persistence initialization."""

    def __init__(self, num_classes=NUM_CLASSES, hidden_channels=HIDDEN_CHANNELS):
        super().__init__()
        self.num_classes = int(num_classes)
        self.hidden_channels = int(hidden_channels)
        self.recurrent = ConvGRUCell(2 * self.num_classes, self.hidden_channels)
        self.delta = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(self.hidden_channels, self.num_classes, 1),
        )
        # E0: next prior equals the current Host logits (semantic persistence).
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def predict_next(self, host_logits_low, probability_error, hidden=None):
        host_probability = F.softmax(host_logits_low, dim=1)
        recurrent_input = torch.cat((host_probability, probability_error), dim=1)
        hidden = self.recurrent(recurrent_input, hidden)
        return host_logits_low + self.delta(hidden), hidden


def _host_frame(model, sample):
    image = load_image(sample)
    with torch.no_grad():
        raw = model.extract_backbone_features(image)
        output_size = tuple(image.shape[-2:])
        logits = model.decode_from_host_feature(HostFeature(raw.c4, raw.c1, output_size))
    low_size = tuple(raw.c1.shape[-2:])
    logits_low = F.interpolate(logits, size=low_size, mode="bilinear", align_corners=False)
    return image, logits, logits_low, output_size


def _upsample_prior(prior_logits_low, output_size):
    return F.interpolate(prior_logits_low, size=output_size, mode="bilinear", align_corners=False)


def _pair_mtc(previous_prediction, current_prediction, flow):
    grid, valid = flow_grid(flow, current_prediction.shape[-2], current_prediction.shape[-1])
    warped = F.grid_sample(
        previous_prediction.float().unsqueeze(0),
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
    ).squeeze(0).squeeze(0).to(torch.int64)
    keep = valid.squeeze(0)
    a = warped[keep].cpu()
    b = current_prediction.squeeze(0)[keep].cpu()
    if not a.numel():
        return float("nan")
    confusion = torch.bincount(
        NUM_CLASSES * a + b,
        minlength=NUM_CLASSES * NUM_CLASSES,
    ).reshape(NUM_CLASSES, NUM_CLASSES)
    return float(torch.nanmean(compute_iou(confusion)).item())


def _new_counts():
    return {
        "host_correct_prior_correct": 0,
        "host_correct_prior_wrong": 0,
        "host_wrong_prior_correct": 0,
        "host_wrong_prior_wrong": 0,
        "valid_pixels": 0,
        "host_correct": 0,
        "host_wrong": 0,
        "prior_correct": 0,
        "prior_wrong": 0,
    }


def _add_counts(total, host_correct, prior_correct, valid):
    hc_pc = valid & host_correct & prior_correct
    hc_pw = valid & host_correct & ~prior_correct
    hw_pc = valid & ~host_correct & prior_correct
    hw_pw = valid & ~host_correct & ~prior_correct
    increments = {
        "host_correct_prior_correct": int(hc_pc.sum().item()),
        "host_correct_prior_wrong": int(hc_pw.sum().item()),
        "host_wrong_prior_correct": int(hw_pc.sum().item()),
        "host_wrong_prior_wrong": int(hw_pw.sum().item()),
        "valid_pixels": int(valid.sum().item()),
        "host_correct": int((valid & host_correct).sum().item()),
        "host_wrong": int((valid & ~host_correct).sum().item()),
        "prior_correct": int((valid & prior_correct).sum().item()),
        "prior_wrong": int((valid & ~prior_correct).sum().item()),
    }
    for key, value in increments.items():
        total[key] += value
    return hw_pc


def _rates(counts):
    valid = max(counts["valid_pixels"], 1)
    host_wrong = max(counts["host_wrong"], 1)
    host_correct = max(counts["host_correct"], 1)
    return {
        **counts,
        "four_way_fraction_of_valid": {
            key: counts[key] / valid
            for key in (
                "host_correct_prior_correct",
                "host_correct_prior_wrong",
                "host_wrong_prior_correct",
                "host_wrong_prior_wrong",
            )
        },
        "recoverable_rate_given_host_wrong": counts["host_wrong_prior_correct"] / host_wrong,
        "damage_rate_given_host_correct": counts["host_correct_prior_wrong"] / host_correct,
        "host_wrong_rate": counts["host_wrong"] / valid,
        "prior_wrong_rate": counts["prior_wrong"] / valid,
    }


def train_sequence(model, prior, samples, optimizer, max_steps=0):
    if len(samples) < 2:
        return {"frames": 0, "windows": 0, "loss": 0.0}

    limit = min(len(samples) - 1, max_steps) if max_steps else len(samples) - 1
    _, _, first_host_low, _ = _host_frame(model, samples[0])
    zero_error = torch.zeros_like(first_host_low)
    pending, hidden = prior.predict_next(first_host_low.detach(), zero_error, None)

    losses = []
    total_loss = 0.0
    frames = 0
    windows = 0

    for offset in range(limit):
        _, _, host_low, output_size = _host_frame(model, samples[offset + 1])
        target = semantic_mask_from_panoptic_png(samples[offset + 1]["mask_path"]).cuda(non_blocking=True)

        prior_logits = _upsample_prior(pending, output_size)
        loss = F.cross_entropy(
            prior_logits,
            target.unsqueeze(0),
            ignore_index=IGNORE_LABEL,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite C-only prior loss")
        losses.append(loss)
        frames += 1

        host_probability_low = F.softmax(host_low.detach(), dim=1)
        prior_probability_low = F.softmax(pending, dim=1)
        probability_error = host_probability_low - prior_probability_low
        previous_hidden = hidden
        next_pending, next_hidden = prior.predict_next(
            host_low.detach(), probability_error, previous_hidden
        )

        boundary = len(losses) == TBPTT_STEPS or offset == limit - 1
        if boundary:
            window_loss = torch.stack(losses).mean()
            optimizer.zero_grad(set_to_none=True)
            window_loss.backward()
            optimizer.step()
            total_loss += float(window_loss.detach().item())
            windows += 1
            # Recompute from the same current-frame inputs with updated weights,
            # while truncating all history before this TBPTT boundary.
            boundary_hidden = (
                previous_hidden.detach() if previous_hidden is not None else None
            )
            pending, hidden = prior.predict_next(
                host_low.detach(), probability_error.detach(), boundary_hidden
            )
            losses = []
        else:
            hidden = next_hidden
            pending = next_pending

    return {
        "frames": frames,
        "windows": windows,
        "loss": total_loss / max(windows, 1),
    }


def train_epoch(model, prior, groups, optimizer, max_steps=0):
    prior.train()
    frame_sum = 0
    window_sum = 0
    weighted_loss = 0.0
    for samples in groups.values():
        row = train_sequence(model, prior, samples, optimizer, max_steps)
        frame_sum += row["frames"]
        window_sum += row["windows"]
        weighted_loss += row["loss"] * row["windows"]
    return {
        "frames": frame_sum,
        "windows": window_sum,
        "loss": weighted_loss / max(window_sum, 1),
    }


@torch.inference_mode()
def evaluate(model, prior, groups, raft):
    names = ("host", "semantic_persistence", "task_space_prior", "repair_only_label_oracle")
    confusion = {
        name: torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
        for name in names
    }
    vc_sums = {name: {8: 0.0, 16: 0.0} for name in names}
    vc_counts = {name: {8: 0, 16: 0} for name in names}
    mtc_sums = {name: 0.0 for name in names}
    mtc_counts = {name: 0 for name in names}
    counts = _new_counts()
    per_sequence = {}
    causal_frames = 0

    prior.eval()
    for sequence in FULL9:
        samples = groups[sequence]
        seq_confusion = {
            name: torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
            for name in names
        }
        seq_vc = {name: VideoConsistency() for name in names}
        seq_mtc_sums = {name: 0.0 for name in names}
        seq_mtc_counts = {name: 0 for name in names}
        seq_counts = _new_counts()

        pending = None
        hidden = None
        previous_image = None
        previous_predictions = {}
        previous_host_low = None

        for index, sample in enumerate(samples):
            image, host_logits, host_low, output_size = _host_frame(model, sample)
            host_prediction = host_logits.argmax(1)
            gt_cpu = semantic_mask_from_panoptic_png(sample["mask_path"])
            gt = gt_cpu.cuda(non_blocking=True)

            if index == 0:
                zero_error = torch.zeros_like(host_low)
                pending, hidden = prior.predict_next(host_low, zero_error, None)
                prior_prediction = host_prediction
                persistence_prediction = host_prediction
                oracle_prediction = host_prediction
            else:
                prior_logits = _upsample_prior(pending, output_size)
                prior_prediction = prior_logits.argmax(1)
                persistence_prediction = _upsample_prior(
                    previous_host_low, output_size
                ).argmax(1)

                valid = gt != IGNORE_LABEL
                host_correct = host_prediction.squeeze(0) == gt
                prior_correct = prior_prediction.squeeze(0) == gt
                recoverable = _add_counts(counts, host_correct, prior_correct, valid)
                _add_counts(seq_counts, host_correct, prior_correct, valid)

                oracle_prediction = host_prediction.clone()
                oracle_mask = recoverable.unsqueeze(0)
                oracle_prediction[oracle_mask] = prior_prediction[oracle_mask]
                causal_frames += 1

            predictions = {
                "host": host_prediction,
                "semantic_persistence": persistence_prediction,
                "task_space_prior": prior_prediction,
                "repair_only_label_oracle": oracle_prediction,
            }

            for name, prediction in predictions.items():
                pred_cpu = prediction.squeeze(0).cpu()
                update_confusion_matrix(confusion[name], pred_cpu, gt_cpu)
                update_confusion_matrix(seq_confusion[name], pred_cpu, gt_cpu)
                seq_vc[name].update(gt_cpu, pred_cpu)

            if previous_image is not None:
                flow = raft.current_to_previous(image, previous_image)
                for name, prediction in predictions.items():
                    score = _pair_mtc(previous_predictions[name], prediction, flow)
                    if math.isfinite(score):
                        mtc_sums[name] += score
                        mtc_counts[name] += 1
                        seq_mtc_sums[name] += score
                        seq_mtc_counts[name] += 1

            if index > 0:
                host_probability_low = F.softmax(host_low, dim=1)
                prior_probability_low = F.softmax(pending, dim=1)
                probability_error = host_probability_low - prior_probability_low
                pending, hidden = prior.predict_next(host_low, probability_error, hidden)

            previous_image = image
            previous_host_low = host_low.detach()
            previous_predictions = {
                name: prediction.detach() for name, prediction in predictions.items()
            }

        for name in names:
            stats = seq_vc[name].stats()
            for length in (8, 16):
                vc_sums[name][length] += stats[length]["sum"]
                vc_counts[name][length] += stats[length]["count"]

        seq_metrics = {}
        for name in names:
            seq_metrics[name] = {
                "mIoU": float(torch.nanmean(compute_iou(seq_confusion[name])).item()),
                "mTC": (
                    seq_mtc_sums[name] / seq_mtc_counts[name]
                    if seq_mtc_counts[name]
                    else float("nan")
                ),
                "mVC8": seq_vc[name].values()[8],
                "mVC16": seq_vc[name].values()[16],
                "valid_frame_pairs": seq_mtc_counts[name],
            }
        per_sequence[sequence] = {
            "metrics": seq_metrics,
            "complementarity": _rates(seq_counts),
        }

    metrics = {}
    for name in names:
        metrics[name] = {
            "mIoU": float(torch.nanmean(compute_iou(confusion[name])).item()),
            "mTC": mtc_sums[name] / mtc_counts[name] if mtc_counts[name] else float("nan"),
            "mVC8": vc_sums[name][8] / vc_counts[name][8] if vc_counts[name][8] else float("nan"),
            "mVC16": vc_sums[name][16] / vc_counts[name][16] if vc_counts[name][16] else float("nan"),
            "valid_frame_pairs": mtc_counts[name],
        }

    delta_vs_host = {
        name: {
            key: metrics[name][key] - metrics["host"][key]
            for key in ("mIoU", "mTC", "mVC8", "mVC16")
        }
        for name in names
        if name != "host"
    }
    return {
        "metrics": metrics,
        "delta_vs_host": delta_vs_host,
        "complementarity_causal_frames_only": _rates(counts),
        "per_sequence": per_sequence,
        "causal_frames": causal_frames,
    }


def _parameter_report(prior):
    return {
        "trainable": sum(p.numel() for p in prior.parameters() if p.requires_grad),
        "total": sum(p.numel() for p in prior.parameters()),
    }


def _selection_score(metrics):
    # C-only is trained for semantic prior quality; choose the best prior mIoU.
    return metrics["metrics"]["task_space_prior"]["mIoU"]


def _zero_step_check(model, prior, sample):
    _, _, host_low, _ = _host_frame(model, sample)
    with torch.no_grad():
        pending, _ = prior.predict_next(
            host_low, torch.zeros_like(host_low), None
        )
    error = float((pending - host_low).abs().max().item())
    if error > 1e-6:
        raise RuntimeError(f"C-only E0 persistence check failed: max_abs={error}")
    return {"pending_equals_current_host_low_logits_max_abs": error}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/home/lin/predify/kitti_step")
    parser.add_argument("--fast-b-checkpoint", default=FAST_B_CHECKPOINT_DEFAULT)
    parser.add_argument("--output", default=OUTPUT_DEFAULT)
    parser.add_argument("--result-output", default=RESULT_DEFAULT)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model = load_fast_b_model(args.fast_b_checkpoint).cuda().eval()
    model.requires_grad_(False)
    prior = TaskSpaceTemporalPrior().cuda()
    optimizer = torch.optim.AdamW(
        prior.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    train = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "train")
    val = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "val")
    train_groups = sequence_groups(train)
    all_val_groups = sequence_groups(val)
    missing = [sequence for sequence in FULL9 if sequence not in all_val_groups]
    if missing:
        raise RuntimeError(f"Missing Full9 validation sequences: {missing}")
    val_groups = {sequence: all_val_groups[sequence] for sequence in FULL9}

    output = Path(args.output)
    result_output = Path(args.result_output)
    output.mkdir(parents=True, exist_ok=True)
    result_output.mkdir(parents=True, exist_ok=True)

    zero_step = _zero_step_check(model, prior, next(iter(val_groups.values()))[0])
    raft = FrozenRAFT()
    history = []
    best = None

    print(json.dumps({
        "experiment": "C-only Task-space Temporal Prior",
        "trainable_parameters": _parameter_report(prior),
        "frozen": ["Host", "Adapter", "Writeback", "Decoder", "all existing temporal modules"],
        "zero_step": zero_step,
        "tbptt_steps": TBPTT_STEPS,
        "epochs": args.epochs,
    }, sort_keys=True), flush=True)

    for epoch in range(1, args.epochs + 1):
        train_stats = train_epoch(
            model, prior, train_groups, optimizer, args.max_train_steps
        )
        metrics = evaluate(model, prior, val_groups, raft)
        row = {
            "epoch": epoch,
            "train": train_stats,
            "metrics": metrics,
            "selection_score_prior_mIoU": _selection_score(metrics),
        }
        history.append(row)
        payload = {
            "experiment": "c_only_task_space_temporal_prior",
            "epoch": epoch,
            "model_state_dict": prior.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "architecture": {
                "num_classes": NUM_CLASSES,
                "hidden_channels": HIDDEN_CHANNELS,
                "state_resolution": "Host C1 spatial size (approximately 1/4 input)",
                "prediction": "next low-resolution Host-logit persistence + learned recurrent delta",
                "prediction_error": "Host probability - predicted prior probability",
                "fusion": False,
                "repair_gate": False,
                "feature_writeback": False,
            },
            "metrics": row,
        }
        torch.save(payload, output / f"epoch_{epoch:03d}.pt")

        score = row["selection_score_prior_mIoU"]
        if best is None or score > best["score"]:
            best = {"epoch": epoch, "score": score, "metrics": metrics}
            torch.save(payload, output / "best.pt")

        (result_output / f"epoch_{epoch:03d}.json").write_text(
            json.dumps(row, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(row, sort_keys=True), flush=True)

    result = {
        "experiment": "C-only Task-space Temporal Prior",
        "purpose": (
            "Test whether direct task-space temporal prediction improves temporal-prior "
            "complementarity before implementing any D/fusion gate."
        ),
        "training": {
            "epochs": args.epochs,
            "tbptt_steps": TBPTT_STEPS,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "max_train_steps_per_sequence": args.max_train_steps,
            "train_sequences": sorted(train_groups),
        },
        "validation": {"full9": list(FULL9)},
        "trainable_parameters": _parameter_report(prior),
        "frozen": ["Host", "Adapter", "Writeback", "Decoder", "all existing temporal modules"],
        "zero_step": zero_step,
        "no_fusion": True,
        "no_gate": True,
        "no_feature_correction": True,
        "history": history,
        "best": best,
        "comparison_reference": {
            "stage_t_proxy_recoverable_rate": 0.08345807180168258,
            "stage_t_proxy_damage_rate": 0.014210056995095902,
            "stage_t_proxy_repair_oracle_delta_mIoU": 0.020257415304355098,
            "switch_probe_pooled_auroc": 0.6174806894207491,
        },
        "development_checks_for_next_step": {
            "recoverable_rate_gt_10pct": bool(
                best and best["metrics"]["complementarity_causal_frames_only"]["recoverable_rate_given_host_wrong"] > 0.10
            ),
            "damage_rate_lt_stage_t_proxy": bool(
                best and best["metrics"]["complementarity_causal_frames_only"]["damage_rate_given_host_correct"] < 0.014210056995095902
            ),
            "repair_oracle_delta_mIoU_gt_stage_t_proxy": bool(
                best and best["metrics"]["delta_vs_host"]["repair_only_label_oracle"]["mIoU"] > 0.020257415304355098
            ),
        },
    }
    (result_output / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "best_epoch": best["epoch"] if best else None,
        "best_prior_mIoU": best["score"] if best else None,
        "result": str(result_output / "summary.json"),
        "checkpoint": str(output / "best.pt"),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
