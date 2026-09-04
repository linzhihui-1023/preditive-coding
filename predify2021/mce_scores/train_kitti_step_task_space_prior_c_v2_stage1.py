"""Train C-V2 Stage 1 learned motion transport on KITTI-STEP.

Only the new causal transport predictor is trainable. Host/Adapter/Writeback/
Decoder and all existing temporal modules are frozen. RAFT is a frozen teacher
for training and mTC evaluation only; the saved transport model does not use
RAFT at inference. No residual semantic head and no D fusion gate are present.
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
    _host_frame,
    _new_counts,
    _pair_mtc,
    _rates,
    _upsample_prior,
)
from predify2021.mce_scores.train_kitti_step_temporal_joint import FrozenRAFT
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_motion_transport import (
    CausalMotionTransportPredictor,
    downsample_backward_flow,
    normalized_flow_distillation_loss,
    teacher_reachable_mask,
    warp_low_logits,
)

SEED = 0
EPOCHS = 3
TBPTT_STEPS = 8
HIDDEN_CHANNELS = 64
MAX_DISPLACEMENT_LOW = 32.0
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
LAMBDA_SEMANTIC = 1.0
LAMBDA_FLOW = 1.0
OUTPUT_DEFAULT = "/home/lin/predify/experiments/kitti_step_task_space_prior_c_v2_stage1"
RESULT_DEFAULT = "results/kitti_step_task_space_prior_c_v2_stage1"

PERSISTENCE_REFERENCE = {"mIoU": 0.5091145088556011, "mTC": 0.6723652597629691}
# Historical diagnostic only. This used a different full-resolution RAFT-warp
# path with current-Host fallback on invalid flow, so it is not a Stage-1
# teacher upper bound and is never used for a GO/NO-GO check below.
LEGACY_FULLRES_RAFT_REFERENCE = {
    "mIoU": 0.6182763159149198,
    "mTC": 0.7245936440098666,
    "repair_oracle_delta_mIoU": 0.04470385447457059,
    "repair_oracle_delta_mTC": 0.032449255591568194,
}


def _zero_step_check(model, transport, sample):
    _, _, host_low, _ = _host_frame(model, sample)
    with torch.no_grad():
        flow, _ = transport.predict_next(host_low, torch.zeros_like(host_low), None)
        warped, valid = warp_low_logits(host_low, flow)
        row = {
            "pending_flow_max_abs": float(flow.abs().max().item()),
            "zero_flow_warp_logits_max_abs": float((warped - host_low).abs().max().item()),
            "zero_flow_argmax_equals_persistence": bool(
                torch.equal(warped.argmax(1), host_low.argmax(1))
            ),
            "zero_flow_valid_fraction": float(valid.float().mean().item()),
        }
    if row["pending_flow_max_abs"] != 0.0 or not row["zero_flow_argmax_equals_persistence"]:
        raise RuntimeError(f"C-V2 Stage 1 zero-step check failed: {row}")
    return row


def _train_sequence(model, transport, raft, samples, optimizer, args):
    if len(samples) < 2:
        return {
            "frames": 0,
            "windows": 0,
            "semantic_loss": 0.0,
            "flow_loss": 0.0,
            "total_loss": 0.0,
        }
    steps = min(len(samples) - 1, args.max_train_steps) if args.max_train_steps else len(samples) - 1
    previous_image, _, previous_low, _ = _host_frame(model, samples[0])
    pending_flow, hidden = transport.predict_next(
        previous_low.detach(), torch.zeros_like(previous_low), None
    )
    sem_losses, flow_losses = [], []
    sums = {
        "frames": 0,
        "windows": 0,
        "semantic_loss": 0.0,
        "flow_loss": 0.0,
        "total_loss": 0.0,
    }

    for offset in range(steps):
        current_image, _, current_low, output_size = _host_frame(model, samples[offset + 1])
        target = semantic_mask_from_panoptic_png(samples[offset + 1]["mask_path"]).cuda(
            non_blocking=True
        )
        warped_low, _ = warp_low_logits(previous_low.detach(), pending_flow)
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
            pending_flow, teacher_low, transport.max_displacement_low
        )
        if not torch.isfinite(semantic_loss) or not torch.isfinite(flow_loss):
            raise FloatingPointError("Non-finite C-V2 Stage 1 loss")
        sem_losses.append(semantic_loss)
        flow_losses.append(flow_loss)
        sums["frames"] += 1

        probability_error = F.softmax(current_low.detach(), dim=1) - F.softmax(warped_low, dim=1)
        previous_hidden = hidden
        next_flow, next_hidden = transport.predict_next(
            current_low.detach(), probability_error, previous_hidden
        )
        boundary = len(sem_losses) == args.tbptt_steps or offset == steps - 1
        if boundary:
            sem = torch.stack(sem_losses).mean()
            flo = torch.stack(flow_losses).mean()
            total = args.lambda_semantic * sem + args.lambda_flow * flo
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            sums["windows"] += 1
            sums["semantic_loss"] += float(sem.detach().item())
            sums["flow_loss"] += float(flo.detach().item())
            sums["total_loss"] += float(total.detach().item())
            pending_flow, hidden = transport.predict_next(
                current_low.detach(),
                probability_error.detach(),
                previous_hidden.detach() if previous_hidden is not None else None,
            )
            sem_losses, flow_losses = [], []
        else:
            pending_flow, hidden = next_flow, next_hidden
        previous_image, previous_low = current_image, current_low.detach()

    windows = max(sums["windows"], 1)
    for key in ("semantic_loss", "flow_loss", "total_loss"):
        sums[key] /= windows
    return sums


def _train_epoch(model, transport, raft, groups, optimizer, args):
    transport.train()
    total = {
        "frames": 0,
        "windows": 0,
        "semantic_loss": 0.0,
        "flow_loss": 0.0,
        "total_loss": 0.0,
    }
    for samples in groups.values():
        row = _train_sequence(model, transport, raft, samples, optimizer, args)
        total["frames"] += row["frames"]
        for key in ("semantic_loss", "flow_loss", "total_loss"):
            total[key] += row[key] * row["windows"]
        total["windows"] += row["windows"]
    windows = max(total["windows"], 1)
    for key in ("semantic_loss", "flow_loss", "total_loss"):
        total[key] /= windows
    return total


@torch.inference_mode()
def _evaluate(model, transport, groups, raft):
    # raft_low_transport_reference uses the exact same low-resolution warp and
    # invalid-flow fallback as learned_transport. The only difference is the
    # transport field: frozen RAFT teacher versus causal learned prediction.
    names = (
        "host",
        "semantic_persistence",
        "learned_transport",
        "raft_low_transport_reference",
        "repair_only_label_oracle",
    )
    conf = {n: torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64) for n in names}
    mtc_sum, mtc_n = {n: 0.0 for n in names}, {n: 0 for n in names}
    vc_sum = {n: {8: 0.0, 16: 0.0} for n in names}
    vc_n = {n: {8: 0, 16: 0} for n in names}
    counts = _new_counts()
    flow_diag = {
        "pairs": 0,
        "reachable_values": 0,
        "spatial_valid_pixels": 0,
        "reachable_pixels": 0,
        "pred_abs": 0.0,
        "teacher_abs": 0.0,
        "l1": 0.0,
        "pred_valid": 0.0,
        "raft_low_valid": 0.0,
    }
    per_sequence = {}
    transport.eval()

    for sequence in FULL9:
        seq_conf = {n: torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64) for n in names}
        seq_mtc_sum, seq_mtc_n = {n: 0.0 for n in names}, {n: 0 for n in names}
        seq_vc = {n: VideoConsistency() for n in names}
        seq_counts = _new_counts()
        previous_image = previous_low = None
        previous_predictions = {}
        pending_flow = hidden = None

        for index, sample in enumerate(groups[sequence]):
            image, host_logits, host_low, output_size = _host_frame(model, sample)
            host_pred = host_logits.argmax(1)
            gt_cpu = semantic_mask_from_panoptic_png(sample["mask_path"])
            gt = gt_cpu.cuda(non_blocking=True)
            teacher_full = None

            if index == 0:
                pending_flow, hidden = transport.predict_next(
                    host_low, torch.zeros_like(host_low), None
                )
                persist_pred = learned_pred = raft_low_pred = oracle_pred = host_pred
            else:
                persist_pred = _upsample_prior(previous_low, output_size).argmax(1)

                warped_low, pred_valid = warp_low_logits(previous_low, pending_flow)
                learned_pred = _upsample_prior(warped_low, output_size).argmax(1)

                teacher_full = raft.current_to_previous(image, previous_image)
                teacher_low = downsample_backward_flow(
                    teacher_full, tuple(host_low.shape[-2:])
                )
                raft_low_warped, raft_low_valid = warp_low_logits(previous_low, teacher_low)
                raft_low_pred = _upsample_prior(raft_low_warped, output_size).argmax(1)

                valid = gt != IGNORE_LABEL
                host_correct = host_pred.squeeze(0) == gt
                learned_correct = learned_pred.squeeze(0) == gt
                recoverable = _add_counts(counts, host_correct, learned_correct, valid)
                _add_counts(seq_counts, host_correct, learned_correct, valid)
                oracle_pred = host_pred.clone()
                oracle_pred[recoverable.unsqueeze(0)] = learned_pred[recoverable.unsqueeze(0)]

                reachable, spatial_valid = teacher_reachable_mask(
                    teacher_low, transport.max_displacement_low
                )
                mask = reachable.unsqueeze(1).expand_as(pending_flow)
                nv = int(mask.sum().item())
                if nv:
                    flow_diag["reachable_values"] += nv
                    flow_diag["pred_abs"] += float(pending_flow[mask].abs().sum().item())
                    flow_diag["teacher_abs"] += float(teacher_low[mask].abs().sum().item())
                    flow_diag["l1"] += float(
                        (pending_flow[mask] - teacher_low[mask]).abs().sum().item()
                    )
                flow_diag["spatial_valid_pixels"] += int(spatial_valid.sum().item())
                flow_diag["reachable_pixels"] += int(reachable.sum().item())
                flow_diag["pairs"] += 1
                flow_diag["pred_valid"] += float(pred_valid.float().mean().item())
                flow_diag["raft_low_valid"] += float(raft_low_valid.float().mean().item())

            predictions = {
                "host": host_pred,
                "semantic_persistence": persist_pred,
                "learned_transport": learned_pred,
                "raft_low_transport_reference": raft_low_pred,
                "repair_only_label_oracle": oracle_pred,
            }
            for n, pred in predictions.items():
                pc = pred.squeeze(0).cpu()
                update_confusion_matrix(conf[n], pc, gt_cpu)
                update_confusion_matrix(seq_conf[n], pc, gt_cpu)
                seq_vc[n].update(gt_cpu, pc)

            if previous_image is not None:
                for n, pred in predictions.items():
                    score = _pair_mtc(previous_predictions[n], pred, teacher_full)
                    if math.isfinite(score):
                        mtc_sum[n] += score
                        mtc_n[n] += 1
                        seq_mtc_sum[n] += score
                        seq_mtc_n[n] += 1
                error = F.softmax(host_low, dim=1) - F.softmax(warped_low, dim=1)
                pending_flow, hidden = transport.predict_next(host_low, error, hidden)

            previous_image, previous_low = image, host_low.detach()
            previous_predictions = {n: p.detach() for n, p in predictions.items()}

        for n in names:
            st = seq_vc[n].stats()
            for length in (8, 16):
                vc_sum[n][length] += st[length]["sum"]
                vc_n[n][length] += st[length]["count"]
        per_sequence[sequence] = {
            "metrics": {
                n: {
                    "mIoU": float(torch.nanmean(compute_iou(seq_conf[n])).item()),
                    "mTC": seq_mtc_sum[n] / seq_mtc_n[n] if seq_mtc_n[n] else float("nan"),
                    "mVC8": seq_vc[n].values()[8],
                    "mVC16": seq_vc[n].values()[16],
                    "valid_frame_pairs": seq_mtc_n[n],
                }
                for n in names
            },
            "complementarity": _rates(seq_counts),
        }

    metrics = {
        n: {
            "mIoU": float(torch.nanmean(compute_iou(conf[n])).item()),
            "mTC": mtc_sum[n] / mtc_n[n] if mtc_n[n] else float("nan"),
            "mVC8": vc_sum[n][8] / vc_n[n][8] if vc_n[n][8] else float("nan"),
            "mVC16": vc_sum[n][16] / vc_n[n][16] if vc_n[n][16] else float("nan"),
            "valid_frame_pairs": mtc_n[n],
        }
        for n in names
    }
    delta_host = {
        n: {
            k: metrics[n][k] - metrics["host"][k]
            for k in ("mIoU", "mTC", "mVC8", "mVC16")
        }
        for n in names
        if n != "host"
    }
    delta_persist = {
        k: metrics["learned_transport"][k] - metrics["semantic_persistence"][k]
        for k in ("mIoU", "mTC", "mVC8", "mVC16")
    }
    raft_low_delta_persist = {
        k: metrics["raft_low_transport_reference"][k] - metrics["semantic_persistence"][k]
        for k in ("mIoU", "mTC", "mVC8", "mVC16")
    }
    values = max(flow_diag["reachable_values"], 1)
    pairs = max(flow_diag["pairs"], 1)
    spatial_valid_pixels = max(flow_diag["spatial_valid_pixels"], 1)
    return {
        "metrics": metrics,
        "delta_vs_host": delta_host,
        "delta_learned_transport_vs_persistence": delta_persist,
        "delta_raft_low_reference_vs_persistence": raft_low_delta_persist,
        "complementarity_causal_frames_only": _rates(counts),
        "flow_diagnostics": {
            "pairs": flow_diag["pairs"],
            "predicted_abs_mean_low_pixels_reachable": flow_diag["pred_abs"] / values,
            "teacher_abs_mean_low_pixels_reachable": flow_diag["teacher_abs"] / values,
            "l1_mean_low_pixels_reachable": flow_diag["l1"] / values,
            "predicted_transport_valid_fraction": flow_diag["pred_valid"] / pairs,
            "raft_low_transport_valid_fraction": flow_diag["raft_low_valid"] / pairs,
            "teacher_spatial_valid_pixels": flow_diag["spatial_valid_pixels"],
            "teacher_reachable_pixels": flow_diag["reachable_pixels"],
            "reachable_teacher_fraction_of_spatial_valid": (
                flow_diag["reachable_pixels"] / spatial_valid_pixels
            ),
        },
        "per_sequence": per_sequence,
    }


def _checks(result):
    learned = result["metrics"]["learned_transport"]
    persistence = result["metrics"]["semantic_persistence"]
    raft_low = result["metrics"]["raft_low_transport_reference"]
    oracle = result["delta_vs_host"]["repair_only_label_oracle"]
    gain = learned["mIoU"] - persistence["mIoU"]
    raft_low_gain = raft_low["mIoU"] - persistence["mIoU"]
    recovered = gain / raft_low_gain if raft_low_gain > 1e-12 else float("nan")
    return {
        "learned_transport_delta_mIoU_vs_persistence_ge_2pp": bool(gain >= 0.02),
        "learned_transport_mTC_gt_persistence": bool(learned["mTC"] > persistence["mTC"]),
        "stage1_minimum_go": bool(gain >= 0.02 and learned["mTC"] > persistence["mTC"]),
        "raft_low_same_path_reference_mIoU": float(raft_low["mIoU"]),
        "raft_low_same_path_delta_mIoU_vs_persistence": float(raft_low_gain),
        "learned_fraction_of_raft_low_mIoU_gain": float(recovered),
        "learned_reaches_half_raft_low_mIoU_gain_diagnostic": bool(
            math.isfinite(recovered) and recovered >= 0.5
        ),
        "oracle_delta_mTC_ge_0p4pp_diagnostic": bool(oracle["mTC"] >= 0.004),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="/home/lin/predify/kitti_step")
    p.add_argument("--fast-b-checkpoint", default=FAST_B_CHECKPOINT_DEFAULT)
    p.add_argument("--output", default=OUTPUT_DEFAULT)
    p.add_argument("--result-output", default=RESULT_DEFAULT)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--tbptt-steps", type=int, default=TBPTT_STEPS)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--lambda-semantic", type=float, default=LAMBDA_SEMANTIC)
    p.add_argument("--lambda-flow", type=float, default=LAMBDA_FLOW)
    p.add_argument("--max-displacement-low", type=float, default=MAX_DISPLACEMENT_LOW)
    p.add_argument("--max-train-steps", type=int, default=0)
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.epochs <= 0 or args.tbptt_steps <= 0 or args.max_displacement_low <= 0:
        raise ValueError("Invalid training arguments")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model = load_fast_b_model(args.fast_b_checkpoint).cuda().eval()
    model.requires_grad_(False)
    transport = CausalMotionTransportPredictor(
        num_classes=NUM_CLASSES,
        hidden_channels=HIDDEN_CHANNELS,
        max_displacement_low=args.max_displacement_low,
    ).cuda()
    optimizer = torch.optim.AdamW(
        transport.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    raft = FrozenRAFT()

    train = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "train")
    val = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "val")
    train_groups = sequence_groups(train)
    all_val = sequence_groups(val)
    missing = [s for s in FULL9 if s not in all_val]
    if missing:
        raise RuntimeError(f"Missing Full9 validation sequences: {missing}")
    val_groups = {s: all_val[s] for s in FULL9}
    output, result_output = Path(args.output), Path(args.result_output)
    output.mkdir(parents=True, exist_ok=True)
    result_output.mkdir(parents=True, exist_ok=True)
    zero_step = _zero_step_check(model, transport, next(iter(val_groups.values()))[0])

    history, best = [], None
    for epoch in range(1, args.epochs + 1):
        train_stats = _train_epoch(model, transport, raft, train_groups, optimizer, args)
        metrics = _evaluate(model, transport, val_groups, raft)
        checks = _checks(metrics)
        row = {
            "epoch": epoch,
            "train": train_stats,
            "metrics": metrics,
            "stage1_checks": checks,
        }
        history.append(row)
        payload = {
            "experiment": "c_v2_stage1_learned_motion_transport",
            "epoch": epoch,
            "transport_state_dict": transport.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "architecture": {
                "input": "observed Host probability + transport prediction error",
                "prediction": "next-frame backward flow at Host-C1 spatial resolution",
                "max_displacement_low_pixels": args.max_displacement_low,
                "invalid_flow_fallback": "same-coordinate previous Host logits",
                "residual_semantic_head": False,
                "fusion_gate": False,
                "raft_at_inference": False,
            },
            "row": row,
        }
        torch.save(payload, output / f"epoch_{epoch:03d}.pt")
        score = metrics["metrics"]["learned_transport"]["mIoU"]
        if best is None or score > best["score"]:
            best = {
                "epoch": epoch,
                "score": score,
                "metrics": metrics,
                "stage1_checks": checks,
            }
            torch.save(payload, output / "best.pt")
        (result_output / f"epoch_{epoch:03d}.json").write_text(
            json.dumps(row, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(row, sort_keys=True), flush=True)

    summary = {
        "experiment": "C-V2 Stage 1 Learned Motion Transport",
        "training": {
            "epochs": args.epochs,
            "tbptt_steps": args.tbptt_steps,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "lambda_semantic": args.lambda_semantic,
            "lambda_flow": args.lambda_flow,
            "max_displacement_low": args.max_displacement_low,
            "max_train_steps_per_sequence": args.max_train_steps,
        },
        "trainable_parameters": {
            "total": sum(x.numel() for x in transport.parameters()),
            "trainable": sum(x.numel() for x in transport.parameters() if x.requires_grad),
        },
        "frozen": [
            "Host",
            "Adapter",
            "Writeback",
            "Decoder",
            "all existing temporal modules",
        ],
        "raft_role": "training teacher, same-path teacher reference, and metric support only; absent from saved inference model",
        "zero_step": zero_step,
        "history": history,
        "best": best,
        "references": {
            "semantic_persistence": PERSISTENCE_REFERENCE,
            "legacy_fullres_raft_warp_noncomparable": LEGACY_FULLRES_RAFT_REFERENCE,
            "raft_low_same_path_from_best_eval": (
                best["metrics"]["metrics"]["raft_low_transport_reference"] if best else None
            ),
        },
        "decision": (
            "Proceed to C-V2 Stage 2 only if best.stage1_checks.stage1_minimum_go is true. "
            "The comparable teacher reference is raft_low_transport_reference; the older "
            "61.83% full-resolution RAFT result is diagnostic only and not a Stage-1 bound."
        ),
    }
    (result_output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "best_epoch": best["epoch"] if best else None,
                "best_learned_transport_mIoU": best["score"] if best else None,
                "stage1_minimum_go": (
                    best["stage1_checks"]["stage1_minimum_go"] if best else False
                ),
                "result": str(result_output / "summary.json"),
                "checkpoint": str(output / "best.pt"),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
