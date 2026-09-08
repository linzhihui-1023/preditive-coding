"""RMS-normalized runner for the C-V7 semantic decodability diagnostic.

中文：C-V7 语义可读出性判别实验的 RMS（均方根）归一化正式运行入口。

Why this runner exists / 目的
----------------------------
The three frozen probe sources (error_76 / evidence_211 / hidden_32) have
materially different numeric scales.  Comparing zero-init linear probes with
one optimizer/LR on raw features would mix semantic decodability with feature
scale/conditioning.  This runner therefore estimates one frozen per-channel
RMS from the TRAIN split before probe training and divides every feature channel
by that RMS in both training and Full9 evaluation.

Important guardrail / 重要边界
-----------------------------
- RMS statistics use only frozen feature values at low-resolution positions
  where C-V7 reports any valid history.
- No GT, Rescue mask, class label, or correct-history age is used to estimate RMS.
- Mean is NOT subtracted.  Mean subtraction plus a bias-free linear probe would
  create an effective constant offset (-W*mean/std), reintroducing the class-prior
  confound that the bias-free probe was designed to remove.
- Host / E1 / C-V3 / trained CE-only C-V7 remain frozen.
- Probe outputs still bypass the existing C-V7 Correction Head, Gate, tanh and
  g_max.  This is a diagnostic, not a deployable model candidate.
"""

import argparse
import json
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import sequence_groups
from predify2021.mce_scores import diagnose_c_v7_semantic_decodability_probes as base


OUTPUT_DEFAULT = (
    "/home/lin/predify/experiments/"
    "kitti_step_c_v7_semantic_decodability_probes_rms"
)
RESULT_DEFAULT = "results/kitti_step_c_v7_semantic_decodability_probes_rms"
RMS_EPS = 1e-6


def _feature_valid_low(frame):
    """Recover model-derived low-res any-history-valid from exact 211D evidence.

    C-V7 evidence concatenation ends with [valid_fraction, any_history_valid].
    This mask is model/history geometry only; GT is not involved.
    """
    evidence = frame["features"][base.PROBE_EVIDENCE_211]
    if evidence.ndim != 4 or evidence.shape[1] < 2:
        raise ValueError("evidence_211 must be BCHW with at least two scalar channels")
    any_valid = evidence[:, -1:, :, :]
    if not bool(torch.isfinite(any_valid).all().item()):
        raise FloatingPointError("non-finite any-history-valid evidence")
    return any_valid > 0.5


@torch.no_grad()
def _estimate_feature_rms(
    model,
    observer,
    residual,
    correction,
    mask_predictor,
    refiner,
    corrector,
    dynamics,
    groups,
):
    """Estimate per-channel RMS on train features without labels or Rescue masks."""
    channels = base._probe_channels(corrector)
    sumsq = {
        name: torch.zeros(channels[name], dtype=torch.float64)
        for name in base.PROBE_NAMES
    }
    counts = {name: 0 for name in base.PROBE_NAMES}
    frames = 0
    valid_positions = 0

    for samples in groups.values():
        for frame in base._frozen_sequence_frames(
            model,
            observer,
            residual,
            correction,
            mask_predictor,
            refiner,
            corrector,
            dynamics,
            samples,
        ):
            frames += 1
            valid = _feature_valid_low(frame)
            n_valid = int(valid.sum().item())
            if n_valid == 0:
                continue
            valid_positions += n_valid
            valid_float = valid.float()
            for name in base.PROBE_NAMES:
                feature = frame["features"][name].detach().float()
                if feature.shape[-2:] != valid.shape[-2:]:
                    raise RuntimeError(f"{name} spatial size differs from C-V7 validity")
                channel_sumsq = (
                    feature.square() * valid_float
                ).sum(dim=(0, 2, 3)).double().cpu()
                sumsq[name] += channel_sumsq
                counts[name] += n_valid

    if frames == 0 or valid_positions == 0:
        raise RuntimeError("no valid frozen feature positions for RMS calibration")

    rms = {}
    metadata = {
        "frames_seen": frames,
        "valid_low_positions": valid_positions,
        "per_probe": {},
    }
    for name in base.PROBE_NAMES:
        if counts[name] <= 0:
            raise RuntimeError(f"no RMS samples for {name}")
        mean_square = sumsq[name] / float(counts[name])
        raw_rms = mean_square.clamp_min(0.0).sqrt().float()
        near_zero = raw_rms <= RMS_EPS
        safe_rms = torch.where(near_zero, torch.ones_like(raw_rms), raw_rms)
        if not bool(torch.isfinite(safe_rms).all().item()):
            raise FloatingPointError(f"non-finite RMS for {name}")
        rms[name] = safe_rms
        metadata["per_probe"][name] = {
            "samples_per_channel": int(counts[name]),
            "channels": int(safe_rms.numel()),
            "near_zero_channels_set_to_one": int(near_zero.sum().item()),
            "rms_min_nonzero": float(
                raw_rms[~near_zero].min().item()
                if bool((~near_zero).any().item())
                else 0.0
            ),
            "rms_mean": float(raw_rms.mean().item()),
            "rms_max": float(raw_rms.max().item()),
        }
    return rms, metadata


def _normalize_feature(feature, rms):
    """Scale only; never subtract a mean, so zero remains exactly zero."""
    scale = rms.to(device=feature.device, dtype=feature.dtype).view(1, -1, 1, 1)
    if scale.shape[1] != feature.shape[1]:
        raise ValueError("RMS channel count does not match frozen feature")
    return feature / scale


def _train_probe_epoch_rms(
    probes,
    optimizer,
    rms,
    model,
    observer,
    residual,
    correction,
    mask_predictor,
    refiner,
    corrector,
    dynamics,
    groups,
):
    probes.train()
    corrector.eval()
    frames_seen = frames_with_rescue = optimizer_steps = rescue_pixels = 0
    loss_sum = {name: 0.0 for name in base.PROBE_NAMES}
    delta_abs_sum = {name: 0.0 for name in base.PROBE_NAMES}

    for samples in groups.values():
        for frame in base._frozen_sequence_frames(
            model,
            observer,
            residual,
            correction,
            mask_predictor,
            refiner,
            corrector,
            dynamics,
            samples,
        ):
            frames_seen += 1
            rescue = frame["masks"]["rescue"]
            n_rescue = int(rescue.sum().item())
            if n_rescue == 0:
                continue
            frames_with_rescue += 1
            rescue_pixels += n_rescue

            target = frame["gt_cpu"].to(
                frame["c_v3_logits"].device, non_blocking=True
            ).long().unsqueeze(0)
            full_size = tuple(frame["c_v3_logits"].shape[-2:])
            optimizer.zero_grad(set_to_none=True)
            for name in base.PROBE_NAMES:
                normalized = _normalize_feature(frame["features"][name], rms[name])
                delta_full = probes[name](
                    normalized,
                    full_size,
                    frame["any_valid_full"],
                )
                logits = frame["c_v3_logits"] + delta_full
                per_pixel = F.cross_entropy(
                    logits,
                    target,
                    ignore_index=base.c_v5.IGNORE_LABEL,
                    reduction="none",
                )[0]
                loss = per_pixel[rescue].mean()
                if not bool(torch.isfinite(loss).item()):
                    raise FloatingPointError(f"non-finite {name} probe loss")
                loss.backward()
                loss_sum[name] += float(loss.detach().item()) * n_rescue
                delta_abs_sum[name] += float(delta_full.detach().abs().mean().item())
            optimizer.step()
            optimizer_steps += 1

    rescue_den = max(rescue_pixels, 1)
    step_den = max(optimizer_steps, 1)
    return {
        "frames_seen": frames_seen,
        "frames_with_rescue": frames_with_rescue,
        "optimizer_steps": optimizer_steps,
        "rescue_pixels": rescue_pixels,
        "rescue_ce": {
            name: loss_sum[name] / rescue_den for name in base.PROBE_NAMES
        },
        "delta_abs": {
            name: delta_abs_sum[name] / step_den for name in base.PROBE_NAMES
        },
    }


@torch.inference_mode()
def _evaluate_probes_rms(
    probes,
    rms,
    model,
    observer,
    residual,
    correction,
    mask_predictor,
    refiner,
    corrector,
    dynamics,
    groups,
):
    probes.eval()
    confusion = {
        name: torch.zeros(
            (base.c_v5.NUM_CLASSES, base.c_v5.NUM_CLASSES), dtype=torch.int64
        )
        for name in base.ALL_NAMES
    }
    diag = {name: base._new_diag_totals() for name in base.ALL_NAMES}
    correction_frames = 0

    for sequence in base.c_v5.FULL9:
        samples = groups[sequence]
        for host_logits, gt_cpu in base._first_two_host_predictions(model, samples):
            host_pred_cpu = host_logits.argmax(1)[0].cpu()
            for name in base.ALL_NAMES:
                base.c_v5.update_confusion_matrix(confusion[name], host_pred_cpu, gt_cpu)

        for frame in base._frozen_sequence_frames(
            model,
            observer,
            residual,
            correction,
            mask_predictor,
            refiner,
            corrector,
            dynamics,
            samples,
        ):
            correction_frames += 1
            gt_cpu = frame["gt_cpu"]
            gt = gt_cpu.to(frame["c_v3_logits"].device, non_blocking=True).long()
            rescue = frame["masks"]["rescue"]
            protection = frame["masks"]["protection"]
            current_correct = frame["masks"]["current_correct"]
            n_rescue = int(rescue.sum().item())
            n_protect = int(protection.sum().item())
            n_correct = int(current_correct.sum().item())

            candidate_logits = {
                base.BASELINE_C_V3: frame["c_v3_logits"],
                base.BASELINE_C_V7: frame["c_v7_logits"],
            }
            full_size = tuple(frame["c_v3_logits"].shape[-2:])
            for name in base.PROBE_NAMES:
                normalized = _normalize_feature(frame["features"][name], rms[name])
                candidate_logits[name] = frame["c_v3_logits"] + probes[name](
                    normalized,
                    full_size,
                    frame["any_valid_full"],
                )

            current_margin = base._true_class_margin(frame["c_v3_logits"], gt_cpu)
            for name, logits in candidate_logits.items():
                pred = logits.argmax(1)[0]
                base.c_v5.update_confusion_matrix(confusion[name], pred.cpu(), gt_cpu)
                row = diag[name]
                row["rescue_pixels"] += n_rescue
                row["protection_pixels"] += n_protect
                row["current_correct_pixels"] += n_correct

                if n_rescue:
                    margin_gain = base._true_class_margin(logits, gt_cpu) - current_margin
                    row["rescue_recovered"] += int((rescue & (pred == gt)).sum().item())
                    row["rescue_margin_gain_sum"] += float(
                        margin_gain[rescue].sum().item()
                    )
                    row["rescue_positive_margin_gain"] += int(
                        (margin_gain[rescue] > 0.0).sum().item()
                    )
                if n_protect:
                    row["protection_damaged"] += int(
                        (protection & (pred != gt)).sum().item()
                    )
                if n_correct:
                    row["current_correct_damaged"] += int(
                        (current_correct & (pred != gt)).sum().item()
                    )

    metrics = {}
    for name in base.ALL_NAMES:
        row = diag[name]
        rescue_den = max(row["rescue_pixels"], 1)
        protect_den = max(row["protection_pixels"], 1)
        correct_den = max(row["current_correct_pixels"], 1)
        metrics[name] = {
            "mIoU": float(
                torch.nanmean(base.c_v5.compute_iou(confusion[name])).item()
            ),
            "rescue_pixels": row["rescue_pixels"],
            "rescue_recovery_rate": row["rescue_recovered"] / rescue_den,
            "rescue_margin_gain": row["rescue_margin_gain_sum"] / rescue_den,
            "rescue_positive_margin_gain_fraction": (
                row["rescue_positive_margin_gain"] / rescue_den
            ),
            "protection_pixels": row["protection_pixels"],
            "protection_damage_rate": row["protection_damaged"] / protect_den,
            "current_correct_pixels": row["current_correct_pixels"],
            "current_correct_damage_rate": row["current_correct_damaged"] / correct_den,
        }

    return {
        "correction_path_frames": correction_frames,
        "metrics": metrics,
        "delta_mIoU_vs_c_v3": {
            name: metrics[name]["mIoU"] - metrics[base.BASELINE_C_V3]["mIoU"]
            for name in base.ALL_NAMES
        },
        "delta_rescue_recovery_vs_existing_c_v7": {
            name: (
                metrics[name]["rescue_recovery_rate"]
                - metrics[base.BASELINE_C_V7]["rescue_recovery_rate"]
            )
            for name in base.PROBE_NAMES
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/home/lin/predify/kitti_step")
    parser.add_argument(
        "--fast-b-checkpoint", default=base.c_v5.FAST_B_CHECKPOINT_DEFAULT
    )
    parser.add_argument(
        "--observer-checkpoint", default=base.c_v5.OBSERVER_CHECKPOINT_DEFAULT
    )
    parser.add_argument(
        "--residual-checkpoint", default=base.c_v5.RESIDUAL_CHECKPOINT_DEFAULT
    )
    parser.add_argument("--base-checkpoint", default=base.c_v5.BASE_CHECKPOINT_DEFAULT)
    parser.add_argument("--c-v3-checkpoint", default=base.c_v5.C_V3_CHECKPOINT_DEFAULT)
    parser.add_argument("--c-v7-checkpoint", default=base.C_V7_CE_CHECKPOINT_DEFAULT)
    parser.add_argument("--output", default=OUTPUT_DEFAULT)
    parser.add_argument("--result-output", default=RESULT_DEFAULT)
    parser.add_argument("--epochs", type=int, default=base.EPOCHS)
    parser.add_argument("--lr", type=float, default=base.LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=base.WEIGHT_DECAY)
    parser.add_argument("--seed", type=int, default=base.SEED)
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.epochs < 1:
        raise ValueError("epochs must be >= 1")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    (
        model,
        observer,
        residual,
        correction,
        mask_predictor,
        refiner,
        corrector,
        dynamics,
        c_v7_payload,
    ) = base._load_frozen_stack(args)

    train = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "train")
    val = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "val")
    train_groups = sequence_groups(train)
    all_val_groups = sequence_groups(val)
    missing = [
        sequence for sequence in base.c_v5.FULL9 if sequence not in all_val_groups
    ]
    if missing:
        raise RuntimeError(f"missing Full9 validation sequences: {missing}")
    val_groups = {
        sequence: all_val_groups[sequence] for sequence in base.c_v5.FULL9
    }

    # Calibration is a frozen no-label pass over train features.
    rms, rms_metadata = _estimate_feature_rms(
        model,
        observer,
        residual,
        correction,
        mask_predictor,
        refiner,
        corrector,
        dynamics,
        train_groups,
    )

    probes, channels = base._build_probes(corrector, torch.device("cuda"))
    zero_step = base._zero_step_contract(probes, channels)
    optimizer = torch.optim.AdamW(
        probes.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    output = Path(args.output)
    result_output = Path(args.result_output)
    output.mkdir(parents=True, exist_ok=True)
    result_output.mkdir(parents=True, exist_ok=True)

    history = []
    for epoch in range(1, args.epochs + 1):
        train_stats = _train_probe_epoch_rms(
            probes,
            optimizer,
            rms,
            model,
            observer,
            residual,
            correction,
            mask_predictor,
            refiner,
            corrector,
            dynamics,
            train_groups,
        )
        eval_stats = _evaluate_probes_rms(
            probes,
            rms,
            model,
            observer,
            residual,
            correction,
            mask_predictor,
            refiner,
            corrector,
            dynamics,
            val_groups,
        )
        row = {"epoch": epoch, "train": train_stats, "full9": eval_stats}
        history.append(row)
        with (result_output / f"epoch_{epoch:03d}.json").open("w") as handle:
            json.dump(row, handle, indent=2)
        torch.save(
            {
                "experiment": "c_v7_semantic_decodability_probes_rms",
                "epoch": epoch,
                "probe_state_dict": probes.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "probe_channels": channels,
                "feature_rms": {name: value.cpu() for name, value in rms.items()},
                "feature_rms_metadata": rms_metadata,
                "source_c_v7_checkpoint": args.c_v7_checkpoint,
                "source_c_v7_epoch": c_v7_payload.get("epoch"),
                "zero_step": zero_step,
                "train": train_stats,
                "full9": eval_stats,
            },
            output / f"epoch_{epoch:03d}.pt",
        )
        print(json.dumps(row, indent=2), flush=True)

    summary = {
        "experiment": "C-V7 frozen semantic decodability probes with per-channel RMS scaling",
        "diagnostic_only": True,
        "source_checkpoint": {
            "path": args.c_v7_checkpoint,
            "experiment": c_v7_payload.get("experiment"),
            "epoch": c_v7_payload.get("epoch"),
        },
        "feature_scaling": {
            "type": "train-split per-channel RMS, scale-only",
            "mean_subtraction": False,
            "gt_used_for_scaling": False,
            "validity_source": "C-V7 any_history_valid low-resolution evidence",
            "eps": RMS_EPS,
            "metadata": rms_metadata,
        },
        "frozen": {
            "host": True,
            "e1": True,
            "c_v3": True,
            "c_v7_corrector": True,
            "c_v7_gate_bypassed_by_probes": True,
            "c_v7_correction_head_bypassed_by_probes": True,
            "tanh_bypassed_by_probes": True,
            "g_max_bypassed_by_probes": True,
        },
        "probe_definitions": {
            base.PROBE_ERROR_76: "RMS-scaled concat e1..e4 -> bias-free zero-init 1x1 Conv -> 19D DeltaZ",
            base.PROBE_EVIDENCE_211: "RMS-scaled exact C-V7 Error Evidence -> bias-free zero-init 1x1 Conv -> 19D DeltaZ",
            base.PROBE_HIDDEN_32: "RMS-scaled trained C-V7 H_err -> bias-free zero-init 1x1 Conv -> 19D DeltaZ",
        },
        "training_target": (
            "Rescue-only CE: C-V3 wrong AND at least one valid aligned history predicts GT; "
            "no correct-history age target"
        ),
        "zero_step": zero_step,
        "history": history,
        "interpretation_guardrails": [
            "Probe success means linearly decodable semantic correction information is present after scale-only RMS normalization.",
            "Probe failure does not prove information absence; it rejects only this normalized minimal linear readout under the present optimization protocol.",
            "RMS scaling uses no GT and subtracts no mean, preventing an implicit constant class correction.",
            "Probe heads are bias-free so Rescue class priors cannot create a feature-independent correction.",
            "This experiment bypasses Gate/tanh/g_max and is not a deployable model evaluation.",
            "No 32D bottleneck conclusion is allowed unless error/evidence probes separate clearly from hidden_32.",
        ],
    }
    with (result_output / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
