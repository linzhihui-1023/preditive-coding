"""Controlled C-V7 semantic decodability diagnostic.

中文：带空白对照与尺度控制的 C-V7 语义可读出性判别实验。

Formal diagnostic / 正式诊断入口
--------------------------------
Freeze Host / E1 / C-V3 / trained CE-only C-V7 and compare four bias-free,
zero-initialized 1x1 residual probes under the same Rescue-only CE objective:

  constant_valid_1 : model-derived any-history-valid only; null/class-prior control
  error_76         : concat(e1,e2,e3,e4), RMS-scaled
  evidence_211     : exact C-V7 pre-ConvGRU Error Evidence, RMS-scaled
  hidden_32        : trained recurrent Error Memory H_err, RMS-scaled

For every probe:

  feature -> bias-free 1x1 Conv -> DeltaZ_probe
  Z_probe = detach(Z_C-V3) + DeltaZ_probe

The existing C-V7 Correction Head, Gate, tanh and g_max are bypassed.
GT is used only to define training/evaluation roles:

  Rescue = C-V3 wrong AND at least one valid aligned history predicts GT

No correct-history age is supplied to any probe.

Why constant_valid_1 is required / 为什么必须有空白对照
------------------------------------------------------
The exact 211D C-V7 evidence ends with any_history_valid.  On every Rescue pixel
that channel is necessarily 1, so even a bias-free probe can use it as an
implicit constant class correction.  constant_valid_1 measures that null effect.
A semantic representation probe should therefore be interpreted relative to
this control, not only by its absolute Rescue recovery.

Why RMS scaling is required / 为什么做 RMS 缩放
-----------------------------------------------
76D / 211D / 32D frozen sources have different numeric scales.  Per-channel RMS
is estimated on TRAIN features at model-derived valid-history low-res positions.
No GT/Rescue/class label enters this calibration.  Mean is never subtracted, so
scale normalization cannot create an implicit constant offset.

Interpretation / 解释边界
------------------------
Probe success means linearly decodable correction information exists under this
controlled readout.  Probe failure does not prove information absence.  A 32D
bottleneck claim is allowed only if error/evidence probes separate clearly from
hidden_32 beyond the null control while optimization is otherwise comparable.
"""

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import (
    sequence_groups,
)
from predify2021.mce_scores import diagnose_c_v7_semantic_decodability_probes as base
from predify2021.mce_scores import diagnose_c_v7_semantic_decodability_probes_rms as rms_base


CONTROL_CONSTANT = "constant_valid_1"
PROBE_NAMES = (CONTROL_CONSTANT, *base.PROBE_NAMES)
ALL_NAMES = (base.BASELINE_C_V3, base.BASELINE_C_V7, *PROBE_NAMES)

OUTPUT_DEFAULT = (
    "/home/lin/predify/experiments/"
    "kitti_step_c_v7_semantic_decodability_probes_controlled"
)
RESULT_DEFAULT = "results/kitti_step_c_v7_semantic_decodability_probes_controlled"


def _build_controlled_probes(corrector, device):
    probes, channels = base._build_probes(corrector, device)
    probes[CONTROL_CONSTANT] = base.LinearResidualProbe(1, base.c_v5.NUM_CLASSES).to(device)
    channels = dict(channels)
    channels[CONTROL_CONSTANT] = 1
    return probes, channels


def _controlled_zero_step(probes, channels):
    rows = base._zero_step_contract(probes, channels)
    probe = probes[CONTROL_CONSTANT]
    if probe.head.bias is not None:
        raise RuntimeError("constant control must be bias-free")
    weight_max = float(probe.head.weight.detach().abs().max().item())
    if weight_max != 0.0:
        raise RuntimeError("constant control must be exactly zero initialized")
    rows[CONTROL_CONSTANT] = {
        "in_channels": 1,
        "trainable_parameters": sum(
            p.numel() for p in probe.parameters() if p.requires_grad
        ),
        "weight_abs_max": weight_max,
        "bias": False,
        "zero_step_equals_c_v3": True,
    }
    return rows


def _probe_feature(frame, name, feature_rms):
    if name == CONTROL_CONSTANT:
        # On Rescue pixels this is exactly 1 by construction; it is the null control.
        return rms_base._feature_valid_low(frame).to(
            dtype=frame["c_v3_logits"].dtype
        )
    return rms_base._normalize_feature(
        frame["features"][name], feature_rms[name]
    )


def _train_epoch(
    probes,
    optimizer,
    feature_rms,
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
    frames_seen = 0
    frames_with_rescue = 0
    optimizer_steps = 0
    rescue_pixels = 0
    loss_sum = {name: 0.0 for name in PROBE_NAMES}
    delta_abs_sum = {name: 0.0 for name in PROBE_NAMES}
    weight_norm_start = {
        name: float(probes[name].head.weight.detach().norm().item())
        for name in PROBE_NAMES
    }

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

            for name in PROBE_NAMES:
                feature = _probe_feature(frame, name, feature_rms)
                delta_full = probes[name](
                    feature,
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
                # Graphs and parameters are disjoint across probes; frozen inputs are detached.
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
            name: loss_sum[name] / rescue_den for name in PROBE_NAMES
        },
        "delta_abs": {
            name: delta_abs_sum[name] / step_den for name in PROBE_NAMES
        },
        "weight_norm_start": weight_norm_start,
        "weight_norm_end": {
            name: float(probes[name].head.weight.detach().norm().item())
            for name in PROBE_NAMES
        },
    }


@torch.inference_mode()
def _evaluate(
    probes,
    feature_rms,
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
        for name in ALL_NAMES
    }
    diag = {name: base._new_diag_totals() for name in ALL_NAMES}
    correction_frames = 0

    for sequence in base.c_v5.FULL9:
        samples = groups[sequence]
        # Match the existing C-V7 Full9 protocol exactly for the first two frames.
        for host_logits, gt_cpu in base._first_two_host_predictions(model, samples):
            host_pred_cpu = host_logits.argmax(1)[0].cpu()
            for name in ALL_NAMES:
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
            for name in PROBE_NAMES:
                feature = _probe_feature(frame, name, feature_rms)
                candidate_logits[name] = frame["c_v3_logits"] + probes[name](
                    feature,
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
    for name in ALL_NAMES:
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

    null_rescue = metrics[CONTROL_CONSTANT]["rescue_recovery_rate"]
    null_margin = metrics[CONTROL_CONSTANT]["rescue_margin_gain"]
    return {
        "correction_path_frames": correction_frames,
        "metrics": metrics,
        "delta_mIoU_vs_c_v3": {
            name: metrics[name]["mIoU"] - metrics[base.BASELINE_C_V3]["mIoU"]
            for name in ALL_NAMES
        },
        "delta_rescue_recovery_vs_existing_c_v7": {
            name: (
                metrics[name]["rescue_recovery_rate"]
                - metrics[base.BASELINE_C_V7]["rescue_recovery_rate"]
            )
            for name in PROBE_NAMES
        },
        "semantic_probe_gain_over_null": {
            name: {
                "rescue_recovery_rate": (
                    metrics[name]["rescue_recovery_rate"] - null_rescue
                ),
                "rescue_margin_gain": metrics[name]["rescue_margin_gain"] - null_margin,
            }
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

    # One frozen, label-free calibration pass.  Only the three semantic sources need RMS.
    feature_rms, rms_metadata = rms_base._estimate_feature_rms(
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

    device = torch.device("cuda")
    probes, channels = _build_controlled_probes(corrector, device)
    zero_step = _controlled_zero_step(probes, channels)
    optimizer = torch.optim.AdamW(
        probes.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    output = Path(args.output)
    result_output = Path(args.result_output)
    output.mkdir(parents=True, exist_ok=True)
    result_output.mkdir(parents=True, exist_ok=True)

    history = []
    for epoch in range(1, args.epochs + 1):
        train_stats = _train_epoch(
            probes,
            optimizer,
            feature_rms,
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
        eval_stats = _evaluate(
            probes,
            feature_rms,
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
                "experiment": "c_v7_semantic_decodability_probes_controlled",
                "epoch": epoch,
                "probe_state_dict": probes.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "probe_channels": channels,
                "feature_rms": {
                    name: value.cpu() for name, value in feature_rms.items()
                },
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
        "experiment": "C-V7 controlled frozen semantic decodability probes",
        "diagnostic_only": True,
        "formal_runner": True,
        "source_checkpoint": {
            "path": args.c_v7_checkpoint,
            "experiment": c_v7_payload.get("experiment"),
            "epoch": c_v7_payload.get("epoch"),
        },
        "controls": {
            "constant_valid_1": (
                "bias-free 1x1 probe on model-derived any-history-valid; null control for "
                "feature-independent Rescue class correction"
            ),
            "per_channel_rms_scale": True,
            "mean_subtraction": False,
            "gt_used_for_rms": False,
            "rms_metadata": rms_metadata,
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
            CONTROL_CONSTANT: "any_history_valid -> bias-free zero-init 1x1 Conv -> 19D DeltaZ",
            base.PROBE_ERROR_76: "RMS-scaled concat e1..e4 -> bias-free zero-init 1x1 Conv -> 19D DeltaZ",
            base.PROBE_EVIDENCE_211: "RMS-scaled exact C-V7 Error Evidence -> bias-free zero-init 1x1 Conv -> 19D DeltaZ",
            base.PROBE_HIDDEN_32: "RMS-scaled trained C-V7 H_err -> bias-free zero-init 1x1 Conv -> 19D DeltaZ",
        },
        "training_target": (
            "Rescue-only CE: C-V3 wrong AND at least one valid aligned history predicts GT; "
            "no correct-history age target"
        ),
        "primary_diagnostic_comparison": (
            "semantic_probe_gain_over_null on Rescue recovery and true-class margin; "
            "mIoU/current-correct damage are secondary safety diagnostics"
        ),
        "zero_step": zero_step,
        "history": history,
        "interpretation_guardrails": [
            "A semantic probe must first outperform constant_valid_1 before being treated as evidence of representation-specific decodability.",
            "Probe success means linearly decodable correction information exists under this controlled readout.",
            "Probe failure does not prove information absence; it rejects only this controlled minimal linear readout under the present optimization protocol.",
            "RMS scaling is label-free and subtracts no mean, so it cannot introduce a constant class correction.",
            "This experiment bypasses Gate/tanh/g_max and is not a deployable model evaluation.",
            "No 32D bottleneck conclusion is allowed unless error/evidence probes separate clearly from hidden_32 beyond the null control.",
            "One training trajectory is a diagnostic comparison, not an independent-repeat statistical significance claim.",
        ],
    }
    with (result_output / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
