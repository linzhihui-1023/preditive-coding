"""C-V7 frozen semantic-decoding diagnostic with minimal linear probes.

中文：C-V7 冻结语义可读出性判别实验。

This is diagnostic-only. Host / E1 / C-V3 / trained CE-only C-V7 are frozen.
Three representations are compared with the same zero-initialized, bias-free
1x1 linear residual readout:

  error_76      = concat(e1,e2,e3,e4), 76 channels
  evidence_211  = exact C-V7 pre-ConvGRU Error Evidence, 211 channels
  hidden_32     = trained recurrent Error Memory H_err, 32 channels

For every probe:

  feature -> bias-free 1x1 Conv -> DeltaZ_probe
  Z_probe = detach(Z_C-V3) + DeltaZ_probe

The existing C-V7 Correction Head, Gate, tanh, and g_max are bypassed. Probe
training uses GT only to define diagnostic Rescue pixels:

  Rescue = C-V3 wrong AND at least one valid aligned history predicts GT
  L_probe = CE(Z_probe, GT) on Rescue pixels only

No correct-history age target is supplied. Probe success means linearly
decodable semantic correction information is present. Probe failure does not
prove information absence; it rejects only this minimal linear readout.
"""

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import sequence_groups
from predify2021.mce_scores.export_kitti_step_full9_predictions_c_v7 import (
    DEFAULT_MODEL_CHECKPOINT as C_V7_CE_CHECKPOINT_DEFAULT,
    _build_dynamics,
    _load_corrector,
)
from predify2021.mce_scores import (
    train_kitti_step_task_space_prior_c_v5_multiframe_candidate_memory as c_v5,
)
from predify2021.mce_scores import (
    train_kitti_step_task_space_prior_c_v7_error_residual_correction as c_v7,
)

SEED = 0
EPOCHS = 1
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
OUTPUT_DEFAULT = "/home/lin/predify/experiments/kitti_step_c_v7_semantic_decodability_probes"
RESULT_DEFAULT = "results/kitti_step_c_v7_semantic_decodability_probes"

PROBE_ERROR_76 = "error_76"
PROBE_EVIDENCE_211 = "evidence_211"
PROBE_HIDDEN_32 = "hidden_32"
PROBE_NAMES = (PROBE_ERROR_76, PROBE_EVIDENCE_211, PROBE_HIDDEN_32)
BASELINE_C_V3 = "c_v3_base"
BASELINE_C_V7 = "c_v7_existing"
ALL_NAMES = (BASELINE_C_V3, BASELINE_C_V7, *PROBE_NAMES)


class LinearResidualProbe(nn.Module):
    """Bias-free zero-initialized 1x1 linear readout into 19-class DeltaZ."""

    def __init__(self, in_channels, num_classes=19):
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        # No bias: a global Rescue-class prior must not masquerade as feature decoding.
        self.head = nn.Conv2d(self.in_channels, self.num_classes, 1, bias=False)
        nn.init.zeros_(self.head.weight)

    def forward(self, feature_low, full_size, any_valid_full):
        if feature_low.ndim != 4 or feature_low.shape[1] != self.in_channels:
            raise ValueError(
                f"probe expected Bx{self.in_channels}xHxW, got {tuple(feature_low.shape)}"
            )
        delta_low = self.head(feature_low)
        delta_full = F.interpolate(
            delta_low,
            size=tuple(full_size),
            mode="bilinear",
            align_corners=False,
        )
        if any_valid_full.ndim != 4 or any_valid_full.shape[1] != 1:
            raise ValueError("any_valid_full must be Bx1xHxW")
        return delta_full * any_valid_full.to(delta_full.dtype)


def _probe_channels(corrector):
    evidence_channels = int(corrector.error_pre[0].in_channels)
    expected_evidence = (
        2 * c_v5.NUM_CLASSES * corrector.history_length
        + 3 * c_v5.NUM_CLASSES
        + 2
    )
    if evidence_channels != expected_evidence:
        raise RuntimeError(
            f"unexpected C-V7 evidence channels: {evidence_channels} != {expected_evidence}"
        )
    return {
        PROBE_ERROR_76: c_v5.NUM_CLASSES * corrector.history_length,
        PROBE_EVIDENCE_211: evidence_channels,
        PROBE_HIDDEN_32: corrector.hidden_channels,
    }


def _build_probes(corrector, device):
    channels = _probe_channels(corrector)
    probes = nn.ModuleDict(
        {
            name: LinearResidualProbe(channels[name], c_v5.NUM_CLASSES)
            for name in PROBE_NAMES
        }
    ).to(device)
    return probes, channels


@torch.no_grad()
def _build_diagnostic_masks(c_v3_logits, candidate_rows, current_gt_cpu):
    """GT defines diagnostic roles only; no history age is selected."""
    gt = current_gt_cpu.to(c_v3_logits.device, non_blocking=True).long()
    valid_gt = gt != c_v5.IGNORE_LABEL
    current_pred = c_v3_logits.detach().argmax(1)[0]
    current_correct = valid_gt & (current_pred == gt)

    any_history_valid = torch.zeros_like(valid_gt)
    history_can_rescue = torch.zeros_like(valid_gt)
    history_conflict = torch.zeros_like(valid_gt)
    for row in candidate_rows:
        valid_history = row["valid_full"][0].bool() & valid_gt
        history_pred = row["logits"].detach().argmax(1)[0]
        any_history_valid |= valid_history
        history_can_rescue |= valid_history & (history_pred == gt)
        history_conflict |= valid_history & (history_pred != current_pred)

    return {
        "valid_gt": valid_gt,
        "current_correct": current_correct,
        "any_history_valid": any_history_valid,
        "history_can_rescue": history_can_rescue,
        "history_conflict": history_conflict,
        "rescue": valid_gt & (~current_correct) & history_can_rescue,
        "protection": valid_gt & current_correct & history_conflict,
    }


def _true_class_margin(logits, gt_cpu):
    gt = gt_cpu.to(logits.device, non_blocking=True).long()
    safe_gt = gt.clamp(0, c_v5.NUM_CLASSES - 1)
    index = safe_gt.unsqueeze(0).unsqueeze(0)
    true_logit = logits.gather(1, index)[:, 0]
    other_logits = logits.clone()
    other_logits.scatter_(1, index, float("-inf"))
    return (true_logit - other_logits.max(dim=1).values)[0]


def _first_two_host_predictions(model, samples):
    rows = []
    for sample in samples[:2]:
        with torch.no_grad():
            _, host_logits, _, _, _ = c_v5._host_observation(model, sample)
        gt = c_v5.semantic_mask_from_panoptic_png(sample["mask_path"])
        rows.append((host_logits.detach(), gt))
    return rows


def _frozen_sequence_frames(
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
    """Yield detached C-V7 internal representations for frames t>=2."""
    if len(samples) < 3:
        return

    with torch.no_grad():
        frame0 = c_v5._host_observation(model, samples[0])
        frame1 = c_v5._host_observation(model, samples[1])
        pending_motion, motion_hidden = c_v5._initialize_motion(
            observer, residual, frame0, frame1
        )
        _, previous_host_logits, previous_low, previous_c1, _ = frame1

    raw_history = [previous_host_logits.detach()]
    motion_history = []
    transport_hidden = semantic_hidden = mask_hidden = None
    semantic_state_low = torch.zeros_like(previous_low)
    memory_state = None
    error_hidden = None
    dynamics_state = None

    for frame_index in range(2, len(samples)):
        sample = samples[frame_index]
        with torch.no_grad():
            _, host_logits, host_low, current_c1, output_size = c_v5._host_observation(
                model, sample
            )
            current_gt = c_v5.semantic_mask_from_panoptic_png(sample["mask_path"])
            prior_low, _ = c_v5.warp_low_logits(previous_low, pending_motion)
            e1 = c_v5._frozen_e1_step(
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

            memory_row, memory_state, _, c_v3_logits = c_v5._frozen_cv3_step(
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
            candidate_rows = c_v5._build_history_candidates(
                raw_history,
                motion_history,
                pending_motion,
                corrector.history_length,
            )
            evidence = c_v7._correction_evidence(
                corrector,
                dynamics,
                c_v3_logits,
                candidate_rows,
                pending_motion,
                e1["transportability_low"],
                memory_row["memory_reliability"],
                error_hidden,
                dynamics_state,
            )
            error_hidden = evidence["row"]["hidden"].detach()
            dynamics_state = evidence["dynamics_state"].detach()
            masks = _build_diagnostic_masks(c_v3_logits, candidate_rows, current_gt)

            features = {
                PROBE_ERROR_76: torch.cat(
                    [error.detach() for error in evidence["prediction_errors"]], dim=1
                ),
                PROBE_EVIDENCE_211: evidence["row"]["evidence"].detach(),
                PROBE_HIDDEN_32: evidence["row"]["hidden"].detach(),
            }
            expected_channels = _probe_channels(corrector)
            for name in PROBE_NAMES:
                if features[name].shape[1] != expected_channels[name]:
                    raise RuntimeError(
                        f"{name} feature channels {features[name].shape[1]} "
                        f"!= expected {expected_channels[name]}"
                    )

            observed_motion = c_v5._observe_motion(
                observer, previous_low, previous_c1, host_low, current_c1
            )
            motion_error = F.softmax(host_low, dim=1) - F.softmax(prior_low, dim=1)
            next_motion, _, next_motion_hidden = residual.predict_next(
                observed_motion, motion_error, motion_hidden
            )

            frame_row = {
                "sequence_id": str(sample["sequence_id"]),
                "frame_id": str(sample["frame_id"]),
                "gt_cpu": current_gt,
                "c_v3_logits": c_v3_logits.detach(),
                "c_v7_logits": evidence["final_logits_full"].detach(),
                # Preserve the two pre/post bounded-correction tensors for
                # frozen decomposition diagnostics.  They are detached and
                # do not alter the existing probe or evaluation path.
                "delta_z_raw_low": evidence["row"]["delta_z_raw"].detach(),
                "gate_low": evidence["row"]["gate"].detach(),
                "any_valid_full": evidence["any_valid_full"].detach(),
                "masks": {key: value.detach() for key, value in masks.items()},
                "features": features,
            }

            raw_history.insert(0, c_v3_logits.detach())
            raw_history = raw_history[: corrector.history_length]
            motion_history.insert(0, pending_motion.detach())
            motion_history = motion_history[: max(corrector.history_length - 1, 0)]
            previous_low = host_low.detach()
            previous_c1 = current_c1.detach()
            pending_motion = next_motion.detach()
            motion_hidden = next_motion_hidden.detach()
            semantic_state_low = semantic_state_low.detach()
            if memory_state is not None:
                memory_state = memory_state.detach()

        yield frame_row


def _train_probe_epoch(
    probes,
    optimizer,
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
    loss_sum = {name: 0.0 for name in PROBE_NAMES}
    delta_abs_sum = {name: 0.0 for name in PROBE_NAMES}

    for samples in groups.values():
        for frame in _frozen_sequence_frames(
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
                delta_full = probes[name](
                    frame["features"][name], full_size, frame["any_valid_full"]
                )
                logits = frame["c_v3_logits"] + delta_full
                per_pixel = F.cross_entropy(
                    logits,
                    target,
                    ignore_index=c_v5.IGNORE_LABEL,
                    reduction="none",
                )[0]
                loss = per_pixel[rescue].mean()
                if not bool(torch.isfinite(loss).item()):
                    raise FloatingPointError(f"non-finite {name} probe loss")
                # Each probe has disjoint parameters; all frozen features are detached.
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
        "rescue_ce": {name: loss_sum[name] / rescue_den for name in PROBE_NAMES},
        "delta_abs": {name: delta_abs_sum[name] / step_den for name in PROBE_NAMES},
    }


def _new_diag_totals():
    return {
        "rescue_pixels": 0,
        "protection_pixels": 0,
        "current_correct_pixels": 0,
        "rescue_recovered": 0,
        "rescue_margin_gain_sum": 0.0,
        "rescue_positive_margin_gain": 0,
        "protection_damaged": 0,
        "current_correct_damaged": 0,
    }


@torch.inference_mode()
def _evaluate_probes(
    probes,
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
        name: torch.zeros((c_v5.NUM_CLASSES, c_v5.NUM_CLASSES), dtype=torch.int64)
        for name in ALL_NAMES
    }
    diag = {name: _new_diag_totals() for name in ALL_NAMES}
    correction_frames = 0

    for sequence in c_v5.FULL9:
        samples = groups[sequence]
        # Existing protocol: first two frames fall back to Host for all candidates.
        for host_logits, gt_cpu in _first_two_host_predictions(model, samples):
            host_pred_cpu = host_logits.argmax(1)[0].cpu()
            for name in ALL_NAMES:
                c_v5.update_confusion_matrix(confusion[name], host_pred_cpu, gt_cpu)

        for frame in _frozen_sequence_frames(
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
                BASELINE_C_V3: frame["c_v3_logits"],
                BASELINE_C_V7: frame["c_v7_logits"],
            }
            full_size = tuple(frame["c_v3_logits"].shape[-2:])
            for name in PROBE_NAMES:
                candidate_logits[name] = frame["c_v3_logits"] + probes[name](
                    frame["features"][name], full_size, frame["any_valid_full"]
                )

            current_margin = _true_class_margin(frame["c_v3_logits"], gt_cpu)
            for name, logits in candidate_logits.items():
                pred = logits.argmax(1)[0]
                c_v5.update_confusion_matrix(confusion[name], pred.cpu(), gt_cpu)
                row = diag[name]
                row["rescue_pixels"] += n_rescue
                row["protection_pixels"] += n_protect
                row["current_correct_pixels"] += n_correct

                if n_rescue:
                    margin_gain = _true_class_margin(logits, gt_cpu) - current_margin
                    row["rescue_recovered"] += int((rescue & (pred == gt)).sum().item())
                    row["rescue_margin_gain_sum"] += float(margin_gain[rescue].sum().item())
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
            "mIoU": float(torch.nanmean(c_v5.compute_iou(confusion[name])).item()),
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
            name: metrics[name]["mIoU"] - metrics[BASELINE_C_V3]["mIoU"]
            for name in ALL_NAMES
        },
        "delta_rescue_recovery_vs_existing_c_v7": {
            name: (
                metrics[name]["rescue_recovery_rate"]
                - metrics[BASELINE_C_V7]["rescue_recovery_rate"]
            )
            for name in PROBE_NAMES
        },
    }


def _zero_step_contract(probes, channels):
    rows = {}
    for name in PROBE_NAMES:
        probe = probes[name]
        weight_max = float(probe.head.weight.detach().abs().max().item())
        if probe.head.bias is not None:
            raise RuntimeError(f"{name} probe must be bias-free")
        if weight_max != 0.0:
            raise RuntimeError(f"{name} probe must be exactly zero initialized")
        rows[name] = {
            "in_channels": channels[name],
            "trainable_parameters": sum(
                parameter.numel() for parameter in probe.parameters() if parameter.requires_grad
            ),
            "weight_abs_max": weight_max,
            "bias": False,
            "zero_step_equals_c_v3": True,
        }
    return rows


def _load_frozen_stack(args):
    device = torch.device("cuda")
    c_v5._validate_bounded_motion_checkpoint(args.residual_checkpoint)
    model = c_v5.load_fast_b_model(args.fast_b_checkpoint).cuda().eval()
    model.requires_grad_(False)
    observer, _ = c_v5._load_frozen_observer(args.observer_checkpoint)
    residual, _ = c_v5._load_frozen_residual(args.residual_checkpoint, observer)
    correction, mask_predictor, _ = c_v5._load_frozen_e1_base(
        args.base_checkpoint, observer
    )
    refiner, _ = c_v5._load_frozen_cv3_refiner(args.c_v3_checkpoint)
    corrector, payload = _load_corrector(args.c_v7_checkpoint, device)
    dynamics = _build_dynamics(payload)
    for module in (observer, residual, correction, mask_predictor, refiner, corrector):
        if isinstance(module, nn.Module):
            module.eval()
            module.requires_grad_(False)
    return (
        model,
        observer,
        residual,
        correction,
        mask_predictor,
        refiner,
        corrector,
        dynamics,
        payload,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/home/lin/predify/kitti_step")
    parser.add_argument("--fast-b-checkpoint", default=c_v5.FAST_B_CHECKPOINT_DEFAULT)
    parser.add_argument("--observer-checkpoint", default=c_v5.OBSERVER_CHECKPOINT_DEFAULT)
    parser.add_argument("--residual-checkpoint", default=c_v5.RESIDUAL_CHECKPOINT_DEFAULT)
    parser.add_argument("--base-checkpoint", default=c_v5.BASE_CHECKPOINT_DEFAULT)
    parser.add_argument("--c-v3-checkpoint", default=c_v5.C_V3_CHECKPOINT_DEFAULT)
    parser.add_argument("--c-v7-checkpoint", default=C_V7_CE_CHECKPOINT_DEFAULT)
    parser.add_argument("--output", default=OUTPUT_DEFAULT)
    parser.add_argument("--result-output", default=RESULT_DEFAULT)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--seed", type=int, default=SEED)
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
    ) = _load_frozen_stack(args)

    probes, channels = _build_probes(corrector, torch.device("cuda"))
    zero_step = _zero_step_contract(probes, channels)
    optimizer = torch.optim.AdamW(
        probes.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    train = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "train")
    val = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "val")
    train_groups = sequence_groups(train)
    all_val_groups = sequence_groups(val)
    missing = [sequence for sequence in c_v5.FULL9 if sequence not in all_val_groups]
    if missing:
        raise RuntimeError(f"missing Full9 validation sequences: {missing}")
    val_groups = {sequence: all_val_groups[sequence] for sequence in c_v5.FULL9}

    output = Path(args.output)
    result_output = Path(args.result_output)
    output.mkdir(parents=True, exist_ok=True)
    result_output.mkdir(parents=True, exist_ok=True)

    history = []
    for epoch in range(1, args.epochs + 1):
        train_stats = _train_probe_epoch(
            probes,
            optimizer,
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
        eval_stats = _evaluate_probes(
            probes,
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
                "experiment": "c_v7_semantic_decodability_probes",
                "epoch": epoch,
                "probe_state_dict": probes.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "probe_channels": channels,
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
        "experiment": "C-V7 frozen semantic decodability linear probes",
        "diagnostic_only": True,
        "source_checkpoint": {
            "path": args.c_v7_checkpoint,
            "experiment": c_v7_payload.get("experiment"),
            "epoch": c_v7_payload.get("epoch"),
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
            PROBE_ERROR_76: "concat signed e1..e4 -> bias-free zero-init 1x1 Conv -> 19D DeltaZ",
            PROBE_EVIDENCE_211: "exact C-V7 Error Evidence -> bias-free zero-init 1x1 Conv -> 19D DeltaZ",
            PROBE_HIDDEN_32: "trained C-V7 H_err -> bias-free zero-init 1x1 Conv -> 19D DeltaZ",
        },
        "training_target": (
            "Rescue-only CE: C-V3 wrong AND at least one valid aligned history predicts GT; "
            "no correct-history age target"
        ),
        "zero_step": zero_step,
        "history": history,
        "interpretation_guardrails": [
            "Probe success means linearly decodable semantic correction information is present.",
            "Probe failure does not prove information absence; it rejects only this minimal linear readout.",
            "Probe heads are bias-free so Rescue class priors cannot create a feature-independent correction.",
            "This experiment bypasses Gate/tanh/g_max and is not a deployable model evaluation.",
            "No 32D bottleneck conclusion is allowed unless the probe comparison supports it.",
        ],
    }
    with (result_output / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
