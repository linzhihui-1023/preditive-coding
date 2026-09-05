#!/usr/bin/env python3
"""Stage-1B-2 failure diagnosis suite.

This is a diagnostic suite, not a new model stage. It intentionally removes the
2 already-known confounds from the remaining Stage-1B-2 questions:

1) Observer quality is bypassed by using strictly-causal RAFT history motion.
2) Semantic probability error e_t^P is removed from the recurrent input; all
   predictors in this suite are motion-history-only.

One --mode all run executes seven controlled 3-epoch training runs plus two
no-training diagnostics, then evaluates every candidate in ONE Full9 pass.

Questions tested:
  D1 loss conflict: Flow-only vs Semantic-only vs Joint.
  D2 weighted gradient balance: semantic vs flow vs delta regularizer.
  D3 invalid/non-transportable semantic supervision: Joint vs GT+RAFT masked Joint.
  D4 bounded additive residual parameterization: bounded residual vs full-range
     persistence-anchored future-motion predictor, plus true residual feasibility.
  D5 low-resolution transport ceiling: current RAFT low-logit warp vs full-logit warp.
  D6 TBPTT limitation: 4 vs 8 vs 16 on the full-range flow-only predictor.

Boundaries:
  - RAFT is privileged diagnostic supervision/history only, not an inference path.
  - GT+RAFT transport masks are diagnostic-only and cannot be used at inference.
  - No D0, Semantic Residual Stage-2, selector, or final Host fusion is trained.
  - Decisions are SUPPORT / WEAKEN / INCONCLUSIVE, never "proven".
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

import run_c_v2_stage1b_failure_diagnostics as base


SEED = 0
DEFAULT_ROOT = "/home/lin/predify/kitti_step"
DEFAULT_RESULT_ROOT = "results/kitti_step_stage1b2_failure_diagnosis_suite"
DEFAULT_CKPT_ROOT = "/home/lin/predify/experiments/kitti_step_stage1b2_failure_diagnosis_suite"
EPOCHS = 3
LR = 1e-4
WEIGHT_DECAY = 1e-2
MAX_OBSERVED_LOW = 32.0
MAX_RESIDUAL_LOW = 16.0
HIDDEN_CHANNELS = 64
LAMBDA_SEMANTIC = 1.0
LAMBDA_FLOW = 10.0
LAMBDA_DELTA = 1e-2


RUN_SPECS = (
    # key, predictor, loss mode, TBPTT
    ("residual_joint_t8", "bounded_residual", "joint", 8),
    ("residual_flow_t8", "bounded_residual", "flow_only", 8),
    ("residual_semantic_t8", "bounded_residual", "semantic_only", 8),
    ("residual_masked_joint_t8", "bounded_residual", "masked_joint", 8),
    ("fullrange_flow_t8", "full_range", "flow_only", 8),
    ("fullrange_flow_t4", "full_range", "flow_only", 4),
    ("fullrange_flow_t16", "full_range", "flow_only", 16),
)


@dataclass
class CachedSequence:
    """CPU cache for one training sequence."""

    low_logits: List[torch.Tensor]          # each [C,H,W], float16 CPU
    gt: List[torch.Tensor]                  # each [Hfull,Wfull], uint8 CPU
    teacher_low: List[torch.Tensor]         # transition i: frame i+1 -> i, [2,H,W], float16 CPU
    transportable: List[torch.Tensor]       # same transition, [Hfull,Wfull], uint8 {0,1}


class MotionOnlyBoundedResidualPredictor(nn.Module):
    """Motion-history-only version of the Stage-1B-2 bounded additive residual."""

    def __init__(self, cell_cls, hidden_channels=64, max_observed=32.0, max_residual=16.0):
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        self.max_observed_displacement_low = float(max_observed)
        self.max_residual_displacement_low = float(max_residual)
        self.recurrent = cell_cls(2, self.hidden_channels)
        self.delta_head = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(self.hidden_channels, 2, 1),
        )
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)

    def predict_next(self, observed_motion, hidden=None):
        scale = max(self.max_observed_displacement_low, 1e-6)
        normalized = observed_motion / scale
        hidden = self.recurrent(normalized, hidden)
        raw = self.delta_head(hidden)
        correction = self.max_residual_displacement_low * torch.tanh(raw)
        predicted = observed_motion + correction
        return predicted, correction, hidden


class MotionOnlyFullRangePredictor(nn.Module):
    """Persistence-anchored but full-range future-motion predictor.

    E0 is exact lagged motion persistence, like the bounded residual model, so the
    comparison does not confound parameterization with a zero-flow initialization.
    Instead of M_t + 16*tanh(delta), the model adds an unconstrained logit-space
    correction before a +/-32 full-range tanh output:

        base = atanh(M_t / 32)
        M_hat = 32 * tanh(base + correction_logits(H_t))

    Thus the final prediction can move anywhere in the full allowed flow range and
    is not capped to a +/-16 additive correction around M_t.
    """

    def __init__(self, cell_cls, hidden_channels=64, max_observed=32.0):
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        self.max_observed_displacement_low = float(max_observed)
        self.max_residual_displacement_low = float(max_observed)  # diagnostics only
        self.recurrent = cell_cls(2, self.hidden_channels)
        self.logit_head = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(self.hidden_channels, 2, 1),
        )
        nn.init.zeros_(self.logit_head[-1].weight)
        nn.init.zeros_(self.logit_head[-1].bias)

    def predict_next(self, observed_motion, hidden=None):
        scale = max(self.max_observed_displacement_low, 1e-6)
        normalized = (observed_motion / scale).clamp(-0.999, 0.999)
        hidden = self.recurrent(normalized, hidden)
        base_logits = torch.atanh(normalized)
        correction_logits = self.logit_head(hidden)
        predicted = scale * torch.tanh(base_logits + correction_logits)
        correction = predicted - observed_motion
        return predicted, correction, hidden


def _load_project():
    P = base._load_project()
    try:
        from predify2021.model_factory.deeplabv3plus_resnet50.semantic_recurrent_predictor import (
            ConvGRUCell,
        )
    except Exception as exc:
        raise RuntimeError("Could not import ConvGRUCell from the current repository.") from exc
    P.ConvGRUCell = ConvGRUCell
    return P


def _seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if abs(float(b)) > 1e-12 else float("nan")


def _pct(x: float) -> str:
    return "nan" if not math.isfinite(float(x)) else f"{100.0 * float(x):.4f}%"


def _pp(x: float) -> str:
    return "nan" if not math.isfinite(float(x)) else f"{100.0 * float(x):+.4f} pp"


def _device_tensor(x: torch.Tensor, dtype=None) -> torch.Tensor:
    y = x.cuda(non_blocking=True)
    return y if dtype is None else y.to(dtype=dtype)


def _warp_full_logits(previous_logits: torch.Tensor, backward_flow: torch.Tensor, flow_grid_fn):
    h, w = previous_logits.shape[-2:]
    grid, valid = flow_grid_fn(backward_flow, h, w)
    warped = F.grid_sample(
        previous_logits.float(),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    warped = torch.where(valid.unsqueeze(1), warped, previous_logits.float())
    return warped.to(previous_logits.dtype), valid


def _transportable_mask_from_gt(
    previous_gt: torch.Tensor,
    current_gt: torch.Tensor,
    teacher_full: torch.Tensor,
    flow_grid_fn,
    ignore_label: int,
) -> torch.Tensor:
    """Diagnostic-only semantic transportability mask using GT + current RAFT.

    A current pixel is marked transportable only when RAFT maps it to an in-bounds
    previous pixel whose GT semantic class equals the current GT semantic class.
    This excludes out-of-bounds transport and most occlusion/disocclusion/class-change
    regions without pretending this mask is available at inference.
    """
    h, w = current_gt.shape[-2:]
    grid, spatial_valid = flow_grid_fn(teacher_full, h, w)
    previous = previous_gt.unsqueeze(0).unsqueeze(0).float()
    warped_previous = F.grid_sample(
        previous,
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
    ).squeeze(0).squeeze(0).to(torch.int64)
    current = current_gt.to(torch.int64)
    valid = spatial_valid.squeeze(0)
    valid = valid & (current != ignore_label) & (warped_previous != ignore_label)
    return valid & (warped_previous == current)


@torch.inference_mode()
def _materialize_train_cache(model, raft, train_groups, args, P) -> Dict[str, CachedSequence]:
    cache: Dict[str, CachedSequence] = {}
    total_frames = 0
    total_pairs = 0
    transportable_pixels = 0
    valid_pixels = 0

    for seq_index, (sequence, samples) in enumerate(train_groups.items(), start=1):
        if args.max_train_steps:
            samples = samples[: min(len(samples), args.max_train_steps + 2)]
        lows: List[torch.Tensor] = []
        gts: List[torch.Tensor] = []
        flows: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []
        previous_image = None
        previous_gt_gpu = None

        for frame_index, sample in enumerate(samples):
            image, _, host_low, _, _ = P._host_observation(model, sample)
            gt_cpu_long = P.semantic_mask_from_panoptic_png(sample["mask_path"])
            gt_cpu = gt_cpu_long.to(torch.uint8).contiguous()
            lows.append(host_low.squeeze(0).half().cpu().contiguous())
            gts.append(gt_cpu)
            total_frames += 1

            current_gt_gpu = gt_cpu_long.cuda(non_blocking=True)
            if previous_image is not None:
                teacher_full = raft.current_to_previous(image, previous_image)
                teacher_low = P.downsample_backward_flow(
                    teacher_full, tuple(host_low.shape[-2:])
                )
                mask = _transportable_mask_from_gt(
                    previous_gt_gpu,
                    current_gt_gpu,
                    teacher_full,
                    P.flow_grid,
                    P.IGNORE_LABEL,
                )
                flows.append(teacher_low.squeeze(0).half().cpu().contiguous())
                masks.append(mask.to(torch.uint8).cpu().contiguous())
                valid = current_gt_gpu != P.IGNORE_LABEL
                valid_pixels += int(valid.sum().item())
                transportable_pixels += int((mask & valid).sum().item())
                total_pairs += 1

            previous_image = image
            previous_gt_gpu = current_gt_gpu

        cache[str(sequence)] = CachedSequence(lows, gts, flows, masks)
        print(
            json.dumps(
                {
                    "phase": "train_cache",
                    "sequence": str(sequence),
                    "index": seq_index,
                    "sequences_total": len(train_groups),
                    "frames": len(lows),
                    "pairs": len(flows),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    summary = {
        "frames": total_frames,
        "pairs": total_pairs,
        "transportable_fraction_of_valid": _safe_div(transportable_pixels, valid_pixels),
        "cache_dtype_low_logits": "float16_cpu",
        "cache_dtype_teacher_low": "float16_cpu",
        "cache_dtype_gt": "uint8_cpu",
        "cache_dtype_transportable_mask": "uint8_cpu",
    }
    print(json.dumps({"phase": "train_cache", "status": "DONE", **summary}, sort_keys=True), flush=True)
    cache["__summary__"] = summary  # type: ignore[assignment]
    return cache


def _build_predictor(kind: str, args, P) -> nn.Module:
    if kind == "bounded_residual":
        return MotionOnlyBoundedResidualPredictor(
            P.ConvGRUCell,
            hidden_channels=args.hidden_channels,
            max_observed=args.max_observed_low,
            max_residual=args.max_residual_low,
        ).cuda()
    if kind == "full_range":
        return MotionOnlyFullRangePredictor(
            P.ConvGRUCell,
            hidden_channels=args.hidden_channels,
            max_observed=args.max_observed_low,
        ).cuda()
    raise ValueError(f"Unknown predictor kind: {kind}")


def _semantic_loss(
    warped_low: torch.Tensor,
    current_gt: torch.Tensor,
    output_size: Tuple[int, int],
    ignore_label: int,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    logits = F.interpolate(warped_low, size=output_size, mode="bilinear", align_corners=False)
    if mask is None:
        return F.cross_entropy(logits, current_gt.unsqueeze(0), ignore_index=ignore_label)
    per_pixel = F.cross_entropy(
        logits,
        current_gt.unsqueeze(0),
        ignore_index=ignore_label,
        reduction="none",
    ).squeeze(0)
    valid = mask & (current_gt != ignore_label)
    if not bool(valid.any()):
        return per_pixel.sum() * 0.0
    return per_pixel[valid].mean()


def _loss_weights(loss_mode: str, args) -> Tuple[float, float, float, bool]:
    if loss_mode == "flow_only":
        return 0.0, args.lambda_flow, 0.0, False
    if loss_mode == "semantic_only":
        return args.lambda_semantic, 0.0, 0.0, False
    if loss_mode == "joint":
        return args.lambda_semantic, args.lambda_flow, args.lambda_delta, False
    if loss_mode == "masked_joint":
        return args.lambda_semantic, args.lambda_flow, args.lambda_delta, True
    raise ValueError(f"Unknown loss mode: {loss_mode}")


def _train_cached_sequence(
    predictor: nn.Module,
    seq: CachedSequence,
    optimizer,
    loss_mode: str,
    tbptt_steps: int,
    args,
    P,
) -> Mapping[str, float]:
    if len(seq.low_logits) < 3:
        return {"frames": 0, "windows": 0, "semantic_loss": 0.0, "flow_loss": 0.0, "delta_loss": 0.0, "total_loss": 0.0}

    sem_w, flow_w, delta_w, use_mask = _loss_weights(loss_mode, args)
    low0 = _device_tensor(seq.low_logits[0], torch.float32).unsqueeze(0)
    low1 = _device_tensor(seq.low_logits[1], torch.float32).unsqueeze(0)
    history1 = _device_tensor(seq.teacher_low[0], torch.float32).unsqueeze(0)
    pending_motion, pending_correction, hidden = predictor.predict_next(history1, None)

    previous_low = low1
    semantic_losses: List[torch.Tensor] = []
    flow_losses: List[torch.Tensor] = []
    delta_losses: List[torch.Tensor] = []
    sums = {"frames": 0, "windows": 0, "semantic_loss": 0.0, "flow_loss": 0.0, "delta_loss": 0.0, "total_loss": 0.0}

    max_steps = len(seq.low_logits) - 2
    for step_index in range(max_steps):
        current_index = step_index + 2
        current_low = _device_tensor(seq.low_logits[current_index], torch.float32).unsqueeze(0)
        teacher_low = _device_tensor(seq.teacher_low[current_index - 1], torch.float32).unsqueeze(0)
        current_gt = _device_tensor(seq.gt[current_index], torch.long)
        output_size = tuple(current_gt.shape[-2:])
        warped_low, _ = P.warp_low_logits(previous_low, pending_motion)
        transport_mask = None
        if use_mask:
            transport_mask = _device_tensor(seq.transportable[current_index - 1], torch.bool)
        sem = _semantic_loss(
            warped_low,
            current_gt,
            output_size,
            P.IGNORE_LABEL,
            mask=transport_mask,
        )
        flo = P.normalized_flow_distillation_loss(
            pending_motion,
            teacher_low,
            args.max_observed_low,
        )
        delt = (pending_correction / max(args.max_residual_low, 1e-6)).square().mean()
        semantic_losses.append(sem)
        flow_losses.append(flo)
        delta_losses.append(delt)
        sums["frames"] += 1

        previous_hidden = hidden
        next_motion, next_correction, next_hidden = predictor.predict_next(
            teacher_low.detach(),
            previous_hidden,
        )
        boundary = len(semantic_losses) == tbptt_steps or step_index == max_steps - 1
        if boundary:
            sem_mean = torch.stack(semantic_losses).mean()
            flo_mean = torch.stack(flow_losses).mean()
            delt_mean = torch.stack(delta_losses).mean()
            total = sem_w * sem_mean + flow_w * flo_mean + delta_w * delt_mean
            if not torch.isfinite(total):
                raise FloatingPointError(f"Non-finite diagnostic training loss: {loss_mode}")
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            sums["windows"] += 1
            sums["semantic_loss"] += float(sem_mean.detach().item())
            sums["flow_loss"] += float(flo_mean.detach().item())
            sums["delta_loss"] += float(delt_mean.detach().item())
            sums["total_loss"] += float(total.detach().item())

            pending_motion, pending_correction, hidden = predictor.predict_next(
                teacher_low.detach(),
                previous_hidden.detach() if previous_hidden is not None else None,
            )
            semantic_losses, flow_losses, delta_losses = [], [], []
        else:
            pending_motion, pending_correction, hidden = next_motion, next_correction, next_hidden

        previous_low = current_low.detach()

    windows = max(int(sums["windows"]), 1)
    for key in ("semantic_loss", "flow_loss", "delta_loss", "total_loss"):
        sums[key] /= windows
    return sums


def _train_run(
    run_key: str,
    predictor_kind: str,
    loss_mode: str,
    tbptt_steps: int,
    cache: Mapping[str, CachedSequence],
    args,
    P,
    ckpt_root: Path,
    result_root: Path,
):
    _seed(args.seed)
    predictor = _build_predictor(predictor_kind, args, P)
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    run_ckpt = ckpt_root / run_key
    run_result = result_root / "runs" / run_key
    run_ckpt.mkdir(parents=True, exist_ok=True)
    run_result.mkdir(parents=True, exist_ok=True)

    history = []
    for epoch in range(1, args.epochs + 1):
        predictor.train()
        total = {"frames": 0, "windows": 0, "semantic_loss": 0.0, "flow_loss": 0.0, "delta_loss": 0.0, "total_loss": 0.0}
        for sequence, seq in cache.items():
            if sequence == "__summary__":
                continue
            row = _train_cached_sequence(predictor, seq, optimizer, loss_mode, tbptt_steps, args, P)
            total["frames"] += int(row["frames"])
            total["windows"] += int(row["windows"])
            for key in ("semantic_loss", "flow_loss", "delta_loss", "total_loss"):
                total[key] += float(row[key]) * int(row["windows"])
        windows = max(int(total["windows"]), 1)
        for key in ("semantic_loss", "flow_loss", "delta_loss", "total_loss"):
            total[key] /= windows
        history.append({"epoch": epoch, "train": total})
        print(
            json.dumps(
                {
                    "phase": "train",
                    "run": run_key,
                    "epoch": epoch,
                    "loss_mode": loss_mode,
                    "tbptt": tbptt_steps,
                    "total_loss": total["total_loss"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    payload = {
        "experiment": "stage1b2_failure_diagnosis_suite",
        "diagnostic_only": True,
        "run": run_key,
        "predictor_kind": predictor_kind,
        "loss_mode": loss_mode,
        "tbptt_steps": tbptt_steps,
        "epochs": args.epochs,
        "state_dict": predictor.state_dict(),
        "history": history,
    }
    torch.save(payload, run_ckpt / f"epoch_{args.epochs:03d}_final.pt")
    (run_result / "train_summary.json").write_text(json.dumps({k: v for k, v in payload.items() if k != "state_dict"}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    predictor.eval()
    return predictor, payload


def _new_eval_banks(names: Sequence[str], P):
    return base._metric_struct(names, P)


def _true_residual_stats_init():
    return {
        "pairs": 0,
        "components": 0,
        "vectors": 0,
        "abs_sum": 0.0,
        "components_over_bound": 0,
        "vectors_over_bound": 0,
        "clipped_epe_sum": 0.0,
        "persistence_epe_sum": 0.0,
    }


def _flow_diag_init():
    return {
        "pairs": 0,
        "components": 0,
        "vectors": 0,
        "pred_abs": 0.0,
        "teacher_abs": 0.0,
        "correction_abs": 0.0,
        "l1": 0.0,
        "epe": 0.0,
        "cos_sum": 0.0,
        "cos_count": 0,
        "temporal_variation_abs": 0.0,
        "temporal_variation_components": 0,
    }


def _update_flow_diag(diag, predicted, correction, teacher, reachable, previous_predicted=None):
    error = predicted - teacher
    component_mask = reachable.unsqueeze(1).expand_as(predicted)
    vector_count = int(reachable.sum().item())
    component_count = int(component_mask.sum().item())
    if component_count:
        diag["pairs"] += 1
        diag["components"] += component_count
        diag["vectors"] += vector_count
        diag["pred_abs"] += float(predicted[component_mask].abs().sum().item())
        diag["teacher_abs"] += float(teacher[component_mask].abs().sum().item())
        diag["correction_abs"] += float(correction[component_mask].abs().sum().item())
        diag["l1"] += float(error[component_mask].abs().sum().item())
        diag["epe"] += float(torch.linalg.vector_norm(error, dim=1)[reachable].sum().item())
        p = predicted.permute(0, 2, 3, 1)[reachable]
        t = teacher.permute(0, 2, 3, 1)[reachable]
        denom = torch.linalg.vector_norm(p, dim=1) * torch.linalg.vector_norm(t, dim=1)
        keep = denom > 1e-8
        if bool(keep.any()):
            cos = (p[keep] * t[keep]).sum(1) / denom[keep]
            diag["cos_sum"] += float(cos.sum().item())
            diag["cos_count"] += int(cos.numel())
    if previous_predicted is not None:
        variation = predicted - previous_predicted
        diag["temporal_variation_abs"] += float(variation.abs().sum().item())
        diag["temporal_variation_components"] += variation.numel()


def _finalize_flow_diag(diag):
    components = max(diag["components"], 1)
    vectors = max(diag["vectors"], 1)
    return {
        "predicted_pairs": int(diag["pairs"]),
        "predicted_abs_mean_low_pixels": diag["pred_abs"] / components,
        "teacher_abs_mean_low_pixels": diag["teacher_abs"] / components,
        "correction_abs_mean_low_pixels": diag["correction_abs"] / components,
        "predicted_teacher_l1_mean_low_pixels": diag["l1"] / components,
        "predicted_teacher_epe_mean_low_pixels": diag["epe"] / vectors,
        "cosine_similarity_mean_nonzero": _safe_div(diag["cos_sum"], diag["cos_count"]),
        "prediction_temporal_variation_abs_mean": _safe_div(
            diag["temporal_variation_abs"], diag["temporal_variation_components"]
        ),
    }


@torch.inference_mode()
def _evaluate_all_once(model, predictors: Mapping[str, nn.Module], val_groups, raft, args, P):
    base_names = [
        "host",
        "semantic_persistence",
        "lagged_raft",
        "current_raft_low",
        "current_raft_full",
    ]
    names = tuple(base_names + list(predictors.keys()))
    bank = _new_eval_banks(names, P)
    per_sequence = {}
    flow_diag = {name: _flow_diag_init() for name in predictors}
    true_residual = _true_residual_stats_init()

    for predictor in predictors.values():
        predictor.eval()

    for sequence in P.FULL9:
        samples = val_groups[sequence]
        seq_bank = base._new_sequence_metric_struct(names, P)
        previous = None
        previous_teacher_low = None
        pending: Dict[str, torch.Tensor] = {}
        pending_correction: Dict[str, torch.Tensor] = {}
        hidden: Dict[str, Optional[torch.Tensor]] = {name: None for name in predictors}
        previous_candidate_motion: Dict[str, Optional[torch.Tensor]] = {name: None for name in predictors}
        previous_predictions: Dict[str, torch.Tensor] = {}

        for sample in samples:
            image, host_logits, host_low, _, output_size = P._host_observation(model, sample)
            host_pred = host_logits.argmax(1)
            gt_cpu = P.semantic_mask_from_panoptic_png(sample["mask_path"])
            teacher_full = None

            if previous is None:
                persistence = lagged = current_low_pred = current_full_pred = host_pred
                candidate_preds = {name: host_pred for name in predictors}
            else:
                previous_image, previous_full_logits, previous_low = previous
                persistence = P._upsample_prior(previous_low, output_size).argmax(1)
                teacher_full = raft.current_to_previous(image, previous_image)
                teacher_low = P.downsample_backward_flow(teacher_full, tuple(host_low.shape[-2:]))

                current_low_warped, _ = P.warp_low_logits(previous_low, teacher_low)
                current_low_pred = P._upsample_prior(current_low_warped, output_size).argmax(1)
                current_full_warped, _ = _warp_full_logits(previous_full_logits, teacher_full, P.flow_grid)
                current_full_pred = current_full_warped.argmax(1)

                if previous_teacher_low is None:
                    lagged_warped = previous_low
                else:
                    lagged_warped, _ = P.warp_low_logits(previous_low, previous_teacher_low)
                lagged = P._upsample_prior(lagged_warped, output_size).argmax(1)

                candidate_preds = {}
                for name, predictor in predictors.items():
                    if name in pending:
                        warped, _ = P.warp_low_logits(previous_low, pending[name])
                        candidate_preds[name] = P._upsample_prior(warped, output_size).argmax(1)
                        if previous_teacher_low is not None:
                            reachable, _ = P.teacher_reachable_mask(
                                teacher_low, args.max_observed_low
                            )
                            _update_flow_diag(
                                flow_diag[name],
                                pending[name],
                                pending_correction[name],
                                teacher_low,
                                reachable,
                                previous_candidate_motion[name],
                            )
                            previous_candidate_motion[name] = pending[name].detach()
                    else:
                        candidate_preds[name] = persistence

                    next_motion, next_correction, next_hidden = predictor.predict_next(
                        teacher_low.detach(), hidden[name]
                    )
                    pending[name] = next_motion.detach()
                    pending_correction[name] = next_correction.detach()
                    hidden[name] = next_hidden.detach()

                if previous_teacher_low is not None:
                    current_reachable, _ = P.teacher_reachable_mask(teacher_low, args.max_observed_low)
                    previous_reachable, _ = P.teacher_reachable_mask(previous_teacher_low, args.max_observed_low)
                    usable = current_reachable & previous_reachable
                    true_delta = teacher_low - previous_teacher_low
                    bound = float(args.max_residual_low)
                    over_component = true_delta.abs() > bound
                    over_vector = over_component.any(dim=1)
                    clipped = previous_teacher_low + true_delta.clamp(-bound, bound)
                    comp_mask = usable.unsqueeze(1).expand_as(true_delta)
                    if bool(usable.any()):
                        true_residual["pairs"] += 1
                        true_residual["components"] += int(comp_mask.sum().item())
                        true_residual["vectors"] += int(usable.sum().item())
                        true_residual["abs_sum"] += float(true_delta[comp_mask].abs().sum().item())
                        true_residual["components_over_bound"] += int((over_component & comp_mask).sum().item())
                        true_residual["vectors_over_bound"] += int((over_vector & usable).sum().item())
                        true_residual["clipped_epe_sum"] += float(
                            torch.linalg.vector_norm(clipped - teacher_low, dim=1)[usable].sum().item()
                        )
                        true_residual["persistence_epe_sum"] += float(
                            torch.linalg.vector_norm(previous_teacher_low - teacher_low, dim=1)[usable].sum().item()
                        )

                previous_teacher_low = teacher_low.detach()

            predictions = {
                "host": host_pred,
                "semantic_persistence": persistence,
                "lagged_raft": lagged,
                "current_raft_low": current_low_pred,
                "current_raft_full": current_full_pred,
                **candidate_preds,
            }
            base._update_segmentation_metrics(bank, seq_bank, predictions, gt_cpu, P)
            if previous is not None:
                scores = base._batched_pair_mtc(
                    previous_predictions,
                    predictions,
                    teacher_full,
                    P.flow_grid,
                    P.NUM_CLASSES,
                    chunk_size=8,
                )
                base._update_mtc_metrics(bank, seq_bank, scores)

            previous = (image, host_logits.detach(), host_low.detach())
            previous_predictions = {k: v.detach() for k, v in predictions.items()}

        base._close_sequence(bank, seq_bank, names)
        per_sequence[sequence] = base._finalize_sequence_metrics(seq_bank, names, P)
        print(json.dumps({"phase": "full9_eval", "sequence": sequence, "status": "DONE"}, sort_keys=True), flush=True)

    metrics = base._finalize_global_metrics(bank, names, P)
    vectors = max(true_residual["vectors"], 1)
    components = max(true_residual["components"], 1)
    residual_feasibility = {
        "pairs": int(true_residual["pairs"]),
        "true_future_minus_history_abs_mean": true_residual["abs_sum"] / components,
        "component_fraction_exceeding_additive_bound": true_residual["components_over_bound"] / components,
        "vector_fraction_any_component_exceeding_additive_bound": true_residual["vectors_over_bound"] / vectors,
        "best_achievable_clipped_residual_epe": true_residual["clipped_epe_sum"] / vectors,
        "lagged_persistence_epe": true_residual["persistence_epe_sum"] / vectors,
        "additive_bound_low_pixels": float(args.max_residual_low),
    }
    return {
        "metrics": metrics,
        "flow_diagnostics": {name: _finalize_flow_diag(diag) for name, diag in flow_diag.items()},
        "residual_feasibility": residual_feasibility,
        "per_sequence": per_sequence,
    }


def _named_grad_norm(grads: Sequence[Optional[torch.Tensor]], names: Sequence[str], predicate=None) -> float:
    total = 0.0
    for grad, name in zip(grads, names):
        if grad is None:
            continue
        if predicate is not None and not predicate(name):
            continue
        total += float(grad.detach().float().square().sum().item())
    return math.sqrt(total)


def _gradient_diagnostic(
    predictor: nn.Module,
    cache: Mapping[str, CachedSequence],
    args,
    P,
):
    predictor.train()
    params = [(n, p) for n, p in predictor.named_parameters() if p.requires_grad]
    names = [n for n, _ in params]
    tensors = [p for _, p in params]
    rows = []

    for sequence, seq in cache.items():
        if sequence == "__summary__" or len(rows) >= args.gradient_windows:
            continue
        if len(seq.low_logits) < 3:
            continue
        predictor.zero_grad(set_to_none=True)
        history = _device_tensor(seq.teacher_low[0], torch.float32).unsqueeze(0)
        pending, correction, hidden = predictor.predict_next(history, None)
        previous_low = _device_tensor(seq.low_logits[1], torch.float32).unsqueeze(0)
        sem_losses = []
        flow_losses = []
        delta_losses = []
        steps = min(args.gradient_tbptt, len(seq.low_logits) - 2)
        for step in range(steps):
            idx = step + 2
            teacher = _device_tensor(seq.teacher_low[idx - 1], torch.float32).unsqueeze(0)
            gt = _device_tensor(seq.gt[idx], torch.long)
            warped, _ = P.warp_low_logits(previous_low, pending)
            sem_losses.append(_semantic_loss(warped, gt, tuple(gt.shape[-2:]), P.IGNORE_LABEL))
            flow_losses.append(P.normalized_flow_distillation_loss(pending, teacher, args.max_observed_low))
            delta_losses.append((correction / max(args.max_residual_low, 1e-6)).square().mean())
            previous_hidden = hidden
            pending, correction, hidden = predictor.predict_next(teacher.detach(), previous_hidden)
            previous_low = _device_tensor(seq.low_logits[idx], torch.float32).unsqueeze(0).detach()

        sem = args.lambda_semantic * torch.stack(sem_losses).mean()
        flo = args.lambda_flow * torch.stack(flow_losses).mean()
        delt = args.lambda_delta * torch.stack(delta_losses).mean()
        losses = {"weighted_semantic": sem, "weighted_flow": flo, "weighted_delta": delt}
        row = {"sequence": sequence, "steps": steps}
        for loss_name, loss in losses.items():
            grads = torch.autograd.grad(loss, tensors, retain_graph=True, allow_unused=True)
            row[loss_name] = {
                "loss": float(loss.detach().item()),
                "grad_norm_total": _named_grad_norm(grads, names),
                "grad_norm_recurrent": _named_grad_norm(grads, names, lambda n: n.startswith("recurrent.")),
                "grad_norm_head": _named_grad_norm(grads, names, lambda n: "head" in n),
            }
        rows.append(row)

    if not rows:
        raise RuntimeError("No gradient diagnostic windows were available")
    sem_norms = [r["weighted_semantic"]["grad_norm_total"] for r in rows]
    flow_norms = [r["weighted_flow"]["grad_norm_total"] for r in rows]
    delta_norms = [r["weighted_delta"]["grad_norm_total"] for r in rows]
    ratios = [s / max(f, 1e-12) for s, f in zip(sem_norms, flow_norms)]
    ratios_sorted = sorted(ratios)
    median = ratios_sorted[len(ratios_sorted) // 2]
    return {
        "windows": rows,
        "mean_weighted_semantic_grad_norm": sum(sem_norms) / len(sem_norms),
        "mean_weighted_flow_grad_norm": sum(flow_norms) / len(flow_norms),
        "mean_weighted_delta_grad_norm": sum(delta_norms) / len(delta_norms),
        "semantic_to_flow_grad_ratio_per_window": ratios,
        "median_semantic_to_flow_grad_ratio": median,
        "note": "Weights lambda_semantic/lambda_flow/lambda_delta are already applied before gradient norms are measured.",
    }


def _run_metric(eval_summary: Mapping, key: str) -> Mapping[str, float]:
    return eval_summary["metrics"][key]


def _run_flow(eval_summary: Mapping, key: str) -> Mapping[str, float]:
    return eval_summary["flow_diagnostics"][key]


def _judgment(status: str, evidence: Mapping, rule: str) -> Mapping:
    return {"status": status, "evidence": dict(evidence), "rule": rule}


def _automatic_judgments(eval_summary: Mapping, grad: Mapping, args) -> Mapping:
    tol_pp = float(args.metric_tolerance_pp)
    tol = tol_pp / 100.0
    epe_tol = float(args.epe_tolerance)

    # D1: loss conflict
    flow_m = _run_metric(eval_summary, "residual_flow_t8")
    joint_m = _run_metric(eval_summary, "residual_joint_t8")
    sem_m = _run_metric(eval_summary, "residual_semantic_t8")
    d_joint_iou = joint_m["mIoU"] - flow_m["mIoU"]
    d_joint_mtc = joint_m["mTC"] - flow_m["mTC"]
    if d_joint_iou >= tol and d_joint_mtc <= -tol:
        d1 = _judgment(
            "SUPPORTED",
            {"joint_minus_flow_mIoU": d_joint_iou, "joint_minus_flow_mTC": d_joint_mtc, "semantic_only_mIoU": sem_m["mIoU"], "semantic_only_mTC": sem_m["mTC"]},
            "Support when Joint gains >= tolerance mIoU over Flow-only while losing >= tolerance mTC.",
        )
    elif d_joint_iou >= -tol and d_joint_mtc >= -tol:
        d1 = _judgment(
            "WEAKENED",
            {"joint_minus_flow_mIoU": d_joint_iou, "joint_minus_flow_mTC": d_joint_mtc},
            "Weaken when Joint is not materially worse than Flow-only on either mIoU or mTC.",
        )
    else:
        d1 = _judgment("INCONCLUSIVE", {"joint_minus_flow_mIoU": d_joint_iou, "joint_minus_flow_mTC": d_joint_mtc}, "Mixed result outside pre-specified support/weaken rules.")

    # D2: gradient dominance
    ratio = float(grad["median_semantic_to_flow_grad_ratio"])
    if ratio >= args.gradient_dominance_ratio:
        d2 = _judgment("SUPPORTED", {"median_semantic_to_flow_grad_ratio": ratio}, f"Support when weighted semantic gradient >= {args.gradient_dominance_ratio:g}x weighted flow gradient.")
    elif ratio <= 1.0 / args.gradient_dominance_ratio:
        d2 = _judgment("WEAKENED", {"median_semantic_to_flow_grad_ratio": ratio}, f"Weaken when weighted semantic gradient <= {1.0/args.gradient_dominance_ratio:g}x weighted flow gradient.")
    else:
        d2 = _judgment("INCONCLUSIVE", {"median_semantic_to_flow_grad_ratio": ratio}, "Neither loss clearly dominates by the pre-specified ratio.")

    # D3: non-transportable semantic supervision
    masked_m = _run_metric(eval_summary, "residual_masked_joint_t8")
    masked_f = _run_flow(eval_summary, "residual_masked_joint_t8")
    joint_f = _run_flow(eval_summary, "residual_joint_t8")
    d_mask_mtc = masked_m["mTC"] - joint_m["mTC"]
    d_mask_epe = masked_f["predicted_teacher_epe_mean_low_pixels"] - joint_f["predicted_teacher_epe_mean_low_pixels"]
    if d_mask_mtc >= tol and d_mask_epe <= -epe_tol:
        d3 = _judgment("SUPPORTED", {"masked_minus_joint_mTC": d_mask_mtc, "masked_minus_joint_EPE": d_mask_epe, "masked_minus_joint_mIoU": masked_m["mIoU"] - joint_m["mIoU"]}, "Support when masking non-transportable pixels improves mTC and reduces future-flow EPE.")
    elif d_mask_mtc <= -tol and d_mask_epe >= epe_tol:
        d3 = _judgment("WEAKENED", {"masked_minus_joint_mTC": d_mask_mtc, "masked_minus_joint_EPE": d_mask_epe}, "Weaken when masking makes both mTC and future-flow EPE worse.")
    else:
        d3 = _judgment("INCONCLUSIVE", {"masked_minus_joint_mTC": d_mask_mtc, "masked_minus_joint_EPE": d_mask_epe, "masked_minus_joint_mIoU": masked_m["mIoU"] - joint_m["mIoU"]}, "Masking helps only one axis or changes are below thresholds.")

    # D4: bounded residual parameterization
    residual_m = flow_m
    direct_m = _run_metric(eval_summary, "fullrange_flow_t8")
    residual_f = _run_flow(eval_summary, "residual_flow_t8")
    direct_f = _run_flow(eval_summary, "fullrange_flow_t8")
    d_direct_mtc = direct_m["mTC"] - residual_m["mTC"]
    d_direct_epe = direct_f["predicted_teacher_epe_mean_low_pixels"] - residual_f["predicted_teacher_epe_mean_low_pixels"]
    feasibility = eval_summary["residual_feasibility"]
    if d_direct_mtc >= tol and d_direct_epe <= -epe_tol:
        d4 = _judgment("SUPPORTED", {"fullrange_minus_residual_mTC": d_direct_mtc, "fullrange_minus_residual_EPE": d_direct_epe, **feasibility}, "Support when full-range predictor improves mTC and reduces future-flow EPE relative to bounded additive residual.")
    elif d_direct_mtc <= -tol and d_direct_epe >= epe_tol:
        d4 = _judgment("WEAKENED", {"fullrange_minus_residual_mTC": d_direct_mtc, "fullrange_minus_residual_EPE": d_direct_epe, **feasibility}, "Weaken when bounded residual is better on both mTC and EPE.")
    else:
        d4 = _judgment("INCONCLUSIVE", {"fullrange_minus_residual_mTC": d_direct_mtc, "fullrange_minus_residual_EPE": d_direct_epe, **feasibility}, "Parameterizations trade metrics or changes are below thresholds.")

    # D5: low-resolution transport ceiling
    low = _run_metric(eval_summary, "current_raft_low")
    full = _run_metric(eval_summary, "current_raft_full")
    d_full_iou = full["mIoU"] - low["mIoU"]
    if d_full_iou >= args.transport_ceiling_support_pp / 100.0:
        d5 = _judgment("SUPPORTED", {"full_minus_low_mIoU": d_full_iou, "full_minus_low_mTC": full["mTC"] - low["mTC"], "low_mIoU": low["mIoU"], "full_mIoU": full["mIoU"]}, f"Support when full-resolution RAFT transport gains >= {args.transport_ceiling_support_pp:g} pp mIoU over low-resolution transport.")
    elif d_full_iou <= tol:
        d5 = _judgment("WEAKENED", {"full_minus_low_mIoU": d_full_iou, "full_minus_low_mTC": full["mTC"] - low["mTC"]}, "Weaken when full-resolution transport improves mIoU by no more than the metric tolerance.")
    else:
        d5 = _judgment("INCONCLUSIVE", {"full_minus_low_mIoU": d_full_iou, "full_minus_low_mTC": full["mTC"] - low["mTC"]}, "Resolution effect exists but is below the support threshold.")

    # D6: TBPTT
    t4m = _run_metric(eval_summary, "fullrange_flow_t4")
    t8m = _run_metric(eval_summary, "fullrange_flow_t8")
    t16m = _run_metric(eval_summary, "fullrange_flow_t16")
    t4f = _run_flow(eval_summary, "fullrange_flow_t4")
    t8f = _run_flow(eval_summary, "fullrange_flow_t8")
    t16f = _run_flow(eval_summary, "fullrange_flow_t16")
    d16_mtc = t16m["mTC"] - t8m["mTC"]
    d16_epe = t16f["predicted_teacher_epe_mean_low_pixels"] - t8f["predicted_teacher_epe_mean_low_pixels"]
    monotonic_mtc = t4m["mTC"] <= t8m["mTC"] + tol and t8m["mTC"] <= t16m["mTC"] + tol
    if d16_mtc >= tol and d16_epe <= -epe_tol and monotonic_mtc:
        d6 = _judgment("SUPPORTED", {"t4_mTC": t4m["mTC"], "t8_mTC": t8m["mTC"], "t16_mTC": t16m["mTC"], "t16_minus_t8_mTC": d16_mtc, "t16_minus_t8_EPE": d16_epe}, "Support when TBPTT16 improves mTC and EPE over TBPTT8 with non-decreasing mTC trend from 4->8->16.")
    elif d16_mtc <= tol and d16_epe >= -epe_tol:
        d6 = _judgment("WEAKENED", {"t4_mTC": t4m["mTC"], "t8_mTC": t8m["mTC"], "t16_mTC": t16m["mTC"], "t16_minus_t8_mTC": d16_mtc, "t16_minus_t8_EPE": d16_epe}, "Weaken when longer TBPTT does not materially improve either mTC or EPE.")
    else:
        d6 = _judgment("INCONCLUSIVE", {"t4_mTC": t4m["mTC"], "t8_mTC": t8m["mTC"], "t16_mTC": t16m["mTC"], "t16_minus_t8_mTC": d16_mtc, "t16_minus_t8_EPE": d16_epe}, "Mixed TBPTT response.")

    return {
        "D1_loss_conflict": d1,
        "D2_semantic_gradient_dominance": d2,
        "D3_nontransportable_semantic_supervision": d3,
        "D4_bounded_residual_parameterization": d4,
        "D5_low_resolution_transport_ceiling": d5,
        "D6_tbptt_limitation": d6,
        "thresholds": {
            "metric_tolerance_pp": args.metric_tolerance_pp,
            "epe_tolerance": args.epe_tolerance,
            "gradient_dominance_ratio": args.gradient_dominance_ratio,
            "transport_ceiling_support_pp": args.transport_ceiling_support_pp,
        },
    }


def _render_markdown(summary: Mapping) -> str:
    ev = summary["full9_evaluation"]
    j = summary["judgments"]
    lines = [
        "# Stage-1B-2 Failure Diagnosis Suite",
        "",
        "This suite intentionally bypasses the learned Observer with causal RAFT history and removes semantic probability error from recurrent inputs. It diagnoses the remaining Stage-1B-2 questions only.",
        "",
        "## Final judgments",
        "",
        "| Diagnostic | Status |",
        "|---|---|",
    ]
    for key, row in j.items():
        if key == "thresholds":
            continue
        lines.append(f"| {key} | **{row['status']}** |")

    lines += [
        "",
        "## Full9 metrics",
        "",
        "| Path | mIoU | mTC | mVC8 | mVC16 | Future EPE | Future L1 | Temporal variation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    order = ["host", "semantic_persistence", "lagged_raft", "current_raft_low", "current_raft_full"] + [s[0] for s in RUN_SPECS]
    for name in order:
        m = ev["metrics"][name]
        f = ev["flow_diagnostics"].get(name)
        if f:
            epe = f"{f['predicted_teacher_epe_mean_low_pixels']:.4f}"
            l1 = f"{f['predicted_teacher_l1_mean_low_pixels']:.4f}"
            tv = f"{f['prediction_temporal_variation_abs_mean']:.4f}"
        else:
            epe = l1 = tv = "-"
        lines.append(f"| {name} | {_pct(m['mIoU'])} | {_pct(m['mTC'])} | {_pct(m['mVC8'])} | {_pct(m['mVC16'])} | {epe} | {l1} | {tv} |")

    lines += ["", "## Gradient diagnostic", ""]
    g = summary["gradient_diagnostic"]
    lines.append(f"Median weighted semantic/flow gradient ratio: **{g['median_semantic_to_flow_grad_ratio']:.4f}x**")
    lines += ["", "## Residual feasibility", ""]
    rf = ev["residual_feasibility"]
    for k, v in rf.items():
        lines.append(f"- `{k}`: {v}")

    lines += ["", "## Judgment evidence", ""]
    for key, row in j.items():
        if key == "thresholds":
            continue
        lines.append(f"### {key}: {row['status']}")
        lines.append("")
        lines.append(row["rule"])
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(row["evidence"], indent=2, sort_keys=True))
        lines.append("```")
        lines.append("")

    lines += [
        "## Boundary",
        "",
        "- Diagnostic only; no D0 or Semantic Residual Stage-2 is opened automatically.",
        "- RAFT-history and GT+RAFT transport masks are privileged diagnostic signals, not deployment inputs.",
        "- SUPPORTED/WEAKENED means the controlled diagnostic supports/weakens a hypothesis; it does not prove uniqueness or causality outside the tested intervention.",
        "",
    ]
    return "\n".join(lines)


def _self_test() -> None:
    class DummyCell(nn.Module):
        def __init__(self, input_channels, hidden_channels):
            super().__init__()
            self.hidden_channels = hidden_channels
            self.conv = nn.Conv2d(input_channels + hidden_channels, hidden_channels, 1)
            nn.init.zeros_(self.conv.weight)
            nn.init.zeros_(self.conv.bias)
        def forward(self, x, hidden):
            if hidden is None:
                hidden = torch.zeros(x.shape[0], self.hidden_channels, x.shape[2], x.shape[3], dtype=x.dtype)
            return torch.tanh(self.conv(torch.cat([x, hidden], dim=1)))

    x = torch.randn(1, 2, 8, 10).clamp(-20, 20)
    r = MotionOnlyBoundedResidualPredictor(DummyCell, hidden_channels=4, max_observed=32, max_residual=16)
    p, d, _ = r.predict_next(x)
    assert torch.equal(d, torch.zeros_like(d))
    assert torch.equal(p, x)
    q = MotionOnlyFullRangePredictor(DummyCell, hidden_channels=4, max_observed=32)
    p2, d2, _ = q.predict_next(x)
    assert float((p2 - x).abs().max().detach()) < 2e-5
    assert float(d2.abs().max().detach()) < 2e-5
    print(json.dumps({"self_test": "PASS"}, sort_keys=True))


def _validate_args(args):
    if args.epochs <= 0 or args.lr <= 0 or args.max_observed_low <= 0 or args.max_residual_low <= 0:
        raise ValueError("Invalid positive training arguments")
    if args.gradient_windows <= 0 or args.gradient_tbptt <= 0:
        raise ValueError("Gradient diagnostic window counts must be positive")
    if args.metric_tolerance_pp < 0 or args.epe_tolerance < 0:
        raise ValueError("Diagnostic tolerances must be non-negative")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("all", "train", "eval", "self-test"), default="all")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--fast-b-checkpoint", default=None)
    parser.add_argument("--result-root", default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--checkpoint-root", default=DEFAULT_CKPT_ROOT)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--hidden-channels", type=int, default=HIDDEN_CHANNELS)
    parser.add_argument("--max-observed-low", type=float, default=MAX_OBSERVED_LOW)
    parser.add_argument("--max-residual-low", type=float, default=MAX_RESIDUAL_LOW)
    parser.add_argument("--lambda-semantic", type=float, default=LAMBDA_SEMANTIC)
    parser.add_argument("--lambda-flow", type=float, default=LAMBDA_FLOW)
    parser.add_argument("--lambda-delta", type=float, default=LAMBDA_DELTA)
    parser.add_argument("--max-train-steps", type=int, default=0, help="0 means all available steps; otherwise cap per train sequence for debugging.")
    parser.add_argument("--gradient-windows", type=int, default=3)
    parser.add_argument("--gradient-tbptt", type=int, default=8)
    parser.add_argument("--metric-tolerance-pp", type=float, default=0.10)
    parser.add_argument("--epe-tolerance", type=float, default=0.01)
    parser.add_argument("--gradient-dominance-ratio", type=float, default=2.0)
    parser.add_argument("--transport-ceiling-support-pp", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    if args.mode == "self-test":
        _self_test()
        return
    _validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the full diagnostic suite")

    P = _load_project()
    _seed(args.seed)
    checkpoint = args.fast_b_checkpoint or P.FAST_B_CHECKPOINT_DEFAULT
    model = P.load_fast_b_model(checkpoint).cuda().eval()
    model.requires_grad_(False)
    raft = P.FrozenRAFT()

    train_ds = P.KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "train")
    val_ds = P.KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "val")
    train_groups = P.sequence_groups(train_ds)
    all_val = P.sequence_groups(val_ds)
    missing = [s for s in P.FULL9 if s not in all_val]
    if missing:
        raise RuntimeError(f"Missing Full9 validation sequences: {missing}")
    val_groups = {s: all_val[s] for s in P.FULL9}

    result_root = Path(args.result_root)
    ckpt_root = Path(args.checkpoint_root)
    result_root.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    train_cache = None
    predictors: Dict[str, nn.Module] = {}
    train_summaries = {}

    if args.mode in ("all", "train"):
        print(json.dumps({"phase": "train_cache", "status": "START"}), flush=True)
        train_cache = _materialize_train_cache(model, raft, train_groups, args, P)
        for run_key, predictor_kind, loss_mode, tbptt in RUN_SPECS:
            print(json.dumps({"phase": "train", "run": run_key, "status": "START"}), flush=True)
            predictor, payload = _train_run(
                run_key,
                predictor_kind,
                loss_mode,
                tbptt,
                train_cache,
                args,
                P,
                ckpt_root,
                result_root,
            )
            predictors[run_key] = predictor
            train_summaries[run_key] = {k: v for k, v in payload.items() if k != "state_dict"}

        if args.mode == "train":
            out = {
                "experiment": "stage1b2_failure_diagnosis_suite",
                "phase": "train_only",
                "train_cache": train_cache["__summary__"],
                "runs": train_summaries,
            }
            (result_root / "train_only_summary.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(json.dumps({"status": "DONE", "result": str(result_root / "train_only_summary.json")}, sort_keys=True), flush=True)
            return
    else:
        # eval mode loads the fixed final checkpoints produced by a previous train/all run.
        for run_key, predictor_kind, loss_mode, tbptt in RUN_SPECS:
            ckpt = ckpt_root / run_key / f"epoch_{args.epochs:03d}_final.pt"
            if not ckpt.exists():
                raise FileNotFoundError(f"Missing diagnostic checkpoint: {ckpt}")
            payload = torch.load(ckpt, map_location="cpu", weights_only=False)
            predictor = _build_predictor(predictor_kind, args, P)
            predictor.load_state_dict(payload["state_dict"], strict=True)
            predictor.eval()
            predictors[run_key] = predictor
            train_summaries[run_key] = {k: v for k, v in payload.items() if k != "state_dict"}

    print(json.dumps({"phase": "full9_eval", "status": "START", "candidates": list(predictors)}, sort_keys=True), flush=True)
    full9 = _evaluate_all_once(model, predictors, val_groups, raft, args, P)
    (result_root / "full9_evaluation.json").write_text(json.dumps(full9, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # D2 requires the training cache. In eval-only mode, rebuild it because only
    # low-resolution host/RAFT tensors are needed and the result must remain self-contained.
    if train_cache is None:
        print(json.dumps({"phase": "train_cache_for_gradient", "status": "START"}), flush=True)
        train_cache = _materialize_train_cache(model, raft, train_groups, args, P)

    grad_model = predictors["residual_joint_t8"]
    gradient = _gradient_diagnostic(grad_model, train_cache, args, P)
    (result_root / "gradient_diagnostic.json").write_text(json.dumps(gradient, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    judgments = _automatic_judgments(full9, gradient, args)
    summary = {
        "experiment": "Stage-1B-2 Failure Diagnosis Suite",
        "diagnostic_only": True,
        "confounds_removed": {
            "observer_bypassed_with_raft_history": True,
            "semantic_probability_error_removed_from_recurrent_input": True,
        },
        "training_protocol": {
            "epochs_fixed": args.epochs,
            "best_epoch_selection": False,
            "runs": [
                {"key": k, "predictor": p, "loss_mode": l, "tbptt": t}
                for k, p, l, t in RUN_SPECS
            ],
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "lambda_semantic": args.lambda_semantic,
            "lambda_flow": args.lambda_flow,
            "lambda_delta": args.lambda_delta,
        },
        "train_cache": train_cache["__summary__"],
        "run_training": train_summaries,
        "full9_evaluation": full9,
        "gradient_diagnostic": gradient,
        "judgments": judgments,
        "boundaries": {
            "D0": False,
            "semantic_residual_stage2": False,
            "raft_history_is_inference_path": False,
            "gt_raft_transport_mask_is_inference_path": False,
            "automatic_judgments_are_causal_proof": False,
        },
    }
    (result_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (result_root / "summary.md").write_text(_render_markdown(summary) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "DONE",
                "summary_json": str(result_root / "summary.json"),
                "summary_markdown": str(result_root / "summary.md"),
                "judgments": {k: v["status"] for k, v in judgments.items() if k != "thresholds"},
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
