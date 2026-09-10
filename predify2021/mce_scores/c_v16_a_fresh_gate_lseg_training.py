"""
C-V16-A review candidate
========================

Purpose
-------
Test exactly one hypothesis:

Given a frozen, previously trained Proposal generator, can a freshly
initialized Gate learn useful spatial reliability from L_seg only?

This file is intentionally written as a review candidate against the
current C-V14/C-V15 interfaces. It must NOT be treated as a repository
source-of-truth until merged into the actual training entrypoint.

Key constraints
---------------
1. Host / Decoder / Prediction-Error core / Proposal are frozen.
2. Gate is reset AFTER loading the source checkpoint.
3. Gate is the only trainable component.
4. Training objective is exactly L_seg.
5. No L_accept / L_protect / GT-derived gate target backpropagation.
6. Proposal-only decoder and GT diagnostics are excluded from training inner loop.
7. Frozen evidence is detached before entering the trainable Gate whenever possible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from predify2021.datasets.kitti_step import semantic_mask_from_panoptic_png
from predify2021.model_factory.deeplabv3plus_resnet50 import HostFeature

from predify2021.mce_scores import (
    c_v14_post_writeback_reliability_training as c_v14,
)
from predify2021.mce_scores import (
    c_v15_proposal_conditioned_soft_acceptance_training as c_v15,
)
from predify2021.mce_scores import (
    train_kitti_step_task_space_prior_c_v5_multiframe_candidate_memory as c_v5,
)

IGNORE_LABEL = c_v14.IGNORE_LABEL
HISTORY_LENGTH = c_v14.HISTORY_LENGTH

# Development protocol
DEV_SEQUENCES = ("0002", "0010", "0018")
EPOCHS = 3
TBPTT = 8
BLUR_MID_SIGMA = 2.25
BLUR_MAX_SIGMA = 3.0

# Diagnostics
LOG_INTERVAL = 20


@dataclass
class GateDiagnostic:
    step: int
    grad_rms: float
    update_ratio: float
    gate_mean: float
    gate_std: float
    gate_min: float
    gate_max: float
    gate_lt_005: float
    gate_gt_095: float
    proposal_delta_rms_ratio: float


def _is_gate_parameter(name: str) -> bool:
    """
    REVIEW POINT:
    This name rule must be checked against the real corrector module.

    C-V16-A must reset/train every deployed Gate parameter carrying C-V14/C-V15
    BCE history, not merely a layer whose name happens to be `reliability_head`.
    """
    lowered = name.lower()
    keywords = (
        "reliability",
        "acceptance",
        "gate",
    )
    return any(k in lowered for k in keywords)


def collect_gate_parameters(corrector: nn.Module) -> List[Tuple[str, nn.Parameter]]:
    rows = [
        (name, p)
        for name, p in corrector.named_parameters()
        if _is_gate_parameter(name)
    ]
    if not rows:
        raise RuntimeError(
            "No Gate parameters found. Inspect the actual corrector names before running C-V16-A."
        )
    return rows


def reset_fresh_gate_after_checkpoint(corrector: nn.Module) -> None:
    """
    Reset the deployed Gate AFTER source-checkpoint loading.

    Hidden layers:
        standard module reset/init where available

    Final scalar reliability/gate-logit layer:
        zero weight + zero bias, giving initial sigmoid ~= 0.5

    REVIEW POINT:
    The exact final layer must be verified against the real corrector structure.
    """
    gate_modules: List[Tuple[str, nn.Module]] = []
    for name, module in corrector.named_modules():
        if name and _is_gate_parameter(name):
            gate_modules.append((name, module))

    if not gate_modules:
        raise RuntimeError("No Gate modules matched for fresh initialization.")

    # First restore healthy fresh initialization for matched modules.
    for _, module in gate_modules:
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()

    # Then neutralize the final 1-channel Conv/Linear layers.
    final_candidates: List[Tuple[str, nn.Module]] = []
    for name, module in gate_modules:
        if isinstance(module, nn.Conv2d) and module.out_channels == 1:
            final_candidates.append((name, module))
        elif isinstance(module, nn.Linear) and module.out_features == 1:
            final_candidates.append((name, module))

    if not final_candidates:
        raise RuntimeError(
            "Could not identify a final scalar Gate-logit layer. "
            "Do not guess; inspect corrector architecture first."
        )

    # Prefer the deepest/named-last scalar layer.
    final_name, final_module = sorted(
        final_candidates,
        key=lambda x: (x[0].count("."), x[0]),
    )[-1]

    with torch.no_grad():
        nn.init.zeros_(final_module.weight)
        if final_module.bias is not None:
            nn.init.zeros_(final_module.bias)

    print(f"[C-V16-A] fresh Gate initialized; neutral final layer: {final_name}")


def configure_gate_only_training(
    model: nn.Module,
    corrector: nn.Module,
    frozen_modules: Iterable[nn.Module],
) -> List[Tuple[str, nn.Parameter]]:
    """
    Freeze everything except the deployed Gate.
    """

    # Freeze host / decoder / all upstream modules passed by caller.
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    for module in frozen_modules:
        module.eval()
        for p in module.parameters():
            p.requires_grad_(False)

    # Freeze the entire corrector first.
    for p in corrector.parameters():
        p.requires_grad_(False)

    # Re-enable Gate only.
    gate_params = collect_gate_parameters(corrector)
    for _, p in gate_params:
        p.requires_grad_(True)

    # Contract check: no non-Gate trainables.
    unexpected = []
    for name, p in corrector.named_parameters():
        if p.requires_grad and not _is_gate_parameter(name):
            unexpected.append(name)

    if unexpected:
        raise RuntimeError(
            "Unexpected non-Gate trainable parameters: " + ", ".join(unexpected)
        )

    print("[C-V16-A] TRAINABLE PARAMETERS")
    total = 0
    for name, p in gate_params:
        if p.requires_grad:
            print(f"  {name}: {p.numel():,}")
            total += p.numel()
    print(f"[C-V16-A] total trainable Gate params: {total:,}")

    return [(n, p) for n, p in gate_params if p.requires_grad]


def build_optimizer(
    gate_params: List[Tuple[str, nn.Parameter]],
    lr: float,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    params = [p for _, p in gate_params]
    if not params:
        raise RuntimeError("Gate optimizer received no trainable parameters.")
    return torch.optim.AdamW(
        params,
        lr=lr,
        weight_decay=weight_decay,
    )


def decode_c_v16_a_training(
    model: nn.Module,
    corrector: nn.Module,
    observation: Dict,
    error_row: Dict,
    transportability_low: torch.Tensor,
    memory_reliability_low: torch.Tensor,
):
    """
    Fast training decode.

    IMPORTANT:
    There is only ONE Decoder pass here.

    Removed from C-V14/C-V15 training path:
      - proposal_logits Decoder pass
      - proposal_acceptance_targets
      - protection_kl_loss
      - acceptance_bce_loss
      - rescue/damage GT diagnostics
    """

    # Upstream evidence is frozen. Detach it before Gate/Proposal use so autograd
    # does not retain unnecessary history graphs.
    prediction_errors = [
        x.detach() if torch.is_tensor(x) else x
        for x in error_row["prediction_errors"]
    ]

    history_validities_low = [
        x.detach() if torch.is_tensor(x) else x
        for x in error_row["history_validities_low"]
    ]

    temporal_hidden = error_row["temporal_hidden"]
    if torch.is_tensor(temporal_hidden):
        temporal_hidden = temporal_hidden.detach()

    dynamics_error = error_row["dynamics_state"]
    if torch.is_tensor(dynamics_error):
        dynamics_error = dynamics_error.detach()

    # Corrector currently contains both Proposal and Gate.
    # Proposal parameters are frozen; Gate parameters remain trainable.
    row = corrector(
        prediction_errors=prediction_errors,
        history_validities_low=history_validities_low,
        temporal_hidden=temporal_hidden,
        dynamics_error=dynamics_error,
        transportability_low=transportability_low.detach(),
        memory_reliability_low=memory_reliability_low.detach(),
        current_c4=observation["c4"].detach(),
    )

    # REVIEW POINT:
    # In the ideal C-V16-A implementation, bounded_semantic_delta_c4 should be
    # detached before multiplication by Gate while Gate remains differentiable.
    #
    # If corrector internally already computes:
    #     delta_c4 = reliability * bounded_semantic_delta_c4
    # then detaching only after `corrector()` is too late to reduce that graph.
    #
    # Therefore the production patch should split:
    #     frozen proposal generation
    #     trainable reliability/gate application
    #
    # This review version checks that Proposal itself has no trainable parameters,
    # but the final integration should expose these two stages explicitly.

    final_logits = model.decode_from_host_feature(
        HostFeature(
            row["corrected_c4"],
            observation["c1"].detach(),
            observation["output_size"],
        )
    )

    return final_logits, row


def segmentation_loss(final_logits: torch.Tensor, gt_cpu: torch.Tensor):
    """
    The ONLY training objective in C-V16-A.

    Do not branch on ``torch.isfinite(loss)`` here.  A Python boolean check on
    a CUDA scalar forces a GPU->CPU synchronization on every frame.  Finite
    checks are performed sparsely at the optimizer/diagnostic boundary instead.
    """
    target = (
        gt_cpu.to(final_logits.device, non_blocking=True)
        .long()
        .unsqueeze(0)
    )

    return F.cross_entropy(
        final_logits,
        target,
        ignore_index=IGNORE_LABEL,
        reduction="mean",
    )


def grad_rms(parameters: Iterable[nn.Parameter]) -> torch.Tensor:
    params = list(parameters)
    sq_sum = None
    count = 0

    for p in params:
        if p.grad is None:
            continue
        g = p.grad.detach()
        term = g.square().sum()
        sq_sum = term if sq_sum is None else sq_sum + term
        count += g.numel()

    if sq_sum is None or count == 0:
        if params:
            return torch.zeros((), device=params[0].device)
        return torch.tensor(0.0)

    return torch.sqrt(sq_sum / float(count))


def parameter_l2(parameters: Iterable[nn.Parameter]) -> torch.Tensor:
    sq_sum = None

    for p in parameters:
        w = p.detach()
        term = w.square().sum()
        sq_sum = term if sq_sum is None else sq_sum + term

    if sq_sum is None:
        return torch.tensor(0.0)

    return torch.sqrt(sq_sum)


@torch.no_grad()
def collect_gate_diagnostic(
    step: int,
    gate_params: List[nn.Parameter],
    row: Dict,
    lr: float,
    loss: torch.Tensor,
) -> GateDiagnostic:
    # ``gate_params`` is cached once before the frame loop.  Re-scanning every
    # module/parameter at each TBPTT boundary adds avoidable Python overhead.
    g_rms = grad_rms(gate_params)
    g_l2_sq = None
    finite = torch.isfinite(loss.detach())
    for p in gate_params:
        if p.grad is None:
            continue
        grad = p.grad.detach()
        term = grad.square().sum()
        g_l2_sq = term if g_l2_sq is None else g_l2_sq + term
        finite = finite & torch.isfinite(grad).all()
    g_l2 = (
        torch.sqrt(g_l2_sq)
        if g_l2_sq is not None
        else torch.zeros((), device=row["reliability"].device)
    )

    w_l2 = parameter_l2(gate_params).to(g_l2.device)
    update_ratio = lr * g_l2 / (w_l2 + 1e-12)

    gate = row["reliability"].detach()
    proposal = row["bounded_semantic_delta_c4"].detach()
    current = row.get("current_c4", None)

    if current is None:
        # Use final corrected feature scale only as a fallback diagnostic.
        current = row["corrected_c4"].detach()

    current_rms = current.square().mean().sqrt().clamp_min(1e-8)
    proposal_ratio = proposal.square().mean().sqrt() / current_rms

    # Transfer all scalar diagnostics together.  This intentionally creates one
    # CUDA->CPU synchronization instead of one ``.item()`` per statistic.
    packed = torch.stack(
        [
            finite.to(dtype=gate.dtype),
            g_rms.to(gate.device, dtype=gate.dtype),
            update_ratio.to(dtype=gate.dtype),
            gate.mean(),
            gate.std(),
            gate.min(),
            gate.max(),
            gate.lt(0.05).to(gate.dtype).mean(),
            gate.gt(0.95).to(gate.dtype).mean(),
            proposal_ratio.to(dtype=gate.dtype),
        ]
    )
    values = packed.cpu().tolist()

    if values[0] < 0.5:
        raise FloatingPointError(
            "Non-finite C-V16-A L_seg/Gate gradient."
        )

    return GateDiagnostic(
        step=step,
        grad_rms=float(values[1]),
        update_ratio=float(values[2]),
        gate_mean=float(values[3]),
        gate_std=float(values[4]),
        gate_min=float(values[5]),
        gate_max=float(values[6]),
        gate_lt_005=float(values[7]),
        gate_gt_095=float(values[8]),
        proposal_delta_rms_ratio=float(values[9]),
    )


def assert_c_v16_a_contract(
    model: nn.Module,
    corrector: nn.Module,
    frozen_modules: Iterable[nn.Module],
) -> None:
    """
    Hard pre-run checks.
    """
    bad_host = [
        name for name, p in model.named_parameters()
        if p.requires_grad
    ]
    if bad_host:
        raise AssertionError(
            "Host/Decoder still has trainable parameters: "
            + ", ".join(bad_host[:20])
        )

    for module_index, module in enumerate(frozen_modules):
        bad = [
            name for name, p in module.named_parameters()
            if p.requires_grad
        ]
        if bad:
            raise AssertionError(
                f"Frozen module #{module_index} still trainable: "
                + ", ".join(bad[:20])
            )

    gate_trainable = []
    non_gate_trainable = []

    for name, p in corrector.named_parameters():
        if not p.requires_grad:
            continue
        if _is_gate_parameter(name):
            gate_trainable.append(name)
        else:
            non_gate_trainable.append(name)

    if not gate_trainable:
        raise AssertionError("No Gate parameters are trainable.")

    if non_gate_trainable:
        raise AssertionError(
            "Proposal/non-Gate parameters unexpectedly trainable: "
            + ", ".join(non_gate_trainable)
        )


def train_sequence_c_v16_a(
    model,
    observer,
    residual,
    correction,
    mask_predictor,
    refiner,
    c_v4_controller,
    dynamics,
    corrector,
    samples,
    optimizer,
    gradient_accumulation_steps: int = TBPTT,
    log_interval: int = LOG_INTERVAL,
):
    """
    C-V16-A sequence training loop.

    This is adapted from the current C-V14 structure but strips all training-only
    GT gate/protection supervision and the second Decoder pass.
    """

    initial = c_v14._initial_state(
        model,
        observer,
        residual,
        samples,
    )
    if initial is None:
        return None

    _, previous, pending_motion, motion_hidden = initial

    previous_host_logits = previous["host_logits"].detach()
    previous_low = previous["host_low"].detach()
    previous_c1 = previous["c1"].detach()

    raw_history = [previous_host_logits]
    motion_history = []

    transport_hidden = None
    semantic_hidden = None
    mask_hidden = None

    semantic_state_low = torch.zeros_like(previous_low)

    memory_state = None
    controller_hidden = None
    dynamics_state = None

    optimizer.zero_grad(set_to_none=True)

    frames_in_window = 0
    optimizer_steps = 0
    total_frames = 0

    # Cache the trainable Gate parameter list once.  It does not change during
    # C-V16-A training.
    gate_params = [
        p
        for name, p in corrector.named_parameters()
        if p.requires_grad and _is_gate_parameter(name)
    ]
    if not gate_params:
        raise RuntimeError("C-V16-A has no trainable Gate parameters.")

    # Keep CE statistics on-device.  The previous ``loss.detach().item()``
    # forced one GPU->CPU synchronization per frame and serialized the hot loop.
    # We synchronize once when this sequence finishes.
    ce_sum_device = torch.zeros(
        (), device=previous_low.device, dtype=torch.float32
    )
    diagnostic_rows: List[GateDiagnostic] = []

    # Everything upstream of Gate is frozen.
    for frame_index in range(2, len(samples)):
        with torch.no_grad():
            observation = c_v14.host_feature_observation(
                model,
                samples[frame_index],
            )

            prior_low, _ = c_v5.warp_low_logits(
                previous_low,
                pending_motion,
            )

            e1 = c_v5._frozen_e1_step(
                correction,
                mask_predictor,
                observation["c1"],
                observation["host_low"],
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
                observation["c1"],
                observation["host_low"],
                prior_low,
                e1,
                pending_motion,
                memory_state,
                observation["output_size"],
                observation["host_logits"],
            )

            candidate_rows = c_v5._build_history_candidates(
                raw_history,
                motion_history,
                pending_motion,
                corrector.history_length,
            )

            error_row = c_v14.build_prediction_error_and_temporal_state(
                c_v4_controller,
                dynamics,
                c_v3_logits,
                candidate_rows,
                pending_motion,
                e1["transportability_low"],
                memory_row["memory_reliability"],
                controller_hidden,
                dynamics_state,
                corrector.history_length,
            )

            controller_hidden = error_row["temporal_hidden"]
            dynamics_state = error_row["dynamics_state"]

        # IMPORTANT:
        # Exit no_grad here. Gate -> corrected C4 -> Decoder -> L_seg must retain
        # autograd so L_seg can update Gate.
        final_logits, correction_row = decode_c_v16_a_training(
            model=model,
            corrector=corrector,
            observation=observation,
            error_row=error_row,
            transportability_low=e1["transportability_low"],
            memory_reliability_low=memory_row["memory_reliability"],
        )

        # Cheap graph contract: metadata-only checks, no CUDA synchronization.
        # A frozen Proposal should not require gradients, while Gate and the
        # corrected feature must remain connected to L_seg through the Decoder.
        if total_frames == 0:
            proposal = correction_row["bounded_semantic_delta_c4"]
            gate = correction_row["reliability"]
            corrected = correction_row["corrected_c4"]
            if proposal.requires_grad:
                raise AssertionError(
                    "Frozen C-V16-A Proposal still requires grad; split/detach "
                    "Proposal before Gate application."
                )
            if not gate.requires_grad:
                raise AssertionError(
                    "C-V16-A Gate is detached from the training graph."
                )
            if not corrected.requires_grad or not final_logits.requires_grad:
                raise AssertionError(
                    "L_seg -> Decoder -> corrected C4 -> Gate gradient path is broken."
                )

        gt_cpu = semantic_mask_from_panoptic_png(
            samples[frame_index]["mask_path"]
        )

        loss = segmentation_loss(
            final_logits,
            gt_cpu,
        )

        loss.backward()

        frames_in_window += 1
        total_frames += 1
        ce_sum_device.add_(loss.detach().float())

        boundary = (
            frames_in_window >= gradient_accumulation_steps
            or frame_index == len(samples) - 1
        )

        if boundary:
            # Normalize accumulated Gate gradients only.
            for p in gate_params:
                if p.grad is not None:
                    p.grad.div_(float(frames_in_window))

            next_step = optimizer_steps + 1

            # Diagnostics BEFORE optimizer.step because gradients still exist.
            if next_step % log_interval == 0:
                # This is the only regular CUDA->CPU synchronization point in
                # the training hot path.  Keep it sparse (default: every 20
                # optimizer steps).  Finite checking is packed into the same
                # transfer as the Gate diagnostics.
                lr = optimizer.param_groups[0]["lr"]
                diag = collect_gate_diagnostic(
                    step=next_step,
                    gate_params=gate_params,
                    row=correction_row,
                    lr=lr,
                    loss=loss,
                )
                diagnostic_rows.append(diag)
                print(
                    "[C-V16-A] "
                    f"step={diag.step} "
                    f"grad_rms={diag.grad_rms:.3e} "
                    f"update_ratio={diag.update_ratio:.3e} "
                    f"gate={diag.gate_mean:.4f}±{diag.gate_std:.4f} "
                    f"proposal_rms_ratio={diag.proposal_delta_rms_ratio:.4e}"
                )

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            optimizer_steps += 1
            frames_in_window = 0

        # Frozen temporal-state update.
        with torch.no_grad():
            observed_motion = c_v5._observe_motion(
                observer,
                previous_low,
                previous_c1,
                observation["host_low"],
                observation["c1"],
            )

            motion_error = (
                F.softmax(observation["host_low"], dim=1)
                - F.softmax(prior_low, dim=1)
            )

            next_motion, _, next_motion_hidden = residual.predict_next(
                observed_motion,
                motion_error,
                motion_hidden,
            )

            raw_history.insert(0, c_v3_logits.detach())
            raw_history = raw_history[: corrector.history_length]

            motion_history.insert(0, pending_motion.detach())
            motion_history = motion_history[
                : max(corrector.history_length - 1, 0)
            ]

            previous = observation
            previous_low = observation["host_low"].detach()
            previous_c1 = observation["c1"].detach()

            pending_motion = next_motion.detach()
            motion_hidden = next_motion_hidden.detach()

            if memory_state is not None:
                memory_state = memory_state.detach()
            if controller_hidden is not None:
                controller_hidden = controller_hidden.detach()
            if dynamics_state is not None:
                dynamics_state = dynamics_state.detach()

    # One synchronization per sequence for reporting, instead of one per frame.
    final_ce = float(
        (ce_sum_device / float(max(total_frames, 1))).item()
    )

    return {
        "frames": total_frames,
        "optimizer_steps": optimizer_steps,
        "final_ce": final_ce,
        "objective": "L_seg_only",
        "acceptance_bce_backprop": False,
        "protection_kl_backprop": False,
        "gt_gate_target_backprop": False,
        "diagnostics": [
            d.__dict__ for d in diagnostic_rows
        ],
    }


def source_checkpoint_then_fresh_gate(
    *,
    load_checkpoint_fn,
    checkpoint_path: str,
    model: nn.Module,
    corrector: nn.Module,
    frozen_modules: Iterable[nn.Module],
    lr: float,
):
    """
    Enforce the only valid initialization order for C-V16-A.

        load old checkpoint
            ↓
        reset Gate
            ↓
        freeze Proposal/upstream
            ↓
        build Gate-only optimizer
    """

    # 1) Load source C-V15/C-V14-era weights first.
    load_checkpoint_fn(checkpoint_path)

    # 2) Remove ALL inherited BCE-trained deployed-Gate information.
    reset_fresh_gate_after_checkpoint(corrector)

    # 3) Freeze all other components and expose Gate only.
    gate_params = configure_gate_only_training(
        model=model,
        corrector=corrector,
        frozen_modules=frozen_modules,
    )

    # 4) Verify the contract before optimizer creation.
    assert_c_v16_a_contract(
        model=model,
        corrector=corrector,
        frozen_modules=frozen_modules,
    )

    # 5) Optimizer sees Gate only.
    optimizer = build_optimizer(
        gate_params=gate_params,
        lr=lr,
    )

    return optimizer


# ---------------------------------------------------------------------------
# REQUIRED REVIEW BEFORE RUN
# ---------------------------------------------------------------------------
#
# 1. Inspect the real corrector.named_parameters().
#    `_is_gate_parameter()` is intentionally conservative and MUST be checked.
#
# 2. Identify the exact module producing:
#       correction_row["reliability"]
#
# 3. Confirm whether C-V15 deployed Gate combines:
#       base_reliability_c4
#       proposal-conditioned acceptance
#       any other reliability prior
#
#    If yes, every trainable component carrying inherited BCE Gate information
#    must be freshly reinitialized.
#
# 4. Best production implementation:
#       proposal = frozen_proposal(...).detach()
#       gate = trainable_gate(frozen_error_evidence)
#       corrected_c4 = current_c4 + gate * proposal
#
#    Do NOT leave Proposal and Gate inseparably fused inside `corrector.forward`
#    if that causes autograd to retain the entire frozen Proposal graph.
#
# 5. Training loop must NOT call:
#       c_v14.protection_kl_loss(...)
#       c_v14.proposal_acceptance_targets(...)
#       c_v15.acceptance_bce_loss(...)
#
# 6. Training loop must NOT decode proposal_c4 separately.
#    Proposal Decoder #2 belongs only in validation.
#
# 7. Damage / Rescue / reliability separation are validation diagnostics only.
#
# 8. C-V15 output files, checkpoints and result directories must remain untouched.
#
# ---------------------------------------------------------------------------
