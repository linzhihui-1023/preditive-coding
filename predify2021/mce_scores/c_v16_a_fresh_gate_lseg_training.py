"""C-V16-A run-ready fresh-Gate L_seg training facade.

This module keeps the validated C-V16-A training loop in the private base
module and fixes the deployed Gate contract against the actual C-V15 model.

Fresh-state contract / 全新门控契约
--------------------------------
1. The inherited C-V14 ``reliability_head`` is reset to a neutral zero logit
   and remains frozen. C-V15 detaches this prior before the new acceptance path,
   so making it trainable would only add a parameter that never receives L_seg
   gradient.
2. Every C-V15 proposal-conditioned acceptance component is freshly reset and
   trainable: ``proposal_encoder``, ``host_encoder``, ``acceptance_fusion`` and
   ``acceptance_residual_head``.
3. ``acceptance_residual_head`` is zero initialized after the hidden Gate
   layers are reset. Together with the zero frozen reliability prior this gives
   initial deployed g = sigmoid(0 + 0) = 0.5.
4. Proposal generation remains frozen. Training objective remains L_seg only.
5. The base loop retains the reduced CUDA synchronization fix: CE is accumulated
   on device and diagnostics synchronize only sparsely.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import torch
from torch import nn

from predify2021.mce_scores import _c_v16_a_fresh_gate_lseg_training_base as _base


# Exact architecture roots verified against
# ProposalConditionedSoftAcceptanceCorrector.
_RESET_GATE_ROOTS = (
    "reliability_head",
    "proposal_encoder",
    "host_encoder",
    "acceptance_fusion",
    "acceptance_residual_head",
)

# Only the proposal-conditioned C-V15 acceptance path is trainable.  The old
# C-V14 reliability prior is reset to zero but deliberately frozen because the
# C-V15 forward detaches it before composing the final Gate logit.
_TRAINABLE_GATE_ROOTS = (
    "proposal_encoder",
    "host_encoder",
    "acceptance_fusion",
    "acceptance_residual_head",
)


def _matches_root(name: str, roots) -> bool:
    return any(name == root or name.startswith(root + ".") for root in roots)


def _is_gate_parameter(name: str) -> bool:
    """Return True only for parameters that must receive L_seg gradients."""
    return _matches_root(name, _TRAINABLE_GATE_ROOTS)


def collect_gate_parameters(corrector: nn.Module) -> List[Tuple[str, nn.Parameter]]:
    rows = [
        (name, parameter)
        for name, parameter in corrector.named_parameters()
        if _is_gate_parameter(name)
    ]
    if not rows:
        raise RuntimeError("C-V16-A found no trainable proposal-conditioned Gate parameters.")
    return rows


def _reset_module_subtree(module: nn.Module, seen: set[int]) -> None:
    for child in module.modules():
        identity = id(child)
        if identity in seen:
            continue
        seen.add(identity)
        reset = getattr(child, "reset_parameters", None)
        if callable(reset):
            reset()


def _zero_scalar_head(name: str, module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        out_dim = module.out_channels
    elif isinstance(module, nn.Linear):
        out_dim = module.out_features
    else:
        raise RuntimeError(
            f"C-V16-A expected {name} to be Conv2d/Linear, got {type(module).__name__}."
        )
    if out_dim != 1:
        raise RuntimeError(f"C-V16-A expected scalar Gate head {name}; output dim={out_dim}.")
    with torch.no_grad():
        nn.init.zeros_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def reset_fresh_gate_after_checkpoint(corrector: nn.Module) -> None:
    """Remove all inherited C-V14/C-V15 Gate supervision after checkpoint load."""
    modules = dict(corrector.named_modules())
    missing = [root for root in _RESET_GATE_ROOTS if root not in modules]
    if missing:
        raise RuntimeError(
            "C-V16-A Gate architecture mismatch; missing modules: " + ", ".join(missing)
        )

    # Fresh normal initialization for all Gate feature/descriptor layers.
    seen: set[int] = set()
    for root in _RESET_GATE_ROOTS:
        _reset_module_subtree(modules[root], seen)

    # C-V15 deploys sigmoid(base_logit + residual_logit).  Both scalar terms
    # must be neutral after reset; otherwise startup Gate is not exactly 0.5.
    _zero_scalar_head("reliability_head", modules["reliability_head"])
    _zero_scalar_head("acceptance_residual_head", modules["acceptance_residual_head"])

    print(
        "[C-V16-A] fresh Gate reset: reliability prior=0/frozen; "
        "proposal_encoder+host_encoder+acceptance_fusion+acceptance_residual_head=fresh/trainable; "
        "initial deployed g=0.5"
    )


def configure_gate_only_training(
    model: nn.Module,
    corrector: nn.Module,
    frozen_modules: Iterable[nn.Module],
) -> List[Tuple[str, nn.Parameter]]:
    """Freeze Host/Proposal/upstream and expose only the C-V15 acceptance path."""
    model.eval()
    model.requires_grad_(False)

    for module in frozen_modules:
        module.eval()
        module.requires_grad_(False)

    corrector.requires_grad_(False)
    gate_params = collect_gate_parameters(corrector)
    for _, parameter in gate_params:
        parameter.requires_grad_(True)

    trainable_names = [
        name for name, parameter in corrector.named_parameters() if parameter.requires_grad
    ]
    unexpected = [name for name in trainable_names if not _is_gate_parameter(name)]
    if unexpected:
        raise RuntimeError(
            "Unexpected non-Gate trainable parameters: " + ", ".join(unexpected)
        )

    # Explicitly enforce the neutral C-V14 prior is NOT in the optimizer.
    leaked_prior = [
        name
        for name, parameter in corrector.named_parameters()
        if name.startswith("reliability_head.") and parameter.requires_grad
    ]
    if leaked_prior:
        raise RuntimeError(
            "Detached C-V14 reliability prior unexpectedly trainable: " + ", ".join(leaked_prior)
        )

    print("[C-V16-A] TRAINABLE GATE PARAMETERS")
    total = 0
    for name, parameter in gate_params:
        print(f"  {name}: {parameter.numel():,}")
        total += parameter.numel()
    print(f"[C-V16-A] total trainable Gate params: {total:,}")
    return gate_params


def assert_c_v16_a_contract(
    model: nn.Module,
    corrector: nn.Module,
    frozen_modules: Iterable[nn.Module],
) -> None:
    """Hard pre-run parameter contract for the verified C-V15 architecture."""
    bad_host = [name for name, p in model.named_parameters() if p.requires_grad]
    if bad_host:
        raise AssertionError(
            "Host/Decoder still has trainable parameters: " + ", ".join(bad_host[:20])
        )

    for index, module in enumerate(frozen_modules):
        bad = [name for name, p in module.named_parameters() if p.requires_grad]
        if bad:
            raise AssertionError(
                f"Frozen module #{index} still trainable: " + ", ".join(bad[:20])
            )

    names = [name for name, p in corrector.named_parameters() if p.requires_grad]
    if not names:
        raise AssertionError("No C-V16-A Gate parameters are trainable.")
    illegal = [name for name in names if not _is_gate_parameter(name)]
    if illegal:
        raise AssertionError(
            "Proposal/non-Gate parameters unexpectedly trainable: " + ", ".join(illegal)
        )

    required_roots = set(_TRAINABLE_GATE_ROOTS)
    present_roots = {
        root for root in required_roots if any(_matches_root(name, (root,)) for name in names)
    }
    missing_roots = sorted(required_roots - present_roots)
    if missing_roots:
        raise AssertionError(
            "Missing trainable C-V16-A Gate roots: " + ", ".join(missing_roots)
        )

    if any(name.startswith("reliability_head.") for name in names):
        raise AssertionError("C-V14 reliability prior must remain frozen in C-V16-A.")


def build_optimizer(
    gate_params: List[Tuple[str, nn.Parameter]],
    lr: float,
    weight_decay: float = 1e-2,
) -> torch.optim.Optimizer:
    """Match the C-V15 AdamW weight-decay protocol unless explicitly overridden."""
    params = [parameter for _, parameter in gate_params]
    if not params:
        raise RuntimeError("Gate optimizer received no trainable parameters.")
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


# Patch the base module's globals. Functions defined in the base module resolve
# these names dynamically, so its validated hot loop now uses the exact Gate
# contract above without duplicating ~800 lines of sequence logic.
_base._is_gate_parameter = _is_gate_parameter
_base.collect_gate_parameters = collect_gate_parameters
_base.reset_fresh_gate_after_checkpoint = reset_fresh_gate_after_checkpoint
_base.configure_gate_only_training = configure_gate_only_training
_base.assert_c_v16_a_contract = assert_c_v16_a_contract
_base.build_optimizer = build_optimizer


# Re-export the run API expected by the local C-V16-A entrypoint.
GateDiagnostic = _base.GateDiagnostic
IGNORE_LABEL = _base.IGNORE_LABEL
HISTORY_LENGTH = _base.HISTORY_LENGTH
DEV_SEQUENCES = _base.DEV_SEQUENCES
EPOCHS = _base.EPOCHS
TBPTT = _base.TBPTT
BLUR_MID_SIGMA = _base.BLUR_MID_SIGMA
BLUR_MAX_SIGMA = _base.BLUR_MAX_SIGMA
LOG_INTERVAL = _base.LOG_INTERVAL

decode_c_v16_a_training = _base.decode_c_v16_a_training
segmentation_loss = _base.segmentation_loss
grad_rms = _base.grad_rms
parameter_l2 = _base.parameter_l2
collect_gate_diagnostic = _base.collect_gate_diagnostic
train_sequence_c_v16_a = _base.train_sequence_c_v16_a
source_checkpoint_then_fresh_gate = _base.source_checkpoint_then_fresh_gate


__all__ = [
    "GateDiagnostic",
    "IGNORE_LABEL",
    "HISTORY_LENGTH",
    "DEV_SEQUENCES",
    "EPOCHS",
    "TBPTT",
    "BLUR_MID_SIGMA",
    "BLUR_MAX_SIGMA",
    "LOG_INTERVAL",
    "collect_gate_parameters",
    "reset_fresh_gate_after_checkpoint",
    "configure_gate_only_training",
    "assert_c_v16_a_contract",
    "build_optimizer",
    "decode_c_v16_a_training",
    "segmentation_loss",
    "grad_rms",
    "parameter_l2",
    "collect_gate_diagnostic",
    "train_sequence_c_v16_a",
    "source_checkpoint_then_fresh_gate",
]
