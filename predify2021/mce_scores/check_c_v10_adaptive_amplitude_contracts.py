"""CPU/synthetic contract checks for C-V10 Adaptive-Amplitude Acceptance.

中文：C-V10 自适应幅度接受机制契约检查。

No KITTI-STEP, checkpoint, RAFT or GPU is required.
"""

import inspect
import math

import torch

from predify2021.mce_scores import (
    train_kitti_step_task_space_prior_c_v10_adaptive_amplitude_acceptance as c_v10,
)
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_adaptive_amplitude_proposal_corrector import (
    AdaptiveAmplitudeProposalCorrector,
)
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_direct_error_proposal_corrector import (
    DirectErrorProposalCorrector,
)


NUM_CLASSES = 19
HISTORY_LENGTH = 4
HIDDEN = 32


def _inputs(size=5):
    prediction_errors = [
        torch.randn(1, NUM_CLASSES, size, size) * 0.05
        for _ in range(HISTORY_LENGTH)
    ]
    dynamics_error = torch.randn(1, NUM_CLASSES, size, size) * 0.02
    current_margin = torch.randn(1, 1, size, size) * 0.1
    history_margins = [
        torch.randn(1, 1, size, size) * 0.1
        for _ in range(HISTORY_LENGTH)
    ]
    transportability = torch.ones(1, 1, size, size)
    reliability = torch.ones(1, 1, size, size)
    validities = [torch.ones(1, 1, size, size) for _ in range(HISTORY_LENGTH)]
    backward_motion = torch.zeros(1, 2, size, size)
    return {
        "prediction_errors": prediction_errors,
        "dynamics_error": dynamics_error,
        "current_margin": current_margin,
        "history_margins": history_margins,
        "transportability_low": transportability,
        "memory_reliability_low": reliability,
        "history_validities_low": validities,
        "backward_motion_low": backward_motion,
    }


def _model():
    return AdaptiveAmplitudeProposalCorrector(
        num_classes=NUM_CLASSES,
        history_length=HISTORY_LENGTH,
        hidden_channels=HIDDEN,
        acceptance_bias=c_v10.ACCEPTANCE_BIAS,
        amplitude_init=c_v10.AMPLITUDE_INIT,
    )


def _check_predictive_coding_boundary():
    parameters = set(inspect.signature(AdaptiveAmplitudeProposalCorrector.forward).parameters)
    forbidden = {
        "current_logits",
        "current_probability",
        "current_probabilities",
        "history_logits",
        "history_probabilities",
        "raw_history",
    }
    leaked = sorted(parameters & forbidden)
    if leaked:
        raise RuntimeError(f"raw semantic inputs leaked into C-V10 forward: {leaked}")

    model = _model()
    if model.proposal_head.in_channels != NUM_CLASSES * HISTORY_LENGTH:
        raise RuntimeError("Proposal Head input must remain concat(e1..e4) = 76D")
    if model.proposal_head.out_channels != NUM_CLASSES:
        raise RuntimeError("Proposal Head output must remain 19D")
    if model.proposal_head.kernel_size != (1, 1) or model.proposal_head.bias is not None:
        raise RuntimeError("Proposal Head must remain bias-free 1x1 Conv")


def _check_initial_operating_point():
    model = _model().eval()
    out = model(**_inputs())
    expected_acceptance = 1.0 / (1.0 + math.exp(-c_v10.ACCEPTANCE_BIAS))
    expected_amplitude = c_v10.AMPLITUDE_INIT
    expected_alpha = expected_acceptance * expected_amplitude

    if float(model.proposal_head.weight.abs().max().item()) != 0.0:
        raise RuntimeError("C-V10 Proposal Head must be exactly zero initialized")
    for name in ("delta_z_raw", "delta_z"):
        if float(out[name].abs().max().item()) != 0.0:
            raise RuntimeError(f"C-V10 {name} must be zero at initialization")

    for name in ("acceptance", "amplitude", "alpha", "gate"):
        if out[name].shape[1] != 1:
            raise RuntimeError(f"C-V10 {name} must be a single-channel pixel field")

    if not torch.allclose(
        out["acceptance"],
        torch.full_like(out["acceptance"], expected_acceptance),
        atol=1e-7,
        rtol=0.0,
    ):
        raise RuntimeError("initial Acceptance does not match sigmoid(-2)")
    if not torch.allclose(
        out["amplitude"],
        torch.full_like(out["amplitude"], expected_amplitude),
        atol=1e-7,
        rtol=0.0,
    ):
        raise RuntimeError("initial Amplitude does not match 0.25")
    if not torch.allclose(
        out["alpha"],
        torch.full_like(out["alpha"], expected_alpha),
        atol=1e-7,
        rtol=0.0,
    ):
        raise RuntimeError("initial alpha does not match former C-V9 gate")
    if not torch.equal(out["gate"], out["alpha"]):
        raise RuntimeError("shared evaluator gate slot must equal C-V10 alpha")


def _check_shared_control_encoder_and_parameter_delta():
    model = _model()
    if model.decision_channels != 95:
        raise RuntimeError(f"C-V10 decision evidence must remain 95D, got {model.decision_channels}")
    if model.control_pre is not model.gate_pre:
        raise RuntimeError("C-V10 must reuse C-V9 gate_pre as the shared Control Pre")
    if model.acceptance_head is not model.gate_head:
        raise RuntimeError("C-V10 Acceptance Head must reuse the C-V9 gate_head slot")
    if hasattr(model, "amplitude_pre"):
        raise RuntimeError("C-V10 must not add a second 95D->32D control encoder")
    if model.amplitude_head.in_channels != HIDDEN or model.amplitude_head.out_channels != 1:
        raise RuntimeError("C-V10 Amplitude Head must be exactly 32D->1")

    c_v9 = DirectErrorProposalCorrector(
        num_classes=NUM_CLASSES,
        history_length=HISTORY_LENGTH,
        hidden_channels=HIDDEN,
        g_max=0.25,
        gate_bias=-2.0,
    )
    c_v9_params = sum(p.numel() for p in c_v9.parameters() if p.requires_grad)
    c_v10_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    expected_extra = HIDDEN + 1
    if c_v10_params - c_v9_params != expected_extra:
        raise RuntimeError(
            "C-V10 must add only one 32D->1 Amplitude Head: "
            f"expected +{expected_extra}, got +{c_v10_params - c_v9_params}"
        )


def _check_fixed_025_cap_removed():
    model = _model().eval()
    target = 0.8
    logit = math.log(target / (1.0 - target))
    with torch.no_grad():
        model.gate_head.bias.fill_(logit)
        model.amplitude_head.bias.fill_(logit)
    out = model(**_inputs())
    alpha = float(out["alpha"].mean().item())
    if alpha <= 0.25:
        raise RuntimeError("C-V10 alpha is still structurally capped at 0.25")
    if abs(alpha - target * target) > 1e-6:
        raise RuntimeError("C-V10 alpha is not Acceptance * Amplitude")


def _check_proposal_independent_of_temporal_control():
    model = _model().eval()
    with torch.no_grad():
        model.proposal_head.weight.normal_(mean=0.0, std=0.02)
    inputs = _inputs()
    out_a = model(**inputs, hidden=None)
    random_hidden = torch.randn(1, HIDDEN, 5, 5)
    out_b = model(**inputs, hidden=random_hidden)
    if not torch.equal(out_a["delta_z_raw"], out_b["delta_z_raw"]):
        raise RuntimeError("Temporal Error Memory leaked into semantic Proposal content")


def _check_joint_final_gradient():
    model = _model().train()
    with torch.no_grad():
        model.proposal_head.weight.normal_(mean=0.0, std=0.02)
        model.gate_head.weight.normal_(mean=0.0, std=0.02)
        model.amplitude_head.weight.normal_(mean=0.0, std=0.02)
    out = model(**_inputs())
    applied = out["gate"] * out["delta_z"]
    loss = applied.square().mean()
    model.zero_grad(set_to_none=True)
    loss.backward()

    required = {
        "proposal_head": model.proposal_head.weight.grad,
        "control_pre": model.gate_pre[0].weight.grad,
        "acceptance_head": model.gate_head.weight.grad,
        "amplitude_head": model.amplitude_head.weight.grad,
    }
    for name, grad in required.items():
        if grad is None or float(grad.abs().sum().item()) <= 0.0:
            raise RuntimeError(f"final correction loss did not train {name}")


def _check_training_protocol_and_metadata():
    source = inspect.getsource(c_v10.main)
    if "c_v9._train_epoch" not in source:
        raise RuntimeError("C-V10 must reuse the validated C-V9 supervision path")
    forbidden_cli = (
        "--g-max",
        "--acceptance-bias",
        "--amplitude-init",
        "--amplitude-target",
        "--acceptance-target",
    )
    for token in forbidden_cli:
        if token in source:
            raise RuntimeError(f"C-V10 must not expose architecture sweep argument {token}")
    if c_v10.PROPOSAL_LOSS_WEIGHT != 1.0:
        raise RuntimeError("C-V10 must preserve C-V9 Proposal loss weight 1.0")
    if 'diagnostics.pop("g_max", None)' not in source:
        raise RuntimeError("C-V10 must remove legacy evaluator g_max metadata")
    if 'diagnostics["alpha_full_mean"] = diagnostics.pop("gate_mean")' not in source:
        raise RuntimeError("C-V10 must rename legacy gate_mean to alpha_full_mean")

    hook_source = inspect.getsource(c_v10._ControlStats._hook)
    if ".item()" in hook_source:
        raise RuntimeError("C-V10 control hook must not synchronize GPU per frame")


def main():
    _check_predictive_coding_boundary()
    _check_initial_operating_point()
    _check_shared_control_encoder_and_parameter_delta()
    _check_fixed_025_cap_removed()
    _check_proposal_independent_of_temporal_control()
    _check_joint_final_gradient()
    _check_training_protocol_and_metadata()
    print(
        {
            "passed": True,
            "causal_path": "History -> Prediction -> Prediction Error -> Proposal -> Adaptive Error Gain -> Correction",
            "proposal": "same C-V9 concat(e1..e4) 76D->19D",
            "control_pre": "shared C-V9 95D->32D gate_pre",
            "acceptance": "shared-control 32D->1 sigmoid",
            "amplitude": "new shared-control 32D->1 sigmoid",
            "extra_control_parameters_vs_c_v9": HIDDEN + 1,
            "alpha": "acceptance * amplitude in [0,1]",
            "fixed_025_cap_removed": True,
            "initial_alpha_matches_c_v9": True,
            "proposal_supervision": "unchanged C-V9 Rescue-only raw Proposal CE",
            "acceptance_direct_target": False,
            "amplitude_oracle_target": False,
            "temporal_loss": False,
            "inference_uses_raw_history_directly": False,
            "legacy_g_max_metadata_removed": True,
            "legacy_gate_mean_renamed_to_alpha_full_mean": True,
            "per_frame_control_item_sync": False,
        }
    )


if __name__ == "__main__":
    main()
