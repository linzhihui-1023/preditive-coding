"""CPU/synthetic contracts for C-V11 Reliability-Conditioned Expansion."""

import inspect
import math

import torch

from predify2021.mce_scores import c_v11_reliability_training as training
from predify2021.mce_scores import train_kitti_step_task_space_prior_c_v11_reliability_conditioned_expansion as c_v11
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_direct_error_proposal_corrector import DirectErrorProposalCorrector
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_reliability_conditioned_expansion_corrector import ReliabilityConditionedExpansionCorrector

NUM_CLASSES = 19
HISTORY_LENGTH = 4
HIDDEN = 32


def _model():
    return ReliabilityConditionedExpansionCorrector(
        num_classes=NUM_CLASSES,
        history_length=HISTORY_LENGTH,
        hidden_channels=HIDDEN,
        acceptance_bias=c_v11.ACCEPTANCE_BIAS,
        base_gain=c_v11.BASE_GAIN,
        reliability_init=c_v11.RELIABILITY_INIT,
    )


def _inputs(size=5):
    return {
        "prediction_errors": [torch.randn(1, NUM_CLASSES, size, size) * 0.05 for _ in range(HISTORY_LENGTH)],
        "dynamics_error": torch.randn(1, NUM_CLASSES, size, size) * 0.02,
        "current_margin": torch.randn(1, 1, size, size) * 0.1,
        "history_margins": [torch.randn(1, 1, size, size) * 0.1 for _ in range(HISTORY_LENGTH)],
        "transportability_low": torch.ones(1, 1, size, size),
        "memory_reliability_low": torch.ones(1, 1, size, size),
        "history_validities_low": [torch.ones(1, 1, size, size) for _ in range(HISTORY_LENGTH)],
        "backward_motion_low": torch.zeros(1, 2, size, size),
    }


def _check_predictive_coding_boundary():
    parameters = set(inspect.signature(ReliabilityConditionedExpansionCorrector.forward).parameters)
    forbidden = {"current_logits", "current_probability", "history_logits", "history_probabilities", "raw_history"}
    leaked = sorted(parameters & forbidden)
    if leaked:
        raise RuntimeError(f"raw semantic input leaked into C-V11 forward: {leaked}")
    model = _model()
    if model.proposal_head.in_channels != NUM_CLASSES * HISTORY_LENGTH:
        raise RuntimeError("Proposal must remain concat(e1..e4)=76D")
    if model.proposal_head.out_channels != NUM_CLASSES or model.proposal_head.bias is not None:
        raise RuntimeError("Proposal must remain bias-free 76D->19D")


def _check_parameter_delta_and_shared_control():
    model = _model()
    c_v9 = DirectErrorProposalCorrector(
        num_classes=NUM_CLASSES,
        history_length=HISTORY_LENGTH,
        hidden_channels=HIDDEN,
        g_max=0.25,
        gate_bias=-2.0,
    )
    if model.control_pre is not model.gate_pre or model.acceptance_head is not model.gate_head:
        raise RuntimeError("C-V11 must reuse C-V9 Control Pre and Acceptance Head")
    p9 = sum(p.numel() for p in c_v9.parameters() if p.requires_grad)
    p11 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if p11 - p9 != HIDDEN + 1:
        raise RuntimeError(f"C-V11 must add exactly 33 parameters, got {p11 - p9}")


def _check_gain_formula_and_initialization():
    model = _model().eval()
    out = model(**_inputs())
    a0 = 1.0 / (1.0 + math.exp(-c_v11.ACCEPTANCE_BIAS))
    r0 = c_v11.RELIABILITY_INIT
    base0 = c_v11.BASE_GAIN * a0
    alpha0 = base0 + (1.0 - base0) * r0
    if not torch.allclose(out["acceptance"], torch.full_like(out["acceptance"], a0), atol=1e-7, rtol=0.0):
        raise RuntimeError("initial Acceptance mismatch")
    if not torch.allclose(out["expansion_reliability"], torch.full_like(out["expansion_reliability"], r0), atol=1e-7, rtol=0.0):
        raise RuntimeError("initial Reliability mismatch")
    if not torch.allclose(out["alpha"], torch.full_like(out["alpha"], alpha0), atol=1e-7, rtol=0.0):
        raise RuntimeError("initial alpha formula mismatch")
    if float(out["alpha"].min()) < 0.0 or float(out["alpha"].max()) > 1.0:
        raise RuntimeError("alpha must stay in [0,1]")
    if not torch.equal(out["gate"], out["alpha"]):
        raise RuntimeError("shared evaluator gate slot must equal alpha")

    # Near-zero reliability must reduce to the C-V9 base gain.
    with torch.no_grad():
        model.expansion_reliability_head.bias.fill_(-30.0)
    out = model(**_inputs())
    diff = (out["alpha"] - c_v11.BASE_GAIN * out["acceptance"]).abs().max().item()
    if diff > 1e-10:
        raise RuntimeError(f"Reliability-off path does not recover C-V9 base gain: {diff}")

    # Full Reliability must remove the former amplitude reachability limit.
    # Keep Acceptance at its normal low initial value; r -> 1 still must allow alpha -> 1.
    model = _model().eval()
    with torch.no_grad():
        model.expansion_reliability_head.bias.fill_(30.0)
    out = model(**_inputs())
    reachability_error = (1.0 - out["alpha"]).abs().max().item()
    if reachability_error > 1e-6:
        raise RuntimeError(
            "Full Reliability must permit full expansion independent of Acceptance: "
            f"max error from alpha=1 is {reachability_error}"
        )


def _check_reliability_aux_gradient_isolation():
    model = _model().train()
    with torch.no_grad():
        model.proposal_head.weight.normal_(0.0, 0.02)
    row = model(**_inputs(size=4))
    supervised_logit = model.expansion_reliability_head(row["control_hidden"].detach())
    positive_loss = torch.nn.functional.softplus(-supervised_logit[:, :, :2]).mean()
    negative_loss = torch.nn.functional.softplus(supervised_logit[:, :, 2:]).mean()
    loss = 0.5 * (positive_loss + negative_loss)
    model.zero_grad(set_to_none=True)
    loss.backward()
    head_grad = model.expansion_reliability_head.weight.grad
    if head_grad is None or float(head_grad.abs().sum().item()) <= 0.0:
        raise RuntimeError("Reliability auxiliary did not train Reliability Head")
    if model.gate_pre[0].weight.grad is not None:
        raise RuntimeError("Reliability auxiliary leaked into shared Control Pre")
    if model.proposal_head.weight.grad is not None:
        raise RuntimeError("Reliability auxiliary leaked into Proposal Head")


def _check_final_ce_path_is_joint():
    model = _model().train()
    with torch.no_grad():
        model.proposal_head.weight.normal_(0.0, 0.02)
        model.gate_head.weight.normal_(0.0, 0.02)
        model.expansion_reliability_head.weight.normal_(0.0, 0.02)
    row = model(**_inputs())
    loss = (row["gate"] * row["delta_z"]).square().mean()
    model.zero_grad(set_to_none=True)
    loss.backward()
    required = {
        "proposal": model.proposal_head.weight.grad,
        "control_pre": model.gate_pre[0].weight.grad,
        "acceptance": model.gate_head.weight.grad,
        "reliability": model.expansion_reliability_head.weight.grad,
    }
    for name, grad in required.items():
        if grad is None or float(grad.abs().sum().item()) <= 0.0:
            raise RuntimeError(f"deployed final path did not train {name}")


def _check_training_protocol():
    source = inspect.getsource(c_v11.main)
    forbidden_cli = ("--base-gain", "--reliability-init", "--reliability-loss-weight", "--g-max")
    for token in forbidden_cli:
        if token in source:
            raise RuntimeError(f"C-V11 must not expose sweep parameter {token}")
    if training.PROPOSAL_LOSS_WEIGHT != 1.0 or training.RELIABILITY_LOSS_WEIGHT != 1.0:
        raise RuntimeError("C-V11 fixed loss weights must be 1.0")
    helper_source = inspect.getsource(training.train_sequence)
    if 'row["control_hidden"].detach()' not in helper_source:
        raise RuntimeError("Reliability auxiliary must detach shared Control Pre features")
    if "c_v9._proposal_rescue_ce" not in helper_source:
        raise RuntimeError("C-V11 must preserve C-V9 Proposal supervision")
    if 'alpha = 0.25*a + (1-0.25*a)*r' not in source:
        raise RuntimeError("C-V11 metadata must record the corrected reachability formula")


def main():
    _check_predictive_coding_boundary()
    _check_parameter_delta_and_shared_control()
    _check_gain_formula_and_initialization()
    _check_reliability_aux_gradient_isolation()
    _check_final_ce_path_is_joint()
    _check_training_protocol()
    print({
        "passed": True,
        "causal_path": "History -> Prediction -> Prediction Error -> Proposal -> Reliability-Conditioned Gain -> Correction",
        "proposal": "same C-V9 76D->19D",
        "control_pre": "same C-V9 95D->32D",
        "acceptance": "same C-V9 32D->1",
        "expansion_reliability": "new 32D->1",
        "additional_parameters_vs_c_v9": 33,
        "alpha": "0.25*a + (1-0.25*a)*r",
        "reliability_off_recovers_c_v9": True,
        "full_reliability_reaches_alpha_one": True,
        "reliability_aux_only_trains_new_head": True,
        "proposal_supervision": "unchanged C-V9",
        "temporal_loss": False,
    })


if __name__ == "__main__":
    main()
