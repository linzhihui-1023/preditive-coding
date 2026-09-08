"""Contract checks for C-V8 Direct Error Proposal + Proposal-Aware Gate.

中文：C-V8 直接预测误差提议 + 提议感知门控结构契约检查。

No KITTI-STEP, Host, checkpoint, RAFT or GPU is required.
"""

import inspect

import torch

from predify2021.model_factory.deeplabv3plus_resnet50.task_space_direct_error_proposal_corrector import (
    DirectErrorProposalCorrector,
)


NUM_CLASSES = 19
HISTORY_LENGTH = 4
HIDDEN = 32
G_MAX = 0.25
GATE_BIAS = -2.0


def _inputs(device="cpu"):
    batch, height, width = 1, 5, 7
    errors = []
    validities = []
    for index in range(HISTORY_LENGTH):
        error = torch.zeros(batch, NUM_CLASSES, height, width, device=device)
        error[:, index, :, :] = 0.1 * float(index + 1)
        errors.append(error)
        valid = torch.ones(batch, 1, height, width, device=device)
        if index == 3:
            valid[:, :, :, -1] = 0.0
        validities.append(valid)

    return {
        "prediction_errors": errors,
        "dynamics_error": torch.zeros(batch, NUM_CLASSES, height, width, device=device),
        "current_margin": torch.zeros(batch, 1, height, width, device=device),
        "history_margins": [
            torch.zeros(batch, 1, height, width, device=device)
            for _ in range(HISTORY_LENGTH)
        ],
        "transportability_low": torch.ones(batch, 1, height, width, device=device),
        "memory_reliability_low": torch.ones(batch, 1, height, width, device=device),
        "history_validities_low": validities,
        "backward_motion_low": torch.zeros(batch, 2, height, width, device=device),
    }


def _check_interface_and_shapes(model):
    signature = inspect.signature(model.forward)
    names = set(signature.parameters)
    forbidden = {
        "current_logits",
        "history_logits",
        "current_probability",
        "history_probabilities",
        "raw_history",
    }
    leaked = sorted(forbidden & names)
    if leaked:
        raise RuntimeError(f"raw semantic input leaked into corrector interface: {leaked}")

    if model.proposal_head.in_channels != NUM_CLASSES * HISTORY_LENGTH:
        raise RuntimeError("Proposal Head must read exactly concat(e1..e4) = 76 channels")
    if model.proposal_head.out_channels != NUM_CLASSES:
        raise RuntimeError("Proposal Head must output 19 class residual channels")
    if model.proposal_head.kernel_size != (1, 1):
        raise RuntimeError("Proposal Head must remain a 1x1 linear readout")
    if model.proposal_head.bias is not None:
        raise RuntimeError("Proposal Head must be bias-free")
    if hasattr(model, "correction_head"):
        raise RuntimeError("C-V8 must not retain the old H_err-only Correction Head")

    expected_gate_channels = HIDDEN + 2 * NUM_CLASSES + 6 + NUM_CLASSES
    if model.gate_pre[0].in_channels != expected_gate_channels:
        raise RuntimeError(
            f"proposal-aware Gate expected {expected_gate_channels} channels, "
            f"got {model.gate_pre[0].in_channels}"
        )
    if model.gate_head.out_channels != 1:
        raise RuntimeError("Gate must remain single-channel")


def _check_zero_init_and_proposal_source(model, inputs):
    if float(model.proposal_head.weight.detach().abs().max().item()) != 0.0:
        raise RuntimeError("Proposal Head must be exactly zero initialized")
    if float(model.gate_head.weight.detach().abs().max().item()) != 0.0:
        raise RuntimeError("Gate Head weight must retain conservative zero initialization")
    if not torch.allclose(
        model.gate_head.bias.detach(),
        torch.full_like(model.gate_head.bias.detach(), GATE_BIAS),
    ):
        raise RuntimeError("Gate bias initialization changed")

    aggregate = model._aggregate_error_evidence(
        inputs["prediction_errors"],
        inputs["history_validities_low"],
    )
    expected = torch.cat(
        [
            error * validity
            for error, validity in zip(
                inputs["prediction_errors"], inputs["history_validities_low"]
            )
        ],
        dim=1,
    )
    if aggregate["proposal_input"].shape[1] != NUM_CLASSES * HISTORY_LENGTH:
        raise RuntimeError("Proposal input channel count is not 76")
    if not torch.equal(aggregate["proposal_input"], expected):
        raise RuntimeError("Proposal input is not exactly validity-gated concat(e1..e4)")

    row = model(**inputs)
    if float(row["delta_z_raw"].detach().abs().max().item()) != 0.0:
        raise RuntimeError("zero-initialized Proposal Head must produce zero DeltaZ_raw")
    if float(row["delta_z"].detach().abs().max().item()) != 0.0:
        raise RuntimeError("zero-initialized Proposal Head must produce zero bounded DeltaZ")
    expected_gate = G_MAX * torch.sigmoid(torch.tensor(GATE_BIAS))
    if not torch.allclose(
        row["gate"],
        torch.full_like(row["gate"], float(expected_gate.item())),
        atol=1e-7,
        rtol=0.0,
    ):
        raise RuntimeError("initial Gate value changed unexpectedly")


def _check_hidden_does_not_define_proposal(model, inputs):
    # Make a deterministic non-zero proposal mapping from selected error channels.
    with torch.no_grad():
        model.proposal_head.weight.zero_()
        for class_index in range(NUM_CLASSES):
            model.proposal_head.weight[class_index, class_index, 0, 0] = 1.0

    row_zero_history = model(**inputs, hidden=None)
    random_hidden = torch.randn(
        1,
        HIDDEN,
        inputs["backward_motion_low"].shape[-2],
        inputs["backward_motion_low"].shape[-1],
    )
    row_random_history = model(**inputs, hidden=random_hidden)

    if not torch.allclose(
        row_zero_history["delta_z_raw"],
        row_random_history["delta_z_raw"],
        atol=0.0,
        rtol=0.0,
    ):
        raise RuntimeError("H_err leaked into semantic Proposal Head")
    if float(row_zero_history["delta_z_raw"].abs().max().item()) <= 0.0:
        raise RuntimeError("synthetic Prediction Error did not create a semantic proposal")


def _check_proposal_aware_gate_and_joint_gradient(model, inputs):
    row = model(**inputs)
    proposal = row["delta_z_raw"]
    gate_evidence = row["gate_evidence"]
    proposal_slice = gate_evidence[:, -NUM_CLASSES:, :, :]

    if not torch.equal(proposal_slice.detach(), proposal.detach()):
        raise RuntimeError("Gate does not explicitly observe the 19D semantic proposal")
    if not proposal_slice.requires_grad:
        raise RuntimeError("Gate proposal input was unexpectedly detached")

    # The Gate graph must remain connected to the proposal tensor.  With the
    # conservative zero-initialized Gate Head the derivative can be exactly zero,
    # but it must not be None; None would mean a structural detach/cut edge.
    gate_gradient = torch.autograd.grad(
        row["gate"].sum(),
        proposal,
        allow_unused=True,
        retain_graph=True,
    )[0]
    if gate_gradient is None:
        raise RuntimeError("Proposal-aware Gate is not differentiably connected to proposal")

    # The actual residual path must still train Proposal Head on the first step.
    model.zero_grad(set_to_none=True)
    row = model(**inputs)
    loss = (row["gate"] * row["delta_z"]).sum()
    loss.backward()
    grad = model.proposal_head.weight.grad
    if grad is None or float(grad.abs().sum().item()) <= 0.0:
        raise RuntimeError("Proposal Head did not receive first-step correction gradient")


def _check_motion_aligned_error_memory(model):
    hidden = torch.randn(1, HIDDEN, 5, 7)
    zero_motion = torch.zeros(1, 2, 5, 7)
    warped, valid = model._warp_hidden_zero_invalid(hidden, zero_motion)
    if warped.shape != hidden.shape or valid.shape != (1, 5, 7):
        raise RuntimeError("motion-aligned Error Memory shape contract changed")
    if not bool(torch.isfinite(warped).all().item()):
        raise RuntimeError("motion-aligned Error Memory produced non-finite values")


def main():
    model = DirectErrorProposalCorrector(
        num_classes=NUM_CLASSES,
        history_length=HISTORY_LENGTH,
        hidden_channels=HIDDEN,
        g_max=G_MAX,
        gate_bias=GATE_BIAS,
    )
    inputs = _inputs()

    _check_interface_and_shapes(model)
    _check_zero_init_and_proposal_source(model, inputs)
    _check_hidden_does_not_define_proposal(model, inputs)
    _check_proposal_aware_gate_and_joint_gradient(model, inputs)
    _check_motion_aligned_error_memory(model)

    print(
        {
            "passed": True,
            "proposal_input": "validity-gated concat(e1..e4)",
            "proposal_channels": 76,
            "proposal_output_channels": 19,
            "proposal_head": "bias-free zero-init 1x1 Conv",
            "old_h_err_correction_head_present": False,
            "h_err_enters_proposal": False,
            "h_err_role": "temporal context for Gate",
            "proposal_aware_gate": True,
            "proposal_gate_input_detached": False,
            "joint_end_to_end_ce_gradient": True,
            "gate_channels": 1,
            "tanh_bound": True,
            "g_max": G_MAX,
            "raw_history_direct_input": False,
        }
    )


if __name__ == "__main__":
    main()
