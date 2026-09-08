"""CPU/synthetic contract checks for C-V9 Proposal supervision.

中文：C-V9 Proposal（提议）直接监督契约检查。

No KITTI-STEP, Host, checkpoint, RAFT or GPU is required.
"""

import inspect

import torch

from predify2021.mce_scores import (
    train_kitti_step_task_space_prior_c_v9_proposal_supervision as c_v9,
)
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_direct_error_proposal_corrector import (
    DirectErrorProposalCorrector,
)


NUM_CLASSES = 19
HISTORY_LENGTH = 4


def _synthetic_case():
    current = torch.zeros(1, NUM_CLASSES, 2, 2)
    current[:, 0] = 0.5
    raw_low = torch.zeros(1, NUM_CLASSES, 2, 2, requires_grad=True)
    raw_low.data[:, 1, 0, 0] = 1.0
    valid = torch.ones(1, 1, 2, 2)
    gt = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
    rescue = torch.zeros(2, 2, dtype=torch.bool)
    rescue[0, 0] = True
    return current, raw_low, valid, gt, rescue


def _check_fixed_single_change():
    if c_v9.PROPOSAL_LOSS_WEIGHT != 1.0:
        raise RuntimeError("C-V9 Proposal loss weight must be fixed to 1.0")
    source = inspect.getsource(c_v9.main)
    if "--proposal-loss-weight" in source:
        raise RuntimeError("Proposal loss weight must not be exposed as a sweep argument")


def _check_raw_ungated_unbounded_path():
    current, raw_low, valid, gt, rescue = _synthetic_case()
    loss, z_raw, rescue_pixels = c_v9._proposal_rescue_ce(
        current,
        raw_low,
        valid,
        gt,
        rescue,
    )
    if rescue_pixels != 1:
        raise RuntimeError("synthetic Rescue pixel count changed")
    expected = current + raw_low
    if not torch.equal(z_raw, expected):
        raise RuntimeError("Proposal supervision is not using raw unbounded DeltaZ")
    if float(z_raw[0, 1, 0, 0].item()) != 1.0:
        raise RuntimeError("Proposal supervision appears bounded before the loss")
    if not bool(torch.isfinite(loss).item()):
        raise RuntimeError("Proposal Rescue CE is non-finite")


def _check_rescue_only_scope():
    current, raw_low, valid, gt, rescue = _synthetic_case()
    loss_a, _, _ = c_v9._proposal_rescue_ce(
        current,
        raw_low,
        valid,
        gt,
        rescue,
    )
    gt_b = gt.clone()
    gt_b[0, 1] = 5
    gt_b[1, 0] = 6
    gt_b[1, 1] = 7
    loss_b, _, _ = c_v9._proposal_rescue_ce(
        current,
        raw_low,
        valid,
        gt_b,
        rescue,
    )
    if not torch.allclose(loss_a, loss_b, atol=0.0, rtol=0.0):
        raise RuntimeError("non-Rescue labels leaked into Proposal Rescue loss")

    empty = torch.zeros_like(rescue)
    loss_empty, _, count_empty = c_v9._proposal_rescue_ce(
        current,
        raw_low,
        valid,
        gt,
        empty,
    )
    if count_empty != 0 or float(loss_empty.detach().item()) != 0.0:
        raise RuntimeError("empty Rescue mask must create exact zero Proposal loss")


def _check_validity_order():
    current, raw_low, valid, gt, rescue = _synthetic_case()
    valid[:, :, 0, 0] = 0.0
    _, z_raw, _ = c_v9._proposal_rescue_ce(
        current,
        raw_low,
        valid,
        gt,
        rescue,
    )
    if not torch.equal(z_raw[:, :, 0, 0], current[:, :, 0, 0]):
        raise RuntimeError("full-resolution validity did not zero Proposal correction")


def _check_proposal_gradient_and_no_gate_dependency():
    model = DirectErrorProposalCorrector(
        num_classes=NUM_CLASSES,
        history_length=HISTORY_LENGTH,
        hidden_channels=32,
        g_max=0.25,
        gate_bias=-2.0,
    )
    proposal_input = torch.randn(1, NUM_CLASSES * HISTORY_LENGTH, 2, 2)
    delta_raw = model.proposal_head(proposal_input)
    current = torch.zeros(1, NUM_CLASSES, 2, 2)
    valid = torch.ones(1, 1, 2, 2)
    gt = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
    rescue = torch.ones(2, 2, dtype=torch.bool)

    loss, _, _ = c_v9._proposal_rescue_ce(
        current,
        delta_raw,
        valid,
        gt,
        rescue,
    )
    model.zero_grad(set_to_none=True)
    loss.backward()
    proposal_grad = model.proposal_head.weight.grad
    if proposal_grad is None or float(proposal_grad.abs().sum().item()) <= 0.0:
        raise RuntimeError("Proposal Rescue loss did not train Proposal Head")

    for name, parameter in model.named_parameters():
        if name.startswith("proposal_head."):
            continue
        if parameter.grad is not None and float(parameter.grad.abs().sum().item()) != 0.0:
            raise RuntimeError(
                f"Proposal-only auxiliary loss leaked gradient into {name}"
            )


def _check_tbptt_rescue_frame_dilution():
    # Eight final-loss frames but only one Rescue-bearing Proposal-loss frame.
    # The Proposal term must retain weight 1.0, not be divided by eight.
    final_losses = [torch.tensor(float(index + 1)) for index in range(8)]
    proposal_losses = [torch.tensor(10.0)]
    total, final_mean, proposal_mean = c_v9._compose_window_loss(
        final_losses,
        proposal_losses,
    )
    expected_final = torch.tensor(4.5)
    expected_proposal = torch.tensor(10.0)
    expected_total = expected_final + c_v9.PROPOSAL_LOSS_WEIGHT * expected_proposal
    if not torch.allclose(final_mean, expected_final, atol=0.0, rtol=0.0):
        raise RuntimeError("TBPTT final CE mean changed")
    if not torch.allclose(proposal_mean, expected_proposal, atol=0.0, rtol=0.0):
        raise RuntimeError("Proposal loss was diluted by non-Rescue frames")
    if not torch.allclose(total, expected_total, atol=0.0, rtol=0.0):
        raise RuntimeError("C-V9 TBPTT objective does not preserve Proposal weight 1.0")

    total_empty, final_empty, proposal_empty = c_v9._compose_window_loss(
        final_losses,
        [],
    )
    if not torch.allclose(final_empty, expected_final, atol=0.0, rtol=0.0):
        raise RuntimeError("final CE changed when a window has no Rescue frame")
    if float(proposal_empty.item()) != 0.0:
        raise RuntimeError("no-Rescue TBPTT window must have zero Proposal loss")
    if not torch.allclose(total_empty, expected_final, atol=0.0, rtol=0.0):
        raise RuntimeError("no-Rescue TBPTT window objective must equal final CE")


def _check_total_loss_contract_text():
    source = inspect.getsource(c_v9._train_sequence)
    required = (
        "buffered_final_losses.append(segmentation_ce)",
        "buffered_proposal_losses.append(proposal_rescue_ce)",
        "_compose_window_loss",
        "diagnostic._build_diagnostic_masks",
        'masks["rescue"]',
        'evidence["row"]["delta_z_raw"]',
        'evidence["any_valid_full"]',
    )
    for token in required:
        if token not in source:
            raise RuntimeError(f"missing C-V9 supervision contract token: {token}")


def main():
    _check_fixed_single_change()
    _check_raw_ungated_unbounded_path()
    _check_rescue_only_scope()
    _check_validity_order()
    _check_proposal_gradient_and_no_gate_dependency()
    _check_tbptt_rescue_frame_dilution()
    _check_total_loss_contract_text()
    print(
        {
            "passed": True,
            "base_architecture": "unchanged C-V8",
            "total_loss": "mean_all_frames(final CE) + mean_rescue_frames(raw Proposal CE)",
            "proposal_loss_weight": c_v9.PROPOSAL_LOSS_WEIGHT,
            "proposal_loss_weight_sweep": False,
            "proposal_loss_rescue_frame_dilution": False,
            "proposal_path": "Z_cur + full-valid * upsample(DeltaZ_proposal_raw)",
            "proposal_loss_gate_bypass": True,
            "proposal_loss_tanh_bypass": True,
            "proposal_loss_scope": "formal Rescue pixels only",
            "non_rescue_labels_affect_proposal_loss": False,
            "proposal_aux_gradient_target": "Proposal Head only",
            "gate_supervision_changed": False,
            "inference_architecture_changed": False,
        }
    )


if __name__ == "__main__":
    main()
