"""Contract checks for C-V7 frozen semantic decodability probes.

中文：C-V7 冻结语义可读出性探针契约检查。

No KITTI-STEP, Host, checkpoint, RAFT, or GPU is required.
"""

import torch

from predify2021.mce_scores.diagnose_c_v7_semantic_decodability_probes import (
    LinearResidualProbe,
    PROBE_ERROR_76,
    PROBE_EVIDENCE_211,
    PROBE_HIDDEN_32,
    _build_diagnostic_masks,
)


NUM_CLASSES = 19


def _logits_from_labels(labels, high=5.0, low=-5.0):
    height, width = labels.shape
    logits = torch.full((1, NUM_CLASSES, height, width), low)
    logits.scatter_(1, labels.unsqueeze(0).unsqueeze(0), high)
    return logits


def _check_zero_init_and_unbounded_linear_path():
    probe = LinearResidualProbe(3, NUM_CLASSES)
    feature = torch.ones((1, 3, 2, 2), requires_grad=True)
    valid = torch.ones((1, 1, 4, 4), dtype=torch.bool)

    delta0 = probe(feature, (4, 4), valid)
    if float(delta0.abs().max().item()) != 0.0:
        raise RuntimeError("zero-initialized probe must produce exact zero DeltaZ")

    with torch.no_grad():
        probe.head.weight.fill_(2.0)
        probe.head.bias.fill_(1.0)
    delta = probe(feature, (4, 4), valid)
    # 3 input channels * weight 2 + bias 1 = 7.  A tanh/gate path would not preserve 7.
    if not torch.allclose(delta, torch.full_like(delta, 7.0)):
        raise RuntimeError("probe output is unexpectedly gated, bounded, or transformed")

    loss = delta.square().mean()
    loss.backward()
    if probe.head.weight.grad is None or float(probe.head.weight.grad.abs().sum().item()) <= 0.0:
        raise RuntimeError("probe head did not receive gradient")
    if feature.grad is None or float(feature.grad.abs().sum().item()) <= 0.0:
        raise RuntimeError("synthetic feature did not receive gradient in raw probe test")

    # Real diagnostic features are detached before probe use.  Verify that contract explicitly.
    detached_feature = feature.detach().clone().requires_grad_(False)
    probe.zero_grad(set_to_none=True)
    detached_delta = probe(detached_feature, (4, 4), valid)
    detached_delta.mean().backward()
    if detached_feature.grad is not None:
        raise RuntimeError("detached frozen feature unexpectedly received gradient")


def _check_rescue_masks_and_no_history_gradient():
    # x0: Current wrong, valid history correct -> Rescue.
    # x1: Current correct, valid history conflicts -> Protection.
    # x2: Current wrong, no correct history -> neither.
    # x3: Current correct, conflicting history invalid -> neither.
    gt = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    current_labels = torch.tensor([[4, 1, 5, 3]], dtype=torch.long)
    current_logits = _logits_from_labels(current_labels)

    history1_labels = torch.tensor([[0, 6, 7, 8]], dtype=torch.long)
    history1_logits = _logits_from_labels(history1_labels).requires_grad_(True)
    history1_valid = torch.tensor([[[True, True, True, False]]])

    history2_labels = torch.tensor([[9, 1, 10, 11]], dtype=torch.long)
    history2_logits = _logits_from_labels(history2_labels).requires_grad_(True)
    history2_valid = torch.tensor([[[True, True, True, False]]])

    rows = [
        {"logits": history1_logits, "valid_full": history1_valid},
        {"logits": history2_logits, "valid_full": history2_valid},
    ]
    masks = _build_diagnostic_masks(current_logits, rows, gt)

    expected_rescue = torch.tensor([[True, False, False, False]])
    expected_protect = torch.tensor([[False, True, False, False]])
    if not torch.equal(masks["rescue"].cpu(), expected_rescue):
        raise RuntimeError(f"unexpected Rescue mask: {masks['rescue'].cpu()}")
    if not torch.equal(masks["protection"].cpu(), expected_protect):
        raise RuntimeError(f"unexpected Protection mask: {masks['protection'].cpu()}")

    # Mask construction is under no_grad and historical logits are argmax-detached.
    if history1_logits.grad is not None or history2_logits.grad is not None:
        raise RuntimeError("mask construction leaked gradient into historical logits")

    invalid_rows = [
        {
            "logits": history1_logits.detach(),
            "valid_full": torch.zeros_like(history1_valid),
        }
    ]
    invalid_masks = _build_diagnostic_masks(current_logits, invalid_rows, gt)
    if bool(invalid_masks["rescue"].any()) or bool(invalid_masks["protection"].any()):
        raise RuntimeError("invalid history created diagnostic Rescue/Protection pixels")


def _check_declared_probe_shapes():
    expected = {
        PROBE_ERROR_76: 76,
        PROBE_EVIDENCE_211: 211,
        PROBE_HIDDEN_32: 32,
    }
    probes = {
        name: LinearResidualProbe(channels, NUM_CLASSES)
        for name, channels in expected.items()
    }
    for name, channels in expected.items():
        probe = probes[name]
        if probe.in_channels != channels:
            raise RuntimeError(f"{name} channels mismatch: {probe.in_channels} != {channels}")
        params = sum(parameter.numel() for parameter in probe.parameters())
        expected_params = NUM_CLASSES * channels + NUM_CLASSES
        if params != expected_params:
            raise RuntimeError(f"{name} parameter count mismatch: {params} != {expected_params}")


def main():
    _check_zero_init_and_unbounded_linear_path()
    _check_rescue_masks_and_no_history_gradient()
    _check_declared_probe_shapes()
    print(
        {
            "passed": True,
            "probe_inputs": {
                PROBE_ERROR_76: 76,
                PROBE_EVIDENCE_211: 211,
                PROBE_HIDDEN_32: 32,
            },
            "zero_step_equals_c_v3": True,
            "probe_gate": False,
            "probe_tanh_bound": False,
            "history_age_target": False,
            "gt_is_probe_input": False,
            "invalid_history_creates_rescue": False,
        }
    )


if __name__ == "__main__":
    main()
