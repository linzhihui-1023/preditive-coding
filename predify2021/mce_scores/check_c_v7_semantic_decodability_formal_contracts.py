"""Contract checks for the formal C-V7 semantic decodability diagnostic.

中文：C-V7 语义可读出性正式判别实验契约检查。
No KITTI-STEP, checkpoint, Host, RAFT, or GPU is required.
"""

import torch

from predify2021.mce_scores import diagnose_c_v7_semantic_decodability_probes as base
from predify2021.mce_scores import diagnose_c_v7_semantic_decodability_probes_rms as rms_base
from predify2021.mce_scores import diagnose_c_v7_semantic_decodability_probes_formal as formal


def _check_scale_only_rms():
    feature = torch.tensor(
        [[[[0.0, 2.0], [-2.0, 4.0]], [[0.0, 3.0], [-3.0, 6.0]]]],
        dtype=torch.float32,
    )
    scaled = rms_base._normalize_feature(feature, torch.tensor([2.0, 3.0]))
    expected = torch.tensor(
        [[[[0.0, 1.0], [-1.0, 2.0]], [[0.0, 1.0], [-1.0, 2.0]]]],
        dtype=torch.float32,
    )
    if not torch.allclose(scaled, expected):
        raise RuntimeError("RMS normalization is not pure per-channel scaling")
    if float(scaled[..., 0, 0].abs().max().item()) != 0.0:
        raise RuntimeError("RMS normalization introduced an offset")


def _synthetic_frame():
    evidence = torch.zeros((1, 211, 2, 2), dtype=torch.float32)
    evidence[:, -2] = torch.tensor([[0.25, 0.50], [0.75, 1.00]])
    evidence[:, -1] = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    return {
        "features": {
            base.PROBE_ERROR_76: torch.zeros((1, 76, 2, 2)),
            base.PROBE_EVIDENCE_211: evidence,
            base.PROBE_HIDDEN_32: torch.zeros((1, 32, 2, 2)),
        },
        "c_v3_logits": torch.zeros((1, 19, 4, 4)),
    }


def _check_controls():
    frame = _synthetic_frame()
    evidence_rms = torch.ones(211)
    feature_rms = {
        base.PROBE_ERROR_76: torch.ones(76),
        base.PROBE_EVIDENCE_211: evidence_rms,
        base.PROBE_HIDDEN_32: torch.ones(32),
    }
    constant = formal._probe_feature(frame, formal.CONTROL_CONSTANT, feature_rms)
    validity = formal._probe_feature(frame, formal.CONTROL_VALIDITY, feature_rms)

    expected_constant = torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]])
    if not torch.equal(constant, expected_constant):
        raise RuntimeError("constant_valid_1 is not exact any_history_valid")

    expected_validity = frame["features"][base.PROBE_EVIDENCE_211][:, -2:, :, :]
    if not torch.equal(validity, expected_validity):
        raise RuntimeError("validity_2 is not exact final two evidence channels under unit RMS")


def _check_probe_shapes_and_zero_init():
    expected = {
        formal.CONTROL_CONSTANT: 1,
        formal.CONTROL_VALIDITY: 2,
        base.PROBE_ERROR_76: 76,
        base.PROBE_EVIDENCE_211: 211,
        base.PROBE_HIDDEN_32: 32,
    }
    if tuple(formal.PROBE_NAMES) != tuple(expected.keys()):
        raise RuntimeError(f"formal probe ordering mismatch: {formal.PROBE_NAMES}")
    for name, channels in expected.items():
        probe = base.LinearResidualProbe(channels, 19)
        if probe.head.bias is not None:
            raise RuntimeError(f"{name} probe unexpectedly has bias")
        if float(probe.head.weight.detach().abs().max().item()) != 0.0:
            raise RuntimeError(f"{name} probe is not exactly zero initialized")
        params = sum(p.numel() for p in probe.parameters() if p.requires_grad)
        if params != 19 * channels:
            raise RuntimeError(f"{name} parameter count mismatch: {params}")


def _check_unbounded_probe_path():
    probe = base.LinearResidualProbe(1, 19)
    with torch.no_grad():
        probe.head.weight.fill_(3.0)
    feature = torch.ones((1, 1, 2, 2))
    valid = torch.ones((1, 1, 4, 4), dtype=torch.bool)
    delta = probe(feature, (4, 4), valid)
    if not torch.allclose(delta, torch.full_like(delta, 3.0)):
        raise RuntimeError("formal probe output is unexpectedly gated or tanh-bounded")


def main():
    _check_scale_only_rms()
    _check_controls()
    _check_probe_shapes_and_zero_init()
    _check_unbounded_probe_path()
    print(
        {
            "passed": True,
            "formal_runner": "diagnose_c_v7_semantic_decodability_probes_formal.py",
            "controls": [formal.CONTROL_CONSTANT, formal.CONTROL_VALIDITY],
            "semantic_probes": list(base.PROBE_NAMES),
            "probe_bias": False,
            "zero_step_equals_c_v3": True,
            "probe_gate": False,
            "probe_tanh_bound": False,
            "rms_mean_subtraction": False,
            "rms_gt_input": False,
            "history_age_target": False,
            "bottleneck_claim_from_this_test": False,
        }
    )


if __name__ == "__main__":
    main()
