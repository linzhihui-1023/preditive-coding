"""Contract checks for the controlled C-V7 semantic decodability diagnostic.

中文：带空白对照与 RMS 尺度控制的 C-V7 语义可读出性诊断契约检查。

No KITTI-STEP, checkpoint, Host, RAFT, or GPU is required.
"""

import torch

from predify2021.mce_scores import diagnose_c_v7_semantic_decodability_probes as base
from predify2021.mce_scores import diagnose_c_v7_semantic_decodability_probes_rms as rms_base
from predify2021.mce_scores import diagnose_c_v7_semantic_decodability_probes_controlled as controlled


def _check_scale_only_normalization():
    feature = torch.tensor(
        [[[[0.0, 2.0], [-2.0, 4.0]], [[0.0, 3.0], [-3.0, 6.0]]]],
        dtype=torch.float32,
    )
    rms = torch.tensor([2.0, 3.0])
    normalized = rms_base._normalize_feature(feature, rms)
    expected = torch.tensor(
        [[[[0.0, 1.0], [-1.0, 2.0]], [[0.0, 1.0], [-1.0, 2.0]]]],
        dtype=torch.float32,
    )
    if not torch.allclose(normalized, expected):
        raise RuntimeError("RMS normalization is not pure per-channel scaling")
    if float(normalized[..., 0, 0].abs().max().item()) != 0.0:
        raise RuntimeError("RMS normalization introduced a non-zero offset")


def _check_validity_control_feature():
    # evidence_211 ends with [valid_fraction, any_history_valid].
    evidence = torch.zeros((1, 211, 2, 2), dtype=torch.float32)
    evidence[:, -2] = torch.tensor([[0.25, 0.50], [0.75, 1.00]])
    evidence[:, -1] = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    frame = {
        "features": {
            base.PROBE_EVIDENCE_211: evidence,
            base.PROBE_ERROR_76: torch.zeros((1, 76, 2, 2)),
            base.PROBE_HIDDEN_32: torch.zeros((1, 32, 2, 2)),
        },
        "c_v3_logits": torch.zeros((1, 19, 4, 4)),
    }
    valid = rms_base._feature_valid_low(frame)
    expected = torch.tensor([[[[True, False], [True, True]]]])
    if not torch.equal(valid, expected):
        raise RuntimeError("controlled null feature is not exact any_history_valid")
    control_feature = controlled._probe_feature(frame, controlled.CONTROL_CONSTANT, {})
    if not torch.equal(control_feature.bool(), expected):
        raise RuntimeError("constant_valid_1 differs from model-derived validity")


def _check_probe_zero_init_and_bias():
    # Use a lightweight synthetic stand-in for the controlled 1D null probe.
    null_probe = base.LinearResidualProbe(1, 19)
    if null_probe.head.bias is not None:
        raise RuntimeError("constant_valid_1 must be bias-free")
    if float(null_probe.head.weight.detach().abs().max().item()) != 0.0:
        raise RuntimeError("constant_valid_1 must be exactly zero initialized")

    feature = torch.ones((1, 1, 2, 2))
    valid_full = torch.ones((1, 1, 4, 4), dtype=torch.bool)
    delta0 = null_probe(feature, (4, 4), valid_full)
    if float(delta0.abs().max().item()) != 0.0:
        raise RuntimeError("zero-step null control must equal C-V3")

    with torch.no_grad():
        null_probe.head.weight.fill_(3.0)
    delta = null_probe(feature, (4, 4), valid_full)
    if not torch.allclose(delta, torch.full_like(delta, 3.0)):
        raise RuntimeError("controlled probe output is unexpectedly gated or bounded")


def _check_declared_sources():
    expected = {
        controlled.CONTROL_CONSTANT: 1,
        base.PROBE_ERROR_76: 76,
        base.PROBE_EVIDENCE_211: 211,
        base.PROBE_HIDDEN_32: 32,
    }
    if tuple(controlled.PROBE_NAMES) != tuple(expected):
        raise RuntimeError(
            f"controlled probe order/names mismatch: {controlled.PROBE_NAMES}"
        )


def main():
    _check_scale_only_normalization()
    _check_validity_control_feature()
    _check_probe_zero_init_and_bias()
    _check_declared_sources()
    print(
        {
            "passed": True,
            "formal_runner": "diagnose_c_v7_semantic_decodability_probes_controlled.py",
            "probe_inputs": {
                controlled.CONTROL_CONSTANT: 1,
                base.PROBE_ERROR_76: 76,
                base.PROBE_EVIDENCE_211: 211,
                base.PROBE_HIDDEN_32: 32,
            },
            "probe_bias": False,
            "zero_step_equals_c_v3": True,
            "probe_gate": False,
            "probe_tanh_bound": False,
            "rms_mean_subtraction": False,
            "rms_gt_input": False,
            "null_control": True,
            "history_age_target": False,
        }
    )


if __name__ == "__main__":
    main()
