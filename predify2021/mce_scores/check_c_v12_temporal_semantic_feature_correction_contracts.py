"""CPU/synthetic contracts for C-V12 Temporal Semantic Feature Correction."""

import inspect

import torch
from torch.nn import functional as F

from predify2021.mce_scores import c_v12_temporal_semantic_feature_training as training
from predify2021.mce_scores import (
    train_kitti_step_task_space_prior_c_v12_temporal_semantic_feature_correction as c_v12,
)
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_temporal_semantic_feature_correction import (
    TemporalSemanticFeatureCorrector,
)


NUM_CLASSES = 19
K = 4
SEMANTIC = 128
TEMPORAL = 32
C4 = 2048


def _model():
    return TemporalSemanticFeatureCorrector(
        num_classes=NUM_CLASSES,
        history_length=K,
        semantic_channels=SEMANTIC,
        temporal_hidden_channels=TEMPORAL,
        host_channels=C4,
    )


def _inputs(error_size=8, c4_size=4):
    return {
        "prediction_errors": [
            torch.randn(1, NUM_CLASSES, error_size, error_size) * 0.05
            for _ in range(K)
        ],
        "history_validities_low": [
            torch.ones(1, 1, error_size, error_size) for _ in range(K)
        ],
        "temporal_hidden": torch.randn(1, TEMPORAL, error_size, error_size) * 0.1,
        "dynamics_error": torch.randn(1, NUM_CLASSES, error_size, error_size) * 0.02,
        "transportability_low": torch.ones(1, 1, error_size, error_size),
        "memory_reliability_low": torch.ones(1, 1, error_size, error_size),
        "current_c4": torch.randn(1, C4, c4_size, c4_size),
    }


def _check_predictive_coding_boundary():
    parameters = set(inspect.signature(TemporalSemanticFeatureCorrector.forward).parameters)
    forbidden = {
        "raw_history",
        "history_logits",
        "history_probabilities",
        "current_logits",
        "current_probability",
    }
    leaked = sorted(parameters & forbidden)
    if leaked:
        raise RuntimeError(f"Raw semantic history leaked into C-V12 correction forward: {leaked}")
    model = _model()
    if model.semantic_input_channels != K * NUM_CLASSES:
        raise RuntimeError("C-V12 semantic content must be concat(e1..e4)=76D")
    first = model.semantic_encoder[0]
    if first.in_channels != K * NUM_CLASSES or first.out_channels != SEMANTIC:
        raise RuntimeError("C-V12 semantic encoder must start 76D->128D")


def _check_feature_target_and_zero_step():
    model = _model().eval()
    data = _inputs()
    current_c4 = data["current_c4"].clone()
    out = model(**data)
    if out["delta_c4"].shape[1] != C4:
        raise RuntimeError("C-V12 must write a 2048D c4 residual")
    if float(out["delta_c4"].abs().max().item()) != 0.0:
        raise RuntimeError("C-V12 Delta-c4 must be exactly zero at initialization")
    if not torch.equal(out["corrected_c4"], current_c4):
        raise RuntimeError("C-V12 corrected c4 must equal the actual input current_c4 at zero step")
    if not torch.allclose(
        out["temporal_gain"],
        torch.ones_like(out["temporal_gain"]),
        atol=1e-7,
        rtol=0.0,
    ):
        raise RuntimeError("C-V12 temporal modulation must be exactly neutral at initialization")
    if float(model.writeback.output_projection.weight.abs().max().item()) != 0.0:
        raise RuntimeError("C-V12 writeback final projection must be zero initialized")


def _check_zero_step_final_composition():
    class DummyDecoder:
        @staticmethod
        def decode_from_host_feature(host_feature):
            logits = host_feature.tensor[:, :NUM_CLASSES]
            return F.interpolate(
                logits,
                size=host_feature.output_size,
                mode="bilinear",
                align_corners=False,
            )

    corrector = _model().eval()
    data = _inputs(error_size=8, c4_size=4)
    decoder = DummyDecoder()
    output_size = (8, 8)
    host_logits = decoder.decode_from_host_feature(
        training.HostFeature(
            data["current_c4"],
            torch.zeros(1, 1, 8, 8),
            output_size,
        )
    )
    baseline_logits = torch.randn(1, NUM_CLASSES, *output_size)
    observation = {
        "c4": data["current_c4"],
        "c1": torch.zeros(1, 1, 8, 8),
        "output_size": output_size,
        "host_logits": host_logits,
    }
    error_row = {
        "prediction_errors": data["prediction_errors"],
        "history_validities_low": data["history_validities_low"],
        "temporal_hidden": data["temporal_hidden"],
        "dynamics_state": data["dynamics_error"],
    }
    final_logits, feature_delta_logits, _ = training.decode_feature_correction(
        decoder,
        corrector,
        observation,
        baseline_logits,
        error_row,
        data["transportability_low"],
        data["memory_reliability_low"],
    )
    if float(feature_delta_logits.abs().max().item()) != 0.0:
        raise RuntimeError("C-V12 decoded feature effect must be exactly zero at initialization")
    if not torch.equal(final_logits, baseline_logits):
        raise RuntimeError("C-V12 zero-step final logits must exactly equal frozen C-V4 baseline logits")


def _check_deep_history_semantic_content():
    model = _model().eval()
    base = _inputs()
    out_a = model(**base)
    changed = {key: value for key, value in base.items()}
    changed["prediction_errors"] = [value.clone() for value in base["prediction_errors"]]
    changed["prediction_errors"][3] = changed["prediction_errors"][3] + 0.25
    out_b = model(**changed)
    if torch.allclose(out_a["semantic_latent"], out_b["semantic_latent"]):
        raise RuntimeError("t-4 Prediction Error must influence the semantic correction content")


def _check_temporal_branch_is_feature_wise():
    model = _model().eval()
    if model.temporal_modulation_head.out_channels != SEMANTIC:
        raise RuntimeError("Temporal modulation must be 128D feature-wise, not a scalar gate")
    with torch.no_grad():
        model.temporal_modulation_head.weight.normal_(0.0, 0.02)
    data_a = _inputs()
    data_b = {key: value for key, value in data_a.items()}
    data_b["temporal_hidden"] = data_a["temporal_hidden"] + 0.5
    gain_a = model(**data_a)["temporal_gain"]
    gain_b = model(**data_b)["temporal_gain"]
    if torch.allclose(gain_a, gain_b):
        raise RuntimeError("Frozen C-V4 temporal hidden must influence feature-wise modulation")


def _check_joint_gradient_after_writeback_opens():
    model = _model().train()
    with torch.no_grad():
        model.writeback.output_projection.weight.normal_(0.0, 1e-3)
        model.temporal_modulation_head.weight.normal_(0.0, 0.02)
    out = model(**_inputs())
    loss = out["corrected_c4"].square().mean()
    model.zero_grad(set_to_none=True)
    loss.backward()
    required = {
        "semantic_encoder": model.semantic_encoder[0].weight.grad,
        "temporal_pre": model.temporal_pre[0].weight.grad,
        "temporal_modulation": model.temporal_modulation_head.weight.grad,
        "writeback_host_projection": model.writeback.host_projection.weight.grad,
        "writeback_delta_projection": model.writeback.delta_projection.weight.grad,
        "writeback_output_projection": model.writeback.output_projection.weight.grad,
    }
    for name, grad in required.items():
        if grad is None or float(grad.abs().sum().item()) <= 0.0:
            raise RuntimeError(f"C-V12 deployed feature path did not train {name}")


def _check_c_v4_baseline_and_training_protocol():
    helper_source = inspect.getsource(training)
    entry_source = inspect.getsource(c_v12)
    required_helper = (
        'final_logits = baseline_logits.detach() + feature_delta_logits',
        'corrected_host_logits - observation["host_logits"].detach()',
        'RESCUE_LOSS_WEIGHT * rescue_ce',
        'raw_history.insert(0, c_v3_logits.detach())',
    )
    for token in required_helper:
        if token not in helper_source:
            raise RuntimeError(f"C-V12 helper lost required architecture/training contract: {token}")
    required_entry = (
        'reference = metrics["c_v4_frozen"]',
        'c_v4_controller, cv4_payload = _load_frozen_c_v4_controller(args.c_v4_checkpoint)',
        '"c_v4_reference_policy": "actual loaded checkpoint and same-run c_v4_frozen metrics"',
    )
    for token in required_entry:
        if token not in entry_source:
            raise RuntimeError(f"C-V12 entrypoint lost dynamic C-V4 baseline contract: {token}")
    forbidden_control = (
        "c_v10_adaptive_amplitude",
        "c_v11_reliability_conditioned",
        "amplitude_head",
        "expansion_reliability_head",
    )
    for token in forbidden_control:
        if token in helper_source or token in entry_source:
            raise RuntimeError(f"C-V12 must not contain forbidden control machinery: {token}")
    if "requires C-V4 balanced-best Epoch" in entry_source:
        raise RuntimeError("C-V12 entrypoint must not hard-code one C-V4 epoch")
    if training.RESCUE_LOSS_WEIGHT != 1.0:
        raise RuntimeError("C-V12 fixed Rescue loss weight must be 1.0")
    if '"temporal_loss": False' not in entry_source or '"raft_training": False' not in entry_source:
        raise RuntimeError("C-V12 must have no temporal loss and no RAFT training teacher")


def _check_model_selection_prioritizes_both_goals():
    reference = {"mIoU": 0.6635, "mTC": 0.7127}
    good = {
        "c_v4_frozen": reference,
        "c_v12": {"mIoU": reference["mIoU"] + 1e-3, "mTC": reference["mTC"] + 1e-3},
    }
    semantic_fail = {
        "c_v4_frozen": reference,
        "c_v12": {"mIoU": reference["mIoU"] - 1e-5, "mTC": reference["mTC"] + 1e-2},
    }
    temporal_fail = {
        "c_v4_frozen": reference,
        "c_v12": {"mIoU": reference["mIoU"] + 1e-2, "mTC": reference["mTC"] - 1e-5},
    }
    if c_v12._selection_key(good)[0] != 1:
        raise RuntimeError("C-V12 passing semantic+temporal candidate must pass selection gate")
    if c_v12._selection_key(semantic_fail)[0] != 0:
        raise RuntimeError("C-V12 must reject mIoU regression even when temporal metrics improve")
    if c_v12._selection_key(temporal_fail)[0] != 0:
        raise RuntimeError("C-V12 must reject temporal regression even when mIoU improves")


def main():
    _check_predictive_coding_boundary()
    _check_feature_target_and_zero_step()
    _check_zero_step_final_composition()
    _check_deep_history_semantic_content()
    _check_temporal_branch_is_feature_wise()
    _check_joint_gradient_after_writeback_opens()
    _check_c_v4_baseline_and_training_protocol()
    _check_model_selection_prioritizes_both_goals()
    model = _model()
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print({
        "passed": True,
        "architecture": "frozen C-V4 temporal base + K=4 Prediction-Error-driven c4 residual",
        "semantic_content": "e1..e4 -> 128D",
        "temporal_modulation": "frozen C-V4 hidden + dynamics -> 128D channel-wise gain",
        "feature_target": "c4 2048D",
        "zero_step": "exact frozen C-V4 final composition",
        "c_v4_reference": "actual loaded checkpoint and same-run c_v4_frozen metrics",
        "trainable_parameters": trainable,
        "history_feedback": False,
        "temporal_loss": False,
        "raft_training": False,
    })


if __name__ == "__main__":
    main()
