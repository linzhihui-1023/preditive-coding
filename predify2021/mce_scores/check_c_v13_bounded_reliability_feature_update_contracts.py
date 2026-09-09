"""CPU/synthetic contracts for C-V13 bounded reliability feature update."""

import inspect

import torch
from torch.nn import functional as F

from predify2021.mce_scores import c_v13_bounded_reliability_feature_training as training
from predify2021.mce_scores import train_kitti_step_task_space_prior_c_v13_bounded_reliability_feature_update as c_v13
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_bounded_temporal_semantic_feature_correction import BoundedTemporalSemanticFeatureCorrector


NUM_CLASSES = 19
K = 4
SEMANTIC = 128
TEMPORAL = 32
C4 = 2048


def _model():
    return BoundedTemporalSemanticFeatureCorrector(
        num_classes=NUM_CLASSES,
        history_length=K,
        semantic_channels=SEMANTIC,
        temporal_hidden_channels=TEMPORAL,
        host_channels=C4,
        residual_scale=training.RESIDUAL_SCALE,
    )


def _inputs(error_size=8, c4_size=4):
    return {
        "prediction_errors": [torch.randn(1, NUM_CLASSES, error_size, error_size) * 0.05 for _ in range(K)],
        "history_validities_low": [torch.ones(1, 1, error_size, error_size) for _ in range(K)],
        "temporal_hidden": torch.randn(1, TEMPORAL, error_size, error_size) * 0.1,
        "dynamics_error": torch.randn(1, NUM_CLASSES, error_size, error_size) * 0.02,
        "transportability_low": torch.ones(1, 1, error_size, error_size),
        "memory_reliability_low": torch.ones(1, 1, error_size, error_size),
        "current_c4": torch.randn(1, C4, c4_size, c4_size),
    }


def _check_predictive_coding_boundary():
    parameters = set(inspect.signature(BoundedTemporalSemanticFeatureCorrector.forward).parameters)
    forbidden = {"raw_history", "history_logits", "history_probabilities", "current_logits", "current_probability"}
    leaked = sorted(parameters & forbidden)
    if leaked:
        raise RuntimeError(f"Raw semantic history leaked into C-V13 correction forward: {leaked}")
    model = _model()
    if model.semantic_input_channels != K * NUM_CLASSES:
        raise RuntimeError("C-V13 semantic content must be concat(e1..e4)=76D")
    first = model.semantic_encoder[0]
    if first.in_channels != K * NUM_CLASSES or first.out_channels != SEMANTIC:
        raise RuntimeError("C-V13 semantic encoder must start 76D->128D")


def _check_zero_step_and_reliability_range():
    model = _model().eval()
    data = _inputs()
    current = data["current_c4"].clone()
    out = model(**data)
    if float(out["raw_delta_c4"].abs().max().item()) != 0.0:
        raise RuntimeError("C-V13 raw Delta-c4 must be exactly zero at initialization")
    if float(out["delta_c4"].abs().max().item()) != 0.0:
        raise RuntimeError("C-V13 bounded Delta-c4 must be exactly zero at initialization")
    if not torch.equal(out["corrected_c4"], current):
        raise RuntimeError("C-V13 corrected c4 must equal current c4 at zero step")
    reliability = out["temporal_reliability"]
    if float(reliability.min().item()) < 0.0 or float(reliability.max().item()) > 1.0:
        raise RuntimeError("C-V13 temporal reliability must stay in [0,1]")
    if not torch.allclose(reliability, torch.full_like(reliability, 0.5), atol=1e-7, rtol=0.0):
        raise RuntimeError("C-V13 temporal reliability must initialize at 0.5")


def _check_zero_step_final_is_host():
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
    c1 = torch.zeros(1, 1, 8, 8)
    host_logits = decoder.decode_from_host_feature(
        training.HostFeature(data["current_c4"], c1, output_size)
    )
    observation = {
        "c4": data["current_c4"],
        "c1": c1,
        "output_size": output_size,
        "host_logits": host_logits,
    }
    error_row = {
        "prediction_errors": data["prediction_errors"],
        "history_validities_low": data["history_validities_low"],
        "temporal_hidden": data["temporal_hidden"],
        "dynamics_state": data["dynamics_error"],
    }
    final_logits, feature_delta_logits, row = training.decode_bounded_feature_update(
        decoder,
        corrector,
        observation,
        error_row,
        data["transportability_low"],
        data["memory_reliability_low"],
    )
    if float(row["delta_c4"].abs().max().item()) != 0.0:
        raise RuntimeError("C-V13 zero-step deployed path must have zero Delta-c4")
    if float(feature_delta_logits.abs().max().item()) != 0.0:
        raise RuntimeError("C-V13 zero-step decoded feature effect must be exactly zero")
    if not torch.equal(final_logits, host_logits):
        raise RuntimeError("C-V13 zero-step final logits must exactly equal current Host logits")


def _check_bounded_feature_update():
    model = _model().eval()
    with torch.no_grad():
        model.writeback.output_projection.weight.normal_(0.0, 0.2)
    data = _inputs(error_size=8, c4_size=4)
    out = model(**data)
    channel_rms = data["current_c4"].square().mean(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-6)
    cap = training.RESIDUAL_SCALE * channel_rms
    if bool((out["delta_c4"].abs() > cap + 1e-7).any().item()):
        raise RuntimeError("C-V13 Delta-c4 exceeded the fixed 0.10 x per-channel c4 RMS cap")
    global_ratio = out["delta_c4"].square().mean().sqrt() / data["current_c4"].square().mean().sqrt().clamp_min(1e-8)
    if float(global_ratio.item()) > training.RESIDUAL_SCALE + 1e-6:
        raise RuntimeError("C-V13 global Delta-c4 RMS ratio exceeded residual_scale")


def _check_deep_history_and_temporal_conditioning():
    model = _model().eval()
    base = _inputs()
    out_a = model(**base)
    changed = {key: value for key, value in base.items()}
    changed["prediction_errors"] = [value.clone() for value in base["prediction_errors"]]
    changed["prediction_errors"][3] = changed["prediction_errors"][3] + 0.25
    out_b = model(**changed)
    if torch.allclose(out_a["semantic_latent"], out_b["semantic_latent"]):
        raise RuntimeError("t-4 Prediction Error must influence C-V13 semantic correction content")

    with torch.no_grad():
        model.temporal_reliability_head.weight.normal_(0.0, 0.02)
    data_a = _inputs()
    data_b = {key: value for key, value in data_a.items()}
    data_b["temporal_hidden"] = data_a["temporal_hidden"] + 0.5
    rel_a = model(**data_a)["temporal_reliability"]
    rel_b = model(**data_b)["temporal_reliability"]
    if torch.allclose(rel_a, rel_b):
        raise RuntimeError("Frozen C-V4 temporal hidden must influence C-V13 reliability")
    if float(rel_b.min().item()) < 0.0 or float(rel_b.max().item()) > 1.0:
        raise RuntimeError("C-V13 reliability left [0,1] after temporal conditioning")


def _check_joint_gradient_after_writeback_opens():
    model = _model().train()
    with torch.no_grad():
        model.writeback.output_projection.weight.normal_(0.0, 1e-3)
        model.temporal_reliability_head.weight.normal_(0.0, 0.02)
    out = model(**_inputs())
    loss = out["corrected_c4"].square().mean()
    model.zero_grad(set_to_none=True)
    loss.backward()
    required = {
        "semantic_encoder": model.semantic_encoder[0].weight.grad,
        "temporal_pre": model.temporal_pre[0].weight.grad,
        "temporal_reliability": model.temporal_reliability_head.weight.grad,
        "writeback_host_projection": model.writeback.host_projection.weight.grad,
        "writeback_delta_projection": model.writeback.delta_projection.weight.grad,
        "writeback_output_projection": model.writeback.output_projection.weight.grad,
    }
    for name, grad in required.items():
        if grad is None or float(grad.abs().sum().item()) <= 0.0:
            raise RuntimeError(f"C-V13 deployed feature path did not train {name}")


def _check_protection_supervision():
    gt = torch.randint(0, NUM_CLASSES, (8, 8))
    teacher_logits = torch.full((1, NUM_CLASSES, 8, 8), -5.0)
    teacher_logits.scatter_(1, gt.unsqueeze(0).unsqueeze(1), 5.0)
    same = teacher_logits.clone().requires_grad_(True)
    zero_loss, mask = training.protection_kl_loss(same, teacher_logits, gt)
    if int(mask.sum().item()) != gt.numel():
        raise RuntimeError("Synthetic C-V4-correct mask should cover all pixels")
    if float(zero_loss.abs().item()) > 1e-6:
        raise RuntimeError("Protection KL must be zero when C-V13 equals C-V4 teacher")
    perturbed = torch.zeros_like(teacher_logits, requires_grad=True)
    positive_loss, _ = training.protection_kl_loss(perturbed, teacher_logits, gt)
    if float(positive_loss.item()) <= 0.0:
        raise RuntimeError("Protection KL must penalize deviation from correct C-V4 teacher")


def _check_training_and_final_composition_contract():
    helper_source = inspect.getsource(training)
    entry_source = inspect.getsource(c_v13)
    decode_parameters = set(inspect.signature(training.decode_bounded_feature_update).parameters)
    if "baseline_logits" in decode_parameters or "c_v4_logits" in decode_parameters:
        raise RuntimeError("C-V13 deployed decoder must not take C-V4/baseline logits as composition input")
    required_helper = (
        'final_logits = model.decode_from_host_feature(',
        'row["corrected_c4"]',
        'loss = final_ce + PROTECTION_LOSS_WEIGHT * protect_kl',
        'raw_history.insert(0, c_v3_logits.detach())',
    )
    for token in required_helper:
        if token not in helper_source:
            raise RuntimeError(f"C-V13 helper lost required contract: {token}")
    forbidden_helper = (
        'baseline_logits.detach() + feature_delta_logits',
        'RESCUE_LOSS_WEIGHT',
        'rescue_ce =',
        '2.0 * torch.sigmoid',
    )
    for token in forbidden_helper:
        if token in helper_source:
            raise RuntimeError(f"C-V13 helper retained forbidden C-V12 behavior: {token}")
    required_entry = (
        '"rescue_ce": False',
        '"temporal_loss": False',
        '"raft_training": False',
        '"final_composition": "Decoder(c4 + bounded Delta-c4)"',
    )
    for token in required_entry:
        if token not in entry_source:
            raise RuntimeError(f"C-V13 entrypoint lost required experiment contract: {token}")
    if training.RESIDUAL_SCALE != 0.10:
        raise RuntimeError("C-V13 residual scale must be fixed at 0.10")
    if training.PROTECTION_LOSS_WEIGHT != 1.0:
        raise RuntimeError("C-V13 protection loss weight must be fixed at 1.0")


def _check_model_selection():
    reference = {"mIoU": 0.6635, "mTC": 0.7132}
    good = {"c_v4_frozen": reference, "c_v13": {"mIoU": 0.6645, "mTC": 0.7142}}
    semantic_fail = {"c_v4_frozen": reference, "c_v13": {"mIoU": 0.6634, "mTC": 0.7200}}
    temporal_fail = {"c_v4_frozen": reference, "c_v13": {"mIoU": 0.6700, "mTC": 0.7131}}
    if c_v13._selection_key(good)[0] != 1:
        raise RuntimeError("C-V13 candidate preserving both goals must pass")
    if c_v13._selection_key(semantic_fail)[0] != 0:
        raise RuntimeError("C-V13 must reject semantic regression")
    if c_v13._selection_key(temporal_fail)[0] != 0:
        raise RuntimeError("C-V13 must reject temporal regression")


def main():
    _check_predictive_coding_boundary()
    _check_zero_step_and_reliability_range()
    _check_zero_step_final_is_host()
    _check_bounded_feature_update()
    _check_deep_history_and_temporal_conditioning()
    _check_joint_gradient_after_writeback_opens()
    _check_protection_supervision()
    _check_training_and_final_composition_contract()
    _check_model_selection()
    model = _model()
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print({
        "passed": True,
        "architecture": "C-V4 temporal reference + K=4 Prediction Error + bounded c4 feature update",
        "semantic_content": "e1..e4 -> 128D",
        "temporal_reliability": "128D sigmoid in [0,1] before bounded writeback",
        "hard_safety_guarantee": "final Delta-c4 is capped by 0.10 x per-channel RMS(c4)",
        "zero_step": "final logits exactly equal current Host logits",
        "residual_bound": "0.10 x per-channel RMS(c4) x tanh(raw_delta)",
        "final_composition": "Decoder(c4 + bounded Delta-c4)",
        "training": "all-pixel CE + C-V4-correct-region protection KL",
        "rescue_supervision": False,
        "trainable_parameters": trainable,
        "history_feedback": False,
        "temporal_loss": False,
        "raft_training": False,
    })


if __name__ == "__main__":
    main()
