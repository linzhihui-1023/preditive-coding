"""CPU/synthetic contracts for C-V14 post-writeback reliability acceptance."""

import inspect

import torch
from torch.nn import functional as F

from predify2021.mce_scores import c_v14_post_writeback_reliability_training as training
from predify2021.mce_scores import train_kitti_step_task_space_prior_c_v14_post_writeback_reliability_acceptance as c_v14
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_post_writeback_reliability_feature_correction import PostWritebackReliabilityFeatureCorrector


NUM_CLASSES = 19
K = 4
SEMANTIC = 128
TEMPORAL = 32
C4 = 2048


def _model():
    return PostWritebackReliabilityFeatureCorrector(
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
    parameters = set(inspect.signature(PostWritebackReliabilityFeatureCorrector.forward).parameters)
    forbidden = {"raw_history", "history_logits", "history_probabilities", "current_logits", "current_probability", "gt"}
    leaked = sorted(parameters & forbidden)
    if leaked:
        raise RuntimeError(f"Raw history/GT leaked into C-V14 correction forward: {leaked}")
    model = _model()
    if model.semantic_input_channels != K * NUM_CLASSES:
        raise RuntimeError("C-V14 semantic content must be concat(e1..e4)=76D")
    first = model.semantic_encoder[0]
    if first.in_channels != K * NUM_CLASSES or first.out_channels != SEMANTIC:
        raise RuntimeError("C-V14 semantic encoder must start 76D->128D")
    if model.reliability_head.out_channels != 1:
        raise RuntimeError("C-V14 reliability must be single-channel pixel-wise")


def _check_zero_step():
    model = _model().eval()
    data = _inputs()
    current = data["current_c4"].clone()
    out = model(**data)
    for key in ("raw_semantic_delta_c4", "bounded_semantic_delta_c4", "delta_c4"):
        if float(out[key].abs().max().item()) != 0.0:
            raise RuntimeError(f"C-V14 {key} must be exactly zero at initialization")
    if not torch.equal(out["corrected_c4"], current):
        raise RuntimeError("C-V14 corrected c4 must equal current c4 at zero step")
    if not torch.equal(out["proposal_c4"], current):
        raise RuntimeError("C-V14 proposal c4 must equal current c4 at zero step")
    if not torch.allclose(out["reliability"], torch.full_like(out["reliability"], 0.5), atol=1e-7, rtol=0.0):
        raise RuntimeError("C-V14 reliability must initialize at 0.5")


def _check_post_writeback_control_and_bound():
    model = _model().eval()
    with torch.no_grad():
        model.writeback.output_projection.weight.normal_(0.0, 0.2)
    data = _inputs()

    with torch.no_grad():
        model.reliability_head.weight.zero_()
        model.reliability_head.bias.fill_(-20.0)
    off = model(**data)
    with torch.no_grad():
        model.reliability_head.bias.fill_(20.0)
    on = model(**data)

    if not torch.allclose(
        off["bounded_semantic_delta_c4"],
        on["bounded_semantic_delta_c4"],
        atol=1e-7,
        rtol=0.0,
    ):
        raise RuntimeError("Reliability must not alter the semantic proposal before writeback")
    if float(off["delta_c4"].abs().max().item()) >= float(on["delta_c4"].abs().max().item()):
        raise RuntimeError("Post-writeback reliability failed to control final Delta-c4")
    if float(off["delta_c4"].abs().max().item()) > 1e-6:
        raise RuntimeError("g≈0 must force near-zero final Delta-c4")
    if not torch.allclose(
        on["delta_c4"],
        on["bounded_semantic_delta_c4"],
        atol=1e-5,
        rtol=1e-5,
    ):
        raise RuntimeError("g≈1 must pass the bounded semantic proposal")

    channel_rms = data["current_c4"].square().mean(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-6)
    cap = training.RESIDUAL_SCALE * channel_rms
    if bool((on["bounded_semantic_delta_c4"].abs() > cap + 1e-7).any().item()):
        raise RuntimeError("C-V14 semantic proposal exceeded 0.10 x per-channel c4 RMS cap")
    if bool((on["delta_c4"].abs() > on["bounded_semantic_delta_c4"].abs() + 1e-7).any().item()):
        raise RuntimeError("Reliability must not amplify the post-writeback proposal")


def _check_deep_history_and_temporal_conditioning():
    model = _model().eval()
    base = _inputs()
    out_a = model(**base)
    changed = {key: value for key, value in base.items()}
    changed["prediction_errors"] = [value.clone() for value in base["prediction_errors"]]
    changed["prediction_errors"][3] = changed["prediction_errors"][3] + 0.25
    out_b = model(**changed)
    if torch.allclose(out_a["semantic_latent"], out_b["semantic_latent"]):
        raise RuntimeError("t-4 Prediction Error must influence C-V14 semantic proposal content")

    with torch.no_grad():
        model.reliability_head.weight.normal_(0.0, 0.02)
    data_a = _inputs()
    data_b = {key: value for key, value in data_a.items()}
    data_b["temporal_hidden"] = data_a["temporal_hidden"] + 0.5
    rel_a = model(**data_a)["reliability"]
    rel_b = model(**data_b)["reliability"]
    if torch.allclose(rel_a, rel_b):
        raise RuntimeError("Frozen C-V4 temporal hidden must influence C-V14 reliability")
    if float(rel_b.min().item()) < 0.0 or float(rel_b.max().item()) > 1.0:
        raise RuntimeError("C-V14 reliability left [0,1]")


def _check_deployed_gradient_path():
    model = _model().train()
    with torch.no_grad():
        model.writeback.output_projection.weight.normal_(0.0, 1e-3)
        model.reliability_head.weight.normal_(0.0, 0.02)
    out = model(**_inputs())
    loss = out["corrected_c4"].square().mean()
    model.zero_grad(set_to_none=True)
    loss.backward()
    required = {
        "semantic_encoder": model.semantic_encoder[0].weight.grad,
        "temporal_pre": model.temporal_pre[0].weight.grad,
        "reliability_head": model.reliability_head.weight.grad,
        "writeback_host_projection": model.writeback.host_projection.weight.grad,
        "writeback_delta_projection": model.writeback.delta_projection.weight.grad,
        "writeback_output_projection": model.writeback.output_projection.weight.grad,
    }
    for name, grad in required.items():
        if grad is None or float(grad.abs().sum().item()) <= 0.0:
            raise RuntimeError(f"C-V14 deployed path did not train {name}")


def _check_acceptance_targets_and_detached_aux_gradient():
    gt = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    host_logits = torch.full((1, NUM_CLASSES, 1, 4), -5.0)
    proposal_logits = torch.full_like(host_logits, -5.0)
    # Host predictions: [4,1,2,4]; proposal: [0,4,2,4].
    host_pred = torch.tensor([4, 1, 2, 4])
    prop_pred = torch.tensor([0, 4, 2, 4])
    for x in range(4):
        host_logits[0, host_pred[x], 0, x] = 5.0
        proposal_logits[0, prop_pred[x], 0, x] = 5.0
    target, supervised, beneficial, harmful = training.proposal_acceptance_targets(
        host_logits, proposal_logits, gt
    )
    if beneficial.tolist() != [[True, False, False, False]]:
        raise RuntimeError("C-V14 beneficial target definition drifted")
    if harmful.tolist() != [[False, True, False, False]]:
        raise RuntimeError("C-V14 harmful target definition drifted")
    if supervised.tolist() != [[True, True, False, False]]:
        raise RuntimeError("C-V14 must ignore ambiguous/no-change proposal pixels")
    if target.tolist() != [[1.0, 0.0, 0.0, 0.0]]:
        raise RuntimeError("C-V14 acceptance target values are wrong")

    model = _model().train()
    with torch.no_grad():
        model.writeback.output_projection.weight.normal_(0.0, 1e-3)
    row = model(**_inputs(error_size=8, c4_size=4))
    aux_target = torch.zeros((8, 8))
    aux_target[:4] = 1.0
    aux_mask = torch.ones((8, 8), dtype=torch.bool)
    aux_loss, _ = training.acceptance_bce_loss(
        model, row, (8, 8), aux_target, aux_mask
    )
    model.zero_grad(set_to_none=True)
    aux_loss.backward()
    if model.reliability_head.weight.grad is None or float(model.reliability_head.weight.grad.abs().sum().item()) <= 0.0:
        raise RuntimeError("Acceptance BCE must train reliability_head")
    if model.temporal_pre[0].weight.grad is not None and float(model.temporal_pre[0].weight.grad.abs().sum().item()) > 0.0:
        raise RuntimeError("Acceptance BCE must not backpropagate into temporal_pre")
    if model.semantic_encoder[0].weight.grad is not None and float(model.semantic_encoder[0].weight.grad.abs().sum().item()) > 0.0:
        raise RuntimeError("Acceptance BCE must not backpropagate into semantic proposal")


def _check_protection_supervision():
    gt = torch.randint(0, NUM_CLASSES, (8, 8))
    teacher_logits = torch.full((1, NUM_CLASSES, 8, 8), -5.0)
    teacher_logits.scatter_(1, gt.unsqueeze(0).unsqueeze(1), 5.0)
    same = teacher_logits.clone().requires_grad_(True)
    zero_loss, mask = training.protection_kl_loss(same, teacher_logits, gt)
    if int(mask.sum().item()) != gt.numel():
        raise RuntimeError("Synthetic C-V4-correct mask should cover all pixels")
    if float(zero_loss.abs().item()) > 1e-6:
        raise RuntimeError("Protection KL must be zero when C-V14 equals C-V4 teacher")


def _check_training_and_composition_contract():
    helper_source = inspect.getsource(training)
    entry_source = inspect.getsource(c_v14)
    model_source = inspect.getsource(PostWritebackReliabilityFeatureCorrector)
    required_helper = (
        'loss = (',
        'PROTECTION_LOSS_WEIGHT * protect_kl',
        'ACCEPTANCE_LOSS_WEIGHT * accept_bce',
        'raw_history.insert(0, c_v3_logits.detach())',
        'proposal_acceptance_targets(',
        'reliability_logits_from_detached_context',
    )
    for token in required_helper:
        if token not in helper_source:
            raise RuntimeError(f"C-V14 helper lost required contract: {token}")
    required_model = (
        'raw_semantic_delta_c4 = self.writeback(current_c4, semantic_latent_c4)',
        'delta_c4 = bounded_semantic_delta_c4 * reliability_c4',
        'self.reliability_head = nn.Conv2d(',
    )
    for token in required_model:
        if token not in model_source:
            raise RuntimeError(f"C-V14 model lost post-writeback reliability contract: {token}")
    forbidden = (
        'semantic_latent_c4 * temporal_reliability_c4',
        'baseline_logits.detach() + feature_delta_logits',
        'RESCUE_LOSS_WEIGHT',
        'rescue_ce =',
    )
    for token in forbidden:
        if token in helper_source or token in model_source:
            raise RuntimeError(f"C-V14 retained forbidden prior behavior: {token}")
    required_entry = (
        '"beneficial_harmful_acceptance_bce": True',
        '"acceptance_bce_detached_temporal_context": True',
        '"rescue_ce": False',
        '"temporal_loss": False',
        '"raft_training": False',
        '"reliability_position": "after bounded semantic writeback"',
    )
    for token in required_entry:
        if token not in entry_source:
            raise RuntimeError(f"C-V14 entrypoint lost required experiment contract: {token}")
    if training.RESIDUAL_SCALE != 0.10:
        raise RuntimeError("C-V14 residual scale must stay fixed at 0.10")
    if training.PROTECTION_LOSS_WEIGHT != 1.0 or training.ACCEPTANCE_LOSS_WEIGHT != 1.0:
        raise RuntimeError("C-V14 fixed loss weights must all equal 1.0")


def _check_model_selection():
    reference = {"mIoU": 0.6635, "mTC": 0.7132}
    good = {"c_v4_frozen": reference, "c_v14": {"mIoU": 0.6645, "mTC": 0.7142}}
    semantic_fail = {"c_v4_frozen": reference, "c_v14": {"mIoU": 0.6634, "mTC": 0.7200}}
    temporal_fail = {"c_v4_frozen": reference, "c_v14": {"mIoU": 0.6700, "mTC": 0.7131}}
    if c_v14._selection_key(good)[0] != 1:
        raise RuntimeError("C-V14 candidate preserving both goals must pass")
    if c_v14._selection_key(semantic_fail)[0] != 0:
        raise RuntimeError("C-V14 must reject semantic regression")
    if c_v14._selection_key(temporal_fail)[0] != 0:
        raise RuntimeError("C-V14 must reject temporal regression")


def main():
    _check_predictive_coding_boundary()
    _check_zero_step()
    _check_post_writeback_control_and_bound()
    _check_deep_history_and_temporal_conditioning()
    _check_deployed_gradient_path()
    _check_acceptance_targets_and_detached_aux_gradient()
    _check_protection_supervision()
    _check_training_and_composition_contract()
    _check_model_selection()
    model = _model()
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print({
        "passed": True,
        "architecture": "K=4 Prediction Error -> bounded semantic c4 proposal -> post-writeback pixel reliability",
        "semantic_content": "e1..e4 -> 128D",
        "reliability": "single-channel sigmoid after bounded writeback",
        "residual_bound": "0.10 x per-channel RMS(c4)",
        "training": "all-pixel CE + C-V4 protection KL + beneficial/harmful acceptance BCE",
        "acceptance_aux_gradient": "reliability_head only",
        "rescue_supervision": False,
        "trainable_parameters": trainable,
        "history_feedback": False,
        "temporal_loss": False,
        "raft_training": False,
    })


if __name__ == "__main__":
    main()
