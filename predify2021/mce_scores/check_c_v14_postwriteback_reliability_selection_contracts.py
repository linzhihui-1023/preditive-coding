"""CPU/synthetic contracts for C-V14 post-writeback reliability selection."""

import inspect

import torch
from torch.nn import functional as F

from predify2021.mce_scores import c_v14_postwriteback_reliability_training as training
from predify2021.mce_scores import train_kitti_step_task_space_prior_c_v14_postwriteback_reliability_selection as c_v14
from predify2021.model_factory.deeplabv3plus_resnet50 import HostFeature
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_postwriteback_reliability_feature_correction import PostWritebackReliabilityFeatureCorrector


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
    parameters = set(inspect.signature(PostWritebackReliabilityFeatureCorrector.forward).parameters)
    forbidden = {"raw_history", "history_logits", "history_probabilities", "current_logits", "current_probability"}
    leaked = sorted(parameters & forbidden)
    if leaked:
        raise RuntimeError(f"Raw semantic history leaked into C-V14 correction forward: {leaked}")
    model = _model()
    if model.semantic_input_channels != K * NUM_CLASSES:
        raise RuntimeError("C-V14 semantic content must be concat(e1..e4)=76D")
    first = model.semantic_encoder[0]
    if first.in_channels != K * NUM_CLASSES or first.out_channels != SEMANTIC:
        raise RuntimeError("C-V14 semantic encoder must start 76D->128D")
    if model.temporal_reliability_head.out_channels != 1:
        raise RuntimeError("C-V14 Reliability must be one pixel-wise channel")


def _check_zero_step():
    model = _model().eval()
    data = _inputs()
    current = data["current_c4"].clone()
    out = model(**data)
    for key in ("raw_delta_c4_sem", "bounded_delta_c4_sem", "delta_c4"):
        if float(out[key].abs().max().item()) != 0.0:
            raise RuntimeError(f"C-V14 {key} must be exactly zero at initialization")
    if not torch.equal(out["proposal_c4"], current):
        raise RuntimeError("C-V14 proposal c4 must equal current c4 at zero step")
    if not torch.equal(out["corrected_c4"], current):
        raise RuntimeError("C-V14 corrected c4 must equal current c4 at zero step")
    reliability = out["temporal_reliability"]
    if not torch.allclose(reliability, torch.full_like(reliability, 0.5), atol=1e-7, rtol=0.0):
        raise RuntimeError("C-V14 temporal reliability must initialize at 0.5")


def _check_postwriteback_control_and_bound():
    model = _model().eval()
    with torch.no_grad():
        model.writeback.output_projection.weight.normal_(0.0, 0.2)
        model.temporal_reliability_head.weight.zero_()
    data = _inputs(error_size=8, c4_size=4)

    with torch.no_grad():
        model.temporal_reliability_head.bias.fill_(-20.0)
    suppressed = model(**data)
    with torch.no_grad():
        model.temporal_reliability_head.bias.fill_(20.0)
    passed = model(**data)

    if not torch.allclose(
        suppressed["bounded_delta_c4_sem"],
        passed["bounded_delta_c4_sem"],
        atol=1e-7,
        rtol=0.0,
    ):
        raise RuntimeError("Reliability must not change the semantic proposal before post-writeback gating")
    proposal = passed["bounded_delta_c4_sem"]
    if float(proposal.abs().sum().item()) <= 0.0:
        raise RuntimeError("Synthetic C-V14 proposal must be nonzero for post-writeback control test")
    suppressed_ratio = suppressed["delta_c4"].abs().sum() / proposal.abs().sum().clamp_min(1e-8)
    passed_error = (passed["delta_c4"] - proposal).abs().sum() / proposal.abs().sum().clamp_min(1e-8)
    if float(suppressed_ratio.item()) > 1e-6:
        raise RuntimeError("Reliability near zero must strictly suppress deployed Delta-c4")
    if float(passed_error.item()) > 1e-6:
        raise RuntimeError("Reliability near one must pass bounded semantic proposal unchanged")

    channel_rms = data["current_c4"].square().mean(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-6)
    cap = training.RESIDUAL_SCALE * channel_rms
    if bool((proposal.abs() > cap + 1e-7).any().item()):
        raise RuntimeError("C-V14 semantic proposal exceeded 0.10 x per-channel c4 RMS cap")
    if bool((passed["delta_c4"].abs() > cap + 1e-7).any().item()):
        raise RuntimeError("C-V14 deployed Delta-c4 exceeded semantic proposal cap")


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
        model.temporal_reliability_head.weight.normal_(0.0, 0.02)
    data_a = _inputs()
    data_b = {key: value for key, value in data_a.items()}
    data_b["temporal_hidden"] = data_a["temporal_hidden"] + 0.5
    rel_a = model(**data_a)["temporal_reliability"]
    rel_b = model(**data_b)["temporal_reliability"]
    if torch.allclose(rel_a, rel_b):
        raise RuntimeError("Frozen C-V4 temporal hidden must influence C-V14 Reliability")
    if float(rel_b.min().item()) < 0.0 or float(rel_b.max().item()) > 1.0:
        raise RuntimeError("C-V14 Reliability left [0,1]")


def _check_acceptance_targets():
    gt = torch.tensor([[0, 1]], dtype=torch.long)
    current = torch.full((1, NUM_CLASSES, 1, 2), -5.0)
    proposal = torch.full((1, NUM_CLASSES, 1, 2), -5.0)
    # Pixel 0: current wrong class 1 -> proposal correct class 0 => positive.
    current[0, 1, 0, 0] = 5.0
    proposal[0, 0, 0, 0] = 5.0
    # Pixel 1: current correct class 1 -> proposal wrong class 0 => negative.
    current[0, 1, 0, 1] = 5.0
    proposal[0, 0, 0, 1] = 5.0
    row = training.acceptance_targets(current, proposal, gt)
    if not bool(row["positive"][0, 0].item()):
        raise RuntimeError("C-V14 beneficial proposal pixel must be Reliability positive")
    if not bool(row["negative"][0, 1].item()):
        raise RuntimeError("C-V14 harmful proposal pixel must be Reliability negative")
    if int(row["supervised"].sum().item()) != 2:
        raise RuntimeError("Synthetic C-V14 acceptance target must supervise exactly two pixels")


def _check_reliability_gradient_isolation():
    model = _model().train()
    out = model(**_inputs())
    target = torch.ones_like(out["temporal_reliability_logit"])
    loss = F.binary_cross_entropy_with_logits(out["temporal_reliability_logit"], target)
    model.zero_grad(set_to_none=True)
    loss.backward()
    if model.temporal_reliability_head.weight.grad is None or float(model.temporal_reliability_head.weight.grad.abs().sum().item()) <= 0.0:
        raise RuntimeError("Reliability BCE must train the temporal Reliability head")
    if model.temporal_pre[0].weight.grad is None or float(model.temporal_pre[0].weight.grad.abs().sum().item()) <= 0.0:
        raise RuntimeError("Reliability BCE must train temporal evidence encoder")
    if model.semantic_encoder[0].weight.grad is not None and float(model.semantic_encoder[0].weight.grad.abs().sum().item()) > 0.0:
        raise RuntimeError("Reliability BCE must not train Semantic Branch")
    for parameter in model.writeback.parameters():
        if parameter.grad is not None and float(parameter.grad.abs().sum().item()) > 0.0:
            raise RuntimeError("Reliability BCE must not train Semantic Writeback")


def _check_joint_deployed_gradient_after_writeback_opens():
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
        "writeback_output_projection": model.writeback.output_projection.weight.grad,
    }
    for name, grad in required.items():
        if grad is None or float(grad.abs().sum().item()) <= 0.0:
            raise RuntimeError(f"C-V14 deployed path did not train {name}")


def _check_zero_step_final_composition():
    class DummyDecoder:
        @staticmethod
        def decode_from_host_feature(host_feature):
            logits = host_feature.tensor[:, :NUM_CLASSES]
            return F.interpolate(logits, size=host_feature.output_size, mode="bilinear", align_corners=False)

    corrector = _model().eval()
    data = _inputs(error_size=8, c4_size=4)
    decoder = DummyDecoder()
    output_size = (8, 8)
    host_logits = decoder.decode_from_host_feature(
        HostFeature(data["current_c4"], torch.zeros(1, 1, 8, 8), output_size)
    )
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
    final_logits, row = training.forward_feature_update(
        decoder,
        corrector,
        observation,
        error_row,
        data["transportability_low"],
        data["memory_reliability_low"],
    )
    proposal_logits = training.decode_detached_proposal(decoder, observation, row["proposal_c4"])
    if not torch.equal(final_logits, host_logits):
        raise RuntimeError("C-V14 zero-step final logits must exactly equal Host logits")
    if not torch.equal(proposal_logits, host_logits):
        raise RuntimeError("C-V14 zero-step proposal logits must exactly equal Host logits")


def _check_training_contract():
    helper_source = inspect.getsource(training)
    entry_source = inspect.getsource(c_v14)
    required_helper = (
        'proposal_c4.detach()',
        'loss = (',
        'PROTECTION_LOSS_WEIGHT * protect_kl',
        'RELIABILITY_LOSS_WEIGHT * reliability_bce',
        'raw_history.insert(0, c_v3_logits.detach())',
        'current_pred.ne(gt_gpu) & proposal_pred.eq(gt_gpu)',
        'current_pred.eq(gt_gpu) & proposal_pred.ne(gt_gpu)',
    )
    for token in required_helper:
        if token not in helper_source:
            raise RuntimeError(f"C-V14 helper lost required contract: {token}")
    forbidden_helper = (
        'RESCUE_LOSS_WEIGHT',
        'rescue_ce =',
        'baseline_logits.detach() + feature_delta_logits',
    )
    for token in forbidden_helper:
        if token in helper_source:
            raise RuntimeError(f"C-V14 helper retained forbidden behavior: {token}")
    required_entry = (
        '"beneficial_harmful_reliability_bce": True',
        '"proposal_detached_for_reliability_bce": True',
        '"rescue_ce": False',
        '"temporal_loss": False',
        '"raft_training": False',
        '"final_composition": "Decoder(c4 + g_t * bounded Delta-c4-sem)"',
    )
    for token in required_entry:
        if token not in entry_source:
            raise RuntimeError(f"C-V14 entrypoint lost required contract: {token}")
    if training.RESIDUAL_SCALE != 0.10:
        raise RuntimeError("C-V14 residual scale must stay fixed at 0.10")
    if training.PROTECTION_LOSS_WEIGHT != 1.0 or training.RELIABILITY_LOSS_WEIGHT != 1.0:
        raise RuntimeError("C-V14 fixed loss weights must both equal 1.0")


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
    _check_postwriteback_control_and_bound()
    _check_deep_history_and_temporal_conditioning()
    _check_acceptance_targets()
    _check_reliability_gradient_isolation()
    _check_joint_deployed_gradient_after_writeback_opens()
    _check_zero_step_final_composition()
    _check_training_contract()
    _check_model_selection()
    model = _model()
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print({
        "passed": True,
        "architecture": "C-V13 semantic proposal + post-writeback pixel Reliability",
        "semantic_content": "e1..e4 -> 128D -> bounded Delta-c4-sem",
        "temporal_reliability": "1D pixel-wise sigmoid after bounded writeback",
        "postwriteback_control": "Delta-c4-final = g_t * bounded Delta-c4-sem",
        "residual_bound": "0.10 x per-channel RMS(c4)",
        "training": "all-pixel CE + C-V4 protection KL + beneficial/harmful Reliability BCE",
        "proposal_detached_for_reliability_bce": True,
        "rescue_supervision": False,
        "trainable_parameters": trainable,
        "history_feedback": False,
        "temporal_loss": False,
        "raft_training": False,
    })


if __name__ == "__main__":
    main()
