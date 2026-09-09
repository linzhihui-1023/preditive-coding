"""CPU/synthetic contracts for C-V14 post-writeback reliability acceptance."""

import inspect
import tempfile
from pathlib import Path

import torch
from torch.nn import functional as F

from predify2021.mce_scores import c_v14_postwriteback_reliability_training as training
from predify2021.mce_scores import (
    train_kitti_step_task_space_prior_c_v14_postwriteback_reliability_acceptance as c_v14,
)
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_bounded_temporal_semantic_feature_correction import (
    BoundedTemporalSemanticFeatureCorrector,
)
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_postwriteback_reliability_semantic_feature_correction import (
    PostWritebackReliabilitySemanticFeatureCorrector,
)


NUM_CLASSES = 19
K = 4
SEMANTIC = 128
TEMPORAL = 32
C4 = 2048


def _model():
    return PostWritebackReliabilitySemanticFeatureCorrector(
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


def _check_boundary_and_shapes():
    parameters = set(
        inspect.signature(PostWritebackReliabilitySemanticFeatureCorrector.forward).parameters
    )
    forbidden = {
        "raw_history",
        "history_logits",
        "history_probabilities",
        "current_logits",
        "current_probability",
    }
    leaked = sorted(parameters & forbidden)
    if leaked:
        raise RuntimeError(f"Raw semantic history leaked into C-V14 forward: {leaked}")

    model = _model()
    if model.semantic_input_channels != K * NUM_CLASSES:
        raise RuntimeError("C-V14 semantic input must be concat(e1..e4)=76D")
    first = model.semantic_encoder[0]
    if first.in_channels != K * NUM_CLASSES or first.out_channels != SEMANTIC:
        raise RuntimeError("C-V14 semantic encoder must preserve C-V13 76D->128D")
    if model.acceptance_head.in_channels != SEMANTIC:
        raise RuntimeError("C-V14 Acceptance Head must consume 128D temporal context")
    if model.acceptance_head.out_channels != 1:
        raise RuntimeError("C-V14 Acceptance Head must be single-channel pixel-wise")


def _check_acceptance_initialization():
    model = _model().eval()
    data = _inputs()
    out = model(**data)
    expected = torch.full_like(out["acceptance"], model.acceptance_init)
    if not torch.allclose(out["acceptance"], expected, atol=1e-6, rtol=0.0):
        raise RuntimeError("Fresh C-V14 acceptance must initialize at fixed 0.95")
    if abs(model.acceptance_init - 0.95) > 1e-12:
        raise RuntimeError("C-V14 acceptance_init must remain fixed at 0.95")
    if float(model.acceptance_head.weight.abs().max().item()) != 0.0:
        raise RuntimeError("Fresh C-V14 Acceptance Head weights must initialize to zero")


def _check_postwriteback_hard_control_and_bound():
    model = _model().eval()
    data = _inputs()
    with torch.no_grad():
        model.writeback.output_projection.weight.normal_(0.0, 0.05)
        model.acceptance_head.weight.zero_()
        model.acceptance_head.bias.fill_(-20.0)
    suppressed = model(**data)

    with torch.no_grad():
        model.acceptance_head.bias.fill_(20.0)
    accepted = model(**data)

    if not torch.allclose(
        suppressed["proposal_delta_c4"],
        accepted["proposal_delta_c4"],
        atol=1e-7,
        rtol=1e-6,
    ):
        raise RuntimeError("Acceptance changed the semantic proposal before post-writeback gating")

    channel_rms = (
        data["current_c4"]
        .square()
        .mean(dim=(-2, -1), keepdim=True)
        .sqrt()
        .clamp_min(1e-6)
    )
    cap = training.RESIDUAL_SCALE * channel_rms
    if bool((accepted["proposal_delta_c4"].abs() > cap + 1e-7).any().item()):
        raise RuntimeError("C-V14 semantic proposal exceeded 0.10 x per-channel c4 RMS")
    if bool(
        (
            accepted["final_delta_c4"].abs()
            > accepted["proposal_delta_c4"].abs() + 1e-7
        ).any().item()
    ):
        raise RuntimeError("C-V14 Acceptance amplified the bounded semantic proposal")

    proposal_norm = float(accepted["proposal_delta_c4"].abs().mean().item())
    if proposal_norm <= 0.0:
        raise RuntimeError("Synthetic C-V14 proposal failed to open")
    suppressed_norm = float(suppressed["final_delta_c4"].abs().mean().item())
    accepted_norm = float(accepted["final_delta_c4"].abs().mean().item())
    if suppressed_norm > proposal_norm * 1e-6:
        raise RuntimeError("Acceptance near zero did not suppress deployed Delta-c4")
    if accepted_norm < proposal_norm * 0.999:
        raise RuntimeError("Acceptance near one did not pass the bounded proposal")


def _check_temporal_branch_controls_acceptance_only():
    model = _model().eval()
    with torch.no_grad():
        model.writeback.output_projection.weight.normal_(0.0, 0.02)
        model.acceptance_head.weight.normal_(0.0, 0.05)
    data_a = _inputs()
    data_b = {key: value for key, value in data_a.items()}
    data_b["temporal_hidden"] = data_a["temporal_hidden"] + 0.5
    out_a = model(**data_a)
    out_b = model(**data_b)

    if not torch.allclose(
        out_a["proposal_delta_c4"],
        out_b["proposal_delta_c4"],
        atol=1e-7,
        rtol=1e-6,
    ):
        raise RuntimeError("Temporal evidence leaked into C-V14 semantic proposal generation")
    if torch.allclose(out_a["acceptance"], out_b["acceptance"]):
        raise RuntimeError("C-V4 temporal hidden must influence C-V14 Acceptance")


def _check_deep_history_semantic_content():
    model = _model().eval()
    base = _inputs()
    out_a = model(**base)
    changed = {key: value for key, value in base.items()}
    changed["prediction_errors"] = [value.clone() for value in base["prediction_errors"]]
    changed["prediction_errors"][3] = changed["prediction_errors"][3] + 0.25
    out_b = model(**changed)
    if torch.allclose(out_a["semantic_latent"], out_b["semantic_latent"]):
        raise RuntimeError("t-4 Prediction Error must influence C-V14 semantic proposal")


def _check_acceptance_targets():
    gt = torch.zeros((4, 4), dtype=torch.long)
    host = torch.full((1, NUM_CLASSES, 4, 4), -5.0)
    proposal = torch.full_like(host, -5.0)
    host[:, 0] = 5.0
    proposal[:, 0] = 5.0

    # Top half: Host wrong -> Proposal correct => beneficial target 1.
    host[:, 0, :2] = -5.0
    host[:, 1, :2] = 5.0
    # Bottom half: Host correct -> Proposal wrong => harmful target 0.
    proposal[:, 0, 2:] = -5.0
    proposal[:, 1, 2:] = 5.0

    target, supervised, beneficial, harmful = training.acceptance_targets(
        host,
        proposal,
        gt,
    )
    if int(beneficial.sum().item()) != 8:
        raise RuntimeError("C-V14 beneficial acceptance target construction is wrong")
    if int(harmful.sum().item()) != 8:
        raise RuntimeError("C-V14 harmful acceptance target construction is wrong")
    if int(supervised.sum().item()) != 16:
        raise RuntimeError("C-V14 must supervise only explicit beneficial/harmful pixels")
    if not bool((target[beneficial] == 1).all().item()):
        raise RuntimeError("Beneficial proposal pixels must have acceptance target 1")
    if not bool((target[harmful] == 0).all().item()):
        raise RuntimeError("Harmful proposal pixels must have acceptance target 0")


def _check_acceptance_aux_gradient_isolation():
    model = _model().train()
    out = model(**_inputs())
    aux_target = torch.zeros_like(out["acceptance_logit_aux"])
    loss = F.binary_cross_entropy_with_logits(out["acceptance_logit_aux"], aux_target)
    model.zero_grad(set_to_none=True)
    loss.backward()

    head_grad = model.acceptance_head.weight.grad
    if head_grad is None or float(head_grad.abs().sum().item()) <= 0.0:
        raise RuntimeError("Acceptance auxiliary loss did not train Acceptance Head")

    forbidden = {
        "temporal_pre": model.temporal_pre[0].weight.grad,
        "semantic_encoder": model.semantic_encoder[0].weight.grad,
        "writeback": model.writeback.output_projection.weight.grad,
    }
    for name, grad in forbidden.items():
        if grad is not None and float(grad.abs().sum().item()) > 0.0:
            raise RuntimeError(f"Acceptance auxiliary gradient leaked into {name}")


def _check_deployed_path_joint_gradient():
    model = _model().train()
    with torch.no_grad():
        model.writeback.output_projection.weight.normal_(0.0, 1e-3)
        model.acceptance_head.weight.normal_(0.0, 0.02)
    out = model(**_inputs())
    loss = out["corrected_c4"].square().mean()
    model.zero_grad(set_to_none=True)
    loss.backward()

    required = {
        "semantic_encoder": model.semantic_encoder[0].weight.grad,
        "temporal_pre": model.temporal_pre[0].weight.grad,
        "acceptance_head": model.acceptance_head.weight.grad,
        "writeback_host": model.writeback.host_projection.weight.grad,
        "writeback_delta": model.writeback.delta_projection.weight.grad,
        "writeback_output": model.writeback.output_projection.weight.grad,
    }
    for name, grad in required.items():
        if grad is None or float(grad.abs().sum().item()) <= 0.0:
            raise RuntimeError(f"C-V14 deployed final path did not train {name}")


def _check_c_v13_transfer():
    source = BoundedTemporalSemanticFeatureCorrector(
        num_classes=NUM_CLASSES,
        history_length=K,
        semantic_channels=SEMANTIC,
        temporal_hidden_channels=TEMPORAL,
        host_channels=C4,
        residual_scale=training.RESIDUAL_SCALE,
    )
    with torch.no_grad():
        source.semantic_encoder[0].weight.fill_(0.123)
        source.temporal_pre[0].weight.fill_(0.234)
        source.writeback.output_projection.weight.fill_(0.345)

    payload = {
        "experiment": "c_v13_bounded_reliability_feature_update",
        "epoch": 3,
        "corrector_state_dict": source.state_dict(),
    }
    target = _model()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "c_v13.pt"
        torch.save(payload, path)
        loaded, transferred = c_v14._load_c_v13_initialization(target, path)

    if int(loaded["epoch"]) != 3 or not transferred:
        raise RuntimeError("C-V14 failed to load C-V13 initialization")
    if not torch.equal(target.semantic_encoder[0].weight, source.semantic_encoder[0].weight):
        raise RuntimeError("C-V14 did not transfer C-V13 semantic encoder")
    if not torch.equal(target.temporal_pre[0].weight, source.temporal_pre[0].weight):
        raise RuntimeError("C-V14 did not transfer C-V13 temporal pre")
    if not torch.equal(
        target.writeback.output_projection.weight,
        source.writeback.output_projection.weight,
    ):
        raise RuntimeError("C-V14 did not transfer C-V13 writeback")
    if float(target.acceptance_head.weight.abs().max().item()) != 0.0:
        raise RuntimeError("C-V14 Acceptance Head must remain fresh after C-V13 transfer")
    expected = torch.full((1,), target.acceptance_init)
    actual = torch.sigmoid(target.acceptance_head.bias.detach().cpu())
    if not torch.allclose(actual, expected, atol=1e-6, rtol=0.0):
        raise RuntimeError("C-V13 transfer changed C-V14 Acceptance initialization")


def _check_protocol_source_contracts():
    model_source = inspect.getsource(PostWritebackReliabilitySemanticFeatureCorrector)
    helper_source = inspect.getsource(training)
    entry_source = inspect.getsource(c_v14)
    joined = model_source + helper_source + entry_source

    required = (
        "proposal_corrected_c4 = current_c4 + proposal_delta_c4",
        "final_delta_c4 = proposal_delta_c4 * acceptance_c4",
        "acceptance_logit_aux = self.acceptance_head(temporal_latent.detach())",
        "raw_history.insert(0, c_v3_logits.detach())",
        "+ ACCEPTANCE_LOSS_WEIGHT * accept_bce",
        '"acceptance_bce_temporal_context_detached": True',
        '"rescue_ce": False',
        '"temporal_loss": False',
        '"raft_training": False',
    )
    for token in required:
        if token not in joined:
            raise RuntimeError(f"C-V14 lost required contract: {token}")

    forbidden = (
        "semantic_latent_c4 * temporal_reliability_c4",
        "baseline_logits.detach() + feature_delta_logits",
        "RESCUE_LOSS_WEIGHT",
        "rescue_ce =",
    )
    for token in forbidden:
        if token in joined:
            raise RuntimeError(f"C-V14 retained forbidden earlier behavior: {token}")

    if training.RESIDUAL_SCALE != 0.10:
        raise RuntimeError("C-V14 residual scale must remain fixed at 0.10")
    if training.PROTECTION_LOSS_WEIGHT != 1.0:
        raise RuntimeError("C-V14 Protection KL weight must remain fixed at 1.0")
    if training.ACCEPTANCE_LOSS_WEIGHT != 1.0:
        raise RuntimeError("C-V14 Acceptance BCE weight must remain fixed at 1.0")


def _check_model_selection():
    reference = {"mIoU": 0.6635, "mTC": 0.7132}
    good = {
        "c_v4_frozen": reference,
        "c_v14": {"mIoU": 0.6645, "mTC": 0.7142},
    }
    semantic_fail = {
        "c_v4_frozen": reference,
        "c_v14": {"mIoU": 0.6634, "mTC": 0.7200},
    }
    temporal_fail = {
        "c_v4_frozen": reference,
        "c_v14": {"mIoU": 0.6700, "mTC": 0.7131},
    }
    if c_v14._selection_key(good)[0] != 1:
        raise RuntimeError("C-V14 preserving both semantic and temporal goals must pass")
    if c_v14._selection_key(semantic_fail)[0] != 0:
        raise RuntimeError("C-V14 must reject semantic regression")
    if c_v14._selection_key(temporal_fail)[0] != 0:
        raise RuntimeError("C-V14 must reject temporal regression")


def main():
    _check_boundary_and_shapes()
    _check_acceptance_initialization()
    _check_postwriteback_hard_control_and_bound()
    _check_temporal_branch_controls_acceptance_only()
    _check_deep_history_semantic_content()
    _check_acceptance_targets()
    _check_acceptance_aux_gradient_isolation()
    _check_deployed_path_joint_gradient()
    _check_c_v13_transfer()
    _check_protocol_source_contracts()
    _check_model_selection()

    model = _model()
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print({
        "passed": True,
        "architecture": "C-V13 semantic proposal + post-writeback pixel-wise acceptance",
        "semantic_content": "e1..e4 -> 128D -> bounded c4 proposal",
        "acceptance": "C-V4 temporal evidence -> 1D pixel-wise sigmoid",
        "acceptance_init": model.acceptance_init,
        "acceptance_position": "after 0.10 bounded writeback",
        "acceptance_aux": "beneficial/harmful BCE; auxiliary gradient only to Acceptance Head",
        "final_composition": "Decoder(c4 + acceptance * bounded semantic Delta-c4)",
        "rescue_supervision": False,
        "history_feedback": False,
        "temporal_loss": False,
        "raft_training": False,
        "trainable_parameters": trainable,
    })


if __name__ == "__main__":
    main()
