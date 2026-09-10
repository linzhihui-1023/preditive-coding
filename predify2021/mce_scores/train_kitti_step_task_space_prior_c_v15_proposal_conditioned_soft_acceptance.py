"""C-V15: Proposal-Conditioned Soft Acceptance.

中文：C-V15 修正提案条件化软接受。

Research question / 研究问题
---------------------------
Does acceptance discrimination emerge when the gate explicitly observes the
specific bounded semantic feature proposal it is deciding whether to apply?

Fixed from aligned C-V14:
- K=4 raw frozen C-V3 history, motion aligned once.
- Frozen C-V4 temporal hidden/dynamics/reference teacher.
- Frozen C-V14 E3 semantic encoder + host-conditioned writeback + 0.10 bound.
- Same all-pixel CE + C-V4-correct protection KL + beneficial/harmful BCE.
- Same Host-relative acceptance labels.
- No Rescue CE, no temporal loss, no RAFT training teacher.

Only trainable C-V15 modules:
- normalized Proposal Encoder: 2048 -> 32;
- current Host c4 Encoder: 2048 -> 32;
- Acceptance Fusion: [temporal 128, semantic 128, proposal 32, host 32] -> 128;
- Acceptance residual head: 128 -> 1.

The residual head is zero initialized on top of the frozen aligned C-V14 gate,
so pre-training C-V15 reproduces the loaded C-V14 acceptance behavior.
"""

import argparse
import json
import random
from pathlib import Path

import torch

from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import sequence_groups
from predify2021.mce_scores.train_kitti_step_temporal_joint import FrozenRAFT
from predify2021.mce_scores import (
    c_v15_proposal_conditioned_soft_acceptance_training as training,
)
from predify2021.mce_scores import (
    train_kitti_step_task_space_prior_c_v5_multiframe_candidate_memory as c_v5,
)
from predify2021.mce_scores import (
    train_kitti_step_task_space_prior_c_v12_temporal_semantic_feature_correction as c_v12_entry,
)
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_semantic_hysteresis import (
    EulerDynamicsError,
)
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_proposal_conditioned_soft_acceptance import (
    ProposalConditionedSoftAcceptanceCorrector,
)


EXPERIMENT = "c_v15_proposal_conditioned_soft_acceptance"
C_V14_ALIGNED_CHECKPOINT_DEFAULT = (
    "/home/lin/predify/experiments/"
    "kitti_step_task_space_prior_c_v14_post_writeback_reliability_acceptance/best.pt"
)
OUTPUT_DEFAULT = (
    "/home/lin/predify/experiments/"
    "kitti_step_task_space_prior_c_v15_proposal_conditioned_soft_acceptance"
)
RESULT_DEFAULT = (
    "results/kitti_step_task_space_prior_c_v15_proposal_conditioned_soft_acceptance"
)
EPOCHS = 3
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
SEED = 0
C_V14_REQUIRED_EPOCH = 3


def _delta(candidate, reference):
    return {
        key: candidate[key] - reference[key]
        for key in ("mIoU", "mTC", "mVC8", "mVC16")
    }


def _selection_key(metrics):
    candidate = metrics["c_v15"]
    reference = metrics["c_v4_frozen"]
    semantic_preserved = candidate["mIoU"] >= reference["mIoU"]
    temporal_preserved = candidate["mTC"] >= reference["mTC"]
    if semantic_preserved and temporal_preserved:
        return (1, candidate["mIoU"], candidate["mTC"])
    return (0, candidate["mIoU"], candidate["mTC"])


def _load_aligned_cv14_payload(path):
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("experiment") != "c_v14_post_writeback_reliability_acceptance":
        raise RuntimeError("C-V15 requires a C-V14 post-writeback reliability checkpoint")
    if int(payload.get("epoch", -1)) != C_V14_REQUIRED_EPOCH:
        raise RuntimeError(
            f"C-V15 requires aligned C-V14 Epoch {C_V14_REQUIRED_EPOCH}; "
            f"got Epoch {payload.get('epoch')}"
        )
    architecture = payload.get("architecture", {})
    if not bool(architecture.get("aligned_reliability_revision", False)):
        raise RuntimeError("C-V15 requires the aligned C-V14 reliability checkpoint")
    if "corrector_state_dict" not in payload:
        raise RuntimeError("C-V14 checkpoint missing corrector_state_dict")
    return payload


def _trainable_parameter_summary(corrector):
    expected_prefixes = (
        "proposal_encoder.",
        "host_encoder.",
        "acceptance_fusion.",
        "acceptance_residual_head.",
    )
    names = [name for name, parameter in corrector.named_parameters() if parameter.requires_grad]
    illegal = [name for name in names if not name.startswith(expected_prefixes)]
    if illegal:
        raise RuntimeError(f"Unexpected trainable C-V15 parameters: {illegal}")
    count = sum(
        parameter.numel()
        for parameter in corrector.parameters()
        if parameter.requires_grad
    )
    return names, count


def _architecture_metadata(corrector, cv14_path, c_v4_epoch):
    return {
        "name": "C-V15 Proposal-Conditioned Soft Acceptance",
        "causal_path": "History -> Prediction -> Prediction Error -> Semantic Proposal -> Proposal-Conditioned Acceptance -> c4 Correction",
        "history_length": corrector.history_length,
        "history_source": "raw detached frozen C-V3 logits",
        "semantic_proposal_source": "frozen aligned C-V14 E3",
        "semantic_content": "concat validity-gated K=4 prediction errors",
        "semantic_encoder_frozen": True,
        "writeback_frozen": True,
        "temporal_pre_frozen": True,
        "cv14_reliability_prior_frozen": True,
        "cv14_checkpoint": str(cv14_path),
        "cv14_required_epoch": C_V14_REQUIRED_EPOCH,
        "temporal_reference": f"frozen C-V4 Epoch {int(c_v4_epoch)} Stateful Semantic Hysteresis",
        "feature_target": "DeepLabV3+ ResNet50 current c4, 2048 channels",
        "bounded_residual_scale": corrector.residual_scale,
        "proposal_descriptor": "normalized bounded Delta-c4 -> 1x1 Conv -> 32D",
        "host_descriptor": "current c4 -> 1x1 Conv -> 32D",
        "acceptance_inputs": "temporal latent 128D + semantic latent 128D + proposal descriptor 32D + host descriptor 32D",
        "acceptance_fusion": "concat 320D -> 3x3 Conv 128D -> 1x1 residual logit",
        "acceptance_prior": "frozen aligned C-V14 deployed c4 reliability",
        "acceptance_update": "g = sigmoid(logit(g_CV14) + proposal_conditioned_residual_logit)",
        "acceptance_semantics": "proposal-conditioned soft acceptance coefficient",
        "zero_residual_initialization": True,
        "final_update": "Delta-c4-final = g * frozen bounded semantic Delta-c4",
        "final_composition": "Decoder(c4 + Delta-c4-final)",
        "c_v4_logit_composition": False,
        "raw_history_direct_feature_fusion": False,
        "host_frozen": True,
        "c_v3_frozen": True,
        "c_v4_frozen": True,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/home/lin/predify/kitti_step")
    parser.add_argument("--fast-b-checkpoint", default=c_v5.FAST_B_CHECKPOINT_DEFAULT)
    parser.add_argument("--observer-checkpoint", default=c_v5.OBSERVER_CHECKPOINT_DEFAULT)
    parser.add_argument("--residual-checkpoint", default=c_v5.RESIDUAL_CHECKPOINT_DEFAULT)
    parser.add_argument("--base-checkpoint", default=c_v5.BASE_CHECKPOINT_DEFAULT)
    parser.add_argument("--c-v3-checkpoint", default=c_v5.C_V3_CHECKPOINT_DEFAULT)
    parser.add_argument("--c-v4-checkpoint", default=training.C_V4_CHECKPOINT_DEFAULT)
    parser.add_argument("--c-v14-checkpoint", default=C_V14_ALIGNED_CHECKPOINT_DEFAULT)
    parser.add_argument("--output", default=OUTPUT_DEFAULT)
    parser.add_argument("--result-output", default=RESULT_DEFAULT)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=GRADIENT_ACCUMULATION_STEPS,
    )
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--dynamics-tau-e", type=float, default=c_v5.DYNAMICS_TAU_E)
    parser.add_argument("--dynamics-k-e", type=float, default=c_v5.DYNAMICS_K_E)
    parser.add_argument("--dynamics-dt", type=float, default=c_v5.DYNAMICS_DT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.epochs <= 0 or args.gradient_accumulation_steps <= 0:
        raise ValueError("epochs and gradient-accumulation-steps must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    c_v5._validate_bounded_motion_checkpoint(args.residual_checkpoint)
    model = c_v5.load_fast_b_model(args.fast_b_checkpoint).cuda().eval()
    model.requires_grad_(False)
    observer, _ = c_v5._load_frozen_observer(args.observer_checkpoint)
    residual, residual_payload = c_v5._load_frozen_residual(
        args.residual_checkpoint,
        observer,
    )
    correction, mask_predictor, base_payload = c_v5._load_frozen_e1_base(
        args.base_checkpoint,
        observer,
    )
    refiner, cv3_payload = c_v5._load_frozen_cv3_refiner(args.c_v3_checkpoint)
    c_v4_controller, cv4_payload = c_v12_entry._load_frozen_c_v4_controller(
        args.c_v4_checkpoint
    )
    c_v4_epoch = int(cv4_payload["epoch"])

    cv14_payload = _load_aligned_cv14_payload(args.c_v14_checkpoint)
    corrector = ProposalConditionedSoftAcceptanceCorrector(
        num_classes=c_v5.NUM_CLASSES,
        history_length=training.HISTORY_LENGTH,
        semantic_channels=training.SEMANTIC_CHANNELS,
        temporal_hidden_channels=c_v5.CONTROLLER_HIDDEN_CHANNELS,
        host_channels=training.C4_CHANNELS,
        residual_scale=training.RESIDUAL_SCALE,
    ).cuda()
    corrector.load_frozen_cv14_state_dict(cv14_payload["corrector_state_dict"])
    trainable_names, trainable_count = _trainable_parameter_summary(corrector)

    dynamics = EulerDynamicsError(
        tau_e=args.dynamics_tau_e,
        k_e=args.dynamics_k_e,
        dt=args.dynamics_dt,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in corrector.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    train_dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(
        Path(args.root),
        "train",
    )
    val_dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(
        Path(args.root),
        "val",
    )
    train_groups = sequence_groups(train_dataset)
    all_val_groups = sequence_groups(val_dataset)
    missing = [sequence for sequence in c_v5.FULL9 if sequence not in all_val_groups]
    if missing:
        raise RuntimeError(f"Missing Full9 validation sequences: {missing}")
    val_groups = {sequence: all_val_groups[sequence] for sequence in c_v5.FULL9}

    output = Path(args.output)
    result_output = Path(args.result_output)
    output.mkdir(parents=True, exist_ok=True)
    result_output.mkdir(parents=True, exist_ok=True)
    raft_metric = FrozenRAFT()

    history = []
    best = None
    for epoch in range(1, args.epochs + 1):
        train_stats = training.train_epoch(
            model,
            observer,
            residual,
            correction,
            mask_predictor,
            refiner,
            c_v4_controller,
            dynamics,
            corrector,
            train_groups,
            optimizer,
            args.gradient_accumulation_steps,
        )
        metrics, diagnostics = training.evaluate(
            model,
            observer,
            residual,
            correction,
            mask_predictor,
            refiner,
            c_v4_controller,
            dynamics,
            corrector,
            val_groups,
            raft_metric,
        )
        row = {
            "epoch": epoch,
            "train": train_stats,
            "metrics": metrics,
            "diagnostics": diagnostics,
            "delta_vs_host": _delta(metrics["c_v15"], metrics["host"]),
            "delta_vs_c_v3": _delta(metrics["c_v15"], metrics["c_v3_base"]),
            "delta_vs_c_v4": _delta(metrics["c_v15"], metrics["c_v4_frozen"]),
            "delta_vs_source_c_v14": _delta(
                metrics["c_v15"],
                cv14_payload["metrics"]["c_v14"],
            ),
        }
        row["selection_key"] = list(_selection_key(metrics))
        history.append(row)
        with (result_output / f"epoch_{epoch:03d}.json").open("w") as handle:
            json.dump(row, handle, indent=2)

        if best is None or tuple(row["selection_key"]) > tuple(best["selection_key"]):
            best = row
            torch.save(
                {
                    "experiment": EXPERIMENT,
                    "epoch": epoch,
                    "corrector_state_dict": corrector.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": metrics,
                    "diagnostics": diagnostics,
                    "train": train_stats,
                    "architecture": _architecture_metadata(
                        corrector,
                        args.c_v14_checkpoint,
                        c_v4_epoch,
                    ),
                    "training_supervision": {
                        "final_all_pixel_ce": True,
                        "c_v4_correct_region_protection_kl": True,
                        "protection_loss_weight": training.PROTECTION_LOSS_WEIGHT,
                        "beneficial_harmful_acceptance_bce": True,
                        "acceptance_loss_weight": training.ACCEPTANCE_LOSS_WEIGHT,
                        "acceptance_target_reference": "Host vs full frozen C-V14 semantic proposal",
                        "rescue_ce": False,
                        "temporal_loss": False,
                        "raft_training": False,
                        "gt_inference_input": False,
                    },
                    "source_cv14_checkpoint": args.c_v14_checkpoint,
                    "source_cv14_epoch": int(cv14_payload["epoch"]),
                    "source_cv14_metrics": cv14_payload["metrics"]["c_v14"],
                    "dynamics": dynamics.config(),
                    "frozen_c_v4_epoch": c_v4_epoch,
                    "frozen_c_v4_metrics": metrics["c_v4_frozen"],
                    "trainable_parameter_names": trainable_names,
                    "trainable_parameter_count": trainable_count,
                },
                output / "best.pt",
            )
        print(json.dumps(row, indent=2), flush=True)

    if best is None:
        raise RuntimeError("C-V15 produced no epoch result")

    frozen_reference = best["metrics"]["c_v4_frozen"]
    summary = {
        "experiment": "C-V15 Proposal-Conditioned Soft Acceptance",
        "research_question": (
            "Does explicit proposal conditioning create beneficial/harmful "
            "acceptance discrimination while keeping the C-V14 proposal fixed?"
        ),
        "best": best,
        "history": history,
        "selection_rule": {
            "reference": "same-run frozen C-V4",
            "semantic_floor": frozen_reference["mIoU"],
            "temporal_floor": frozen_reference["mTC"],
            "objective_if_both_pass": "maximize mIoU, then mTC",
            "fallback": "highest mIoU, then mTC",
        },
        "architecture": _architecture_metadata(
            corrector,
            args.c_v14_checkpoint,
            c_v4_epoch,
        ),
        "source_cv14": {
            "checkpoint": args.c_v14_checkpoint,
            "epoch": int(cv14_payload["epoch"]),
            "metrics": cv14_payload["metrics"]["c_v14"],
        },
        "frozen_checkpoints": {
            "fast_b": args.fast_b_checkpoint,
            "observer": args.observer_checkpoint,
            "residual": args.residual_checkpoint,
            "e1_base": args.base_checkpoint,
            "c_v3_base": args.c_v3_checkpoint,
            "c_v4": args.c_v4_checkpoint,
            "c_v14_aligned": args.c_v14_checkpoint,
            "c_v3_checkpoint_epoch": cv3_payload.get("epoch"),
            "c_v4_checkpoint_epoch": c_v4_epoch,
            "residual_experiment": residual_payload.get("experiment"),
            "e1_experiment": base_payload.get("experiment"),
        },
        "trainable_parameter_names": trainable_names,
        "trainable_parameter_count": trainable_count,
        "epochs": args.epochs,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
    }
    with (result_output / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
