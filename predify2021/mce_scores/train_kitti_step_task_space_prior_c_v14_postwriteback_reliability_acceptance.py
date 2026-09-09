"""C-V14: post-writeback pixel-wise reliability with explicit acceptance supervision.

中文：C-V14 = C-V13 语义修正提议 + 后回写像素级可靠性 + 显式接受监督。

Fixed experiment / 固定实验
---------------------------
- Initialize C-V13's learned semantic_encoder, temporal_pre, and writeback from
  the C-V13 best checkpoint; only the new one-channel Acceptance Head is fresh.
- K=4 raw frozen C-V3 History Bank, motion aligned once.
- Semantic proposal remains e1..e4 -> 128D -> Host-conditioned Writeback.
- Proposal Delta-c4 remains bounded by 0.10 x per-channel current-c4 RMS.
- Pixel-wise acceptance g_t in [0,1] is applied AFTER bounded Writeback.
- Acceptance auxiliary target: current Host wrong & proposal correct -> 1;
  current Host correct & proposal wrong -> 0; all other pixels ignored.
- Training = all-pixel CE + C-V4-correct-region protection KL + acceptance BCE.
- Acceptance BCE updates only Acceptance Head through detached temporal context.
- No Rescue CE, no temporal loss, no RAFT training teacher.
"""

import argparse
import json
import random
from pathlib import Path

import torch

from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import sequence_groups
from predify2021.mce_scores.train_kitti_step_temporal_joint import FrozenRAFT
from predify2021.mce_scores import c_v14_postwriteback_reliability_training as training
from predify2021.mce_scores import train_kitti_step_task_space_prior_c_v5_multiframe_candidate_memory as c_v5
from predify2021.mce_scores import train_kitti_step_task_space_prior_c_v12_temporal_semantic_feature_correction as c_v12_entry
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_semantic_hysteresis import EulerDynamicsError
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_postwriteback_reliability_semantic_feature_correction import PostWritebackReliabilitySemanticFeatureCorrector


EXPERIMENT = "c_v14_postwriteback_reliability_acceptance"
OUTPUT_DEFAULT = (
    "/home/lin/predify/experiments/"
    "kitti_step_task_space_prior_c_v14_postwriteback_reliability_acceptance"
)
RESULT_DEFAULT = (
    "results/kitti_step_task_space_prior_c_v14_postwriteback_reliability_acceptance"
)
EPOCHS = 3
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
SEED = 0


def _delta(candidate, reference):
    return {
        key: candidate[key] - reference[key]
        for key in ("mIoU", "mTC", "mVC8", "mVC16")
    }


def _selection_key(metrics):
    candidate = metrics["c_v14"]
    reference = metrics["c_v4_frozen"]
    semantic_preserved = candidate["mIoU"] >= reference["mIoU"]
    temporal_preserved = candidate["mTC"] >= reference["mTC"]
    if semantic_preserved and temporal_preserved:
        return (1, candidate["mIoU"], candidate["mTC"])
    return (0, candidate["mIoU"], candidate["mTC"])


def _load_c_v13_initialization(corrector, path):
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("experiment") != "c_v13_bounded_reliability_feature_update":
        raise RuntimeError("Expected a C-V13 bounded reliability feature checkpoint")
    if "corrector_state_dict" not in payload:
        raise RuntimeError("C-V13 checkpoint missing corrector_state_dict")
    state = payload["corrector_state_dict"]
    prefixes = ("semantic_encoder.", "temporal_pre.", "writeback.")
    transferred = {key: value for key, value in state.items() if key.startswith(prefixes)}
    if not transferred:
        raise RuntimeError("C-V13 checkpoint provided no transferable proposal/temporal weights")
    incompatible = corrector.load_state_dict(transferred, strict=False)
    allowed_missing = {"acceptance_head.weight", "acceptance_head.bias"}
    actual_missing = set(incompatible.missing_keys)
    if actual_missing != allowed_missing:
        raise RuntimeError(
            f"Unexpected C-V14 missing keys after C-V13 transfer: {sorted(actual_missing)}"
        )
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"Unexpected C-V13 transfer keys: {sorted(incompatible.unexpected_keys)}"
        )
    return payload, sorted(transferred)


def _architecture_metadata(corrector, c_v4_epoch, c_v13_epoch):
    return {
        "initialization": f"C-V13 best Epoch {int(c_v13_epoch)} semantic_encoder + temporal_pre + writeback",
        "temporal_reference": f"frozen C-V4 Epoch {int(c_v4_epoch)} Stateful Semantic Hysteresis",
        "history_length": corrector.history_length,
        "history_source": "raw detached frozen C-V3 logits",
        "history_feedback": False,
        "causal_path": "History -> Prediction -> Prediction Error -> Semantic Proposal -> Acceptance -> Feature Correction",
        "semantic_content": "concat validity-gated e1..e4",
        "semantic_input_channels": corrector.semantic_input_channels,
        "semantic_latent_channels": corrector.semantic_channels,
        "proposal_target": "DeepLabV3+ ResNet50 current c4, 2048 channels",
        "proposal_bound": "0.10 * per-channel RMS(c4) * tanh(raw proposal Delta-c4)",
        "residual_scale": corrector.residual_scale,
        "temporal_context": "frozen C-V4 hidden + Dynamics Error + T/Q/validity/error agreement",
        "acceptance": "single-channel pixel-wise sigmoid in [0,1]",
        "acceptance_position": "after bounded Host-conditioned Writeback",
        "final_update": "final Delta-c4 = acceptance * bounded semantic proposal",
        "final_composition": "Decoder(c4 + final Delta-c4)",
        "raw_history_direct_feature_fusion": False,
        "whole_feature_reconstruction": False,
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
    parser.add_argument("--c-v13-checkpoint", default=training.C_V13_CHECKPOINT_DEFAULT)
    parser.add_argument("--output", default=OUTPUT_DEFAULT)
    parser.add_argument("--result-output", default=RESULT_DEFAULT)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=GRADIENT_ACCUMULATION_STEPS)
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
    residual, residual_payload = c_v5._load_frozen_residual(args.residual_checkpoint, observer)
    correction, mask_predictor, base_payload = c_v5._load_frozen_e1_base(args.base_checkpoint, observer)
    refiner, cv3_payload = c_v5._load_frozen_cv3_refiner(args.c_v3_checkpoint)
    c_v4_controller, cv4_payload = c_v12_entry._load_frozen_c_v4_controller(args.c_v4_checkpoint)
    c_v4_epoch = int(cv4_payload["epoch"])

    corrector = PostWritebackReliabilitySemanticFeatureCorrector(
        num_classes=c_v5.NUM_CLASSES,
        history_length=training.HISTORY_LENGTH,
        semantic_channels=training.SEMANTIC_CHANNELS,
        temporal_hidden_channels=c_v5.CONTROLLER_HIDDEN_CHANNELS,
        host_channels=training.C4_CHANNELS,
        residual_scale=training.RESIDUAL_SCALE,
    ).cuda()
    c_v13_payload, transferred_keys = _load_c_v13_initialization(
        corrector,
        args.c_v13_checkpoint,
    )
    c_v13_epoch = int(c_v13_payload.get("epoch", -1))
    if c_v13_epoch <= 0:
        raise RuntimeError("C-V13 initialization checkpoint has invalid epoch metadata")

    dynamics = EulerDynamicsError(
        tau_e=args.dynamics_tau_e,
        k_e=args.dynamics_k_e,
        dt=args.dynamics_dt,
    )
    optimizer = torch.optim.AdamW(
        corrector.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    train_dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "train")
    val_dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "val")
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
            model, observer, residual, correction, mask_predictor, refiner,
            c_v4_controller, dynamics, corrector, train_groups, optimizer,
            args.gradient_accumulation_steps,
        )
        metrics, diagnostics = training.evaluate(
            model, observer, residual, correction, mask_predictor, refiner,
            c_v4_controller, dynamics, corrector, val_groups, raft_metric,
        )
        row = {
            "epoch": epoch,
            "train": train_stats,
            "metrics": metrics,
            "diagnostics": diagnostics,
            "delta_vs_host": _delta(metrics["c_v14"], metrics["host"]),
            "delta_vs_c_v3": _delta(metrics["c_v14"], metrics["c_v3_base"]),
            "delta_vs_c_v4": _delta(metrics["c_v14"], metrics["c_v4_frozen"]),
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
                        corrector, c_v4_epoch, c_v13_epoch
                    ),
                    "training_supervision": {
                        "final_all_pixel_ce": True,
                        "c_v4_correct_region_protection_kl": True,
                        "protection_loss_weight": training.PROTECTION_LOSS_WEIGHT,
                        "acceptance_bce": True,
                        "acceptance_loss_weight": training.ACCEPTANCE_LOSS_WEIGHT,
                        "acceptance_target": "Host wrong/proposal correct=1; Host correct/proposal wrong=0; others ignore",
                        "acceptance_bce_temporal_context_detached": True,
                        "rescue_ce": False,
                        "temporal_loss": False,
                        "raft_training": False,
                        "gt_inference_input": False,
                    },
                    "dynamics": dynamics.config(),
                    "frozen_c_v4_epoch": c_v4_epoch,
                    "frozen_c_v4_metrics": metrics["c_v4_frozen"],
                    "c_v13_initialization_checkpoint": args.c_v13_checkpoint,
                    "c_v13_initialization_epoch": c_v13_epoch,
                    "c_v13_transferred_keys": transferred_keys,
                },
                output / "best.pt",
            )
        print(json.dumps(row, indent=2), flush=True)

    if best is None:
        raise RuntimeError("C-V14 produced no epoch result")

    frozen_reference = best["metrics"]["c_v4_frozen"]
    summary = {
        "experiment": "C-V14 Post-Writeback Reliability Acceptance",
        "architecture_decision": (
            "keep C-V13 semantic proposal and bound; move reliability after writeback "
            "and supervise pixel-wise acceptance on explicit beneficial/harmful proposals"
        ),
        "best": best,
        "history": history,
        "selection_rule": {
            "reference": "same-run c_v4_frozen",
            "semantic_floor": frozen_reference["mIoU"],
            "temporal_floor": frozen_reference["mTC"],
            "objective_if_both_pass": "maximize mIoU, then mTC",
            "fallback": "highest mIoU, then mTC",
        },
        "architecture": _architecture_metadata(corrector, c_v4_epoch, c_v13_epoch),
        "frozen_checkpoints": {
            "fast_b": args.fast_b_checkpoint,
            "observer": args.observer_checkpoint,
            "residual": args.residual_checkpoint,
            "e1_base": args.base_checkpoint,
            "c_v3_base": args.c_v3_checkpoint,
            "c_v4": args.c_v4_checkpoint,
            "c_v13_initialization": args.c_v13_checkpoint,
            "c_v3_checkpoint_epoch": cv3_payload.get("epoch"),
            "c_v4_checkpoint_epoch": c_v4_epoch,
            "c_v13_checkpoint_epoch": c_v13_epoch,
            "residual_experiment": residual_payload.get("experiment"),
            "e1_experiment": base_payload.get("experiment"),
        },
        "epochs": args.epochs,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
    }
    with (result_output / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
