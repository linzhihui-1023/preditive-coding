"""C-V12: C-V4 temporal base + Prediction-Error-driven c4 semantic correction.

中文：C-V12 = 冻结 C-V4 时序底座 + 预测误差驱动的 c4 语义特征修正。

Fixed experiment / 固定实验
---------------------------
- K=4 raw frozen C-V3 History Bank（历史库）, motion aligned once.
- Frozen C-V4 E2 Stateful Semantic Hysteresis（有状态语义滞回） supplies the
  temporal baseline and temporal hidden state.
- New semantic branch sees only e1..e4 and writes a residual into Host c4.
- Frozen Host Decoder converts Delta-c4 into a semantic logit effect.
- Final = frozen C-V4 logits + [Decoder(c4+Delta-c4)-Decoder(c4)].
- Training: all-pixel final CE + formal Rescue-pixel CE, both weight 1.
- No temporal training loss and no RAFT teacher. RAFT is metric-only.
"""

import argparse
import json
import random
from pathlib import Path

import torch

from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import sequence_groups
from predify2021.mce_scores.train_kitti_step_temporal_joint import FrozenRAFT
from predify2021.mce_scores import c_v12_temporal_semantic_feature_training as training
from predify2021.mce_scores import (
    train_kitti_step_task_space_prior_c_v5_multiframe_candidate_memory as c_v5,
)
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_semantic_hysteresis import (
    EulerDynamicsError,
)
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_temporal_semantic_feature_correction import (
    TemporalSemanticFeatureCorrector,
)


EXPERIMENT = "c_v12_temporal_semantic_feature_correction"
OUTPUT_DEFAULT = (
    "/home/lin/predify/experiments/"
    "kitti_step_task_space_prior_c_v12_temporal_semantic_feature_correction"
)
RESULT_DEFAULT = (
    "results/kitti_step_task_space_prior_c_v12_temporal_semantic_feature_correction"
)
EPOCHS = 3
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
SEED = 0
C_V4_MIOU_FLOOR = training.C_V4_REFERENCE_MIOU
C_V4_MTC_FLOOR = 0.7126707488339122


def _delta(candidate, reference):
    return {
        key: candidate[key] - reference[key]
        for key in ("mIoU", "mTC", "mVC8", "mVC16")
    }


def _selection_key(metrics):
    candidate = metrics["c_v12"]
    semantic_preserved = candidate["mIoU"] >= C_V4_MIOU_FLOOR
    temporal_preserved = candidate["mTC"] >= C_V4_MTC_FLOOR
    if semantic_preserved and temporal_preserved:
        # The new branch exists to recover semantic gain; mIoU is primary.
        return (1, candidate["mIoU"], candidate["mTC"])
    return (0, candidate["mIoU"], candidate["mTC"])


def _architecture_metadata(corrector):
    return {
        "temporal_base": "frozen C-V4 E2 Stateful Semantic Hysteresis",
        "history_length": corrector.history_length,
        "history_source": "raw detached frozen C-V3 logits",
        "history_feedback": False,
        "causal_path": "History -> Prediction -> Prediction Error -> Semantic Feature Correction",
        "semantic_content": "concat validity-gated e1..e4",
        "semantic_input_channels": corrector.semantic_input_channels,
        "semantic_latent_channels": corrector.semantic_channels,
        "temporal_context": "frozen C-V4 hidden + Dynamics Error + T/Q/validity/error agreement",
        "temporal_modulation": "128D channel-wise, neutral initialized",
        "feature_target": "DeepLabV3+ ResNet50 c4, 2048 channels",
        "feature_update": "c4_corrected = c4 + Delta-c4",
        "writeback": "zero-initialized HostConditionedResidualWriteback",
        "decoded_effect": "Decoder(c4+Delta-c4) - Decoder(c4)",
        "final_composition": "frozen C-V4 logits + decoded feature effect",
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
    c_v4_controller, cv4_payload = training.load_frozen_c_v4_controller(args.c_v4_checkpoint)

    corrector = TemporalSemanticFeatureCorrector(
        num_classes=c_v5.NUM_CLASSES,
        history_length=training.HISTORY_LENGTH,
        semantic_channels=training.SEMANTIC_CHANNELS,
        temporal_hidden_channels=c_v5.CONTROLLER_HIDDEN_CHANNELS,
        host_channels=training.C4_CHANNELS,
    ).cuda()
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
            "delta_vs_host": _delta(metrics["c_v12"], metrics["host"]),
            "delta_vs_c_v3": _delta(metrics["c_v12"], metrics["c_v3_base"]),
            "delta_vs_c_v4": _delta(metrics["c_v12"], metrics["c_v4_frozen"]),
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
                    "architecture": _architecture_metadata(corrector),
                    "training_supervision": {
                        "final_all_pixel_ce": True,
                        "formal_rescue_ce": True,
                        "rescue_loss_weight": training.RESCUE_LOSS_WEIGHT,
                        "temporal_loss": False,
                        "raft_training": False,
                        "gt_inference_input": False,
                    },
                    "dynamics": dynamics.config(),
                    "frozen_c_v4_epoch": cv4_payload.get("epoch"),
                },
                output / "best.pt",
            )
        print(json.dumps(row, indent=2), flush=True)

    if best is None:
        raise RuntimeError("C-V12 produced no epoch result")

    summary = {
        "experiment": "C-V12 Temporal-Conditioned Semantic Feature Correction",
        "architecture_decision": (
            "return to frozen C-V4 temporal baseline; use K=4 Prediction Error "
            "to correct Host c4 features through the frozen Decoder"
        ),
        "best": best,
        "history": history,
        "selection_rule": {
            "semantic_floor": C_V4_MIOU_FLOOR,
            "temporal_floor": C_V4_MTC_FLOOR,
            "objective_if_both_pass": "maximize mIoU, then mTC",
            "fallback": "highest mIoU, then mTC",
        },
        "architecture": _architecture_metadata(corrector),
        "frozen_checkpoints": {
            "fast_b": args.fast_b_checkpoint,
            "observer": args.observer_checkpoint,
            "residual": args.residual_checkpoint,
            "e1_base": args.base_checkpoint,
            "c_v3_base": args.c_v3_checkpoint,
            "c_v4": args.c_v4_checkpoint,
            "c_v3_checkpoint_epoch": cv3_payload.get("epoch"),
            "c_v4_checkpoint_epoch": cv4_payload.get("epoch"),
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
