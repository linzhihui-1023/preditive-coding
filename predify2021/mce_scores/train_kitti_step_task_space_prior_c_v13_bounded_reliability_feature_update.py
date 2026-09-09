"""C-V13: bounded reliability-controlled semantic feature update.

中文：C-V13 = C-V4 时序参考 + K=4 预测误差语义修正 + 有界可靠性特征更新。

Fixed experiment / 固定实验
---------------------------
- K=4 raw frozen C-V3 History Bank（历史库）, motion aligned once.
- Frozen C-V4 Stateful Semantic Hysteresis（有状态语义滞回）provides temporal
  hidden state and a training-time protection teacher.
- Semantic branch sees only e1..e4.
- Temporal reliability is feature-wise and strictly in [0,1]; it cannot amplify.
- Delta-c4 is tanh bounded and capped at 0.10 x per-channel current-c4 RMS.
- Final output is Decoder(c4 + bounded Delta-c4), never C-V4 logits + decoder delta.
- Training = all-pixel CE + C-V4-correct-region protection KL, both weight 1.
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
from predify2021.mce_scores import c_v13_bounded_reliability_feature_training as training
from predify2021.mce_scores import train_kitti_step_task_space_prior_c_v5_multiframe_candidate_memory as c_v5
from predify2021.mce_scores import train_kitti_step_task_space_prior_c_v12_temporal_semantic_feature_correction as c_v12_entry
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_semantic_hysteresis import EulerDynamicsError
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_bounded_temporal_semantic_feature_correction import BoundedTemporalSemanticFeatureCorrector


EXPERIMENT = "c_v13_bounded_reliability_feature_update"
OUTPUT_DEFAULT = "/home/lin/predify/experiments/kitti_step_task_space_prior_c_v13_bounded_reliability_feature_update"
RESULT_DEFAULT = "results/kitti_step_task_space_prior_c_v13_bounded_reliability_feature_update"
EPOCHS = 3
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
SEED = 0


def _delta(candidate, reference):
    return {key: candidate[key] - reference[key] for key in ("mIoU", "mTC", "mVC8", "mVC16")}


def _selection_key(metrics):
    candidate = metrics["c_v13"]
    reference = metrics["c_v4_frozen"]
    semantic_preserved = candidate["mIoU"] >= reference["mIoU"]
    temporal_preserved = candidate["mTC"] >= reference["mTC"]
    if semantic_preserved and temporal_preserved:
        return (1, candidate["mIoU"], candidate["mTC"])
    return (0, candidate["mIoU"], candidate["mTC"])


def _architecture_metadata(corrector, c_v4_epoch):
    return {
        "temporal_reference": f"frozen C-V4 Epoch {int(c_v4_epoch)} Stateful Semantic Hysteresis",
        "history_length": corrector.history_length,
        "history_source": "raw detached frozen C-V3 logits",
        "history_feedback": False,
        "causal_path": "History -> Prediction -> Prediction Error -> Semantic Feature Correction",
        "semantic_content": "concat validity-gated e1..e4",
        "semantic_input_channels": corrector.semantic_input_channels,
        "semantic_latent_channels": corrector.semantic_channels,
        "temporal_context": "frozen C-V4 hidden + Dynamics Error + T/Q/validity/error agreement",
        "temporal_reliability": "128D channel-wise sigmoid in [0,1]",
        "feature_target": "DeepLabV3+ ResNet50 current c4, 2048 channels",
        "bounded_update": "0.10 * per-channel RMS(c4) * tanh(raw Delta-c4)",
        "residual_scale": corrector.residual_scale,
        "final_composition": "Decoder(c4 + bounded Delta-c4)",
        "c_v4_logit_composition": False,
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
    c_v4_controller, cv4_payload = c_v12_entry._load_frozen_c_v4_controller(args.c_v4_checkpoint)
    c_v4_epoch = int(cv4_payload["epoch"])

    corrector = BoundedTemporalSemanticFeatureCorrector(
        num_classes=c_v5.NUM_CLASSES,
        history_length=training.HISTORY_LENGTH,
        semantic_channels=training.SEMANTIC_CHANNELS,
        temporal_hidden_channels=c_v5.CONTROLLER_HIDDEN_CHANNELS,
        host_channels=training.C4_CHANNELS,
        residual_scale=training.RESIDUAL_SCALE,
    ).cuda()
    dynamics = EulerDynamicsError(
        tau_e=args.dynamics_tau_e,
        k_e=args.dynamics_k_e,
        dt=args.dynamics_dt,
    )
    optimizer = torch.optim.AdamW(corrector.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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
            "delta_vs_host": _delta(metrics["c_v13"], metrics["host"]),
            "delta_vs_c_v3": _delta(metrics["c_v13"], metrics["c_v3_base"]),
            "delta_vs_c_v4": _delta(metrics["c_v13"], metrics["c_v4_frozen"]),
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
                    "architecture": _architecture_metadata(corrector, c_v4_epoch),
                    "training_supervision": {
                        "final_all_pixel_ce": True,
                        "c_v4_correct_region_protection_kl": True,
                        "protection_loss_weight": training.PROTECTION_LOSS_WEIGHT,
                        "rescue_ce": False,
                        "temporal_loss": False,
                        "raft_training": False,
                        "gt_inference_input": False,
                    },
                    "dynamics": dynamics.config(),
                    "frozen_c_v4_epoch": c_v4_epoch,
                    "frozen_c_v4_metrics": metrics["c_v4_frozen"],
                },
                output / "best.pt",
            )
        print(json.dumps(row, indent=2), flush=True)

    if best is None:
        raise RuntimeError("C-V13 produced no epoch result")

    frozen_reference = best["metrics"]["c_v4_frozen"]
    summary = {
        "experiment": "C-V13 Bounded Reliability-Controlled Feature Update",
        "architecture_decision": (
            "retain Prediction-Error semantic feature correction but bound Delta-c4, "
            "make temporal modulation suppressive-only, and decode corrected c4 directly"
        ),
        "best": best,
        "history": history,
        "selection_rule": {
            "reference": "same-run c_v4_frozen from the actually loaded checkpoint",
            "semantic_floor": frozen_reference["mIoU"],
            "temporal_floor": frozen_reference["mTC"],
            "objective_if_both_pass": "maximize mIoU, then mTC",
            "fallback": "highest mIoU, then mTC",
        },
        "architecture": _architecture_metadata(corrector, c_v4_epoch),
        "frozen_checkpoints": {
            "fast_b": args.fast_b_checkpoint,
            "observer": args.observer_checkpoint,
            "residual": args.residual_checkpoint,
            "e1_base": args.base_checkpoint,
            "c_v3_base": args.c_v3_checkpoint,
            "c_v4": args.c_v4_checkpoint,
            "c_v3_checkpoint_epoch": cv3_payload.get("epoch"),
            "c_v4_checkpoint_epoch": c_v4_epoch,
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
