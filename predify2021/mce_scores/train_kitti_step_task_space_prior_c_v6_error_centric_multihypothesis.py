"""C-V6: Error-Centric Multi-Hypothesis Temporal Predictive Coding.

中文：C-V6 误差中心多假设时序预测编码。

This branch preserves the frozen C-V5 foundation (Host, motion observer,
motion residual predictor, E1 correction, C-V3 semantic memory and K=4
candidate bank) while replacing the top-level generic semantic selector with
an error-only controller.

Hard constraint / 硬约束：
- raw Current / History semantic probabilities never enter the controller;
- every historical candidate first becomes a validity-gated prediction error
  e_k = V_k * (P_current - P_history_k);
- the t-1 error continues to drive the explicit Euler Dynamics Error state;
- full semantic candidates remain outside the controller and are used only
  after the selector chooses Current / t-1 / ... / t-K.

The frozen upstream boundary is detached. Therefore this stage changes only the
new C-V6 selector optimization and cannot alter C-V3, motion or Host weights.
"""

import argparse
import json
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import sequence_groups
from predify2021.mce_scores.train_kitti_step_frozen_error_utility_probe import load_fast_b_model
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v2_stage1b_observer import _host_observation
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v2_stage1b_residual import (
    _load_frozen_observer,
)
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v2_stage1b_role_vs_mask import (
    _load_frozen_residual,
)
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v2_stage1b_role_vs_mask_r2 import (
    _validate_bounded_motion_checkpoint,
)
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v2_stage1b_utility_from_e1_r2 import (
    _load_frozen_e1_base,
)
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v3_temporal_semantic_prediction import (
    _load_frozen_cv3_refiner,
)
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v4_stateful_semantic_hysteresis_main import (
    _warp_low_state_zero_invalid,
)
from predify2021.mce_scores.train_kitti_step_temporal_joint import FrozenRAFT
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_multihypothesis_error_selector import (
    MultiHypothesisErrorSelector,
    build_multihypothesis_error_evidence,
)
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_semantic_hysteresis import (
    EulerDynamicsError,
)

# Reuse the validated C-V5 sequence engine, candidate bank, targets and metrics.
# Only its module-level selector-evidence function is replaced below.
from predify2021.mce_scores import (
    train_kitti_step_task_space_prior_c_v5_multiframe_candidate_memory as c_v5,
)


EXPERIMENT = "c_v6_error_centric_multihypothesis"
OUTPUT_DEFAULT = (
    "/home/lin/predify/experiments/"
    "kitti_step_task_space_prior_c_v6_error_centric_multihypothesis"
)
RESULT_DEFAULT = "results/kitti_step_task_space_prior_c_v6_error_centric_multihypothesis"


def _selector_evidence(
    selector,
    dynamics,
    c_v3_logits,
    candidate_rows,
    pending_motion,
    transportability_low,
    memory_reliability_low,
    selector_hidden,
    dynamics_state,
):
    """Build C-V6 controller evidence without raw semantic shortcut inputs."""
    low_size = tuple(transportability_low.shape[-2:])
    current_low = F.interpolate(
        c_v3_logits.detach(),
        size=low_size,
        mode="bilinear",
        align_corners=False,
    )
    current_probability = F.softmax(current_low, dim=1)
    history_probabilities, history_validities = c_v5._pad_history_for_controller(
        current_probability,
        candidate_rows,
        selector.history_length,
    )

    multi = build_multihypothesis_error_evidence(
        current_probability,
        history_probabilities,
        history_validities,
    )
    prediction_errors = multi["prediction_errors"]
    primary_error = prediction_errors[0]
    history1_valid = history_validities[0].detach()

    if dynamics_state is None:
        warped_dynamics = None
    else:
        warped_dynamics, _ = _warp_low_state_zero_invalid(
            dynamics_state,
            pending_motion,
        )
        warped_dynamics = (
            warped_dynamics
            * history1_valid
            * transportability_low.detach().clamp(0.0, 1.0)
        )

    dynamics_state = dynamics.step(
        primary_error,
        warped_dynamics,
    ).detach()

    row = selector(
        prediction_errors,
        dynamics_state,
        multi["current_margin"],
        multi["history_margins"],
        transportability_low.detach(),
        memory_reliability_low.detach(),
        [validity.detach() for validity in history_validities],
        selector_hidden,
    )

    selector_logits_full = F.interpolate(
        row["selector_logits"],
        size=tuple(c_v3_logits.shape[-2:]),
        mode="bilinear",
        align_corners=False,
    )
    for index in range(selector.history_length):
        channel = index + 1
        if index < len(candidate_rows):
            valid = candidate_rows[index]["valid_full"]
            selector_logits_full[:, channel][~valid] = -1e4
        else:
            selector_logits_full[:, channel].fill_(-1e4)

    return {
        "row": row,
        "selector_logits_full": selector_logits_full,
        # Legacy scalar diagnostic in the reused C-V5 train engine refers to t-1.
        "prediction_error": primary_error,
        "prediction_errors": prediction_errors,
        "dynamics_state": dynamics_state,
        "history_validities": history_validities,
    }


def _rename_metrics(metrics):
    metrics = dict(metrics)
    metrics["c_v6"] = metrics.pop("c_v5")
    return metrics


def _annotate_diagnostics(diagnostics):
    diagnostics = dict(diagnostics)
    diagnostics.update(
        {
            "controller_semantic_input": (
                "validity-gated class-wise prediction errors e1..eK only"
            ),
            "raw_current_probability_in_controller": False,
            "raw_history_probability_in_controller": False,
            "prediction_error_reference": "t-1..t-K frozen C-V3 hypotheses",
            "deep_history_error_validity_gated": True,
            "dynamics_error_reference": "validity-gated t-1 prediction error",
            "upstream_error_boundary_detached": True,
        }
    )
    return diagnostics


def _evaluate(*args, **kwargs):
    metrics, diagnostics = c_v5._evaluate(*args, **kwargs)
    return _rename_metrics(metrics), _annotate_diagnostics(diagnostics)


def _selection_key(metrics):
    base = metrics["c_v3_base"]
    candidate = metrics["c_v6"]
    preserved = candidate["mIoU"] >= base["mIoU"]
    if preserved:
        return (1, candidate["mTC"], candidate["mIoU"])
    return (0, candidate["mIoU"], candidate["mTC"])


def _delta_metrics(candidate, reference):
    return {
        key: candidate[key] - reference[key]
        for key in ("mIoU", "mTC", "mVC8", "mVC16")
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/home/lin/predify/kitti_step")
    parser.add_argument("--fast-b-checkpoint", default=c_v5.FAST_B_CHECKPOINT_DEFAULT)
    parser.add_argument("--observer-checkpoint", default=c_v5.OBSERVER_CHECKPOINT_DEFAULT)
    parser.add_argument("--residual-checkpoint", default=c_v5.RESIDUAL_CHECKPOINT_DEFAULT)
    parser.add_argument("--base-checkpoint", default=c_v5.BASE_CHECKPOINT_DEFAULT)
    parser.add_argument("--c-v3-checkpoint", default=c_v5.C_V3_CHECKPOINT_DEFAULT)
    parser.add_argument("--output", default=OUTPUT_DEFAULT)
    parser.add_argument("--result-output", default=RESULT_DEFAULT)
    parser.add_argument("--epochs", type=int, default=c_v5.EPOCHS)
    parser.add_argument("--tbptt-steps", type=int, default=c_v5.TBPTT_STEPS)
    parser.add_argument("--lr", type=float, default=c_v5.LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=c_v5.WEIGHT_DECAY)
    parser.add_argument("--dynamics-tau-e", type=float, default=c_v5.DYNAMICS_TAU_E)
    parser.add_argument("--dynamics-k-e", type=float, default=c_v5.DYNAMICS_K_E)
    parser.add_argument("--dynamics-dt", type=float, default=c_v5.DYNAMICS_DT)
    parser.add_argument("--oracle-only", action="store_true")
    parser.add_argument("--seed", type=int, default=c_v5.SEED)
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    _validate_bounded_motion_checkpoint(args.residual_checkpoint)
    model = load_fast_b_model(args.fast_b_checkpoint).cuda().eval()
    model.requires_grad_(False)
    observer, _ = _load_frozen_observer(args.observer_checkpoint)
    residual, residual_payload = _load_frozen_residual(
        args.residual_checkpoint,
        observer,
    )
    correction, mask_predictor, base_payload = _load_frozen_e1_base(
        args.base_checkpoint,
        observer,
    )
    refiner, cv3_payload = _load_frozen_cv3_refiner(args.c_v3_checkpoint)

    selector = MultiHypothesisErrorSelector(
        num_classes=c_v5.NUM_CLASSES,
        history_length=c_v5.HISTORY_LENGTH,
        hidden_channels=c_v5.CONTROLLER_HIDDEN_CHANNELS,
    ).cuda()
    dynamics = EulerDynamicsError(
        tau_e=args.dynamics_tau_e,
        k_e=args.dynamics_k_e,
        dt=args.dynamics_dt,
    )
    zero_step = c_v5._zero_step_check(selector)

    # Patch only the evidence builder used by the validated C-V5 train/eval engine.
    original_selector_evidence = c_v5._selector_evidence
    c_v5._selector_evidence = _selector_evidence

    optimizer = torch.optim.AdamW(
        selector.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    raft = FrozenRAFT()

    train = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "train")
    val = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "val")
    train_groups = sequence_groups(train)
    all_val_groups = sequence_groups(val)
    missing = [sequence for sequence in c_v5.FULL9 if sequence not in all_val_groups]
    if missing:
        raise RuntimeError(f"Missing Full9 validation sequences: {missing}")
    val_groups = {sequence: all_val_groups[sequence] for sequence in c_v5.FULL9}

    output = Path(args.output)
    result_output = Path(args.result_output)
    output.mkdir(parents=True, exist_ok=True)
    result_output.mkdir(parents=True, exist_ok=True)

    try:
        oracle_metrics, oracle_diagnostics = _evaluate(
            model,
            observer,
            residual,
            correction,
            mask_predictor,
            refiner,
            selector,
            dynamics,
            val_groups,
            raft,
        )
        oracle_precheck = {
            "metrics": oracle_metrics,
            "diagnostics": oracle_diagnostics,
            "delta_multiframe_oracle_vs_c_v3": _delta_metrics(
                oracle_metrics["multiframe_semantic_temporal_oracle"],
                oracle_metrics["c_v3_base"],
            ),
            "single_frame_semantic_temporal_oracle_mTC": c_v5.SINGLE_FRAME_ORACLE_MTC,
            "multiframe_oracle_exceeds_single_frame_oracle_mTC": (
                oracle_metrics["multiframe_semantic_temporal_oracle"]["mTC"]
                > c_v5.SINGLE_FRAME_ORACLE_MTC
            ),
        }
        with (result_output / "oracle_precheck.json").open("w") as handle:
            json.dump(oracle_precheck, handle, indent=2)
        print(json.dumps({"oracle_precheck": oracle_precheck}, indent=2), flush=True)

        if args.oracle_only:
            return

        history = []
        best = None
        for epoch in range(1, args.epochs + 1):
            train_stats = c_v5._train_epoch(
                model,
                observer,
                residual,
                correction,
                mask_predictor,
                refiner,
                selector,
                dynamics,
                raft,
                train_groups,
                optimizer,
                args.tbptt_steps,
            )
            metrics, diagnostics = _evaluate(
                model,
                observer,
                residual,
                correction,
                mask_predictor,
                refiner,
                selector,
                dynamics,
                val_groups,
                raft,
            )

            row = {
                "epoch": epoch,
                "train": train_stats,
                "metrics": metrics,
                "diagnostics": diagnostics,
                "delta_vs_c_v3_base": {
                    name: _delta_metrics(metrics[name], metrics["c_v3_base"])
                    for name in (
                        "c_v6",
                        "multiframe_semantic_oracle",
                        "multiframe_semantic_temporal_oracle",
                    )
                },
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
                        "selector_state_dict": selector.state_dict(),
                        "metrics": metrics,
                        "diagnostics": diagnostics,
                        "train": train_stats,
                        "dynamics": dynamics.config(),
                        "architecture": {
                            "history_length": c_v5.HISTORY_LENGTH,
                            "history_source": "raw detached frozen C-V3 logits",
                            "candidate_bank_semantics_enter_controller": False,
                            "controller_semantic_input": "validity-gated e1..eK",
                            "signed_error_encoding": "[ReLU(e), ReLU(-e)]",
                            "deep_history_error_validity_gated": True,
                            "dynamics_error_reference": "validity-gated e1",
                            "upstream_error_boundary_detached": True,
                            "history_logits_resampling": "one final warp per candidate",
                            "controller_output_feedback": False,
                            "selector_hidden_channels": c_v5.CONTROLLER_HIDDEN_CHANNELS,
                            "target_priority": "semantic correctness then temporal consistency",
                            "raft_inference": False,
                        },
                    },
                    output / "best.pt",
                )
            print(json.dumps(row, indent=2), flush=True)

        if best is None:
            raise RuntimeError("C-V6 produced no epoch result")

        base_metrics = best["metrics"]["c_v3_base"]
        candidate = best["metrics"]["c_v6"]
        summary = {
            "experiment": "C-V6 Error-Centric Multi-Hypothesis Temporal Predictive Coding",
            "oracle_precheck": oracle_precheck,
            "best": best,
            "history": history,
            "zero_step": zero_step,
            "dynamics": dynamics.config(),
            "frozen_checkpoints": {
                "fast_b": args.fast_b_checkpoint,
                "observer": args.observer_checkpoint,
                "residual": args.residual_checkpoint,
                "e1_base": args.base_checkpoint,
                "c_v3_base": args.c_v3_checkpoint,
                "c_v3_checkpoint_epoch": cv3_payload.get("epoch"),
                "residual_experiment": residual_payload.get("experiment"),
                "e1_experiment": base_payload.get("experiment"),
            },
            "architecture": {
                "history_length": c_v5.HISTORY_LENGTH,
                "history_source": "raw detached frozen C-V3 logits",
                "candidate_bank_semantics_enter_controller": False,
                "controller_semantic_input": "validity-gated class-wise e1..eK",
                "signed_error_encoding": "[ReLU(e), ReLU(-e)]",
                "deep_history_error_validity_gated": True,
                "dynamics_error_role": "motion-compensated T-masked explicit error state from e1",
                "upstream_error_boundary_detached": True,
                "history_logits_resampling": "one final warp per candidate",
                "controller_output_feedback": False,
                "training_loss": "direct per-pixel K+1 cross entropy on supervised decision pixels",
                "raft_inference": False,
            },
            "selection_rule": {
                "hard_constraint": "C-V6 mIoU >= frozen C-V3 Base mIoU",
                "objective_after_constraint": "maximize C-V6 mTC, then mIoU",
            },
            "target": {
                "reference_c_v3_mIoU": base_metrics["mIoU"],
                "expected_reference_c_v3_mIoU": c_v5.MIOU_REFERENCE,
                "mIoU_preserved": candidate["mIoU"] >= base_metrics["mIoU"],
                "mTC_target_low": c_v5.MTC_TARGET_LOW,
                "mTC_target_high": c_v5.MTC_TARGET_HIGH,
                "mTC_reached_74": candidate["mTC"] >= c_v5.MTC_TARGET_LOW,
            },
            "tbptt_steps": args.tbptt_steps,
            "epochs": args.epochs,
        }
        with (result_output / "summary.json").open("w") as handle:
            json.dump(summary, handle, indent=2)
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        c_v5._selector_evidence = original_selector_evidence


if __name__ == "__main__":
    main()
