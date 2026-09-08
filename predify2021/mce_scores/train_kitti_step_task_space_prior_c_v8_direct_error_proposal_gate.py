"""C-V8 Direct Prediction-Error Proposal + Temporal Error Memory + Proposal-Aware Gate.

中文：C-V8 直接预测误差提议 + 时序误差记忆 + 提议感知门控。

Architecture decision implemented from the frozen C-V7 diagnostics:
- concat(e1..e4) directly produces the 19-class semantic proposal;
- H_err remains motion-aligned recurrent temporal context for Gate decisions;
- Gate explicitly observes the detached semantic proposal;
- tanh bound and g_max=0.25 are retained;
- Raw History never enters Proposal Head or Gate directly;
- training remains final segmentation CE only; no Temporal Loss.

This script intentionally reuses the validated C-V7 History/Prediction-Error,
TBPTT, Full9 mIoU/mTC/mVC and validity pipeline. Only the corrector structure is
changed, so the experiment answers the architecture question rather than mixing
new data/evaluation/training protocols.
"""

import argparse
import json
import random
from pathlib import Path

import torch

from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import (
    sequence_groups,
)
from predify2021.mce_scores.train_kitti_step_temporal_joint import FrozenRAFT
from predify2021.mce_scores import (
    train_kitti_step_task_space_prior_c_v5_multiframe_candidate_memory as c_v5,
)
from predify2021.mce_scores import (
    train_kitti_step_task_space_prior_c_v7_error_residual_correction as c_v7,
)
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_direct_error_proposal_corrector import (
    DirectErrorProposalCorrector,
)
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_semantic_hysteresis import (
    EulerDynamicsError,
)


EXPERIMENT = "c_v8_direct_error_proposal_gate"
OUTPUT_DEFAULT = (
    "/home/lin/predify/experiments/"
    "kitti_step_task_space_prior_c_v8_direct_error_proposal_gate"
)
RESULT_DEFAULT = "results/kitti_step_task_space_prior_c_v8_direct_error_proposal_gate"
MIOU_HARD_FLOOR = c_v7.MIOU_HARD_FLOOR
G_MAX = c_v7.G_MAX
GATE_BIAS = c_v7.GATE_BIAS
MODEL_KEY = "c_v8"


def _rename_c_v7_metric_slot(metrics):
    """C-V7 evaluator is structure-agnostic; rename its candidate slot to C-V8."""
    if "c_v7" not in metrics:
        raise KeyError("shared evaluator did not return c_v7 candidate slot")
    return {
        "host": metrics["host"],
        "c_v3_base": metrics["c_v3_base"],
        MODEL_KEY: metrics["c_v7"],
    }


def _selection_key(metrics):
    candidate = metrics[MODEL_KEY]
    passed = candidate["mIoU"] >= MIOU_HARD_FLOOR
    if passed:
        return (1, candidate["mTC"], candidate["mIoU"])
    return (0, candidate["mIoU"], candidate["mTC"])


@torch.no_grad()
def _zero_step_equality_check(
    model,
    observer,
    residual,
    correction,
    mask_predictor,
    refiner,
    corrector,
    dynamics,
    groups,
):
    """Real-frame contract: zero proposal must make C-V8 exactly equal C-V3."""
    samples = next((rows for rows in groups.values() if len(rows) >= 3), None)
    if samples is None:
        raise RuntimeError("No sequence with at least three frames for C-V8 zero-step check")

    frame0 = c_v5._host_observation(model, samples[0])
    frame1 = c_v5._host_observation(model, samples[1])
    pending_motion, _ = c_v5._initialize_motion(observer, residual, frame0, frame1)
    _, previous_host_logits, previous_low, _, _ = frame1
    _, host_logits, host_low, current_c1, output_size = c_v5._host_observation(
        model, samples[2]
    )
    prior_low, _ = c_v5.warp_low_logits(previous_low, pending_motion)
    semantic_state_low = torch.zeros_like(previous_low)
    e1 = c_v5._frozen_e1_step(
        correction,
        mask_predictor,
        current_c1,
        host_low,
        prior_low,
        pending_motion,
        semantic_state_low,
        None,
        None,
        None,
    )
    memory_row, _, _, c_v3_logits = c_v5._frozen_cv3_step(
        refiner,
        current_c1,
        host_low,
        prior_low,
        e1,
        pending_motion,
        None,
        output_size,
        host_logits,
    )
    candidate_rows = c_v5._build_history_candidates(
        [previous_host_logits.detach()],
        [],
        pending_motion,
        corrector.history_length,
    )
    evidence = c_v7._correction_evidence(
        corrector,
        dynamics,
        c_v3_logits,
        candidate_rows,
        pending_motion,
        e1["transportability_low"],
        memory_row["memory_reliability"],
        None,
        None,
    )

    proposal_weight_max = float(corrector.proposal_head.weight.abs().max().item())
    delta_raw_max = float(evidence["row"]["delta_z_raw"].abs().max().item())
    equality_max = float(
        (evidence["final_logits_full"] - c_v3_logits.detach()).abs().max().item()
    )
    if proposal_weight_max != 0.0 or delta_raw_max != 0.0 or equality_max != 0.0:
        raise RuntimeError(
            "C-V8 zero-step equality failed: "
            f"proposal_w={proposal_weight_max} delta_raw={delta_raw_max} "
            f"equality={equality_max}"
        )
    return {
        "proposal_head_weight_abs_max": proposal_weight_max,
        "proposal_head_bias": False,
        "delta_z_raw_abs_max": delta_raw_max,
        "c_v8_vs_c_v3_logit_abs_max": equality_max,
        "gate_bias": float(corrector.gate_head.bias.detach().mean().item()),
        "g_max": corrector.g_max,
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
    parser.add_argument("--seed", type=int, default=c_v5.SEED)
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

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

    # Gate operating point is fixed to C-V7 values for this architecture test.
    # Do not turn the first C-V8 run into a g_max / gate-bias sweep.
    corrector = DirectErrorProposalCorrector(
        num_classes=c_v5.NUM_CLASSES,
        history_length=c_v5.HISTORY_LENGTH,
        hidden_channels=c_v5.CONTROLLER_HIDDEN_CHANNELS,
        g_max=G_MAX,
        gate_bias=GATE_BIAS,
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

    train = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "train")
    val = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "val")
    train_groups = sequence_groups(train)
    all_val_groups = sequence_groups(val)
    missing = [sequence for sequence in c_v5.FULL9 if sequence not in all_val_groups]
    if missing:
        raise RuntimeError(f"Missing Full9 validation sequences: {missing}")
    val_groups = {sequence: all_val_groups[sequence] for sequence in c_v5.FULL9}

    zero_step = _zero_step_equality_check(
        model,
        observer,
        residual,
        correction,
        mask_predictor,
        refiner,
        corrector,
        dynamics,
        train_groups,
    )

    output = Path(args.output)
    result_output = Path(args.result_output)
    output.mkdir(parents=True, exist_ok=True)
    result_output.mkdir(parents=True, exist_ok=True)

    # RAFT is metric-only, exactly as in the reviewed C-V7 evaluator.
    raft_metric = FrozenRAFT()

    history = []
    best = None
    for epoch in range(1, args.epochs + 1):
        train_stats = c_v7._train_epoch(
            model,
            observer,
            residual,
            correction,
            mask_predictor,
            refiner,
            corrector,
            dynamics,
            train_groups,
            optimizer,
            args.tbptt_steps,
        )
        raw_metrics, diagnostics = c_v7._evaluate(
            model,
            observer,
            residual,
            correction,
            mask_predictor,
            refiner,
            corrector,
            dynamics,
            val_groups,
            raft_metric,
        )
        metrics = _rename_c_v7_metric_slot(raw_metrics)
        diagnostics.pop("raw_history_enters_correction_head", None)
        diagnostics.update(
            {
                "semantic_proposal_source": "concat validity-gated e1..e4",
                "proposal_head": "bias-free zero-init 1x1 Conv, 76D->19D for K=4/C=19",
                "raw_history_enters_proposal_head": False,
                "raw_history_enters_gate": False,
                "error_hidden_enters_proposal_head": False,
                "error_hidden_role": "motion-aligned temporal context for proposal acceptance",
                "proposal_aware_gate": True,
                "proposal_gate_input_detached": True,
                "proposal_gate_input_channels": c_v5.NUM_CLASSES,
                "tanh_bound_retained": True,
                "fixed_gate_operating_point": True,
            }
        )

        row = {
            "epoch": epoch,
            "train": train_stats,
            "metrics": metrics,
            "diagnostics": diagnostics,
            "delta_vs_host": c_v7._delta_metrics(metrics[MODEL_KEY], metrics["host"]),
            "delta_vs_c_v3": c_v7._delta_metrics(
                metrics[MODEL_KEY], metrics["c_v3_base"]
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
                    "dynamics": dynamics.config(),
                    "architecture": {
                        "history_length": c_v5.HISTORY_LENGTH,
                        "history_source": "raw detached frozen C-V3 logits",
                        "causal_path": "History -> Prediction -> Prediction Error -> Proposal -> Gated Correction",
                        "prediction_error": "V_k * (P_cur - P_hist_k)",
                        "semantic_proposal_source": "concat(e1..e4)",
                        "proposal_head": "bias-free zero-init 1x1 Conv",
                        "proposal_input_channels": c_v5.NUM_CLASSES * c_v5.HISTORY_LENGTH,
                        "proposal_output_channels": c_v5.NUM_CLASSES,
                        "raw_history_enters_proposal_head": False,
                        "raw_history_enters_gate": False,
                        "error_memory_evidence_aggregation": True,
                        "error_hidden_motion_aligned": True,
                        "error_hidden_reliability_gated": True,
                        "error_hidden_enters_proposal_head": False,
                        "error_hidden_role": "temporal context for Gate",
                        "proposal_aware_gate": True,
                        "proposal_gate_input_detached": True,
                        "gate_channels": 1,
                        "g_max": corrector.g_max,
                        "gate_bias_init": corrector.gate_bias,
                        "fixed_gate_operating_point": True,
                        "z_cur_detached": True,
                        "output": "Z_final = Z_cur + g * tanh(DeltaZ_proposal)",
                        "validity_order": "upsample first, full-resolution mask second",
                        "training_loss": "final segmentation CE only",
                        "temporal_loss": False,
                        "raft_training": False,
                        "raft_inference_decision": False,
                    },
                    "zero_step": zero_step,
                },
                output / "best.pt",
            )
        print(json.dumps(row, indent=2), flush=True)

    if best is None:
        raise RuntimeError("C-V8 produced no epoch result")

    summary = {
        "experiment": "C-V8 Direct Prediction-Error Proposal + Temporal Error Memory + Proposal-Aware Gate",
        "best": best,
        "history": history,
        "zero_step": zero_step,
        "selection_rule": {
            "hard_constraint": f"C-V8 mIoU >= fixed C-V4 E2 floor {MIOU_HARD_FLOOR:.16f}",
            "objective_after_constraint": "maximize C-V8 mTC, then mIoU",
            "fallback_if_all_fail_floor": "highest mIoU, then mTC",
        },
        "architecture": {
            "history_length": c_v5.HISTORY_LENGTH,
            "history_source": "raw detached frozen C-V3 logits",
            "causal_path": "History -> Prediction -> Prediction Error -> Proposal -> Gated Correction",
            "semantic_proposal_source": "concat validity-gated e1..e4",
            "proposal_head": "bias-free zero-init 1x1 Conv",
            "error_memory_role": "motion-aligned temporal context for Gate",
            "proposal_aware_gate": True,
            "proposal_gate_input_detached": True,
            "raw_history_direct_fusion": False,
            "z_cur_detached": True,
            "tanh_bound": True,
            "g_max": corrector.g_max,
            "gate_bias_init": corrector.gate_bias,
            "fixed_gate_operating_point": True,
            "training_loss": "final segmentation CE only",
            "temporal_loss": False,
            "raft_training": False,
            "raft_metric_only": True,
        },
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
        "tbptt_steps": args.tbptt_steps,
        "epochs": args.epochs,
    }
    with (result_output / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
