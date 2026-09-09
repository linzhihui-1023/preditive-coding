"""C-V11 Reliability-Conditioned Expansion training entrypoint.

中文：C-V11 可靠性条件幅度扩张训练入口。
"""

import argparse
import json
import math
import random
from pathlib import Path

import torch

from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import sequence_groups
from predify2021.mce_scores.train_kitti_step_temporal_joint import FrozenRAFT
from predify2021.mce_scores import c_v11_reliability_training as training
from predify2021.mce_scores import train_kitti_step_task_space_prior_c_v5_multiframe_candidate_memory as c_v5
from predify2021.mce_scores import train_kitti_step_task_space_prior_c_v7_error_residual_correction as c_v7
from predify2021.mce_scores import train_kitti_step_task_space_prior_c_v9_proposal_supervision as c_v9
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_reliability_conditioned_expansion_corrector import ReliabilityConditionedExpansionCorrector
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_semantic_hysteresis import EulerDynamicsError

EXPERIMENT = "c_v11_reliability_conditioned_expansion"
OUTPUT_DEFAULT = "/home/lin/predify/experiments/kitti_step_task_space_prior_c_v11_reliability_conditioned_expansion"
RESULT_DEFAULT = "results/kitti_step_task_space_prior_c_v11_reliability_conditioned_expansion"
MODEL_KEY = "c_v11"
MIOU_HARD_FLOOR = c_v9.MIOU_HARD_FLOOR
ACCEPTANCE_BIAS = -2.0
BASE_GAIN = training.BASE_GAIN
RELIABILITY_INIT = 1e-3


def _rename_metrics(raw_metrics):
    if "c_v7" not in raw_metrics:
        raise KeyError("shared C-V7 evaluator did not return candidate slot")
    return {"host": raw_metrics["host"], "c_v3_base": raw_metrics["c_v3_base"], MODEL_KEY: raw_metrics["c_v7"]}


def _selection_key(metrics):
    candidate = metrics[MODEL_KEY]
    passed = candidate["mIoU"] >= MIOU_HARD_FLOOR
    return (1, candidate["mTC"], candidate["mIoU"]) if passed else (0, candidate["mIoU"], candidate["mTC"])


@torch.no_grad()
def _zero_step_equality_check(model, observer, residual, correction, mask_predictor, refiner, corrector, dynamics, groups):
    samples = next((rows for rows in groups.values() if len(rows) >= 3), None)
    if samples is None:
        raise RuntimeError("No sequence with at least three frames for C-V11 zero-step check")
    frame0 = c_v5._host_observation(model, samples[0])
    frame1 = c_v5._host_observation(model, samples[1])
    pending_motion, _ = c_v5._initialize_motion(observer, residual, frame0, frame1)
    _, previous_host_logits, previous_low, _, _ = frame1
    _, host_logits, host_low, current_c1, output_size = c_v5._host_observation(model, samples[2])
    prior_low, _ = c_v5.warp_low_logits(previous_low, pending_motion)
    e1 = c_v5._frozen_e1_step(
        correction, mask_predictor, current_c1, host_low, prior_low, pending_motion,
        torch.zeros_like(previous_low), None, None, None,
    )
    memory_row, _, _, c_v3_logits = c_v5._frozen_cv3_step(
        refiner, current_c1, host_low, prior_low, e1, pending_motion, None,
        output_size, host_logits,
    )
    candidate_rows = c_v5._build_history_candidates(
        [previous_host_logits.detach()], [], pending_motion, corrector.history_length
    )
    evidence = c_v7._correction_evidence(
        corrector, dynamics, c_v3_logits, candidate_rows, pending_motion,
        e1["transportability_low"], memory_row["memory_reliability"], None, None,
    )
    row = evidence["row"]
    proposal_weight_max = float(corrector.proposal_head.weight.abs().max().item())
    delta_raw_max = float(row["delta_z_raw"].abs().max().item())
    equality_max = float((evidence["final_logits_full"] - c_v3_logits.detach()).abs().max().item())
    acceptance = float(row["acceptance"].mean().item())
    reliability = float(row["expansion_reliability"].mean().item())
    base_alpha = float(row["base_alpha"].mean().item())
    alpha = float(row["alpha"].mean().item())
    expected_a = 1.0 / (1.0 + math.exp(-ACCEPTANCE_BIAS))
    expected_base = BASE_GAIN * expected_a
    expected_alpha = expected_base + (1.0 - expected_base) * RELIABILITY_INIT
    if proposal_weight_max != 0.0 or delta_raw_max != 0.0 or equality_max != 0.0:
        raise RuntimeError("C-V11 zero-step C-V3 equality failed")
    if abs(acceptance - expected_a) > 1e-7 or abs(reliability - RELIABILITY_INIT) > 1e-7:
        raise RuntimeError("C-V11 initial control operating point mismatch")
    if abs(base_alpha - expected_base) > 1e-7 or abs(alpha - expected_alpha) > 1e-7:
        raise RuntimeError("C-V11 initial alpha mismatch")
    return {
        "proposal_head_weight_abs_max": proposal_weight_max,
        "delta_z_raw_abs_max": delta_raw_max,
        "c_v11_vs_c_v3_logit_abs_max": equality_max,
        "acceptance_mean": acceptance,
        "expansion_reliability_mean": reliability,
        "base_alpha_mean": base_alpha,
        "alpha_mean": alpha,
        "initial_expansion_over_c_v9": alpha - expected_base,
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
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    c_v5._validate_bounded_motion_checkpoint(args.residual_checkpoint)
    model = c_v5.load_fast_b_model(args.fast_b_checkpoint).cuda().eval(); model.requires_grad_(False)
    observer, _ = c_v5._load_frozen_observer(args.observer_checkpoint)
    residual, residual_payload = c_v5._load_frozen_residual(args.residual_checkpoint, observer)
    correction, mask_predictor, base_payload = c_v5._load_frozen_e1_base(args.base_checkpoint, observer)
    refiner, cv3_payload = c_v5._load_frozen_cv3_refiner(args.c_v3_checkpoint)
    corrector = ReliabilityConditionedExpansionCorrector(
        num_classes=c_v5.NUM_CLASSES,
        history_length=c_v5.HISTORY_LENGTH,
        hidden_channels=c_v5.CONTROLLER_HIDDEN_CHANNELS,
        acceptance_bias=ACCEPTANCE_BIAS,
        base_gain=BASE_GAIN,
        reliability_init=RELIABILITY_INIT,
    ).cuda()
    dynamics = EulerDynamicsError(tau_e=args.dynamics_tau_e, k_e=args.dynamics_k_e, dt=args.dynamics_dt)
    optimizer = torch.optim.AdamW(corrector.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "train")
    val = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "val")
    train_groups = sequence_groups(train)
    all_val_groups = sequence_groups(val)
    missing = [s for s in c_v5.FULL9 if s not in all_val_groups]
    if missing:
        raise RuntimeError(f"Missing Full9 validation sequences: {missing}")
    val_groups = {s: all_val_groups[s] for s in c_v5.FULL9}
    zero_step = _zero_step_equality_check(
        model, observer, residual, correction, mask_predictor, refiner,
        corrector, dynamics, train_groups,
    )

    output = Path(args.output); result_output = Path(args.result_output)
    output.mkdir(parents=True, exist_ok=True); result_output.mkdir(parents=True, exist_ok=True)
    raft_metric = FrozenRAFT()
    history = []; best = None
    for epoch in range(1, args.epochs + 1):
        with training.ControlStats(corrector) as train_control:
            train_stats = training.train_epoch(
                model, observer, residual, correction, mask_predictor, refiner,
                corrector, dynamics, train_groups, optimizer, args.tbptt_steps,
            )
        train_stats["control_lowres"] = train_control.result()
        with training.ControlStats(corrector) as eval_control:
            raw_metrics, diagnostics = c_v7._evaluate(
                model, observer, residual, correction, mask_predictor, refiner,
                corrector, dynamics, val_groups, raft_metric,
            )
        diagnostics["control_lowres"] = eval_control.result()
        if "gate_mean" in diagnostics:
            diagnostics["alpha_full_mean"] = diagnostics.pop("gate_mean")
        diagnostics.pop("g_max", None)
        diagnostics.pop("raw_history_enters_correction_head", None)
        metrics = _rename_metrics(raw_metrics)
        diagnostics.update({
            "semantic_proposal_source": "concat validity-gated e1..e4",
            "control_pre": "shared C-V9 95D->32D gate_pre",
            "acceptance_head": "shared C-V9 32D->1 gate_head",
            "expansion_reliability_head": "new shared-control 32D->1 sigmoid",
            "additional_control_parameters_vs_c_v9": 33,
            "base_gain": BASE_GAIN,
            "adaptive_error_gain": "alpha = 0.25*a + (1-0.25*a)*r",
            "full_reliability_reaches_alpha_one": True,
            "reliability_supervision": True,
            "reliability_positive": "current wrong AND 0.25 proposal wrong AND full proposal correct",
            "reliability_negative": "current correct AND full proposal wrong",
            "reliability_aux_detaches_control_hidden": True,
            "reliability_loss_weight": training.RELIABILITY_LOSS_WEIGHT,
            "proposal_supervision": True,
            "proposal_loss_weight": training.PROPOSAL_LOSS_WEIGHT,
            "temporal_loss": False,
            "tanh_bound_retained": True,
        })
        row = {
            "epoch": epoch,
            "train": train_stats,
            "metrics": metrics,
            "diagnostics": diagnostics,
            "delta_vs_host": c_v7._delta_metrics(metrics[MODEL_KEY], metrics["host"]),
            "delta_vs_c_v3": c_v7._delta_metrics(metrics[MODEL_KEY], metrics["c_v3_base"]),
        }
        row["selection_key"] = list(_selection_key(metrics)); history.append(row)
        with (result_output / f"epoch_{epoch:03d}.json").open("w") as handle:
            json.dump(row, handle, indent=2)
        if best is None or tuple(row["selection_key"]) > tuple(best["selection_key"]):
            best = row
            torch.save({
                "experiment": EXPERIMENT,
                "epoch": epoch,
                "corrector_state_dict": corrector.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
                "diagnostics": diagnostics,
                "train": train_stats,
                "dynamics": dynamics.config(),
                "architecture": {
                    "base": "C-V9 Proposal-Supervised Predictive-Error Correction",
                    "causal_path": "History -> Prediction -> Prediction Error -> Proposal -> Reliability-Conditioned Gain -> Correction",
                    "proposal_head": "same C-V9 76D->19D",
                    "control_pre": "same C-V9 95D->32D",
                    "acceptance_head": "same C-V9 32D->1",
                    "expansion_reliability_head": "new 32D->1",
                    "additional_control_parameters_vs_c_v9": 33,
                    "base_gain": BASE_GAIN,
                    "reliability_init": RELIABILITY_INIT,
                    "alpha": "0.25*a + (1-0.25*a)*r",
                    "full_reliability_reaches_alpha_one": True,
                    "raw_history_direct_fusion": False,
                },
                "training_supervision": {
                    "final_ce": True,
                    "proposal_rescue_ce": True,
                    "expansion_reliability_bce": True,
                    "reliability_aux_control_hidden_detached": True,
                    "temporal_loss": False,
                    "raft_training": False,
                    "gt_inference_input": False,
                },
                "zero_step": zero_step,
            }, output / "best.pt")
        print(json.dumps(row, indent=2), flush=True)

    if best is None:
        raise RuntimeError("C-V11 produced no epoch result")
    summary = {
        "experiment": "C-V11 Reliability-Conditioned Expansion",
        "architecture_decision": "retain C-V9 safe base gain; allow >0.25 correction only through supervised Expansion Reliability",
        "best": best,
        "history": history,
        "zero_step": zero_step,
        "selection_rule": {
            "hard_constraint": f"C-V11 mIoU >= fixed C-V4 E2 floor {MIOU_HARD_FLOOR:.16f}",
            "objective_after_constraint": "maximize C-V11 mTC, then mIoU",
            "fallback_if_all_fail_floor": "highest mIoU, then mTC",
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
