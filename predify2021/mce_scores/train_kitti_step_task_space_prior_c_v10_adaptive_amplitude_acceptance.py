"""C-V10: adaptive-amplitude acceptance for prediction-error correction.

中文：C-V10 预测误差修正的自适应幅度接受机制。

Architecture decision / 架构决策
--------------------------------
C-V9 Gate Acceptance Decomposition showed that the fixed g_max=0.25 cap makes
75.7% of beneficial pixels unreachable, while the existing Gate already has
useful acceptance/protection behavior. C-V10 therefore keeps the validated C-V9
95D->32D control feature extractor and splits only its scalar output role:

  acceptance a_t = sigmoid(l_acc) : whether to accept the correction;
  amplitude  s_t = sigmoid(l_amp) : how strongly to apply it;
  alpha_t = a_t * s_t             : learned error gain in [0, 1].

Relative to C-V9, the control path adds only one 32D->1 Amplitude Head.
The predictive-coding causal boundary is unchanged:
History -> Prediction -> Prediction Error -> Proposal -> Error-gain Correction.

Training supervision / 训练监督
------------------------------
Exactly reuse C-V9 supervision:
  L = mean_all_frames(L_final_CE)
      + mean_rescue_frames(L_proposal_rescue_CE)

No Acceptance target, no Amplitude Oracle target, no temporal loss, no RAFT
teacher. The experimental variable is the fixed-cap scalar Gate becoming
Acceptance * Adaptive Amplitude while preserving the C-V9 control encoder.
"""

import argparse
import json
import math
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
from predify2021.mce_scores import (
    train_kitti_step_task_space_prior_c_v9_proposal_supervision as c_v9,
)
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_adaptive_amplitude_proposal_corrector import (
    AdaptiveAmplitudeProposalCorrector,
)
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_semantic_hysteresis import (
    EulerDynamicsError,
)


EXPERIMENT = "c_v10_adaptive_amplitude_acceptance"
OUTPUT_DEFAULT = (
    "/home/lin/predify/experiments/"
    "kitti_step_task_space_prior_c_v10_adaptive_amplitude_acceptance"
)
RESULT_DEFAULT = (
    "results/kitti_step_task_space_prior_c_v10_adaptive_amplitude_acceptance"
)
MODEL_KEY = "c_v10"
MIOU_HARD_FLOOR = c_v9.MIOU_HARD_FLOOR
PROPOSAL_LOSS_WEIGHT = c_v9.PROPOSAL_LOSS_WEIGHT
ACCEPTANCE_BIAS = -2.0
AMPLITUDE_INIT = 0.25


class _ControlStats:
    """Accumulate detached low-resolution control means without per-frame sync."""

    def __init__(self, corrector):
        self.corrector = corrector
        self.handle = None
        self.count = 0
        self.sums = {
            "acceptance_mean": None,
            "amplitude_mean": None,
            "alpha_mean": None,
        }

    def _hook(self, _module, _inputs, output):
        if not isinstance(output, dict):
            return
        values = {
            "acceptance_mean": output.get("acceptance"),
            "amplitude_mean": output.get("amplitude"),
            "alpha_mean": output.get("alpha"),
        }
        if any(value is None for value in values.values()):
            return
        for key, value in values.items():
            scalar = value.detach().mean()
            if self.sums[key] is None:
                self.sums[key] = scalar
            else:
                self.sums[key] = self.sums[key] + scalar
        self.count += 1

    def __enter__(self):
        self.handle = self.corrector.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def result(self):
        if self.count == 0:
            return {
                "control_frames": 0,
                "acceptance_mean": 0.0,
                "amplitude_mean": 0.0,
                "alpha_mean": 0.0,
            }
        return {
            "control_frames": self.count,
            **{
                key: float((value / self.count).item())
                for key, value in self.sums.items()
            },
        }


def _rename_metrics(raw_metrics):
    if "c_v7" not in raw_metrics:
        raise KeyError("shared C-V7 evaluator did not return its candidate slot")
    return {
        "host": raw_metrics["host"],
        "c_v3_base": raw_metrics["c_v3_base"],
        MODEL_KEY: raw_metrics["c_v7"],
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
    """Real-frame contract: C-V10 starts at C-V3 with C-V9 initial alpha."""
    samples = next((rows for rows in groups.values() if len(rows) >= 3), None)
    if samples is None:
        raise RuntimeError("No sequence with at least three frames for C-V10 zero-step check")

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

    row = evidence["row"]
    proposal_weight_max = float(corrector.proposal_head.weight.abs().max().item())
    delta_raw_max = float(row["delta_z_raw"].abs().max().item())
    equality_max = float(
        (evidence["final_logits_full"] - c_v3_logits.detach()).abs().max().item()
    )
    acceptance_mean = float(row["acceptance"].mean().item())
    amplitude_mean = float(row["amplitude"].mean().item())
    alpha_mean = float(row["alpha"].mean().item())
    expected_acceptance = 1.0 / (1.0 + math.exp(-ACCEPTANCE_BIAS))
    expected_alpha = expected_acceptance * AMPLITUDE_INIT

    if proposal_weight_max != 0.0 or delta_raw_max != 0.0 or equality_max != 0.0:
        raise RuntimeError(
            "C-V10 zero-step equality failed: "
            f"proposal_w={proposal_weight_max} delta_raw={delta_raw_max} "
            f"equality={equality_max}"
        )
    if abs(acceptance_mean - expected_acceptance) > 1e-7:
        raise RuntimeError("C-V10 initial Acceptance operating point mismatch")
    if abs(amplitude_mean - AMPLITUDE_INIT) > 1e-7:
        raise RuntimeError("C-V10 initial Amplitude operating point mismatch")
    if abs(alpha_mean - expected_alpha) > 1e-7:
        raise RuntimeError("C-V10 initial alpha does not match C-V9 initial gate")

    return {
        "proposal_head_weight_abs_max": proposal_weight_max,
        "proposal_head_bias": False,
        "delta_z_raw_abs_max": delta_raw_max,
        "c_v10_vs_c_v3_logit_abs_max": equality_max,
        "acceptance_bias": corrector.acceptance_bias,
        "amplitude_bias": corrector.amplitude_bias,
        "acceptance_mean": acceptance_mean,
        "amplitude_mean": amplitude_mean,
        "alpha_mean": alpha_mean,
        "former_c_v9_initial_gate": 0.25 * expected_acceptance,
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

    corrector = AdaptiveAmplitudeProposalCorrector(
        num_classes=c_v5.NUM_CLASSES,
        history_length=c_v5.HISTORY_LENGTH,
        hidden_channels=c_v5.CONTROLLER_HIDDEN_CHANNELS,
        acceptance_bias=ACCEPTANCE_BIAS,
        amplitude_init=AMPLITUDE_INIT,
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

    raft_metric = FrozenRAFT()

    history = []
    best = None
    for epoch in range(1, args.epochs + 1):
        with _ControlStats(corrector) as train_control:
            train_stats = c_v9._train_epoch(
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
        train_stats["control_lowres"] = train_control.result()
        if "gate_mean" in train_stats:
            train_stats["alpha_full_mean"] = train_stats.pop("gate_mean")

        with _ControlStats(corrector) as eval_control:
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
        diagnostics["control_lowres"] = eval_control.result()
        if "gate_mean" in diagnostics:
            diagnostics["alpha_full_mean"] = diagnostics.pop("gate_mean")
        diagnostics.pop("g_max", None)
        metrics = _rename_metrics(raw_metrics)
        diagnostics.pop("raw_history_enters_correction_head", None)
        diagnostics.update(
            {
                "semantic_proposal_source": "concat validity-gated e1..e4",
                "proposal_head": "same C-V9 bias-free zero-init 1x1 Conv 76D->19D",
                "proposal_supervision": True,
                "proposal_supervision_scope": "formal Rescue pixels only",
                "proposal_loss_weight": PROPOSAL_LOSS_WEIGHT,
                "proposal_loss_gate_bypass": True,
                "proposal_loss_tanh_bypass": True,
                "control_pre": "shared C-V9 95D->32D gate_pre",
                "acceptance_head": "shared-control 32D->1 sigmoid",
                "amplitude_head": "new shared-control 32D->1 sigmoid",
                "acceptance_amplitude_share_pre": True,
                "additional_control_parameters_vs_c_v9": corrector.hidden_channels + 1,
                "adaptive_error_gain": "alpha = acceptance * amplitude",
                "fixed_g_max_removed": True,
                "alpha_range": [0.0, 1.0],
                "initial_operating_point_matches_c_v9": True,
                "acceptance_direct_target": False,
                "amplitude_oracle_target": False,
                "joint_final_ce_gradient": True,
                "tanh_bound_retained": True,
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
                        "base": "C-V9 Proposal-Supervised Predictive-Error Correction",
                        "history_length": c_v5.HISTORY_LENGTH,
                        "history_source": "raw detached frozen C-V3 logits",
                        "causal_path": "History -> Prediction -> Prediction Error -> Proposal -> Adaptive Error Gain -> Correction",
                        "prediction_error": "V_k * (P_cur - P_hist_k)",
                        "semantic_proposal_source": "concat(e1..e4)",
                        "proposal_head": "bias-free zero-init 1x1 Conv",
                        "proposal_input_channels": c_v5.NUM_CLASSES * c_v5.HISTORY_LENGTH,
                        "proposal_output_channels": c_v5.NUM_CLASSES,
                        "error_hidden_role": "motion-aligned temporal reliability context",
                        "decision_evidence_channels": corrector.decision_channels,
                        "control_pre": "shared C-V9 95D->32D gate_pre",
                        "acceptance_head": "32D->1 sigmoid",
                        "amplitude_head": "32D->1 sigmoid",
                        "acceptance_amplitude_share_input": True,
                        "acceptance_amplitude_share_pre": True,
                        "additional_control_parameters_vs_c_v9": corrector.hidden_channels + 1,
                        "alpha": "acceptance * amplitude",
                        "alpha_range": [0.0, 1.0],
                        "fixed_g_max": False,
                        "acceptance_bias_init": corrector.acceptance_bias,
                        "amplitude_init": corrector.amplitude_init,
                        "amplitude_bias_init": corrector.amplitude_bias,
                        "z_cur_detached": True,
                        "output": "Z_final = Z_cur + alpha * tanh(DeltaZ_proposal)",
                        "validity_order": "upsample first, full-resolution mask second",
                        "raw_history_direct_fusion": False,
                    },
                    "training_supervision": {
                        "final_loss": "full-image segmentation CE",
                        "proposal_loss": "raw Proposal CE on formal Rescue pixels only",
                        "proposal_loss_weight": PROPOSAL_LOSS_WEIGHT,
                        "proposal_loss_window_normalization": "mean over Rescue-bearing frames only",
                        "acceptance_target": False,
                        "amplitude_target": False,
                        "temporal_loss": False,
                        "raft_training": False,
                        "gt_inference_input": False,
                    },
                    "zero_step": zero_step,
                },
                output / "best.pt",
            )
        print(json.dumps(row, indent=2), flush=True)

    if best is None:
        raise RuntimeError("C-V10 produced no epoch result")

    summary = {
        "experiment": "C-V10 Adaptive-Amplitude Proposal Acceptance",
        "architecture_decision": "retain C-V9 control encoder; replace fixed 0.25 cap with Acceptance * Adaptive Amplitude",
        "best": best,
        "history": history,
        "zero_step": zero_step,
        "selection_rule": {
            "hard_constraint": f"C-V10 mIoU >= fixed C-V4 E2 floor {MIOU_HARD_FLOOR:.16f}",
            "objective_after_constraint": "maximize C-V10 mTC, then mIoU",
            "fallback_if_all_fail_floor": "highest mIoU, then mTC",
        },
        "training_supervision": {
            "same_as_c_v9": True,
            "proposal_loss_weight": PROPOSAL_LOSS_WEIGHT,
            "acceptance_direct_supervision": False,
            "amplitude_oracle_supervision": False,
            "temporal_loss": False,
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
