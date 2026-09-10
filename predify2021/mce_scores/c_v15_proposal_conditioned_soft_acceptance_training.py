"""Training/evaluation wrapper for C-V15 proposal-conditioned soft acceptance.

中文：C-V15 修正提案条件化软接受训练与评测层。

Training supervision deliberately remains Host-relative so C-V15 changes only
one experimental variable: whether Acceptance sees the specific proposal.
Validation reports three separate diagnostic layers:
1. Host-reference acceptance separation: the target actually used by BCE.
2. C-V4-reference acceptance separation: evaluation-only, never backpropagated.
3. C-V4 Damage/Rescue: downstream effect of the final soft-gated correction.

This separation prevents a Host/C-V4 reference mismatch from being mistaken for
failure of proposal conditioning itself.
"""

import torch
from torch.nn import functional as F

from predify2021.mce_scores import (
    c_v14_post_writeback_reliability_training as _base,
)


# Fixed protocol aliases used by the C-V15 entrypoint.
HISTORY_LENGTH = _base.HISTORY_LENGTH
SEMANTIC_CHANNELS = _base.SEMANTIC_CHANNELS
C4_CHANNELS = _base.C4_CHANNELS
RESIDUAL_SCALE = _base.RESIDUAL_SCALE
PROTECTION_LOSS_WEIGHT = _base.PROTECTION_LOSS_WEIGHT
ACCEPTANCE_LOSS_WEIGHT = _base.ACCEPTANCE_LOSS_WEIGHT
C_V4_CHECKPOINT_DEFAULT = _base.C_V4_CHECKPOINT_DEFAULT
NUM_CLASSES = _base.NUM_CLASSES
IGNORE_LABEL = _base.IGNORE_LABEL
FULL9 = _base.FULL9


def proposal_conditioned_acceptance_probability(correction_row, output_size):
    """Return the exact deployed C-V15 gate resized from c4 to output space."""
    return F.interpolate(
        correction_row["reliability"],
        size=tuple(output_size),
        mode="bilinear",
        align_corners=False,
    )[0, 0].clamp(0.0, 1.0)


def acceptance_bce_loss(corrector, correction_row, output_size, target, supervised):
    """Host-relative beneficial/harmful BCE on the deployed C-V15 gate."""
    if int(supervised.sum().item()) == 0:
        zero = correction_row["reliability_logit"].sum() * 0.0
        return zero, None
    probability_full = proposal_conditioned_acceptance_probability(
        correction_row,
        output_size,
    )
    probability = probability_full[supervised].clamp(1e-6, 1.0 - 1e-6)
    bce = F.binary_cross_entropy(
        probability,
        target[supervised],
        reduction="mean",
    )
    return bce, probability_full


# C-V14 train_sequence resolves acceptance_bce_loss from its module globals.
# Patch only the loss implementation. Its Host-relative target definition,
# sequence construction, K=4 history, and losses remain unchanged.
_base.acceptance_bce_loss = acceptance_bce_loss


def reference_acceptance_targets(reference_logits, proposal_logits, gt_gpu):
    """Evaluation-only proposal labels relative to an arbitrary reference.

    positive: reference wrong and full frozen proposal correct.
    negative: reference correct and full frozen proposal wrong.
    Other pixels are ignored.
    """
    valid = gt_gpu.ne(IGNORE_LABEL)
    reference_pred = reference_logits.argmax(1)[0]
    proposal_pred = proposal_logits.argmax(1)[0]
    positive = valid & reference_pred.ne(gt_gpu) & proposal_pred.eq(gt_gpu)
    negative = valid & reference_pred.eq(gt_gpu) & proposal_pred.ne(gt_gpu)
    return positive, negative


def _add_acceptance_diagnostics(row):
    if row is None:
        return row
    row["reliability_separation"] = (
        row.get("positive_reliability_mean", 0.0)
        - row.get("negative_reliability_mean", 0.0)
    )
    row["host_reference_reliability_separation"] = row["reliability_separation"]
    row["damage_rescue_ratio"] = (
        row.get("baseline_correct_damaged", 0)
        / max(row.get("rescue_recovered", 0), 1)
    )
    return row


def train_sequence(*args, **kwargs):
    # Training diagnostics remain Host-reference because training labels remain
    # unchanged by design in this controlled C-V15 experiment.
    return _add_acceptance_diagnostics(_base.train_sequence(*args, **kwargs))


def train_epoch(*args, **kwargs):
    return _add_acceptance_diagnostics(_base.train_epoch(*args, **kwargs))


def _accumulate_reference_stats(
    diag,
    prefix,
    positive,
    negative,
    reliability_full,
):
    positive_count = int(positive.sum().item())
    negative_count = int(negative.sum().item())
    diag[f"{prefix}_positive_pixels"] += positive_count
    diag[f"{prefix}_negative_pixels"] += negative_count
    if positive_count:
        diag[f"{prefix}_positive_reliability_sum"] += float(
            reliability_full[positive].sum().item()
        )
    if negative_count:
        diag[f"{prefix}_negative_reliability_sum"] += float(
            reliability_full[negative].sum().item()
        )
    diag[f"{prefix}_hard_positive_accepted"] += int(
        (positive & reliability_full.gt(0.5)).sum().item()
    )
    diag[f"{prefix}_hard_negative_rejected"] += int(
        (negative & reliability_full.le(0.5)).sum().item()
    )


def _finalize_reference_stats(diag, prefix):
    positive_count = max(diag[f"{prefix}_positive_pixels"], 1)
    negative_count = max(diag[f"{prefix}_negative_pixels"], 1)
    positive_mean = diag[f"{prefix}_positive_reliability_sum"] / positive_count
    negative_mean = diag[f"{prefix}_negative_reliability_sum"] / negative_count
    diag[f"{prefix}_positive_reliability_mean"] = positive_mean
    diag[f"{prefix}_negative_reliability_mean"] = negative_mean
    diag[f"{prefix}_reliability_separation"] = positive_mean - negative_mean
    diag[f"{prefix}_hard_positive_accept_recall"] = (
        diag[f"{prefix}_hard_positive_accepted"] / positive_count
    )
    diag[f"{prefix}_hard_negative_reject_rate"] = (
        diag[f"{prefix}_hard_negative_rejected"] / negative_count
    )


@torch.inference_mode()
def evaluate(
    model,
    observer,
    residual,
    correction,
    mask_predictor,
    refiner,
    c_v4_controller,
    dynamics,
    corrector,
    groups,
    raft_metric,
):
    """Full9 evaluation with Host- and C-V4-reference acceptance diagnostics."""
    names = ("host", "c_v3_base", "c_v4_frozen", "c_v15")
    confusion = {
        name: torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
        for name in names
    }
    mtc_sum = {name: 0.0 for name in names}
    mtc_count = {name: 0 for name in names}
    vc_sum = {name: {8: 0.0, 16: 0.0} for name in names}
    vc_count = {name: {8: 0, 16: 0} for name in names}

    diag = {
        "correction_frames": 0,
        "delta_c4_abs": 0.0,
        "delta_c4_rms_ratio": 0.0,
        "bounded_semantic_delta_rms_ratio": 0.0,
        "raw_semantic_delta_c4_abs": 0.0,
        "feature_delta_logits_abs": 0.0,
        "reliability_mean": 0.0,
        "reliability_std": 0.0,
        "rescue_pixels": 0,
        "rescue_recovered": 0,
        "baseline_correct_pixels": 0,
        "baseline_correct_damaged": 0,
    }
    reference_prefixes = (
        "host_reference",
        "cv4_reference",
        "source_cv14_host_reference",
        "source_cv14_cv4_reference",
    )
    for prefix in reference_prefixes:
        diag.update(
            {
                f"{prefix}_positive_pixels": 0,
                f"{prefix}_negative_pixels": 0,
                f"{prefix}_positive_reliability_sum": 0.0,
                f"{prefix}_negative_reliability_sum": 0.0,
                f"{prefix}_hard_positive_accepted": 0,
                f"{prefix}_hard_negative_rejected": 0,
            }
        )

    corrector.eval()
    c_v4_controller.eval()
    model.eval()

    for sequence in FULL9:
        samples = groups[sequence]
        if len(samples) < 2:
            continue
        first = _base.host_feature_observation(model, samples[0])
        second = _base.host_feature_observation(model, samples[1])
        seq_vc = {name: _base.VideoConsistency() for name in names}
        previous_predictions = None

        for index, observation in enumerate((first, second)):
            gt_cpu = _base.semantic_mask_from_panoptic_png(samples[index]["mask_path"])
            host_pred = observation["host_logits"].argmax(1)
            predictions = {name: host_pred for name in names}
            for name, prediction in predictions.items():
                pred_cpu = prediction[0].cpu()
                _base.update_confusion_matrix(confusion[name], pred_cpu, gt_cpu)
                seq_vc[name].update(gt_cpu, pred_cpu)
            if previous_predictions is not None:
                teacher = raft_metric.current_to_previous(
                    observation["image"], first["image"]
                )
                for name, prediction in predictions.items():
                    score = _base._pair_mtc(
                        previous_predictions[name], prediction, teacher
                    )
                    if torch.isfinite(torch.tensor(score)):
                        mtc_sum[name] += score
                        mtc_count[name] += 1
            previous_predictions = {
                name: pred.detach() for name, pred in predictions.items()
            }

        pending_motion, motion_hidden = _base.c_v5._initialize_motion(
            observer,
            residual,
            _base.legacy_observation(first),
            _base.legacy_observation(second),
        )
        previous = second
        previous_low = second["host_low"].detach()
        previous_c1 = second["c1"].detach()
        raw_history = [second["host_logits"].detach()]
        motion_history = []
        transport_hidden = semantic_hidden = mask_hidden = None
        semantic_state_low = torch.zeros_like(previous_low)
        memory_state = controller_hidden = dynamics_state = None

        for frame_index in range(2, len(samples)):
            observation = _base.host_feature_observation(model, samples[frame_index])
            gt_cpu = _base.semantic_mask_from_panoptic_png(
                samples[frame_index]["mask_path"]
            )
            prior_low, _ = _base.c_v5.warp_low_logits(previous_low, pending_motion)
            e1 = _base.c_v5._frozen_e1_step(
                correction,
                mask_predictor,
                observation["c1"],
                observation["host_low"],
                prior_low,
                pending_motion,
                semantic_state_low,
                transport_hidden,
                semantic_hidden,
                mask_hidden,
            )
            transport_hidden = e1["transport_hidden"]
            semantic_hidden = e1["semantic_hidden"]
            mask_hidden = e1["mask_hidden"]
            semantic_state_low = e1["semantic_state_low"]
            memory_row, memory_state, _, c_v3_logits = _base.c_v5._frozen_cv3_step(
                refiner,
                observation["c1"],
                observation["host_low"],
                prior_low,
                e1,
                pending_motion,
                memory_state,
                observation["output_size"],
                observation["host_logits"],
            )
            candidate_rows = _base.c_v5._build_history_candidates(
                raw_history,
                motion_history,
                pending_motion,
                corrector.history_length,
            )
            error_row = _base.build_prediction_error_and_temporal_state(
                c_v4_controller,
                dynamics,
                c_v3_logits,
                candidate_rows,
                pending_motion,
                e1["transportability_low"],
                memory_row["memory_reliability"],
                controller_hidden,
                dynamics_state,
                corrector.history_length,
            )
            controller_hidden = error_row["temporal_hidden"]
            dynamics_state = error_row["dynamics_state"]
            c_v4_logits, _ = _base.cv4_baseline_logits(
                c_v3_logits,
                candidate_rows,
                error_row["temporal_row"],
            )
            final_logits, proposal_logits, feature_delta_logits, correction_row = (
                _base.decode_post_writeback_feature_update(
                    model,
                    corrector,
                    observation,
                    error_row,
                    e1["transportability_low"],
                    memory_row["memory_reliability"],
                )
            )

            predictions = {
                "host": observation["host_logits"].argmax(1),
                "c_v3_base": c_v3_logits.argmax(1),
                "c_v4_frozen": c_v4_logits.argmax(1),
                "c_v15": final_logits.argmax(1),
            }
            for name, prediction in predictions.items():
                pred_cpu = prediction[0].cpu()
                _base.update_confusion_matrix(confusion[name], pred_cpu, gt_cpu)
                seq_vc[name].update(gt_cpu, pred_cpu)

            teacher = raft_metric.current_to_previous(
                observation["image"], previous["image"]
            )
            for name, prediction in predictions.items():
                score = _base._pair_mtc(
                    previous_predictions[name], prediction, teacher
                )
                if torch.isfinite(torch.tensor(score)):
                    mtc_sum[name] += score
                    mtc_count[name] += 1
            previous_predictions = {
                name: pred.detach() for name, pred in predictions.items()
            }

            rescue = _base.formal_rescue_mask(c_v4_logits, candidate_rows, gt_cpu)
            gt_gpu = gt_cpu.to(final_logits.device, non_blocking=True)
            valid = gt_gpu.ne(IGNORE_LABEL)
            c_v4_pred = c_v4_logits.argmax(1)[0]
            final_pred = final_logits.argmax(1)[0]
            baseline_correct = valid & c_v4_pred.eq(gt_gpu)

            host_positive, host_negative = reference_acceptance_targets(
                observation["host_logits"].detach(), proposal_logits, gt_gpu
            )
            cv4_positive, cv4_negative = reference_acceptance_targets(
                c_v4_logits.detach(), proposal_logits, gt_gpu
            )
            reliability_full = proposal_conditioned_acceptance_probability(
                correction_row,
                observation["output_size"],
            )
            source_cv14_reliability_full = F.interpolate(
                correction_row["base_reliability_c4"],
                size=observation["output_size"],
                mode="bilinear",
                align_corners=False,
            )[0, 0].clamp(0.0, 1.0)

            _accumulate_reference_stats(
                diag,
                "host_reference",
                host_positive,
                host_negative,
                reliability_full,
            )
            _accumulate_reference_stats(
                diag,
                "cv4_reference",
                cv4_positive,
                cv4_negative,
                reliability_full,
            )
            _accumulate_reference_stats(
                diag,
                "source_cv14_host_reference",
                host_positive,
                host_negative,
                source_cv14_reliability_full,
            )
            _accumulate_reference_stats(
                diag,
                "source_cv14_cv4_reference",
                cv4_positive,
                cv4_negative,
                source_cv14_reliability_full,
            )

            diag["correction_frames"] += 1
            diag["rescue_pixels"] += int(rescue.sum().item())
            diag["rescue_recovered"] += int(
                (rescue & final_pred.eq(gt_gpu)).sum().item()
            )
            diag["baseline_correct_pixels"] += int(baseline_correct.sum().item())
            diag["baseline_correct_damaged"] += int(
                (baseline_correct & final_pred.ne(gt_gpu)).sum().item()
            )
            current = observation["c4"]
            current_rms = current.square().mean().sqrt().clamp_min(1e-8)
            final_delta = correction_row["delta_c4"]
            proposal_delta = correction_row["bounded_semantic_delta_c4"]
            diag["delta_c4_abs"] += float(final_delta.abs().mean().item())
            diag["delta_c4_rms_ratio"] += float(
                final_delta.square().mean().sqrt().div(current_rms).item()
            )
            diag["bounded_semantic_delta_rms_ratio"] += float(
                proposal_delta.square().mean().sqrt().div(current_rms).item()
            )
            diag["raw_semantic_delta_c4_abs"] += float(
                correction_row["raw_semantic_delta_c4"].abs().mean().item()
            )
            diag["feature_delta_logits_abs"] += float(
                feature_delta_logits.abs().mean().item()
            )
            diag["reliability_mean"] += float(
                correction_row["reliability"].mean().item()
            )
            diag["reliability_std"] += float(
                correction_row["reliability"].std().item()
            )

            observed_motion = _base.c_v5._observe_motion(
                observer,
                previous_low,
                previous_c1,
                observation["host_low"],
                observation["c1"],
            )
            motion_error = (
                F.softmax(observation["host_low"], dim=1)
                - F.softmax(prior_low, dim=1)
            )
            next_motion, _, next_motion_hidden = residual.predict_next(
                observed_motion,
                motion_error,
                motion_hidden,
            )
            raw_history.insert(0, c_v3_logits.detach())
            raw_history = raw_history[: corrector.history_length]
            motion_history.insert(0, pending_motion.detach())
            motion_history = motion_history[: max(corrector.history_length - 1, 0)]
            previous = observation
            previous_low = observation["host_low"].detach()
            previous_c1 = observation["c1"].detach()
            pending_motion = next_motion.detach()
            motion_hidden = next_motion_hidden.detach()
            memory_state = memory_state.detach() if memory_state is not None else None
            controller_hidden = (
                controller_hidden.detach() if controller_hidden is not None else None
            )
            dynamics_state = (
                dynamics_state.detach() if dynamics_state is not None else None
            )

        for name in names:
            stats = seq_vc[name].stats()
            for length in (8, 16):
                vc_sum[name][length] += stats[length]["sum"]
                vc_count[name][length] += stats[length]["count"]

    metrics = {
        name: {
            "mIoU": float(torch.nanmean(_base.compute_iou(confusion[name])).item()),
            "mTC": mtc_sum[name] / mtc_count[name]
            if mtc_count[name]
            else float("nan"),
            "mVC8": vc_sum[name][8] / vc_count[name][8]
            if vc_count[name][8]
            else float("nan"),
            "mVC16": vc_sum[name][16] / vc_count[name][16]
            if vc_count[name][16]
            else float("nan"),
        }
        for name in names
    }

    frames = max(diag["correction_frames"], 1)
    for key in (
        "delta_c4_abs",
        "delta_c4_rms_ratio",
        "bounded_semantic_delta_rms_ratio",
        "raw_semantic_delta_c4_abs",
        "feature_delta_logits_abs",
        "reliability_mean",
        "reliability_std",
    ):
        diag[key] /= frames

    for prefix in reference_prefixes:
        _finalize_reference_stats(diag, prefix)

    # Backward-compatible names remain Host-reference, because that is the
    # actual BCE target used in training.
    diag["acceptance_positive_pixels"] = diag["host_reference_positive_pixels"]
    diag["acceptance_negative_pixels"] = diag["host_reference_negative_pixels"]
    diag["positive_reliability_mean"] = diag[
        "host_reference_positive_reliability_mean"
    ]
    diag["negative_reliability_mean"] = diag[
        "host_reference_negative_reliability_mean"
    ]
    diag["hard_positive_accept_recall"] = diag[
        "host_reference_hard_positive_accept_recall"
    ]
    diag["hard_negative_reject_rate"] = diag[
        "host_reference_hard_negative_reject_rate"
    ]
    diag["reliability_separation"] = diag[
        "host_reference_reliability_separation"
    ]

    diag["host_reference_separation_gain_vs_source_cv14"] = (
        diag["host_reference_reliability_separation"]
        - diag["source_cv14_host_reference_reliability_separation"]
    )
    diag["cv4_reference_separation_gain_vs_source_cv14"] = (
        diag["cv4_reference_reliability_separation"]
        - diag["source_cv14_cv4_reference_reliability_separation"]
    )
    diag["rescue_recovery_rate"] = (
        diag["rescue_recovered"] / max(diag["rescue_pixels"], 1)
    )
    diag["baseline_correct_damage_rate"] = (
        diag["baseline_correct_damaged"] / max(diag["baseline_correct_pixels"], 1)
    )
    diag["damage_rescue_ratio"] = (
        diag["baseline_correct_damaged"] / max(diag["rescue_recovered"], 1)
    )
    diag.update(
        {
            "acceptance_semantics": "proposal-conditioned soft acceptance coefficient",
            "training_acceptance_reference": "Host",
            "validation_secondary_acceptance_reference": "frozen C-V4",
            "cv4_reference_labels_backpropagated": False,
            "acceptance_inputs": (
                "frozen temporal latent + frozen semantic latent + normalized "
                "bounded Delta-c4 descriptor + current c4 descriptor"
            ),
            "acceptance_supervision_mapping": (
                "deployed reliability_c4 -> output resize -> Host-relative "
                "beneficial/harmful BCE"
            ),
            "damage_rescue_reference": "frozen C-V4",
            "cv14_proposal_generator_frozen": True,
            "cv14_reliability_prior_frozen": True,
        }
    )
    return metrics, diag


# Fixed-protocol re-exports used by contracts and entrypoint.
decode_post_writeback_feature_update = _base.decode_post_writeback_feature_update
protection_kl_loss = _base.protection_kl_loss
proposal_acceptance_targets = _base.proposal_acceptance_targets
host_feature_observation = _base.host_feature_observation
legacy_observation = _base.legacy_observation
build_prediction_error_and_temporal_state = _base.build_prediction_error_and_temporal_state
cv4_baseline_logits = _base.cv4_baseline_logits
formal_rescue_mask = _base.formal_rescue_mask
