"""C-V5 corrected main.

中文：C-V5 非自回归多帧语义候选记忆正式入口。

This entrypoint supersedes the first C-V5 draft for execution.
修正三项会影响结构结论的问题：
1. History Memory（历史记忆）不再以 Host logits 启动；首个真正 C-V3 输出只用于播种历史，
   从下一帧起才启用多帧 Selector（选择器）与 Oracle（上限诊断）。
2. Multi-frame Oracle（多帧 Oracle）在同一次 Full9 前向中同时计算 K=1 与 K=4，
   结构判据使用同协议 K4-K1，而不是旧实验硬编码值。
3. Best Epoch（最佳轮次）的语义硬约束提升到当前 Pareto 点 C-V4 E2 mIoU=66.3774%。

其余 C-V5 结构保持不变：
- History 仅来自 detached frozen C-V3 raw logits（切断梯度的冻结 C-V3 原始得分）；
- Controller output（控制器输出）不回灌；
- 深历史不进入 Prediction Error（预测误差）/ Dynamics Error（动力学误差）基准；
- 每个历史 logits 只做一次最终 warp（运动搬运）。
"""

import math

import torch
from torch.nn import functional as F

import predify2021.mce_scores.train_kitti_step_task_space_prior_c_v5_multiframe_candidate_memory as base


CV4_E2_MIOU_REFERENCE = 0.663774


def _train_sequence_corrected(
    model,
    observer,
    residual,
    correction,
    mask_predictor,
    refiner,
    selector,
    dynamics,
    raft,
    samples,
    optimizer,
    tbptt_steps,
):
    if len(samples) < 4:
        return None

    frame0 = base._host_observation(model, samples[0])
    frame1 = base._host_observation(model, samples[1])
    pending_motion, motion_hidden = base._initialize_motion(
        observer, residual, frame0, frame1
    )
    previous_image, _, previous_low, previous_c1, _ = frame1

    # Corrected warmup: Host is never inserted into C-V5 semantic history.
    raw_history = []
    motion_history = []
    previous_cv3_logits = None

    transport_hidden = semantic_hidden = mask_hidden = None
    semantic_state_low = torch.zeros_like(previous_low)
    memory_state = None
    selector_hidden = None
    dynamics_state = None

    loss_sums = []
    loss_counts = []
    frames_in_window = 0
    totals = {
        "frames": 0,
        "supervised_frames": 0,
        "optimizer_steps": 0,
        "supervised_pixels": 0,
        "selector_ce_per_pixel": 0.0,
        "prediction_error_abs": 0.0,
        "dynamics_error_abs": 0.0,
        "selected_candidate_counts": [0] * (selector.history_length + 1),
    }
    target_totals = base._new_target_totals(selector.history_length)

    for frame_index in range(2, len(samples)):
        current_image, host_logits, host_low, current_c1, output_size = base._host_observation(
            model, samples[frame_index]
        )
        current_gt = base.semantic_mask_from_panoptic_png(samples[frame_index]["mask_path"])

        with torch.no_grad():
            prior_low, _ = base.warp_low_logits(previous_low, pending_motion)
            e1 = base._frozen_e1_step(
                correction,
                mask_predictor,
                current_c1,
                host_low,
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

            memory_row, memory_state, _, c_v3_logits = base._frozen_cv3_step(
                refiner,
                current_c1,
                host_low,
                prior_low,
                e1,
                pending_motion,
                memory_state,
                output_size,
                host_logits,
            )

            observed_motion = base._observe_motion(
                observer, previous_low, previous_c1, host_low, current_c1
            )
            motion_error = F.softmax(host_low, dim=1) - F.softmax(prior_low, dim=1)
            next_motion, _, next_motion_hidden = residual.predict_next(
                observed_motion, motion_error, motion_hidden
            )

        # First true C-V3 output only seeds history. No fake Host history is used.
        if not raw_history:
            raw_history = [c_v3_logits.detach()]
            previous_cv3_logits = c_v3_logits.detach()
            previous_image = current_image
            previous_low = host_low.detach()
            previous_c1 = current_c1.detach()
            pending_motion = next_motion.detach()
            motion_hidden = next_motion_hidden.detach()
            totals["frames"] += 1
            continue

        with torch.no_grad():
            candidate_rows = base._build_history_candidates(
                raw_history,
                motion_history,
                pending_motion,
                selector.history_length,
            )

        evidence = base._selector_evidence(
            selector,
            dynamics,
            c_v3_logits,
            candidate_rows,
            pending_motion,
            e1["transportability_low"],
            memory_row["memory_reliability"],
            selector_hidden,
            dynamics_state,
        )
        selector_hidden = evidence["row"]["hidden"]
        dynamics_state = evidence["dynamics_state"]

        with torch.no_grad():
            teacher_full = raft.current_to_previous(current_image, previous_image)
            _, temporal_target, supervised, _, target_diag = base._build_multiframe_target(
                c_v3_logits,
                candidate_rows,
                previous_cv3_logits,
                current_gt,
                teacher_full,
            )
            base._add_target_totals(target_totals, target_diag)

        if bool(supervised.any()):
            logits = evidence["selector_logits_full"][0].permute(1, 2, 0)[supervised]
            target = temporal_target[supervised]
            pixel_loss_sum = F.cross_entropy(logits, target, reduction="sum")
            count = int(supervised.sum().item())
            loss_sums.append(pixel_loss_sum)
            loss_counts.append(count)
            totals["supervised_frames"] += 1
            totals["supervised_pixels"] += count
            totals["selector_ce_per_pixel"] += float(pixel_loss_sum.detach().item())

        with torch.no_grad():
            hard_selection = evidence["selector_logits_full"].argmax(1)[0]
            for index in range(selector.history_length + 1):
                totals["selected_candidate_counts"][index] += int(
                    (hard_selection == index).sum().item()
                )
            totals["prediction_error_abs"] += float(
                evidence["prediction_error"].abs().mean().item()
            )
            totals["dynamics_error_abs"] += float(dynamics_state.abs().mean().item())

        totals["frames"] += 1
        frames_in_window += 1
        boundary = frames_in_window >= tbptt_steps or frame_index == len(samples) - 1
        if boundary:
            if loss_sums:
                total_count = max(sum(loss_counts), 1)
                window_loss = torch.stack(loss_sums).sum() / float(total_count)
                optimizer.zero_grad(set_to_none=True)
                window_loss.backward()
                optimizer.step()
                totals["optimizer_steps"] += 1
            loss_sums = []
            loss_counts = []
            frames_in_window = 0
            if selector_hidden is not None:
                selector_hidden = selector_hidden.detach()
            if memory_state is not None:
                memory_state = memory_state.detach()
            if dynamics_state is not None:
                dynamics_state = dynamics_state.detach()

        raw_history.insert(0, c_v3_logits.detach())
        raw_history = raw_history[: selector.history_length]
        motion_history.insert(0, pending_motion.detach())
        motion_history = motion_history[: max(selector.history_length - 1, 0)]

        previous_image = current_image
        previous_low = host_low.detach()
        previous_c1 = current_c1.detach()
        previous_cv3_logits = c_v3_logits.detach()
        pending_motion = next_motion.detach()
        motion_hidden = next_motion_hidden.detach()

    frames = max(totals["frames"], 1)
    totals["selector_ce_per_pixel"] /= max(totals["supervised_pixels"], 1)
    totals["prediction_error_abs"] /= frames
    totals["dynamics_error_abs"] /= frames
    totals["targets"] = base._target_rates(target_totals)
    return totals


@torch.inference_mode()
def _evaluate_corrected(
    model,
    observer,
    residual,
    correction,
    mask_predictor,
    refiner,
    selector,
    dynamics,
    groups,
    raft,
):
    selector.eval()
    refiner.eval()
    correction.eval()
    mask_predictor.eval()

    names = base.CANDIDATES
    confusion = {
        name: torch.zeros((base.NUM_CLASSES, base.NUM_CLASSES), dtype=torch.int64)
        for name in names
    }
    k1_confusion = torch.zeros((base.NUM_CLASSES, base.NUM_CLASSES), dtype=torch.int64)
    mtc_sum = {name: 0.0 for name in names}
    mtc_count = {name: 0 for name in names}
    k1_mtc_sum = 0.0
    k1_mtc_count = 0
    vc_sum = {name: {8: 0.0, 16: 0.0} for name in names}
    vc_count = {name: {8: 0, 16: 0} for name in names}
    k1_vc_sum = {8: 0.0, 16: 0.0}
    k1_vc_count = {8: 0, 16: 0}
    target_totals = base._new_target_totals(selector.history_length)
    k1_target_totals = base._new_target_totals(1)
    selected_counts = [0] * (selector.history_length + 1)

    for sequence in base.FULL9:
        previous = None
        pending_motion = motion_hidden = None
        transport_hidden = semantic_hidden = mask_hidden = None
        semantic_state_low = None
        memory_state = None
        selector_hidden = None
        dynamics_state = None
        raw_history = []
        motion_history = []
        previous_cv3_logits = None
        previous_predictions = {}
        previous_k1_prediction = None
        seq_vc = {name: base.VideoConsistency() for name in names}
        seq_k1_vc = base.VideoConsistency()

        for sample in groups[sequence]:
            image, host_logits, host_low, current_c1, output_size = base._host_observation(
                model, sample
            )
            host_pred = host_logits.argmax(1)
            gt_cpu = base.semantic_mask_from_panoptic_png(sample["mask_path"])
            previous_image_for_mtc = previous[0] if previous is not None else None
            teacher_full = (
                raft.current_to_previous(image, previous_image_for_mtc)
                if previous_image_for_mtc is not None
                else None
            )

            if previous is None:
                e1_pred = c_v3_pred = c_v5_pred = host_pred
                semantic_oracle_pred = temporal_oracle_pred = k1_oracle_pred = host_pred
                semantic_state_low = torch.zeros_like(host_low)
                previous = (image, host_low.detach(), current_c1.detach())

            elif pending_motion is None:
                previous_image, previous_low, previous_c1 = previous
                observed = base._observe_motion(
                    observer, previous_low, previous_c1, host_low, current_c1
                )
                error = F.softmax(host_low, dim=1) - F.softmax(previous_low, dim=1)
                pending_motion, _, motion_hidden = residual.predict_next(observed, error, None)
                e1_pred = c_v3_pred = c_v5_pred = host_pred
                semantic_oracle_pred = temporal_oracle_pred = k1_oracle_pred = host_pred
                previous = (image, host_low.detach(), current_c1.detach())

            else:
                previous_image, previous_low, previous_c1 = previous
                prior_low, _ = base.warp_low_logits(previous_low, pending_motion)
                e1 = base._frozen_e1_step(
                    correction,
                    mask_predictor,
                    current_c1,
                    host_low,
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

                memory_row, memory_state, e1_logits, c_v3_logits = base._frozen_cv3_step(
                    refiner,
                    current_c1,
                    host_low,
                    prior_low,
                    e1,
                    pending_motion,
                    memory_state,
                    output_size,
                    host_logits,
                )

                observed = base._observe_motion(
                    observer, previous_low, previous_c1, host_low, current_c1
                )
                motion_error = F.softmax(host_low, dim=1) - F.softmax(prior_low, dim=1)
                next_motion, _, next_motion_hidden = residual.predict_next(
                    observed, motion_error, motion_hidden
                )

                e1_pred = e1_logits.argmax(1)
                c_v3_pred = c_v3_logits.argmax(1)

                # First actual C-V3 output seeds memory only.
                if not raw_history:
                    c_v5_pred = c_v3_pred
                    semantic_oracle_pred = temporal_oracle_pred = k1_oracle_pred = c_v3_pred
                    raw_history = [c_v3_logits.detach()]
                    previous_cv3_logits = c_v3_logits.detach()
                else:
                    candidate_rows = base._build_history_candidates(
                        raw_history,
                        motion_history,
                        pending_motion,
                        selector.history_length,
                    )
                    evidence = base._selector_evidence(
                        selector,
                        dynamics,
                        c_v3_logits,
                        candidate_rows,
                        pending_motion,
                        e1["transportability_low"],
                        memory_row["memory_reliability"],
                        selector_hidden,
                        dynamics_state,
                    )
                    selector_hidden = evidence["row"]["hidden"]
                    dynamics_state = evidence["dynamics_state"]

                    semantic_target, temporal_target, _, _, target_diag = base._build_multiframe_target(
                        c_v3_logits,
                        candidate_rows,
                        previous_cv3_logits,
                        gt_cpu,
                        teacher_full,
                    )
                    base._add_target_totals(target_totals, target_diag)

                    k1_rows = candidate_rows[:1]
                    _, k1_temporal_target, _, _, k1_diag = base._build_multiframe_target(
                        c_v3_logits,
                        k1_rows,
                        previous_cv3_logits,
                        gt_cpu,
                        teacher_full,
                    )
                    base._add_target_totals(k1_target_totals, k1_diag)

                    current_pred, history_preds = base._candidate_predictions(
                        c_v3_logits, candidate_rows
                    )
                    hard_selection = evidence["selector_logits_full"].argmax(1)[0]
                    for index in range(selector.history_length + 1):
                        selected_counts[index] += int((hard_selection == index).sum().item())

                    c_v5_pred = base._apply_selection(
                        current_pred, history_preds, hard_selection
                    )
                    semantic_oracle_pred = base._apply_selection(
                        current_pred, history_preds, semantic_target
                    )
                    temporal_oracle_pred = base._apply_selection(
                        current_pred, history_preds, temporal_target
                    )
                    k1_oracle_pred = base._apply_selection(
                        current_pred,
                        history_preds[:1],
                        k1_temporal_target,
                    )

                    raw_history.insert(0, c_v3_logits.detach())
                    raw_history = raw_history[: selector.history_length]
                    motion_history.insert(0, pending_motion.detach())
                    motion_history = motion_history[: max(selector.history_length - 1, 0)]
                    previous_cv3_logits = c_v3_logits.detach()

                previous = (image, host_low.detach(), current_c1.detach())
                pending_motion = next_motion.detach()
                motion_hidden = next_motion_hidden.detach()

            predictions = {
                "host": host_pred,
                "e1_base": e1_pred,
                "c_v3_base": c_v3_pred,
                "c_v5": c_v5_pred,
                "multiframe_semantic_oracle": semantic_oracle_pred,
                "multiframe_semantic_temporal_oracle": temporal_oracle_pred,
            }
            for name, prediction in predictions.items():
                pred_cpu = prediction.squeeze(0).cpu()
                base.update_confusion_matrix(confusion[name], pred_cpu, gt_cpu)
                seq_vc[name].update(gt_cpu, pred_cpu)

            k1_cpu = k1_oracle_pred.squeeze(0).cpu()
            base.update_confusion_matrix(k1_confusion, k1_cpu, gt_cpu)
            seq_k1_vc.update(gt_cpu, k1_cpu)

            if teacher_full is not None:
                for name, prediction in predictions.items():
                    if name in previous_predictions:
                        score = base._pair_mtc(previous_predictions[name], prediction, teacher_full)
                        if math.isfinite(score):
                            mtc_sum[name] += score
                            mtc_count[name] += 1
                if previous_k1_prediction is not None:
                    score = base._pair_mtc(previous_k1_prediction, k1_oracle_pred, teacher_full)
                    if math.isfinite(score):
                        k1_mtc_sum += score
                        k1_mtc_count += 1

            previous_predictions = {
                name: prediction.detach() for name, prediction in predictions.items()
            }
            previous_k1_prediction = k1_oracle_pred.detach()

        for name in names:
            stats = seq_vc[name].stats()
            for length in (8, 16):
                vc_sum[name][length] += stats[length]["sum"]
                vc_count[name][length] += stats[length]["count"]
        k1_stats = seq_k1_vc.stats()
        for length in (8, 16):
            k1_vc_sum[length] += k1_stats[length]["sum"]
            k1_vc_count[length] += k1_stats[length]["count"]

    metrics = {
        name: {
            "mIoU": float(torch.nanmean(base.compute_iou(confusion[name])).item()),
            "mTC": mtc_sum[name] / mtc_count[name] if mtc_count[name] else float("nan"),
            "mVC8": vc_sum[name][8] / vc_count[name][8] if vc_count[name][8] else float("nan"),
            "mVC16": vc_sum[name][16] / vc_count[name][16] if vc_count[name][16] else float("nan"),
        }
        for name in names
    }
    k1_metrics = {
        "mIoU": float(torch.nanmean(base.compute_iou(k1_confusion)).item()),
        "mTC": k1_mtc_sum / k1_mtc_count if k1_mtc_count else float("nan"),
        "mVC8": k1_vc_sum[8] / k1_vc_count[8] if k1_vc_count[8] else float("nan"),
        "mVC16": k1_vc_sum[16] / k1_vc_count[16] if k1_vc_count[16] else float("nan"),
    }

    # Base main reads this value immediately after _evaluate returns.
    # Setting it here converts the old comparison field into a same-run K=1 reference.
    base.SINGLE_FRAME_ORACLE_MTC = k1_metrics["mTC"]

    k4_metrics = metrics["multiframe_semantic_temporal_oracle"]
    diagnostics = {
        "target_distribution": base._target_rates(target_totals),
        "single_frame_target_distribution": base._target_rates(k1_target_totals),
        "selected_candidate_counts": selected_counts,
        "history_length": selector.history_length,
        "semantic_history_source": "detached raw frozen C-V3 logits only; Host never enters history",
        "history_logits_resampling": "one final warp per candidate",
        "controller_output_feedback": False,
        "prediction_error_reference": "t-1 frozen C-V3 only",
        "dynamics_error_reference": "t-1 prediction error only",
        "dynamics_motion_compensated": True,
        "dynamics_transportability_masked": True,
        "raft_inference": False,
        "same_run_oracle_comparison": {
            "k1": k1_metrics,
            "k4": k4_metrics,
            "delta_k4_minus_k1": base._delta_metrics(k4_metrics, k1_metrics),
            "decision_question": "Do t-2..t-4 add semantic/temporal information beyond t-1 under the same Full9 run?",
        },
    }
    return metrics, diagnostics


def _selection_key_corrected(metrics):
    candidate = metrics["c_v5"]
    preserved = candidate["mIoU"] >= CV4_E2_MIOU_REFERENCE
    if preserved:
        return (1, candidate["mTC"], candidate["mIoU"])
    return (0, candidate["mIoU"], candidate["mTC"])


def main(argv=None):
    # Patch only the three corrected behaviors; all other C-V5 code remains shared.
    base._train_sequence = _train_sequence_corrected
    base._evaluate = _evaluate_corrected
    base._selection_key = _selection_key_corrected
    base.MIOU_REFERENCE = CV4_E2_MIOU_REFERENCE
    return base.main(argv)


if __name__ == "__main__":
    main()
