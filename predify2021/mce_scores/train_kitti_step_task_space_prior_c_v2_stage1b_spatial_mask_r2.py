"""C-V2 Stage 1B current mainline: Spatial Mask on bounded r=2 motion.

Role-only inference is retired. Training-only role supervision remains because it
teaches the transport and semantic-innovation correction heads their respective
regions. The only deployed/evaluated correction is:

    L_out = L_host + T * DeltaL_transport + (1-T) * DeltaL_semantic

T is recurrent transportability, not Host/Prior fusion. Host, Motion Observer,
and bounded r=2 Task-Alignment Residual are frozen.
"""

import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from predify2021.datasets.kitti_step import (
    KITTISTEPSegmentationDataset,
    semantic_mask_from_panoptic_png,
)
from predify2021.mce_scores.evaluate_kitti_step_clean_fast_iss_vss import VideoConsistency
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import sequence_groups
from predify2021.mce_scores.evaluate_kitti_step_static_baseline import (
    compute_iou,
    update_confusion_matrix,
)
from predify2021.mce_scores.train_kitti_step_frozen_error_utility_probe import (
    FAST_B_CHECKPOINT_DEFAULT,
    load_fast_b_model,
)
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_only import (
    FULL9,
    NUM_CLASSES,
    _pair_mtc,
    _upsample_prior,
)
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v2_stage1b_observer import (
    _host_observation,
)
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v2_stage1b_residual import (
    OBSERVER_CHECKPOINT_DEFAULT,
    _load_frozen_observer,
    _observe_motion,
)
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v2_stage1b_role_vs_mask import (
    _load_frozen_residual,
    _train_epoch,
)
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_v2_stage1b_role_vs_mask_r2 import (
    RESIDUAL_CHECKPOINT_DEFAULT,
    _validate_bounded_motion_checkpoint,
)
from predify2021.mce_scores.train_kitti_step_temporal_joint import FrozenRAFT
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_motion_transport import (
    warp_low_logits,
)
from predify2021.model_factory.deeplabv3plus_resnet50.task_space_role_decoupling import (
    RecurrentTransportabilityMask,
    RoleSeparatedTaskCorrection,
)

SEED = 0
EPOCHS = 3
TBPTT_STEPS = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
OUTPUT_DEFAULT = (
    "/home/lin/predify/experiments/"
    "kitti_step_task_space_prior_c_v2_stage1b_spatial_mask_r2"
)
RESULT_DEFAULT = "results/kitti_step_task_space_prior_c_v2_stage1b_spatial_mask_r2"
CANDIDATES = ("host", "spatial_mask")


@torch.inference_mode()
def _evaluate(model, observer, residual, correction, mask_predictor, groups, raft):
    correction.eval()
    mask_predictor.eval()
    confusion = {
        name: torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
        for name in CANDIDATES
    }
    mtc_sum = {name: 0.0 for name in CANDIDATES}
    mtc_count = {name: 0 for name in CANDIDATES}
    vc_sum = {name: {8: 0.0, 16: 0.0} for name in CANDIDATES}
    vc_count = {name: {8: 0, 16: 0} for name in CANDIDATES}
    diagnostics = {
        "frames_with_temporal_prior": 0,
        "mask_mean": 0.0,
        "delta_transport_abs": 0.0,
        "delta_semantic_abs": 0.0,
    }

    for sequence in FULL9:
        previous = None
        pending_motion = motion_hidden = None
        transport_hidden = semantic_hidden = mask_hidden = None
        previous_predictions = {}
        seq_vc = {name: VideoConsistency() for name in CANDIDATES}

        for sample in groups[sequence]:
            image, host_logits, host_low, current_c1, output_size = _host_observation(
                model, sample
            )
            host_pred = host_logits.argmax(1)
            gt_cpu = semantic_mask_from_panoptic_png(sample["mask_path"])
            previous_image_for_mtc = previous[0] if previous is not None else None

            if previous is None:
                spatial_pred = host_pred
                previous = (image, host_low.detach(), current_c1.detach())
            elif pending_motion is None:
                previous_image, previous_low, previous_c1 = previous
                observed = _observe_motion(
                    observer, previous_low, previous_c1, host_low, current_c1
                )
                error = F.softmax(host_low, dim=1) - F.softmax(previous_low, dim=1)
                pending_motion, _, motion_hidden = residual.predict_next(observed, error, None)
                spatial_pred = host_pred
                previous = (image, host_low.detach(), current_c1.detach())
            else:
                previous_image, previous_low, previous_c1 = previous
                prior_low, _ = warp_low_logits(previous_low, pending_motion)
                row = correction(
                    current_c1,
                    host_low,
                    prior_low,
                    pending_motion,
                    transport_hidden,
                    semantic_hidden,
                )
                transport_hidden = row["transport_hidden"]
                semantic_hidden = row["semantic_hidden"]
                delta_transport = _upsample_prior(row["delta_transport"], output_size)
                delta_semantic = _upsample_prior(row["delta_semantic"], output_size)

                mask_logits_low, mask_hidden = mask_predictor(
                    current_c1,
                    host_low,
                    prior_low,
                    pending_motion,
                    mask_hidden,
                )
                transportability = torch.sigmoid(
                    _upsample_prior(mask_logits_low, output_size)
                )
                spatial_pred = (
                    host_logits
                    + transportability * delta_transport
                    + (1.0 - transportability) * delta_semantic
                ).argmax(1)

                diagnostics["frames_with_temporal_prior"] += 1
                diagnostics["mask_mean"] += float(transportability.mean().item())
                diagnostics["delta_transport_abs"] += float(delta_transport.abs().mean().item())
                diagnostics["delta_semantic_abs"] += float(delta_semantic.abs().mean().item())

                observed = _observe_motion(
                    observer, previous_low, previous_c1, host_low, current_c1
                )
                error = F.softmax(host_low, dim=1) - F.softmax(prior_low, dim=1)
                pending_motion, _, motion_hidden = residual.predict_next(
                    observed, error, motion_hidden
                )
                previous = (image, host_low.detach(), current_c1.detach())

            predictions = {"host": host_pred, "spatial_mask": spatial_pred}
            for name, prediction in predictions.items():
                pc = prediction.squeeze(0).cpu()
                update_confusion_matrix(confusion[name], pc, gt_cpu)
                seq_vc[name].update(gt_cpu, pc)

            if previous_image_for_mtc is not None:
                teacher_full = raft.current_to_previous(image, previous_image_for_mtc)
                for name, prediction in predictions.items():
                    score = _pair_mtc(previous_predictions[name], prediction, teacher_full)
                    if math.isfinite(score):
                        mtc_sum[name] += score
                        mtc_count[name] += 1

            previous_predictions = {
                name: prediction.detach() for name, prediction in predictions.items()
            }

        for name in CANDIDATES:
            stats = seq_vc[name].stats()
            for length in (8, 16):
                vc_sum[name][length] += stats[length]["sum"]
                vc_count[name][length] += stats[length]["count"]

    metrics = {
        name: {
            "mIoU": float(torch.nanmean(compute_iou(confusion[name])).item()),
            "mTC": mtc_sum[name] / mtc_count[name] if mtc_count[name] else float("nan"),
            "mVC8": vc_sum[name][8] / vc_count[name][8] if vc_count[name][8] else float("nan"),
            "mVC16": vc_sum[name][16] / vc_count[name][16] if vc_count[name][16] else float("nan"),
        }
        for name in CANDIDATES
    }
    frames = max(diagnostics["frames_with_temporal_prior"], 1)
    for key in ("mask_mean", "delta_transport_abs", "delta_semantic_abs"):
        diagnostics[key] /= frames
    return metrics, diagnostics


def _selection_key(metrics):
    """Preserve Host semantic quality first, then maximize temporal consistency.

    No new metric or threshold is introduced. If the Spatial Mask is at least as
    accurate as Host, prefer the epoch with higher mTC; otherwise fall back to the
    highest mIoU epoch.
    """
    host = metrics["host"]
    spatial = metrics["spatial_mask"]
    semantic_non_degraded = spatial["mIoU"] >= host["mIoU"]
    if semantic_non_degraded:
        return (1, spatial["mTC"], spatial["mIoU"])
    return (0, spatial["mIoU"], spatial["mTC"])


def _checkpoint_payload(
    epoch,
    args,
    correction,
    mask_predictor,
    optimizer,
    train_stats,
    metrics,
    diagnostics,
    residual_payload,
):
    return {
        "experiment": "c_v2_stage1b_spatial_mask_r2",
        "epoch": epoch,
        "architecture": {
            "inference_route": "spatial_mask_only",
            "output": "Host + T*DeltaL_transport + (1-T)*DeltaL_semantic",
            "T_definition": "recurrent transportability; not Host/Prior fusion",
            "role_only_inference_retired": True,
            "training_role_supervision_retained": True,
            "shared_frozen_motion": True,
            "mask_semantic_ce_gradient": False,
            "motion_gradient_from_correction": False,
        },
        "training_contract": {
            "role_target": "GT + frozen RAFT current-to-previous correspondence",
            "transport_head": "CE on transportable; zero residual off-role",
            "semantic_head": "CE on non-transportable; zero residual off-role",
            "mask_head": "class-balanced direct transportability BCE only",
            "role_supervision_is_training_signal_not_inference_candidate": True,
        },
        "frozen": [
            "Host",
            "Adapter",
            "Writeback",
            "Decoder",
            "Motion Observer",
            "bounded r=2 Task-Alignment Residual",
        ],
        "observer_checkpoint": args.observer_checkpoint,
        "residual_checkpoint": args.residual_checkpoint,
        "residual_checkpoint_epoch": residual_payload.get("epoch"),
        "train": train_stats,
        "metrics": metrics,
        "diagnostics": diagnostics,
        "correction_state_dict": correction.state_dict(),
        "mask_state_dict": mask_predictor.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/home/lin/predify/kitti_step")
    parser.add_argument("--fast-b-checkpoint", default=FAST_B_CHECKPOINT_DEFAULT)
    parser.add_argument("--observer-checkpoint", default=OBSERVER_CHECKPOINT_DEFAULT)
    parser.add_argument("--residual-checkpoint", default=RESIDUAL_CHECKPOINT_DEFAULT)
    parser.add_argument("--output", default=OUTPUT_DEFAULT)
    parser.add_argument("--result-output", default=RESULT_DEFAULT)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--tbptt-steps", type=int, default=TBPTT_STEPS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.epochs <= 0 or args.tbptt_steps <= 0:
        raise ValueError("epochs and tbptt-steps must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    _validate_bounded_motion_checkpoint(args.residual_checkpoint)
    model = load_fast_b_model(args.fast_b_checkpoint).cuda().eval()
    model.requires_grad_(False)
    observer, _ = _load_frozen_observer(args.observer_checkpoint)
    residual, residual_payload = _load_frozen_residual(args.residual_checkpoint, observer)
    correction = RoleSeparatedTaskCorrection(
        c1_channels=256,
        num_classes=NUM_CLASSES,
        motion_scale=observer.max_displacement_low,
    ).cuda()
    mask_predictor = RecurrentTransportabilityMask(
        c1_channels=256,
        num_classes=NUM_CLASSES,
        motion_scale=observer.max_displacement_low,
    ).cuda()
    optimizer = torch.optim.AdamW(
        list(correction.parameters()) + list(mask_predictor.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    raft = FrozenRAFT()

    train = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "train")
    val = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "val")
    train_groups = sequence_groups(train)
    all_val_groups = sequence_groups(val)
    missing = [sequence for sequence in FULL9 if sequence not in all_val_groups]
    if missing:
        raise RuntimeError(f"Missing Full9 validation sequences: {missing}")
    val_groups = {sequence: all_val_groups[sequence] for sequence in FULL9}

    output = Path(args.output)
    result_output = Path(args.result_output)
    output.mkdir(parents=True, exist_ok=True)
    result_output.mkdir(parents=True, exist_ok=True)

    history = []
    best = None
    for epoch in range(1, args.epochs + 1):
        train_stats = _train_epoch(
            model,
            observer,
            residual,
            correction,
            mask_predictor,
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
            val_groups,
            raft,
        )
        delta_vs_host = {
            metric: metrics["spatial_mask"][metric] - metrics["host"][metric]
            for metric in ("mIoU", "mTC", "mVC8", "mVC16")
        }
        row = {
            "epoch": epoch,
            "train": train_stats,
            "metrics": metrics,
            "diagnostics": diagnostics,
            "delta_vs_host": delta_vs_host,
        }
        history.append(row)
        payload = _checkpoint_payload(
            epoch,
            args,
            correction,
            mask_predictor,
            optimizer,
            train_stats,
            metrics,
            diagnostics,
            residual_payload,
        )
        torch.save(payload, output / f"epoch_{epoch:03d}.pt")
        (result_output / f"epoch_{epoch:03d}.json").write_text(
            json.dumps(row, indent=2, sort_keys=True) + "\n"
        )

        key = _selection_key(metrics)
        if best is None or key > best["selection_key"]:
            best = {
                "epoch": epoch,
                "selection_key": key,
                "metrics": metrics["spatial_mask"],
                "delta_vs_host": delta_vs_host,
            }
            torch.save(payload, output / "best.pt")
        print(json.dumps(row, sort_keys=True), flush=True)

    summary = {
        "experiment": "C-V2 Stage 1B Spatial Mask r2",
        "purpose": "Current mainline; Role-only inference retired.",
        "training": {
            "epochs": args.epochs,
            "tbptt_steps": args.tbptt_steps,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "role_supervision_retained": True,
        },
        "frozen_motion": {
            "observer_checkpoint": args.observer_checkpoint,
            "residual_checkpoint": args.residual_checkpoint,
        },
        "history": history,
        "best": best,
        "selection_rule": (
            "If Spatial Mask mIoU >= Host, select highest mTC; otherwise select highest mIoU. "
            "Uses existing main metrics only."
        ),
        "next_focus": (
            "Stabilize Semantic Correction over time; do not reopen Role-only inference."
        ),
    }
    (result_output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "best": best,
                "checkpoint": str(output / "best.pt"),
                "result": str(result_output / "summary.json"),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
