import csv
import json
import os
import random
from pathlib import Path

import torch
from torch.nn import functional as F
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

from predify2021.datasets.kitti_step import (
    KITTISTEPSegmentationDataset,
    semantic_mask_from_panoptic_png,
)
from predify2021.mce_scores.evaluate_kitti_step_adaptive_dynamic_correction import (
    detach_state,
)
from predify2021.mce_scores.evaluate_kitti_step_closed_loop_dynamic_correction import (
    ADAPTER_CHECKPOINT_DEFAULT,
    PREDICTOR_CHECKPOINT_DEFAULT,
    STATIC_CHECKPOINT_DEFAULT,
    add_frame_noise,
    phase_statistics,
    posterior_state,
    state_metrics,
)
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import (
    encode_image,
    load_image,
    predict_current,
    sequence_groups,
    update_dynamic_error,
)
from predify2021.mce_scores.evaluate_kitti_step_error_correction import (
    corrected_host_feature,
)
from predify2021.mce_scores.evaluate_kitti_step_static_baseline import (
    compute_iou,
    update_confusion_matrix,
)
from predify2021.mce_scores.train_kitti_step_state_predictor import (
    load_static_kitti_checkpoint,
)
from predify2021.model_factory.deeplabv3plus_resnet50 import (
    MultiLayerPredictor,
    UnifiedFeatures,
    build_deeplabv3plus_resnet50_host,
)
from predify2021.model_factory.deeplabv3plus_resnet50.corrections import (
    ErrorGainCorrection,
)
from predify2021.model_factory.deeplabv3plus_resnet50.flow_error_alignment import (
    backward_flow_to_state_warp,
)


CLOSED_LOOP_CORRECTION_CHECKPOINT_DEFAULT = (
    "/home/lin/predify/experiments/kitti_step_closed_loop_dynamic_correction_9820f4e/"
    "best_closed_loop_correction.pt"
)
GAMMA = 0.207
HISTORY_WEIGHT = 0.793


class FrozenRAFT:
    def __init__(self):
        self.weights = Raft_Large_Weights.DEFAULT
        self.model = raft_large(weights=self.weights, progress=True).cuda().eval()
        self.model.requires_grad_(False)

    @torch.no_grad()
    def backward_flow(self, current_clean, previous_clean):
        height, width = current_clean.shape[-2:]
        pad_height = (-height) % 8
        pad_width = (-width) % 8
        current = F.pad(
            current_clean, (0, pad_width, 0, pad_height), mode="replicate"
        )
        previous = F.pad(
            previous_clean, (0, pad_width, 0, pad_height), mode="replicate"
        )
        current, previous = self.weights.transforms()(current, previous)
        flow = self.model(current, previous)[-1]
        return flow[..., :height, :width]


def update_flow_dynamic_error(error, previous_error, backward_flow):
    if previous_error is None:
        previous_error = UnifiedFeatures(
            *(torch.zeros_like(value) for value in error.as_tuple())
        )
    aligned_z1, valid_z1 = backward_flow_to_state_warp(
        previous_error.z1, backward_flow
    )
    aligned_z4, valid_z4 = backward_flow_to_state_warp(
        previous_error.z4, backward_flow
    )
    state = UnifiedFeatures(
        GAMMA * error.z1 + HISTORY_WEIGHT * aligned_z1,
        GAMMA * error.z2 + HISTORY_WEIGHT * previous_error.z2,
        GAMMA * error.z3 + HISTORY_WEIGHT * previous_error.z3,
        GAMMA * error.z4 + HISTORY_WEIGHT * aligned_z4,
    )
    return state, valid_z1.float().mean().item(), valid_z4.float().mean().item()


def write_rows(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def evaluate_paths(model, predictor, corrections, raft, groups, sigma):
    confusion = {
        name: torch.zeros((19, 19), dtype=torch.int64)
        for name in ("clean", "noisy", "prediction", "unaligned", "flow_aligned")
    }
    rows = []
    finite = True
    model.eval()
    predictor.eval()
    corrections.eval()
    with torch.no_grad():
        for sequence_id, samples in groups.items():
            observation_previous_previous = None
            observation_previous = None
            unaligned_previous_previous = None
            unaligned_previous = None
            flow_previous_previous = None
            flow_previous = None
            unaligned_error = None
            flow_error = None
            previous_clean = None
            for frame_index, sample in enumerate(samples):
                clean_image = load_image(sample)
                noisy_image = add_frame_noise(clean_image, sigma)
                observation = encode_image(model, noisy_image)
                if observation_previous is None:
                    observation_previous = observation
                    unaligned_previous = observation
                    flow_previous = observation
                    previous_clean = clean_image
                    continue
                if observation_previous_previous is None:
                    observation_previous_previous = observation_previous
                    observation_previous = observation
                    unaligned_previous_previous = unaligned_previous
                    unaligned_previous = observation
                    flow_previous_previous = flow_previous
                    flow_previous = observation
                    previous_clean = clean_image
                    continue

                prediction_prior, _ = predict_current(
                    predictor,
                    observation_previous_previous,
                    observation_previous,
                    observation,
                )
                unaligned_prior, unaligned_instant = predict_current(
                    predictor,
                    unaligned_previous_previous,
                    unaligned_previous,
                    observation,
                )
                flow_prior, flow_instant = predict_current(
                    predictor,
                    flow_previous_previous,
                    flow_previous,
                    observation,
                )
                backward_flow = raft.backward_flow(clean_image, previous_clean)
                unaligned_error = update_dynamic_error(
                    unaligned_instant, unaligned_error
                )
                flow_error, valid_z1, valid_z4 = update_flow_dynamic_error(
                    flow_instant, flow_error, backward_flow
                )
                unaligned_posterior = posterior_state(
                    unaligned_prior,
                    unaligned_instant,
                    unaligned_error,
                    observation,
                    corrections,
                )
                flow_posterior = posterior_state(
                    flow_prior,
                    flow_instant,
                    flow_error,
                    observation,
                    corrections,
                )
                noisy_features = model.extract_backbone_features(noisy_image)
                output_size = tuple(clean_image.shape[-2:])
                prediction_host = corrected_host_feature(
                    model,
                    noisy_features,
                    observation,
                    prediction_prior,
                    output_size,
                )
                unaligned_host = corrected_host_feature(
                    model,
                    noisy_features,
                    observation,
                    unaligned_posterior,
                    output_size,
                )
                flow_host = corrected_host_feature(
                    model,
                    noisy_features,
                    observation,
                    flow_posterior,
                    output_size,
                )
                logits = {
                    "clean": model(clean_image),
                    "noisy": model(noisy_image),
                    "prediction": model.decode_from_host_feature(prediction_host),
                    "unaligned": model.decode_from_host_feature(unaligned_host),
                    "flow_aligned": model.decode_from_host_feature(flow_host),
                }
                mask = semantic_mask_from_panoptic_png(sample["mask_path"])
                for name, value in logits.items():
                    prediction = value.argmax(dim=1).squeeze(0).cpu().to(torch.int64)
                    update_confusion_matrix(confusion[name], prediction, mask)
                finite = finite and all(
                    torch.isfinite(value).all().item()
                    for value in (
                        flow_instant.z1,
                        flow_instant.z4,
                        flow_error.z1,
                        flow_error.z4,
                        flow_posterior.z1,
                        flow_posterior.z4,
                    )
                )
                row = {
                    "sequence_id": sequence_id,
                    "frame_id": sample["frame_id"],
                    "frame_index": frame_index,
                    "valid_warp_ratio_z1": valid_z1,
                    "valid_warp_ratio_z4": valid_z4,
                }
                row.update(
                    state_metrics(
                        flow_instant,
                        flow_error,
                        flow_posterior,
                        observation,
                    )
                )
                rows.append(row)
                unaligned_error = detach_state(unaligned_error)
                flow_error = detach_state(flow_error)
                observation_previous_previous = observation_previous
                observation_previous = observation
                unaligned_previous_previous = unaligned_previous
                unaligned_previous = detach_state(unaligned_posterior)
                flow_previous_previous = flow_previous
                flow_previous = detach_state(flow_posterior)
                previous_clean = clean_image
    metrics = {
        name: float(torch.nanmean(compute_iou(value)).item())
        for name, value in confusion.items()
    }
    return metrics, rows, finite


def decision_label(flow_aligned_miou, unaligned_miou):
    improvement = flow_aligned_miou - unaligned_miou
    if improvement <= 0.0:
        return "ALIGNMENT NO-GO"
    if improvement < 0.005:
        return "ALIGNMENT INCONCLUSIVE"
    if improvement < 0.010:
        return "ALIGNMENT GO"
    return "ALIGNMENT STRONG GO"


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("RAFT flow alignment expects GPU 0.")
    seed = int(os.environ.get("PREDIFY_SEED", "0"))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    root = Path(os.environ.get("PREDIFY_KITTI_STEP_ROOT", "/home/lin/predify/kitti_step"))
    static_checkpoint = Path(
        os.environ.get("PREDIFY_KITTI_STEP_STATIC_CHECKPOINT", STATIC_CHECKPOINT_DEFAULT)
    )
    adapter_checkpoint = Path(
        os.environ.get("PREDIFY_KITTI_STEP_ADAPTER_CHECKPOINT", ADAPTER_CHECKPOINT_DEFAULT)
    )
    predictor_checkpoint = Path(
        os.environ.get("PREDIFY_KITTI_STEP_PREDICTOR_CHECKPOINT", PREDICTOR_CHECKPOINT_DEFAULT)
    )
    correction_checkpoint = Path(
        os.environ.get(
            "PREDIFY_KITTI_STEP_CLOSED_LOOP_CORRECTION_CHECKPOINT",
            CLOSED_LOOP_CORRECTION_CHECKPOINT_DEFAULT,
        )
    )
    output_dir = Path(
        os.environ.get(
            "PREDIFY_KITTI_STEP_RAFT_ALIGNMENT_OUTPUT_DIR",
            "/home/lin/predify/experiments/kitti_step_raft_error_memory_alignment",
        )
    )
    sigma = 0.10
    model = build_deeplabv3plus_resnet50_host(
        checkpoint_path=os.environ.get("PREDIFY_DEEPLABV3PLUS_CITYSCAPES_CHECKPOINT"),
    ).cuda()
    load_static_kitti_checkpoint(model, static_checkpoint)
    adapter_payload = torch.load(adapter_checkpoint, map_location="cpu", weights_only=False)
    model.multi_layer_adapter.load_state_dict(adapter_payload["adapter_state_dict"], strict=True)
    predictor = MultiLayerPredictor().cuda()
    predictor_payload = torch.load(predictor_checkpoint, map_location="cpu", weights_only=False)
    predictor.load_state_dict(predictor_payload["predictor_state_dict"], strict=True)
    corrections = torch.nn.ModuleList(
        [ErrorGainCorrection(), ErrorGainCorrection()]
    ).cuda()
    correction_payload = torch.load(
        correction_checkpoint, map_location="cpu", weights_only=False
    )
    corrections.load_state_dict(
        correction_payload["correction_state_dict"], strict=True
    )
    model.requires_grad_(False)
    predictor.requires_grad_(False)
    corrections.requires_grad_(False)
    raft = FrozenRAFT()
    trainable_parameter_count = sum(
        parameter.numel()
        for module in (model, predictor, corrections, raft.model)
        for parameter in module.parameters()
        if parameter.requires_grad
    )
    dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(root, "val")
    groups = sequence_groups(dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    metrics, rows, finite = evaluate_paths(
        model, predictor, corrections, raft, groups, sigma
    )
    write_rows(output_dir / "per_frame.csv", rows)
    phases, phase_stable = phase_statistics(rows)
    flow_recovery = (metrics["flow_aligned"] - metrics["noisy"]) / (
        metrics["clean"] - metrics["noisy"]
    )
    summary = {
        "experiment": "kitti_step_raft_error_memory_alignment",
        "git_revision": os.environ.get("PREDIFY_GIT_REVISION"),
        "checkpoints": {
            "static_host": str(static_checkpoint),
            "fixed_adapter": str(adapter_checkpoint),
            "fixed_predictor": str(predictor_checkpoint),
            "fixed_closed_loop_correction": str(correction_checkpoint),
            "raft_weights": raft.weights.url,
        },
        "config": {
            "gaussian_noise_sigma": sigma,
            "gamma": GAMMA,
            "history_weight": HISTORY_WEIGHT,
            "flow_aligned_layers": [1, 4],
            "raft": "torchvision.models.optical_flow.raft_large",
            "raft_weight_enum": str(raft.weights),
            "trainable_parameter_count": trainable_parameter_count,
            "optimizer": None,
        },
        "dataset": {
            "split": "val",
            "sequence_count": len(groups),
            "frame_count": len(dataset.samples),
        },
        "gradient_boundary": {
            "host_trainable": any(parameter.requires_grad for parameter in model.parameters()),
            "adapter_trainable": any(
                parameter.requires_grad for parameter in model.multi_layer_adapter.parameters()
            ),
            "predictor_trainable": any(
                parameter.requires_grad for parameter in predictor.parameters()
            ),
            "correction_trainable": any(
                parameter.requires_grad for parameter in corrections.parameters()
            ),
            "raft_trainable": any(parameter.requires_grad for parameter in raft.model.parameters()),
        },
        "metrics": {
            "mIoU_clean_static": metrics["clean"],
            "mIoU_noisy_static": metrics["noisy"],
            "mIoU_continuous_prediction_only": metrics["prediction"],
            "mIoU_closed_loop_no_alignment": metrics["unaligned"],
            "mIoU_closed_loop_raft_flow_aligned": metrics["flow_aligned"],
            "flow_aligned_minus_unaligned": metrics["flow_aligned"] - metrics["unaligned"],
            "flow_aligned_minus_prediction": metrics["flow_aligned"] - metrics["prediction"],
            "flow_aligned_minus_noisy": metrics["flow_aligned"] - metrics["noisy"],
            "recovery_flow_aligned": flow_recovery,
        },
        "flow_alignment": {
            "z1_mean_valid_warp_ratio": sum(row["valid_warp_ratio_z1"] for row in rows) / len(rows),
            "z4_mean_valid_warp_ratio": sum(row["valid_warp_ratio_z4"] for row in rows) / len(rows),
        },
        "stability": {
            "finite": finite,
            "phase_stable": phase_stable,
            "phase_statistics": phases,
        },
        "decision": decision_label(metrics["flow_aligned"], metrics["unaligned"]),
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
