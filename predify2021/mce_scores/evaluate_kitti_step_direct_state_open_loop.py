import json
import os
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from predify2021.datasets.kitti_step import (
    KITTISTEPSegmentationDataset,
    semantic_mask_from_panoptic_png,
)
from predify2021.mce_scores.evaluate_kitti_step_closed_loop_dynamic_correction import (
    add_frame_noise,
)
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import (
    encode_image,
    load_image,
    predict_current,
    PREDICTOR_CHECKPOINT_DEFAULT,
    sequence_groups,
    update_dynamic_error,
)
from predify2021.mce_scores.evaluate_kitti_step_error_correction import (
    corrected_host_feature,
)
from predify2021.mce_scores.evaluate_kitti_step_oracle_upper_bound import (
    load_writeback_checkpoint,
)
from predify2021.mce_scores.evaluate_kitti_step_static_baseline import (
    compute_iou,
    update_confusion_matrix,
)
from predify2021.mce_scores.train_kitti_step_direct_state_correction import (
    CORRECTION_INDICES,
    direct_posterior,
)
from predify2021.mce_scores.train_kitti_step_fixed_adapter_predictor import (
    ADAPTER_CHECKPOINT_DEFAULT,
    STATIC_CHECKPOINT_DEFAULT,
)
from predify2021.mce_scores.train_kitti_step_state_predictor import (
    load_static_kitti_checkpoint,
)
from predify2021.model_factory.deeplabv3plus_resnet50 import (
    DirectStateCorrection,
    MultiLayerPredictor,
    build_deeplabv3plus_resnet50_host,
)


SEED = 0
SIGMA = 0.10
EXPECTED_CLEAN_MIOU = 0.6552125562
EXPECTED_NOISY_MIOU = 0.2940764905
EXPECTED_SEQUENCES = 9
EXPECTED_TOTAL_FRAMES = 2981
EXPECTED_EVALUATED_FRAMES = 2963

WRITEBACK_CHECKPOINT_DEFAULT = (
    "/home/lin/predify/experiments/kitti_step_host_conditioned_writeback/"
    "host_conditioned_writeback_epoch3.pt"
)
DIRECT_CHECKPOINT_DEFAULT = (
    "/home/lin/predify/experiments/kitti_step_direct_state_correction/"
    "direct_state_correction_epoch3.pt"
)


def load_direct_checkpoint(corrections, checkpoint_path):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    for position, index in enumerate(CORRECTION_INDICES):
        corrections[position].load_state_dict(
            payload["direct_corrections"][str(index)], strict=True
        )
    return payload


def open_loop_history(previous_previous, previous, observation):
    """Keep noisy observation history after the current frame is decoded."""
    return previous, observation


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Direct state open-loop evaluation requires CUDA.")
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    root = Path(os.environ.get("PREDIFY_KITTI_STEP_ROOT", "/home/lin/predify/kitti_step"))
    output_dir = Path(
        os.environ.get(
            "PREDIFY_KITTI_STEP_DIRECT_OPEN_LOOP_OUTPUT_DIR",
            "results/kitti_step_direct_state_open_loop",
        )
    )
    static_checkpoint = Path(
        os.environ.get("PREDIFY_KITTI_STEP_STATIC_CHECKPOINT", STATIC_CHECKPOINT_DEFAULT)
    )
    adapter_checkpoint = Path(
        os.environ.get("PREDIFY_KITTI_STEP_ADAPTER_CHECKPOINT", ADAPTER_CHECKPOINT_DEFAULT)
    )
    predictor_checkpoint = Path(
        os.environ.get("PREDIFY_KITTI_STEP_PREDICTOR_CHECKPOINT", PREDICTOR_CHECKPOINT_DEFAULT)
    )
    writeback_checkpoint = Path(
        os.environ.get(
            "PREDIFY_KITTI_STEP_WRITEBACK_CHECKPOINT",
            WRITEBACK_CHECKPOINT_DEFAULT,
        )
    )
    direct_checkpoint = Path(
        os.environ.get(
            "PREDIFY_KITTI_STEP_DIRECT_CORRECTION_CHECKPOINT",
            DIRECT_CHECKPOINT_DEFAULT,
        )
    )

    model = build_deeplabv3plus_resnet50_host().cuda()
    load_static_kitti_checkpoint(model, static_checkpoint)
    adapter_payload = torch.load(adapter_checkpoint, map_location="cpu", weights_only=False)
    model.multi_layer_adapter.load_state_dict(adapter_payload["adapter_state_dict"], strict=True)
    load_writeback_checkpoint(model, writeback_checkpoint)
    predictor = MultiLayerPredictor().cuda()
    predictor_payload = torch.load(predictor_checkpoint, map_location="cpu", weights_only=False)
    predictor.load_state_dict(predictor_payload["predictor_state_dict"], strict=True)
    corrections = torch.nn.ModuleList(
        [DirectStateCorrection(), DirectStateCorrection()]
    ).cuda()
    load_direct_checkpoint(corrections, direct_checkpoint)
    model.eval()
    predictor.eval()
    corrections.eval()
    modules = (model, predictor, corrections)
    parameter_snapshot = [
        parameter.detach().cpu().clone()
        for module in modules
        for parameter in module.parameters()
    ]

    dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(root, "val")
    groups = sequence_groups(dataset)
    if len(groups) != EXPECTED_SEQUENCES or len(dataset.samples) != EXPECTED_TOTAL_FRAMES:
        raise RuntimeError("KITTI-STEP validation protocol mismatch.")
    confusion = torch.zeros((19, 19), dtype=torch.int64)
    noisy_mse = {"z1": 0.0, "z4": 0.0}
    direct_mse = {"z1": 0.0, "z4": 0.0}
    finite = True
    evaluated_frame_count = 0

    with torch.inference_mode():
        for samples in groups.values():
            previous_previous = None
            previous = None
            dynamic_error = None
            for sample in samples:
                clean_image = load_image(sample)
                noisy_image = add_frame_noise(clean_image, SIGMA)
                clean_state = encode_image(model, clean_image)
                observation = encode_image(model, noisy_image)
                if previous is None:
                    previous = observation
                    continue
                if previous_previous is None:
                    previous_previous = previous
                    previous = observation
                    continue
                predicted, error = predict_current(
                    predictor, previous_previous, previous, observation
                )
                dynamic_error = update_dynamic_error(error, dynamic_error)
                posterior = direct_posterior(observation, error, dynamic_error, corrections)
                noisy_mse["z1"] += F.mse_loss(observation.z1, clean_state.z1).item()
                noisy_mse["z4"] += F.mse_loss(observation.z4, clean_state.z4).item()
                direct_mse["z1"] += F.mse_loss(posterior.z1, clean_state.z1).item()
                direct_mse["z4"] += F.mse_loss(posterior.z4, clean_state.z4).item()
                raw_features = model.extract_backbone_features(noisy_image)
                host_feature = corrected_host_feature(
                    model, raw_features, observation, posterior, tuple(clean_image.shape[-2:])
                )
                logits = model.decode_from_host_feature(host_feature)
                mask = semantic_mask_from_panoptic_png(sample["mask_path"])
                update_confusion_matrix(
                    confusion,
                    logits.argmax(dim=1).squeeze(0).cpu().to(torch.int64),
                    mask,
                )
                finite = finite and all(
                    torch.isfinite(value).all().item()
                    for value in (
                        observation.z1,
                        observation.z4,
                        error.z1,
                        error.z4,
                        dynamic_error.z1,
                        dynamic_error.z4,
                        posterior.z1,
                        posterior.z4,
                        logits,
                    )
                )
                evaluated_frame_count += 1
                previous_previous, previous = open_loop_history(
                    previous_previous, previous, observation
                )
                dynamic_error = type(dynamic_error)(
                    *(value.detach() for value in dynamic_error.as_tuple())
                )

    if evaluated_frame_count != EXPECTED_EVALUATED_FRAMES:
        raise RuntimeError("KITTI-STEP evaluated frame count mismatch.")
    noisy_average = sum(noisy_mse.values()) / (2 * evaluated_frame_count)
    direct_average = sum(direct_mse.values()) / (2 * evaluated_frame_count)
    direct_miou = float(torch.nanmean(compute_iou(confusion)).item())
    current_parameters = [
        parameter for module in modules for parameter in module.parameters()
    ]
    parameters_unchanged = all(
        torch.equal(parameter.detach().cpu(), snapshot)
        for parameter, snapshot in zip(current_parameters, parameter_snapshot)
    )
    summary = {
        "experiment": "kitti_step_direct_state_open_loop_diagnostic",
        "git_revision": os.environ.get("PREDIFY_GIT_REVISION"),
        "protocol": {
            "split": "val",
            "sequence_count": len(groups),
            "total_frame_count": len(dataset.samples),
            "evaluated_frame_count": evaluated_frame_count,
            "seed": SEED,
            "sigma": SIGMA,
        },
        "checkpoints": {
            "static_host": str(static_checkpoint),
            "adapter": str(adapter_checkpoint),
            "predictor": str(predictor_checkpoint),
            "writeback": str(writeback_checkpoint),
            "direct_correction": str(direct_checkpoint),
        },
        "state_mse": {
            "noisy_to_clean": {
                "z1": noisy_mse["z1"] / evaluated_frame_count,
                "z4": noisy_mse["z4"] / evaluated_frame_count,
                "mean": noisy_average,
            },
            "direct_open_loop_to_clean": {
                "z1": direct_mse["z1"] / evaluated_frame_count,
                "z4": direct_mse["z4"] / evaluated_frame_count,
                "mean": direct_average,
            },
        },
        "mIoU_direct_open_loop": direct_miou,
        "historical_references": {
            "mIoU_noisy_static": 0.2940764905,
            "mIoU_direct_closed_loop": 0.2870732213,
        },
        "checks": {
            "parameters_unchanged": parameters_unchanged,
            "history_source": "noisy_observation",
            "clean_state_in_history": False,
            "finite": finite,
            "protocol": len(groups) == EXPECTED_SEQUENCES
            and len(dataset.samples) == EXPECTED_TOTAL_FRAMES
            and evaluated_frame_count == EXPECTED_EVALUATED_FRAMES,
        },
        "decision": (
            "STATE CORRECTION FAILURE"
            if direct_average >= noisy_average
            else "MSE-SEMANTIC OBJECTIVE MISMATCH"
            if direct_miou <= EXPECTED_NOISY_MIOU
            else "CLOSED-LOOP ACCUMULATION BOTTLENECK"
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
