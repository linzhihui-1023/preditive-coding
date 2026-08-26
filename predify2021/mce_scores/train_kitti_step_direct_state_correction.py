import json
import os
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset
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
from predify2021.mce_scores.evaluate_kitti_step_oracle_upper_bound import (
    load_writeback_checkpoint,
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
    UnifiedFeatures,
    build_deeplabv3plus_resnet50_host,
)


SEED = 0
SIGMA = 0.10
EPOCHS = 3
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
EXPECTED_TRAIN_SEQUENCES = 12
EXPECTED_TRAIN_FRAMES = 5027
CORRECTION_INDICES = (0, 3)

WRITEBACK_CHECKPOINT_DEFAULT = (
    "/home/lin/predify/experiments/kitti_step_host_conditioned_writeback/"
    "host_conditioned_writeback_epoch3.pt"
)


def build_corrections():
    return torch.nn.ModuleList([DirectStateCorrection(), DirectStateCorrection()]).cuda()


def configure_direct_correction(corrections):
    corrections.requires_grad_(True)
    parameters = list(corrections.parameters())
    return parameters


def direct_posterior(observation, error, dynamic_error, corrections):
    delta = (
        corrections[0](observation.z1, error.z1, dynamic_error.z1),
        corrections[1](observation.z4, error.z4, dynamic_error.z4),
    )
    return UnifiedFeatures(
        observation.z1 + delta[0],
        observation.z2,
        observation.z3,
        observation.z4 + delta[1],
    )


def correction_loss(posterior, clean_state):
    return F.mse_loss(posterior.z1, clean_state.z1) + F.mse_loss(
        posterior.z4, clean_state.z4
    )


def run_epoch(model, predictor, corrections, groups, optimizer, training):
    model.eval()
    predictor.eval()
    corrections.train(training)
    total_loss = 0.0
    frame_count = 0
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
            with torch.no_grad():
                predicted, error = predict_current(
                    predictor, previous_previous, previous, observation
                )
                dynamic_error = update_dynamic_error(error, dynamic_error)
            posterior = direct_posterior(observation, error, dynamic_error, corrections)
            loss = correction_loss(posterior, clean_state)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.detach().item()
            frame_count += 1
            dynamic_error = UnifiedFeatures(
                *(value.detach() for value in dynamic_error.as_tuple())
            )
            previous_previous = previous
            previous = UnifiedFeatures(*(value.detach() for value in posterior.as_tuple()))
    return total_loss / frame_count, frame_count


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Direct state correction training requires CUDA.")
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    root = Path(os.environ.get("PREDIFY_KITTI_STEP_ROOT", "/home/lin/predify/kitti_step"))
    output_dir = Path(
        os.environ.get(
            "PREDIFY_KITTI_STEP_DIRECT_CORRECTION_OUTPUT_DIR",
            "/home/lin/predify/experiments/kitti_step_direct_state_correction",
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
    model = build_deeplabv3plus_resnet50_host().cuda()
    load_static_kitti_checkpoint(model, static_checkpoint)
    adapter_payload = torch.load(adapter_checkpoint, map_location="cpu", weights_only=False)
    model.multi_layer_adapter.load_state_dict(adapter_payload["adapter_state_dict"], strict=True)
    load_writeback_checkpoint(model, writeback_checkpoint)
    predictor = MultiLayerPredictor().cuda()
    predictor_payload = torch.load(predictor_checkpoint, map_location="cpu", weights_only=False)
    predictor.load_state_dict(predictor_payload["predictor_state_dict"], strict=True)
    model.requires_grad_(False)
    predictor.requires_grad_(False)
    corrections = build_corrections()
    parameters = configure_direct_correction(corrections)
    optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(root, "train")
    groups = sequence_groups(dataset)
    sequence_count = len(groups)
    frame_count = len(dataset.samples)
    if sequence_count != EXPECTED_TRAIN_SEQUENCES or frame_count != EXPECTED_TRAIN_FRAMES:
        raise RuntimeError("KITTI-STEP train protocol mismatch.")
    history = []
    for epoch in range(1, EPOCHS + 1):
        loss, visited = run_epoch(model, predictor, corrections, groups, optimizer, True)
        history.append({"epoch": epoch, "average_correction_mse": loss, "trained_frame_count": visited})
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "direct_state_correction_epoch3.pt"
    torch.save(
        {
            "direct_corrections": {str(index): corrections[position].state_dict() for position, index in enumerate(CORRECTION_INDICES)},
            "epoch": EPOCHS,
            "config": {"epochs": EPOCHS, "sigma": SIGMA, "seed": SEED, "optimizer": "AdamW", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY},
        },
        checkpoint_path,
    )
    summary = {
        "experiment": "kitti_step_direct_state_correction_training",
        "git_revision": os.environ.get("PREDIFY_GIT_REVISION"),
        "checkpoint": str(checkpoint_path),
        "base_writeback_checkpoint": str(writeback_checkpoint),
        "dataset": {"split": "train", "sequence_count": sequence_count, "frame_count": frame_count, "labels_used": False},
        "trainable_parameter_count": sum(parameter.numel() for parameter in parameters),
        "history": history,
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
