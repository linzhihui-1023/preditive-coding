import json
import os
import random
from pathlib import Path

import torch

from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset
from predify2021.mce_scores.role_separated_dynamic_error_correction import (
    ADAPTER_CHECKPOINT_DEFAULT,
    ROLE_PREDICTOR_CHECKPOINT_DEFAULT,
    STATIC_CHECKPOINT_DEFAULT,
    WRITEBACK_CHECKPOINT_DEFAULT,
    DYNAMIC_ERROR_GAIN,
    DYNAMIC_ERROR_SAMPLE_TIME,
    DYNAMIC_ERROR_TIME_CONSTANT,
    add_frame_noise,
    correction_loss,
    correction_posterior,
    detach_state,
    encode_image,
    error_state,
    load_components,
    load_image,
    next_role_prediction,
    update_dynamic_error,
    zero_state,
)
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import sequence_groups
from predify2021.model_factory.deeplabv3plus_resnet50.corrections import ErrorGainCorrection


EPOCHS = 3
SIGMA = 0.10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01


def run_epoch(model, predictor, corrections, groups, optimizer=None):
    corrections.train(optimizer is not None)
    total_loss = 0.0
    frame_count = 0
    for samples in groups.values():
        hidden = (None, None, None, None)
        dynamic_error = None
        pending_dynamics = None
        pending_semantic = None
        for frame_index, sample in enumerate(samples):
            clean_image = load_image(sample)
            noisy_image = add_frame_noise(clean_image, SIGMA)
            clean_state = encode_image(model, clean_image)
            observation = encode_image(model, noisy_image)
            with torch.no_grad():
                if frame_index == 0:
                    pending_dynamics, pending_semantic, *hidden = next_role_prediction(
                        predictor, observation, zero_state(observation), hidden
                    )
                    continue
                error = error_state(observation, pending_dynamics)
                if frame_index == 1:
                    pending_dynamics, pending_semantic, *hidden = next_role_prediction(
                        predictor, observation, error, hidden
                    )
                    continue
                dynamic_error = update_dynamic_error(error, dynamic_error)
            posterior, _ = correction_posterior(
                pending_semantic, error, dynamic_error, corrections
            )
            loss = correction_loss(posterior, clean_state)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.detach().item()
            frame_count += 1
            with torch.no_grad():
                pending_dynamics, pending_semantic, *hidden = next_role_prediction(
                    predictor, observation, error, hidden
                )
                dynamic_error = detach_state(dynamic_error)
    return total_loss / frame_count, frame_count


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Role-separated dynamic correction training requires CUDA")
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    root = Path(os.environ.get("PREDIFY_KITTI_STEP_ROOT", "/home/lin/predify/kitti_step"))
    output = Path(os.environ.get("PREDIFY_ROLE_SEPARATED_DYNAMIC_CORRECTION_OUTPUT_DIR", "/home/lin/predify/experiments/kitti_step_role_separated_dynamic_error_correction"))
    static_checkpoint = os.environ.get("PREDIFY_KITTI_STEP_STATIC_CHECKPOINT", STATIC_CHECKPOINT_DEFAULT)
    adapter_checkpoint = os.environ.get("PREDIFY_KITTI_STEP_ADAPTER_CHECKPOINT", ADAPTER_CHECKPOINT_DEFAULT)
    predictor_checkpoint = os.environ.get("PREDIFY_ROLE_SEPARATED_PREDICTOR_CHECKPOINT", ROLE_PREDICTOR_CHECKPOINT_DEFAULT)
    writeback_checkpoint = os.environ.get("PREDIFY_KITTI_STEP_WRITEBACK_CHECKPOINT", WRITEBACK_CHECKPOINT_DEFAULT)
    model, predictor = load_components(static_checkpoint, adapter_checkpoint, predictor_checkpoint, writeback_checkpoint)
    corrections = torch.nn.ModuleList([ErrorGainCorrection(), ErrorGainCorrection()]).cuda()
    corrections.requires_grad_(True)
    parameters = list(corrections.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train = KITTISTEPSegmentationDataset.from_kitti_step_root(root, "train")
    val = KITTISTEPSegmentationDataset.from_kitti_step_root(root, "val")
    train_groups = sequence_groups(train)
    val_groups = sequence_groups(val)
    history = []
    best = None
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_frames = run_epoch(model, predictor, corrections, train_groups, optimizer)
        with torch.no_grad():
            val_loss, val_frames = run_epoch(model, predictor, corrections, val_groups)
        record = {
            "epoch": epoch,
            "train_correction_state_mse": train_loss,
            "train_frame_count": train_frames,
            "val_correction_state_mse": val_loss,
            "val_frame_count": val_frames,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if best is None or val_loss < best["val_correction_state_mse"]:
            best = record
            output.mkdir(parents=True, exist_ok=True)
            torch.save({"correction_state_dict": corrections.state_dict(), "epoch": epoch, "val_correction_state_mse": val_loss}, output / "best_role_separated_dynamic_correction.pt")
    summary = {
        "experiment": "kitti_step_role_separated_dynamic_error_correction_training",
        "git_revision": os.environ.get("PREDIFY_GIT_REVISION"),
        "checkpoint": str(output / "best_role_separated_dynamic_correction.pt"),
        "static_host_checkpoint": static_checkpoint,
        "adapter_checkpoint": adapter_checkpoint,
        "predictor_checkpoint": predictor_checkpoint,
        "writeback_checkpoint": writeback_checkpoint,
        "trainable_parameter_count": sum(parameter.numel() for parameter in parameters),
        "config": {"epochs": EPOCHS, "optimizer": "AdamW", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "seed": 0, "gaussian_noise_sigma": SIGMA, "dynamic_error_sample_time": DYNAMIC_ERROR_SAMPLE_TIME, "dynamic_error_time_constant": DYNAMIC_ERROR_TIME_CONSTANT, "dynamic_error_gain": DYNAMIC_ERROR_GAIN, "labels_used": False},
        "dataset": {"train_sequence_count": len(train_groups), "val_sequence_count": len(val_groups), "train_frame_count": len(train.samples), "val_frame_count": len(val.samples)},
        "history": history,
        "best": best,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
