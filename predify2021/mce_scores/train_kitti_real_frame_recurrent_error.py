import json
import os
import random
from pathlib import Path

import torch

from predify2021.mce_scores.kitti_pairs import KITTINextFramePairDataset
from predify2021.model_factory import get_model


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TRAIN_DRIVES = (
    "2011_09_26/2011_09_26_drive_0005_sync",
    "2011_09_26/2011_09_26_drive_0013_sync",
    "2011_09_26/2011_09_26_drive_0014_sync",
    "2011_09_26/2011_09_26_drive_0036_sync",
)
VAL_DRIVES = (
    "2011_09_26/2011_09_26_drive_0011_sync",
    "2011_09_26/2011_09_26_drive_0039_sync",
)
FORBIDDEN_TEST_DRIVE_IDS = ("drive_0051_sync", "drive_0056_sync")


def build_model(weights_path):
    model = get_model(
        "pvgg_tf",
        pretrained=True,
        pcoder_weights=weights_path,
        task="real_frame_pc",
        dynamic_error=True,
        error_state_mode="ema",
        error_sample_time=0.1035,
        error_time_constant=(0.5,) * 5,
        error_gain=(1.0,) * 5,
        real_frame_transition_mode="convgru_error",
    ).to(DEVICE).eval()
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    transition_parameters = list(model.recurrent_transition_modules.parameters())
    if {id(parameter) for parameter in trainable} != {
        id(parameter) for parameter in transition_parameters
    }:
        raise RuntimeError("Only recurrent transition parameters may be trainable.")
    return model


def build_datasets(root, drives, camera, fixed_dt_s, tolerance):
    return {
        drive: KITTINextFramePairDataset(
            root,
            drive,
            camera=camera,
            fixed_dt_s=fixed_dt_s,
            dt_tolerance_s=tolerance,
        )
        for drive in drives
    }


def segment_raw_frames(dataset):
    for sample_segment in dataset.valid_sample_segments:
        starts = tuple(
            int(dataset.valid_start_indices[index]) for index in sample_segment
        )
        yield (*starts, starts[-1] + 1)


def run_epoch(model, datasets, optimizer=None):
    training = optimizer is not None
    model.eval()
    model.recurrent_transition_modules.train(training)
    total_loss = 0.0
    update_count = 0
    per_drive = {}

    for drive, dataset in datasets.items():
        drive_loss = 0.0
        drive_updates = 0
        for raw_frames in segment_raw_frames(dataset):
            model.reset()
            for raw_index, next_raw_index in zip(raw_frames[:-1], raw_frames[1:]):
                frame = dataset._load_frame(dataset.frame_paths[raw_index])
                frame = frame.unsqueeze(0).to(DEVICE)
                next_frame = dataset._load_frame(dataset.frame_paths[next_raw_index])
                next_frame = next_frame.unsqueeze(0).to(DEVICE)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    model.step_frame(frame)
                    loss = model.collect_recurrent_transition_loss(next_frame)
                    if loss is not None:
                        loss.backward()
                        optimizer.step()
                else:
                    with torch.no_grad():
                        model.step_frame(frame)
                        loss = model.collect_recurrent_transition_loss(next_frame)

                if loss is not None:
                    value = float(loss.detach().item())
                    total_loss += value
                    drive_loss += value
                    update_count += 1
                    drive_updates += 1
                del frame, next_frame

        per_drive[drive] = {
            "mean_prediction_mse": drive_loss / drive_updates,
            "transition_frames": drive_updates,
        }

    return {
        "mean_prediction_mse": total_loss / update_count,
        "transition_frames": update_count,
        "per_drive": per_drive,
    }


def main():
    if DEVICE.type != "cuda":
        raise RuntimeError("Formal recurrent-error training requires GPU 0.")
    revision = os.environ["PREDIFY_GIT_REVISION"]
    output_dir = Path(os.environ["PREDIFY_RECURRENT_ERROR_TRAIN_OUTPUT_DIR"])
    root = os.environ["PREDIFY_KITTI_ROOT"]
    camera = os.environ.get("PREDIFY_KITTI_CAMERA", "image_02")
    weights_path = os.environ["PREDIFY_PCODER_WEIGHTS"]
    epochs = int(os.environ.get("PREDIFY_EPOCHS", "5"))
    learning_rate = float(os.environ.get("PREDIFY_LR", "1e-4"))
    seed = int(os.environ.get("PREDIFY_SEED", "0"))
    fixed_dt_s = float(os.environ.get("PREDIFY_FIXED_TS_S", "0.1035"))
    tolerance = float(os.environ.get("PREDIFY_FIXED_TS_TOL_S", "0.001"))

    if any(
        forbidden in drive
        for forbidden in FORBIDDEN_TEST_DRIVE_IDS
        for drive in (*TRAIN_DRIVES, *VAL_DRIVES)
    ):
        raise RuntimeError("Frozen Test drive entered the training protocol.")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_datasets = build_datasets(
        root, TRAIN_DRIVES, camera, fixed_dt_s, tolerance
    )
    val_datasets = build_datasets(root, VAL_DRIVES, camera, fixed_dt_s, tolerance)
    model = build_model(weights_path)
    optimizer = torch.optim.Adam(
        model.recurrent_transition_modules.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
    )
    frozen_versions = {
        name: parameter._version
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    history = []
    best_val = float("inf")
    best_epoch = None
    checkpoint_path = output_dir / "best_recurrent_transition.pt"
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_datasets, optimizer=optimizer)
        val_metrics = run_epoch(model, val_datasets)
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if val_metrics["mean_prediction_mse"] < best_val:
            best_val = val_metrics["mean_prediction_mse"]
            best_epoch = epoch
            torch.save(
                {
                    "git_revision": revision,
                    "epoch": epoch,
                    "val_prediction_mse": best_val,
                    "transition_mode": "prediction_error_driven_convgru",
                    "top_down_feedback": True,
                    "recurrent_error_input": "dynamic",
                    "current_feedforward_transition_input": False,
                    "dynamic_error": "epsilon_t=0.207*e_t+0.793*epsilon_(t-1)",
                    "instant_error": "e_t=F_t-Fhat_t",
                    "training_target": "F_(t+1)",
                    "train_drives": TRAIN_DRIVES,
                    "val_drives": VAL_DRIVES,
                    "recurrent_transition_state_dict": (
                        model.recurrent_transition_modules.state_dict()
                    ),
                },
                checkpoint_path,
            )

    if any(
        parameter._version != frozen_versions[name]
        for name, parameter in model.named_parameters()
        if name in frozen_versions
    ):
        raise RuntimeError("A frozen Predify parameter changed during training.")

    summary = {
        "experiment": "real_frame_learned_error_driven_training",
        "git_revision": revision,
        "device": str(DEVICE),
        "gpu_name": torch.cuda.get_device_name(DEVICE),
        "seed": seed,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": 0.0,
        "train_drives": TRAIN_DRIVES,
        "val_drives": VAL_DRIVES,
        "frozen_test_drives_read": False,
        "best_epoch": best_epoch,
        "best_val_prediction_mse": best_val,
        "checkpoint": str(checkpoint_path),
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in model.recurrent_transition_modules.parameters()
        ),
        "objective": "mean next-frame per-layer PCoder prediction MSE",
        "transition": (
            "ConvGRU(previous_state,current_dynamic_prediction_error,feedback)"
        ),
        "current_feedforward_transition_input": False,
        "bptt": False,
        "cross_frame_state_detached": True,
        "history": history,
    }
    with (output_dir / "training_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
