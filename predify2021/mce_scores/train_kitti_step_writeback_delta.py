"""Train only host-conditioned residual writeback for c1 and c4.

For paired clean/noisy views of the same KITTI-STEP Train frame, the frozen host
and frozen input adapters produce (F_clean, F_noisy) and (Z_clean, Z_noisy).
The frozen D1/D4 coordinate converters provide the base correction.  Independent
P/H/Q branches learn the remaining host-feature delta from F_noisy and delta_Z.
Predictor, correction, labels, and closed-loop state remain outside the objective.
"""

import json
import os
import random
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset
from predify2021.mce_scores.evaluate_kitti_step_closed_loop_dynamic_correction import (
    add_frame_noise,
)
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import (
    load_image,
    sequence_groups,
)
from predify2021.mce_scores.train_kitti_step_fixed_adapter_predictor import (
    ADAPTER_CHECKPOINT_DEFAULT,
    STATIC_CHECKPOINT_DEFAULT,
)
from predify2021.mce_scores.train_kitti_step_state_predictor import (
    load_static_kitti_checkpoint,
)
from predify2021.model_factory.deeplabv3plus_resnet50 import (
    build_deeplabv3plus_resnet50_host,
)


SEED = 0
SIGMA = 0.10
EPOCHS = 1
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
EXPECTED_TRAIN_SEQUENCES = 12
EXPECTED_TRAINABLE_PARAMETERS = 884_736
WRITEBACK_INDICES = (0, 3)
BASE_WRITEBACK_CHECKPOINT_DEFAULT = (
    "/home/lin/predify/experiments/kitti_step_writeback_delta/"
    "writeback_delta_epoch1.pt"
)


def configure_writeback_only(model):
    model.requires_grad_(False)
    model.host_conditioned_writeback_enabled = True
    writebacks = [
        model.host_conditioned_writebacks[str(index)] for index in WRITEBACK_INDICES
    ]
    output_adapters = [
        model.multi_layer_adapter.output_adapters[index] for index in WRITEBACK_INDICES
    ]
    for writeback in writebacks:
        writeback.requires_grad_(True)
    parameters = [
        parameter for writeback in writebacks for parameter in writeback.parameters()
    ]
    details = [
        {
            "name": f"host_conditioned_writebacks.{index}.{name}",
            "parameter_count": parameter.numel(),
        }
        for index, writeback in zip(WRITEBACK_INDICES, writebacks)
        for name, parameter in writeback.named_parameters()
    ]
    actual_count = sum(parameter.numel() for parameter in parameters)
    if actual_count != EXPECTED_TRAINABLE_PARAMETERS:
        raise RuntimeError(
            "Writeback parameter count mismatch: "
            f"expected {EXPECTED_TRAINABLE_PARAMETERS}, got {actual_count}; "
            f"details={details}"
        )
    return writebacks, output_adapters, parameters, details


def load_base_writeback_checkpoint(model, checkpoint_path):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    for index in WRITEBACK_INDICES:
        model.multi_layer_adapter.output_adapters[index].load_state_dict(
            payload["output_adapters"][str(index)], strict=True
        )
    return payload


def writeback_loss(
    writebacks,
    output_adapters,
    clean_features,
    noisy_features,
    clean_state,
    noisy_state,
):
    delta_z1 = clean_state.z1 - noisy_state.z1
    delta_z4 = clean_state.z4 - noisy_state.z4
    delta_c1 = clean_features.c1 - noisy_features.c1
    delta_c4 = clean_features.c4 - noisy_features.c4
    predicted_c1 = output_adapters[0](delta_z1) + writebacks[0](
        noisy_features.c1, delta_z1
    )
    predicted_c4 = output_adapters[1](delta_z4) + writebacks[1](
        noisy_features.c4, delta_z4
    )
    loss_z1_c1 = F.mse_loss(predicted_c1, delta_c1)
    loss_z4_c4 = F.mse_loss(predicted_c4, delta_c4)
    return 0.5 * (loss_z1_c1 + loss_z4_c4), loss_z1_c1, loss_z4_c4


def collate_samples(samples):
    return samples


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Writeback delta training requires CUDA.")

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    root = Path(os.environ.get("PREDIFY_KITTI_STEP_ROOT", "/home/lin/predify/kitti_step"))
    static_checkpoint = Path(
        os.environ.get("PREDIFY_KITTI_STEP_STATIC_CHECKPOINT", STATIC_CHECKPOINT_DEFAULT)
    )
    adapter_checkpoint = Path(
        os.environ.get("PREDIFY_KITTI_STEP_ADAPTER_CHECKPOINT", ADAPTER_CHECKPOINT_DEFAULT)
    )
    base_writeback_checkpoint = Path(
        os.environ.get(
            "PREDIFY_KITTI_STEP_WRITEBACK_CHECKPOINT",
            BASE_WRITEBACK_CHECKPOINT_DEFAULT,
        )
    )
    output_dir = Path(
        os.environ.get(
            "PREDIFY_KITTI_STEP_HOST_CONDITIONED_WRITEBACK_OUTPUT_DIR",
            "/home/lin/predify/experiments/kitti_step_host_conditioned_writeback",
        )
    )
    batch_size = int(os.environ.get("PREDIFY_KITTI_STEP_WRITEBACK_BATCH_SIZE", "1"))
    num_workers = int(os.environ.get("PREDIFY_KITTI_STEP_NUM_WORKERS", "4"))

    model = build_deeplabv3plus_resnet50_host(
        checkpoint_path=os.environ.get("PREDIFY_DEEPLABV3PLUS_CITYSCAPES_CHECKPOINT"),
    ).cuda()
    load_static_kitti_checkpoint(model, static_checkpoint)
    adapter_payload = torch.load(adapter_checkpoint, map_location="cpu", weights_only=False)
    model.multi_layer_adapter.load_state_dict(
        adapter_payload["adapter_state_dict"], strict=True
    )
    load_base_writeback_checkpoint(model, base_writeback_checkpoint)
    writebacks, output_adapters, trainable_parameters, parameter_details = (
        configure_writeback_only(model)
    )
    trainable_ids = {id(parameter) for parameter in trainable_parameters}

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    optimizer_exact = optimizer_ids == trainable_ids
    if not optimizer_exact:
        raise RuntimeError("Optimizer parameter set is not exactly c1/c4 P, H, and Q.")

    dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(root, "train")
    sequence_count = len(sequence_groups(dataset))
    if sequence_count != EXPECTED_TRAIN_SEQUENCES:
        raise RuntimeError(
            f"Dataset protocol mismatch: expected {EXPECTED_TRAIN_SEQUENCES} train "
            f"sequences, got {sequence_count}."
        )
    loader = DataLoader(
        dataset.samples,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=collate_samples,
        generator=torch.Generator().manual_seed(SEED),
    )

    model.eval()
    for writeback in writebacks:
        writeback.train()
    total_loss = 0.0
    total_z1_c1_loss = 0.0
    total_z4_c4_loss = 0.0
    trained_frame_count = 0

    for samples in loader:
        clean_images = torch.cat([load_image(sample) for sample in samples], dim=0)
        noisy_images = add_frame_noise(clean_images, SIGMA)
        with torch.no_grad():
            clean_features = model.extract_backbone_features(clean_images)
            noisy_features = model.extract_backbone_features(noisy_images)
            clean_state = model.encode_backbone_features(clean_features)
            noisy_state = model.encode_backbone_features(noisy_features)

        optimizer.zero_grad(set_to_none=True)
        loss, loss_z1_c1, loss_z4_c4 = writeback_loss(
            writebacks,
            output_adapters,
            clean_features,
            noisy_features,
            clean_state,
            noisy_state,
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Non-finite L_write encountered.")
        loss.backward()
        optimizer.step()

        current_batch_size = len(samples)
        total_loss += loss.detach().item() * current_batch_size
        total_z1_c1_loss += loss_z1_c1.detach().item() * current_batch_size
        total_z4_c4_loss += loss_z4_c4.detach().item() * current_batch_size
        trained_frame_count += current_batch_size

    frozen_gradients_absent = all(
        parameter.grad is None
        for parameter in model.parameters()
        if id(parameter) not in trainable_ids
    )
    if not frozen_gradients_absent:
        raise RuntimeError("A frozen model parameter received a gradient.")
    if trained_frame_count != len(dataset.samples):
        raise RuntimeError("Training did not consume every KITTI-STEP Train frame once.")

    average_loss = total_loss / trained_frame_count
    average_z1_c1_loss = total_z1_c1_loss / trained_frame_count
    average_z4_c4_loss = total_z4_c4_loss / trained_frame_count
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "host_conditioned_writeback_epoch1.pt"
    fixed_config = {
        "epochs": EPOCHS,
        "sigma": SIGMA,
        "seed": SEED,
        "batch_size": batch_size,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "writeback_indices": list(WRITEBACK_INDICES),
    }
    torch.save(
        {
            "output_adapters": {
                str(index): adapter.state_dict()
                for index, adapter in zip(WRITEBACK_INDICES, output_adapters)
            },
            "host_conditioned_writebacks": {
                str(index): writeback.state_dict()
                for index, writeback in zip(WRITEBACK_INDICES, writebacks)
            },
            "epoch": EPOCHS,
            "train_average_l_write": average_loss,
            "config": fixed_config,
        },
        checkpoint_path,
    )

    parameter_gate = {
        "expected_trainable_parameter_count": EXPECTED_TRAINABLE_PARAMETERS,
        "actual_trainable_parameter_count": sum(
            parameter.numel() for parameter in trainable_parameters
        ),
        "parameter_details": parameter_details,
        "optimizer_parameters_exact": optimizer_exact,
        "frozen_gradients_absent": frozen_gradients_absent,
        "predictor_or_correction_executed": False,
    }
    parameter_gate["pass"] = (
        parameter_gate["actual_trainable_parameter_count"]
        == parameter_gate["expected_trainable_parameter_count"]
        and parameter_gate["optimizer_parameters_exact"]
        and parameter_gate["frozen_gradients_absent"]
        and not parameter_gate["predictor_or_correction_executed"]
    )
    summary = {
        "experiment": "kitti_step_host_conditioned_writeback_training",
        "git_revision": os.environ.get("PREDIFY_GIT_REVISION"),
        "objective": (
            "0.5 * (MSE(D1(delta_z1) + Q1(P1(c1_noisy) * H1(delta_z1)), "
            "delta_c1) + MSE(D4(delta_z4) + Q4(P4(c4_noisy) * H4(delta_z4)), "
            "delta_c4))"
        ),
        "config": fixed_config,
        "dataset": {
            "split": "train",
            "sequence_count": sequence_count,
            "trained_frame_count": trained_frame_count,
            "labels_used": False,
        },
        "checkpoints": {
            "static_host": str(static_checkpoint),
            "fixed_adapter_initialization": str(adapter_checkpoint),
            "frozen_base_writeback": str(base_writeback_checkpoint),
            "trained_writeback": str(checkpoint_path),
        },
        "train": {
            "average_l_write": average_loss,
            "average_z1_to_c1_mse": average_z1_c1_loss,
            "average_z4_to_c4_mse": average_z4_c4_loss,
            "finite": True,
        },
        "parameter_gate": parameter_gate,
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
