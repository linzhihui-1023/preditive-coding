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
    sequence_groups,
    update_dynamic_error,
    PREDICTOR_CHECKPOINT_DEFAULT,
)
from predify2021.mce_scores.evaluate_kitti_step_error_correction import (
    corrected_host_feature,
)
from predify2021.mce_scores.evaluate_kitti_step_oracle_upper_bound import (
    load_writeback_checkpoint,
)
from predify2021.mce_scores.train_kitti_step_direct_state_correction import (
    CORRECTION_INDICES,
    correction_loss,
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
    HostFeature,
    MultiLayerPredictor,
    build_deeplabv3plus_resnet50_host,
)


SEED = 0
SIGMA = 0.10
EPOCHS = 3
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
EXPECTED_TRAIN_SEQUENCES = 12
EXPECTED_TRAIN_FRAMES = 5027
CORRECTION_CHANNELS = 128

WRITEBACK_CHECKPOINT_DEFAULT = (
    "/home/lin/predify/experiments/kitti_step_host_conditioned_writeback/"
    "host_conditioned_writeback_epoch3.pt"
)


def build_corrections():
    return torch.nn.ModuleList(
        [DirectStateCorrection(), DirectStateCorrection()]
    ).cuda()


def semantic_kl_loss(clean_logits, corrected_logits):
    target = F.softmax(clean_logits.detach(), dim=1)
    log_prediction = F.log_softmax(corrected_logits, dim=1)
    target = target.permute(0, 2, 3, 1).reshape(-1, target.shape[1])
    log_prediction = log_prediction.permute(0, 2, 3, 1).reshape(-1, log_prediction.shape[1])
    return F.kl_div(log_prediction, target, reduction="batchmean")


def semantic_state_loss(model, clean_features, clean_state, noisy_features, observation,
                        predicted, error, dynamic_error, corrections, output_size):
    posterior = direct_posterior(
        observation, error, dynamic_error, corrections
    )
    clean_host = HostFeature(
        tensor=clean_features.c4,
        low_level=clean_features.c1,
        output_size=output_size,
    )
    corrected_host = corrected_host_feature(
        model,
        noisy_features,
        observation,
        posterior,
        output_size,
    )
    clean_logits = model.decode_from_host_feature(clean_host)
    corrected_logits = model.decode_from_host_feature(corrected_host)
    semantic_loss = semantic_kl_loss(clean_logits, corrected_logits)
    state_loss = correction_loss(posterior, clean_state) * 0.5
    return semantic_loss, state_loss, posterior


def run_epoch(model, predictor, corrections, groups, optimizer, training, mode):
    model.eval()
    predictor.eval()
    corrections.train(training)
    totals = {"semantic": 0.0, "state": 0.0, "total": 0.0, "frames": 0}
    for samples in groups.values():
        previous_previous = None
        previous = None
        dynamic_error = None
        for sample in samples:
            clean_image = load_image(sample)
            noisy_image = add_frame_noise(clean_image, SIGMA)
            with torch.no_grad():
                clean_features = model.extract_backbone_features(clean_image)
                noisy_features = model.extract_backbone_features(noisy_image)
                clean_state = model.encode_backbone_features(clean_features)
                observation = model.encode_backbone_features(noisy_features)
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
                if mode == "observation_only":
                    error_input = torch.zeros_like(error.z1)
                    dynamic_input = torch.zeros_like(dynamic_error.z1)
                    error_state = type(error)(
                        error_input,
                        torch.zeros_like(error.z2),
                        torch.zeros_like(error.z3),
                        torch.zeros_like(error.z4),
                    )
                    dynamic_state = type(dynamic_error)(
                        dynamic_input,
                        torch.zeros_like(dynamic_error.z2),
                        torch.zeros_like(dynamic_error.z3),
                        torch.zeros_like(dynamic_error.z4),
                    )
                else:
                    error_state = error
                    dynamic_state = dynamic_error
            semantic_loss, state_loss, posterior = semantic_state_loss(
                model,
                clean_features,
                clean_state,
                noisy_features,
                observation,
                predicted,
                error_state,
                dynamic_state,
                corrections,
                tuple(clean_image.shape[-2:]),
            )
            total_loss = semantic_loss + 10.0 * state_loss
            if training:
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                optimizer.step()
            totals["semantic"] += semantic_loss.detach().item()
            totals["state"] += state_loss.detach().item()
            totals["total"] += total_loss.detach().item()
            totals["frames"] += 1
            dynamic_error = type(dynamic_error)(
                *(value.detach() for value in dynamic_error.as_tuple())
            )
            previous_previous = previous
            previous = type(observation)(
                *(value.detach() for value in observation.as_tuple())
            )
    frames = totals["frames"]
    return {key: value / frames for key, value in totals.items() if key != "frames"}, frames


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Semantic direct correction training requires CUDA.")
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    mode = os.environ.get("PREDIFY_KITTI_SEMANTIC_CORRECTION_MODE", "full_error")
    if mode not in ("full_error", "observation_only"):
        raise RuntimeError("Unknown semantic correction mode.")
    root = Path(os.environ.get("PREDIFY_KITTI_STEP_ROOT", "/home/lin/predify/kitti_step"))
    output_dir = Path(
        os.environ.get(
            "PREDIFY_KITTI_SEMANTIC_CORRECTION_OUTPUT_DIR",
            f"/home/lin/predify/experiments/kitti_step_semantic_direct_{mode}",
        )
    )
    static_checkpoint = Path(os.environ.get("PREDIFY_KITTI_STEP_STATIC_CHECKPOINT", STATIC_CHECKPOINT_DEFAULT))
    adapter_checkpoint = Path(os.environ.get("PREDIFY_KITTI_STEP_ADAPTER_CHECKPOINT", ADAPTER_CHECKPOINT_DEFAULT))
    predictor_checkpoint = Path(os.environ.get("PREDIFY_KITTI_STEP_PREDICTOR_CHECKPOINT", PREDICTOR_CHECKPOINT_DEFAULT))
    writeback_checkpoint = Path(os.environ.get("PREDIFY_KITTI_STEP_WRITEBACK_CHECKPOINT", WRITEBACK_CHECKPOINT_DEFAULT))
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
    parameters = list(corrections.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(root, "train")
    groups = sequence_groups(dataset)
    if len(groups) != EXPECTED_TRAIN_SEQUENCES or len(dataset.samples) != EXPECTED_TRAIN_FRAMES:
        raise RuntimeError("KITTI-STEP train protocol mismatch.")
    history = []
    for epoch in range(1, EPOCHS + 1):
        averages, frame_count = run_epoch(
            model, predictor, corrections, groups, optimizer, True, mode
        )
        history.append({"epoch": epoch, **averages, "trained_frame_count": frame_count})
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "semantic_direct_correction_epoch3.pt"
    torch.save(
        {
            "mode": mode,
            "corrections": {str(index): corrections[position].state_dict() for position, index in enumerate(CORRECTION_INDICES)},
            "epoch": EPOCHS,
            "config": {"epochs": EPOCHS, "sigma": SIGMA, "seed": SEED, "optimizer": "AdamW", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY},
        },
        checkpoint_path,
    )
    summary = {
        "experiment": "kitti_step_semantic_direct_correction_training",
        "git_revision": os.environ.get("PREDIFY_GIT_REVISION"),
        "mode": mode,
        "checkpoint": str(checkpoint_path),
        "base_writeback_checkpoint": str(writeback_checkpoint),
        "dataset": {"split": "train", "sequence_count": len(groups), "frame_count": len(dataset.samples), "effective_frame_count_per_epoch": history[0]["trained_frame_count"], "labels_used": False},
        "trainable_parameter_count": sum(parameter.numel() for parameter in parameters),
        "history": history,
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
