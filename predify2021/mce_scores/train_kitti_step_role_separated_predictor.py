import json
import os
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import load_image, sequence_groups
from predify2021.mce_scores.evaluate_kitti_step_oracle_upper_bound import load_writeback_checkpoint
from predify2021.mce_scores.train_kitti_step_fixed_adapter_predictor import ADAPTER_CHECKPOINT_DEFAULT, STATIC_CHECKPOINT_DEFAULT
from predify2021.mce_scores.train_kitti_step_semantic_recurrent_predictor import (
    STATE_LOSS_WEIGHT,
    VALIDATION_CLIP_LIMIT,
    clips,
    prediction_step,
    zero_error,
)
from predify2021.mce_scores.train_kitti_step_state_predictor import load_static_kitti_checkpoint
from predify2021.model_factory.deeplabv3plus_resnet50 import (
    HostFeature,
    RoleSeparatedRecurrentPredictor,
    UnifiedFeatures,
    build_deeplabv3plus_resnet50_host,
)

WRITEBACK_CHECKPOINT = "/home/lin/predify/experiments/kitti_step_host_conditioned_writeback/host_conditioned_writeback_epoch3.pt"


def role_separated_clip_loss(model, predictor, samples):
    images = torch.cat([load_image(sample) for sample in samples], dim=0)
    with torch.no_grad():
        features = model.extract_backbone_features(images)
        states = model.encode_backbone_features(features)
        teacher = model.decode_from_host_feature(HostFeature(features.c4, features.c1, tuple(images.shape[-2:])))
    h4_dyn, h4_sem, h1_dyn, h1_sem = predictor.initial_state()
    dynamics_error = zero_error(UnifiedFeatures(*(value[:1] for value in states.as_tuple())))
    semantic_total = states.z1.new_zeros(())
    dynamics_total = states.z1.new_zeros(())
    for index in range(len(samples) - 1):
        current = UnifiedFeatures(*(value[index:index + 1] for value in states.as_tuple()))
        target = UnifiedFeatures(*(value[index + 1:index + 2] for value in states.as_tuple()))
        dynamics, diagnostic, h4_dyn, h4_sem, h1_dyn, h1_sem = predictor.step(
            current, dynamics_error, h4_dyn, h4_sem, h1_dyn, h1_sem
        )
        diagnostic_logits = model.decode_from_host_feature(prediction_step(model, diagnostic, tuple(images.shape[-2:])))
        semantic_total = semantic_total + F.kl_div(
            F.log_softmax(diagnostic_logits, dim=1),
            F.softmax(teacher[index + 1:index + 2], dim=1),
            reduction="none",
        ).sum(dim=1).mean()
        dynamics_total = dynamics_total + 0.5 * (
            F.mse_loss(dynamics.z1, target.z1) + F.mse_loss(dynamics.z4, target.z4)
        )
        dynamics_error = UnifiedFeatures(*(value.detach() for value in (
            target.z1 - dynamics.z1,
            target.z2 - dynamics.z2,
            target.z3 - dynamics.z3,
            target.z4 - dynamics.z4,
        )))
    return semantic_total / 7, dynamics_total / 7


def run_epoch(model, predictor, clip_list, optimizer):
    predictor.train(optimizer is not None)
    semantic = dynamics = 0.0
    for samples in clip_list:
        semantic_loss, dynamics_loss = role_separated_clip_loss(model, predictor, samples)
        total = semantic_loss + STATE_LOSS_WEIGHT * dynamics_loss
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
        semantic += semantic_loss.detach().item()
        dynamics += dynamics_loss.detach().item()
    count = len(clip_list)
    return {
        "clip_count": count,
        "semantic_loss": semantic / count,
        "dynamics_loss": dynamics / count,
        "combined_loss": (semantic + STATE_LOSS_WEIGHT * dynamics) / count,
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Role-separated predictor training requires CUDA")
    random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    root = Path(os.environ.get("PREDIFY_KITTI_STEP_ROOT", "/home/lin/predify/kitti_step"))
    output = Path(os.environ.get("PREDIFY_ROLE_SEPARATED_OUTPUT_DIR", "/home/lin/predify/experiments/kitti_step_role_separated_predictor"))
    model = build_deeplabv3plus_resnet50_host().cuda()
    load_static_kitti_checkpoint(model, Path(STATIC_CHECKPOINT_DEFAULT))
    adapter = torch.load(ADAPTER_CHECKPOINT_DEFAULT, map_location="cpu", weights_only=False)
    model.multi_layer_adapter.load_state_dict(adapter["adapter_state_dict"], strict=True)
    load_writeback_checkpoint(model, Path(WRITEBACK_CHECKPOINT))
    model.requires_grad_(False); model.eval()
    predictor = RoleSeparatedRecurrentPredictor().cuda()
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=1e-4, weight_decay=0.01)
    train = KITTISTEPSegmentationDataset.from_kitti_step_root(root, "train")
    val = KITTISTEPSegmentationDataset.from_kitti_step_root(root, "val")
    train_clips = clips(sequence_groups(train))
    val_clips = clips(sequence_groups(val))[:VALIDATION_CLIP_LIMIT]
    history = []
    best = None
    for epoch in range(1, 4):
        train_metrics = run_epoch(model, predictor, train_clips, optimizer)
        with torch.no_grad():
            val_metrics = run_epoch(model, predictor, val_clips, None)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if best is None or val_metrics["combined_loss"] < best["val"]["combined_loss"]:
            best = row
            output.mkdir(parents=True, exist_ok=True)
            torch.save({"predictor_state_dict": predictor.state_dict(), "epoch": epoch, "val_metrics": val_metrics}, output / "best_role_separated_predictor.pt")
    summary = {
        "experiment": "kitti_step_role_separated_predictor",
        "git_revision": os.environ.get("PREDIFY_GIT_REVISION"),
        "checkpoint": str(output / "best_role_separated_predictor.pt"),
        "trainable_parameter_count": sum(parameter.numel() for parameter in predictor.parameters()),
        "config": {"epochs": 3, "clip_length": 8, "batch_size": 1, "optimizer": "AdamW", "learning_rate": 1e-4, "weight_decay": 0.01, "seed": 0, "labels_used": False, "state_loss_weight": STATE_LOSS_WEIGHT, "validation_clip_limit": VALIDATION_CLIP_LIMIT},
        "dataset": {"train_clip_count": len(train_clips), "validation_clip_count": len(val_clips)},
        "history": history,
        "best": best,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
