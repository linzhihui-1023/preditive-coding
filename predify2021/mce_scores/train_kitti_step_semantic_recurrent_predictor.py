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
from predify2021.mce_scores.train_kitti_step_state_predictor import load_static_kitti_checkpoint
from predify2021.model_factory.deeplabv3plus_resnet50 import HostFeature, SemanticRecurrentPredictor, UnifiedFeatures, build_deeplabv3plus_resnet50_host

PREDICTOR_CHECKPOINT = "/home/lin/predify/experiments/kitti_step_fixed_adapter_predictor_41aa5cc/best_predictor.pt"
WRITEBACK_CHECKPOINT = "/home/lin/predify/experiments/kitti_step_host_conditioned_writeback/host_conditioned_writeback_epoch3.pt"
STATE_LOSS_WEIGHT = 1593.4333312535177
VALIDATION_CLIP_LIMIT = 64


def clips(groups, length=8):
    return [samples[index:index + length] for samples in groups.values() for index in range(0, len(samples) - length + 1, length)]


def zero_error(state):
    return UnifiedFeatures(*(torch.zeros_like(value) for value in state.as_tuple()))


def prediction_step(model, states, output_size):
    decoded = model.decode_adapter_deltas(states)
    return HostFeature(decoded.c4, decoded.c1, output_size)


def clip_loss(model, predictor, samples, training, semantic_normalized=True):
    images = torch.cat([load_image(sample) for sample in samples], dim=0)
    with torch.no_grad():
        features = model.extract_backbone_features(images)
        states = model.encode_backbone_features(features)
        teacher = model.decode_from_host_feature(HostFeature(features.c4, features.c1, tuple(images.shape[-2:])))
    hidden4, hidden1 = predictor.initial_state()
    error = zero_error(UnifiedFeatures(*(value[:1] for value in states.as_tuple())))
    semantic_total = states.z1.new_zeros(())
    state_total = states.z1.new_zeros(())
    for index in range(len(samples) - 1):
        current = UnifiedFeatures(*(value[index:index + 1] for value in states.as_tuple()))
        target = UnifiedFeatures(*(value[index + 1:index + 2] for value in states.as_tuple()))
        predicted, hidden4, hidden1 = predictor.step(current, error, hidden4, hidden1)
        predicted_host = prediction_step(model, predicted, tuple(images.shape[-2:]))
        predicted_logits = model.decode_from_host_feature(predicted_host)
        target_logits = teacher[index + 1:index + 2]
        log_prediction = F.log_softmax(predicted_logits, dim=1)
        target_distribution = F.softmax(target_logits, dim=1)
        if semantic_normalized:
            semantic_total = semantic_total + F.kl_div(log_prediction, target_distribution, reduction="none").sum(dim=1).mean()
        else:
            semantic_total = semantic_total + F.kl_div(log_prediction, target_distribution, reduction="batchmean")
        state_total = state_total + 0.5 * (F.mse_loss(predicted.z1, target.z1) + F.mse_loss(predicted.z4, target.z4))
        error = UnifiedFeatures(*(value.detach() for value in (target.z1 - predicted.z1, target.z2 - predicted.z2, target.z3 - predicted.z3, target.z4 - predicted.z4)))
    return semantic_total / 7, state_total / 7


def run_epoch(model, predictor, clip_list, optimizer):
    predictor.train(optimizer is not None)
    semantic = state = 0.0
    for samples in clip_list:
        semantic_loss, state_loss = clip_loss(model, predictor, samples, optimizer is not None)
        total = semantic_loss + STATE_LOSS_WEIGHT * state_loss
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
        semantic += semantic_loss.detach().item(); state += state_loss.detach().item()
    count = len(clip_list)
    return {"semantic_loss": semantic / count, "state_loss": state / count, "total_loss": (semantic + STATE_LOSS_WEIGHT * state) / count, "clip_count": count}


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Semantic recurrent predictor training requires CUDA")
    random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    root = Path(os.environ.get("PREDIFY_KITTI_STEP_ROOT", "/home/lin/predify/kitti_step"))
    output = Path(os.environ.get("PREDIFY_SEMANTIC_RECURRENT_OUTPUT_DIR", "/home/lin/predify/experiments/kitti_step_semantic_recurrent_predictor"))
    model = build_deeplabv3plus_resnet50_host().cuda(); load_static_kitti_checkpoint(model, Path(STATIC_CHECKPOINT_DEFAULT))
    adapter = torch.load(ADAPTER_CHECKPOINT_DEFAULT, map_location="cpu", weights_only=False); model.multi_layer_adapter.load_state_dict(adapter["adapter_state_dict"], strict=True); load_writeback_checkpoint(model, Path(WRITEBACK_CHECKPOINT)); model.requires_grad_(False); model.eval()
    predictor = SemanticRecurrentPredictor().cuda(); optimizer = torch.optim.AdamW(predictor.parameters(), lr=1e-4, weight_decay=0.01)
    train = KITTISTEPSegmentationDataset.from_kitti_step_root(root, "train"); val = KITTISTEPSegmentationDataset.from_kitti_step_root(root, "val")
    train_clips = clips(sequence_groups(train)); val_clips = clips(sequence_groups(val))[:VALIDATION_CLIP_LIMIT]; history=[]; best=None
    for epoch in range(1, 4):
        train_metrics = run_epoch(model, predictor, train_clips, optimizer)
        with torch.no_grad(): val_metrics = run_epoch(model, predictor, val_clips, None)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}; history.append(row); print(json.dumps(row, sort_keys=True), flush=True)
        if best is None or val_metrics["total_loss"] < best["val"]["total_loss"]:
            best = row; output.mkdir(parents=True, exist_ok=True); torch.save({"predictor_state_dict": predictor.state_dict(), "epoch": epoch, "val_metrics": val_metrics}, output / "best_semantic_recurrent_predictor.pt")
    summary = {"experiment":"kitti_step_semantic_recurrent_predictor", "git_revision":os.environ.get("PREDIFY_GIT_REVISION"), "config":{"epochs":3,"clip_length":8,"batch_size":1,"optimizer":"AdamW","learning_rate":1e-4,"weight_decay":0.01,"seed":0,"labels_used":False,"semantic_loss":"per_pixel_mean_kl","state_loss_weight":STATE_LOSS_WEIGHT,"validation_clip_limit":VALIDATION_CLIP_LIMIT}, "trainable_parameter_count":sum(p.numel() for p in predictor.parameters()), "dataset":{"train_clip_count":len(train_clips),"val_clip_count":len(val_clips)}, "history":history,"best":best,"checkpoint":str(output / "best_semantic_recurrent_predictor.pt")}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__": main()
