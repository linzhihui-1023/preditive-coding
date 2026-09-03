"""Clean-task temporal training for FAST-B's existing architecture.

Only segmentation supervision is optimized.  The backbone, decoder, dynamics
branch, C1 path, and all unrelated modules remain frozen; the trainable set is
the same semantic/error/C4 path used by FAST-B.  No corruption is constructed
in this module.
"""

import argparse
import json
import os
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset, semantic_mask_from_panoptic_png
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import load_image, sequence_groups
from predify2021.mce_scores.evaluate_kitti_step_static_baseline import compute_iou, update_confusion_matrix
from predify2021.mce_scores.kitti_step_persistent_blur import warmup_frame_count
from predify2021.mce_scores.role_separated_dynamic_error_correction import (
    ADAPTER_CHECKPOINT_DEFAULT, ROLE_PREDICTOR_CHECKPOINT_DEFAULT,
    STATIC_CHECKPOINT_DEFAULT, WRITEBACK_CHECKPOINT_DEFAULT,
    error_state, load_components, residual_writeback_host_feature, zero_state,
)
from predify2021.mce_scores.evaluate_kitti_step_oracle_upper_bound import load_writeback_checkpoint
from predify2021.model_factory.deeplabv3plus_resnet50 import (
    ErrorRegulatedSemanticRestorationPredictor, HostFeature, UnifiedFeatures,
)


SEED = 0
EPOCHS = 3
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
TBPTT_STEPS = 8
# Dense supervision: every decoded frame contributes to the task objective.
SEGMENTATION_SUPERVISION_POSITIONS = "every_frame"
VALIDATION_SEQUENCES = ("0002", "0010", "0018")
OUTPUT_DEFAULT = "/home/lin/predify/experiments/kitti_step_clean_task_temporal_fast"
RESULT_DEFAULT = "results/kitti_step_clean_task_temporal_fast"


def configure_trainable_path(model, predictor):
    model.requires_grad_(False)
    predictor.requires_grad_(False)
    predictor.semantic_error_encoder.requires_grad_(True)
    predictor.semantic_state_cell.requires_grad_(True)
    predictor.semantic_restoration_head.requires_grad_(True)
    model.multi_layer_adapter.output_adapters[3].requires_grad_(True)
    model.host_conditioned_writebacks["3"].requires_grad_(True)


def trainable_parameters(model, predictor):
    modules = (
        predictor.semantic_error_encoder,
        predictor.semantic_state_cell,
        predictor.semantic_restoration_head,
        model.multi_layer_adapter.output_adapters[3],
        model.host_conditioned_writebacks["3"],
    )
    return [p for module in modules for p in module.parameters() if p.requires_grad]


def assert_freeze_contract(model, predictor):
    groups = {
        "backbone": model.backbone,
        "decoder": model.decode_head,
        "c1_output_adapter": model.multi_layer_adapter.output_adapters[0],
        "c1_writeback": model.host_conditioned_writebacks["0"],
    }
    for name in predictor.DYNAMICS_MODULES:
        groups[f"dynamics_{name}"] = getattr(predictor, name)
    bad = [name for name, module in groups.items() if any(p.requires_grad for p in module.parameters())]
    if bad:
        raise RuntimeError(f"Frozen-module contract violated: {bad}")


def checkpoint_payload(model, predictor, args, epoch):
    return {
        "experiment": "kitti_step_clean_task_temporal_fast",
        "model_state_dict": predictor.state_dict(),
        "c4_output_adapter_state_dict": model.multi_layer_adapter.output_adapters[3].state_dict(),
        "c4_writeback_state_dict": model.host_conditioned_writebacks["3"].state_dict(),
        "source_dynamics_checkpoint": args.dynamics_checkpoint,
        "epoch": epoch,
        "use_error_temporal_stats": True,
        "joint_c4_training": True,
        "clean_task_training": True,
        "lambda_feature": 0.0,
        "lambda_segmentation": 1.0,
        "tbptt_steps": TBPTT_STEPS,
        "segmentation_supervision_positions": SEGMENTATION_SUPERVISION_POSITIONS,
    }


def encode_clean(model, sample):
    image = load_image(sample)
    with torch.no_grad():
        raw = model.extract_backbone_features(image)
        observation = model.encode_backbone_features(raw)
    return observation, raw, tuple(image.shape[-2:])


def decode_records(model, records):
    features, masks = [], []
    for raw, observation, restored, size, mask in records:
        zero = zero_state(observation)
        delta = UnifiedFeatures(zero.z1, zero.z2, zero.z3, restored.z4 - observation.z4)
        features.append(residual_writeback_host_feature(model, raw, delta, size))
        masks.append(mask)
    host = HostFeature(
        torch.cat([feature.tensor for feature in features], dim=0),
        torch.cat([feature.low_level for feature in features], dim=0),
        features[0].output_size,
    )
    return model.decode_from_host_feature(host), torch.stack(masks).to(host.tensor.device)


def diagnostics_row(diags, model, raw, observation, restored):
    delta_z4 = restored.z4 - observation.z4
    delta_c4 = residual_writeback_host_feature(
        model, raw,
        UnifiedFeatures(torch.zeros_like(observation.z1), torch.zeros_like(observation.z2), torch.zeros_like(observation.z3), delta_z4),
        tuple(raw.c4.shape[-2:]),
    ).tensor - raw.c4
    gain = diags["semantic_update_gain"].detach()
    return {
        "mean_abs_delta_z4": float(delta_z4.detach().abs().mean().item()),
        "mean_abs_delta_c4": float(delta_c4.detach().abs().mean().item()),
        "semantic_state_difference": float((diags["semantic_hidden"].detach() - observation.z4).abs().mean().item()),
        "semantic_update_gain_mean": float(gain.mean().item()),
        "semantic_update_gain_std": float(gain.std(unbiased=False).item()),
    }


def add_diag(total, row):
    for key, value in row.items():
        total[key] = total.get(key, 0.0) + value


def train_sequence(model, predictor, samples, optimizer, max_steps=0):
    if len(samples) < 2:
        return {"steps": 0, "windows": 0, "segmentation_loss": 0.0, "diagnostics": {}}
    first, _, _ = encode_clean(model, samples[0])
    pending, h4_dyn, h1_dyn = predictor.predict_next(first, zero_state(first), None, None)
    semantic_hidden = predictor.initial_semantic_state(first)
    error_stats = predictor.initial_error_temporal_statistics()
    losses, records = [], []
    metrics = {"steps": 0, "windows": 0, "segmentation_loss": 0.0, "diagnostics": {}}
    steps = min(len(samples) - 1, max_steps) if max_steps else len(samples) - 1
    for offset in range(steps):
        observation, raw, size = encode_clean(model, samples[offset + 1])
        prediction_error = error_state(observation, pending)
        restored, semantic_hidden, diag = predictor.restore_current(
            observation, pending, semantic_hidden, error_temporal_state=error_stats
        )
        error_stats = diag["error_temporal_state"]
        losses.append(None)  # the sole objective is assembled from segmentation records
        local_position = len(losses)
        records.append((raw, observation, restored, size, semantic_mask_from_panoptic_png(samples[offset + 1]["mask_path"])))
        add_diag(metrics["diagnostics"], diagnostics_row(diag, model, raw, observation, restored))
        metrics["steps"] += 1
        with torch.no_grad():
            pending, h4_dyn, h1_dyn = predictor.predict_next(observation, prediction_error, h4_dyn, h1_dyn)
        if local_position == TBPTT_STEPS or offset + 1 == steps:
            logits, masks = decode_records(model, records)
            loss = F.cross_entropy(logits, masks, ignore_index=255)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            metrics["windows"] += 1
            metrics["segmentation_loss"] += float(loss.detach().item())
            semantic_hidden = semantic_hidden.detach()
            error_stats = error_stats.detach() if error_stats is not None else None
            pending = UnifiedFeatures(*(value.detach() for value in pending.as_tuple()))
            h4_dyn, h1_dyn = h4_dyn.detach(), h1_dyn.detach()
            losses, records = [], []
    return metrics


def gradient_smoke(model, predictor, samples):
    model.zero_grad(set_to_none=True)
    predictor.zero_grad(set_to_none=True)
    optimizer = torch.optim.AdamW(trainable_parameters(model, predictor), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    row = train_sequence(model, predictor, samples[: min(len(samples), 10)], optimizer)
    def norm(modules):
        values = [p.grad.detach().norm() for module in modules for p in module.parameters() if p.grad is not None]
        return float(torch.stack(values).norm().item()) if values else 0.0
    result = {
        "semantic_grad_norm": norm((predictor.semantic_state_cell, predictor.semantic_restoration_head, predictor.semantic_error_encoder)),
        "error_regulation_grad_norm": norm((predictor.semantic_state_cell.history_projection,)) if predictor.semantic_state_cell.history_projection is not None else 0.0,
        "c4_adapter_grad_norm": norm((model.multi_layer_adapter.output_adapters[3],)),
        "c4_writeback_grad_norm": norm((model.host_conditioned_writebacks["3"],)),
        "backbone_grad_norm": norm((model.backbone,)),
        "decoder_grad_norm": norm((model.decode_head,)),
        "dynamics_grad_norm": norm(tuple(getattr(predictor, name) for name in predictor.DYNAMICS_MODULES)),
        "c1_adapter_grad_norm": norm((model.multi_layer_adapter.output_adapters[0],)),
        "c1_writeback_grad_norm": norm((model.host_conditioned_writebacks["0"],)),
        "loss_finite": bool(torch.isfinite(torch.tensor(row["segmentation_loss"])).item()),
    }
    if any(result[key] <= 0 or not torch.isfinite(torch.tensor(result[key])) for key in ("semantic_grad_norm", "error_regulation_grad_norm", "c4_adapter_grad_norm", "c4_writeback_grad_norm")):
        raise RuntimeError(f"Gradient smoke failed: {result}")
    if any(result[key] != 0 for key in ("backbone_grad_norm", "decoder_grad_norm", "dynamics_grad_norm", "c1_adapter_grad_norm", "c1_writeback_grad_norm")):
        raise RuntimeError(f"Frozen branch received gradient: {result}")
    model.zero_grad(set_to_none=True); predictor.zero_grad(set_to_none=True)
    return result


def evaluate_clean(model, predictor, groups):
    confusion = {name: torch.zeros((19, 19), dtype=torch.int64) for name in ("host", "ours")}
    effective = 0
    sequence_counts = {}
    predictor.eval()
    with torch.inference_mode():
        for sequence_id, samples in groups.items():
            pending = h4_dyn = h1_dyn = semantic_hidden = error_stats = None
            onset = warmup_frame_count(len(samples))
            count = 0
            for frame_index, sample in enumerate(samples):
                observation, raw, size = encode_clean(model, sample)
                if semantic_hidden is None:
                    semantic_hidden = predictor.initial_semantic_state(observation)
                if frame_index == 0:
                    pending, h4_dyn, h1_dyn = predictor.predict_next(observation, zero_state(observation), None, None)
                    continue
                error = error_state(observation, pending)
                if frame_index == 1:
                    pending, h4_dyn, h1_dyn = predictor.predict_next(observation, error, h4_dyn, h1_dyn)
                    continue
                restored, semantic_hidden, diag = predictor.restore_current(observation, pending, semantic_hidden, error_temporal_state=error_stats)
                error_stats = diag["error_temporal_state"]
                if frame_index >= onset:
                    host = HostFeature(raw.c4, raw.c1, size)
                    zero = zero_state(observation)
                    delta = UnifiedFeatures(zero.z1, zero.z2, zero.z3, restored.z4 - observation.z4)
                    ours = residual_writeback_host_feature(model, raw, delta, size)
                    predictions = {"host": model.decode_from_host_feature(host).argmax(1).squeeze(0).cpu(), "ours": model.decode_from_host_feature(ours).argmax(1).squeeze(0).cpu()}
                    mask = semantic_mask_from_panoptic_png(sample["mask_path"])
                    for name, prediction in predictions.items(): update_confusion_matrix(confusion[name], prediction, mask)
                    effective += 1; count += 1
                pending, h4_dyn, h1_dyn = predictor.predict_next(observation, error, h4_dyn, h1_dyn)
                semantic_hidden = semantic_hidden.detach(); pending = UnifiedFeatures(*(v.detach() for v in pending.as_tuple()))
                h4_dyn, h1_dyn = h4_dyn.detach(), h1_dyn.detach()
                error_stats = error_stats.detach() if error_stats is not None else None
            sequence_counts[sequence_id] = {"total_frame_count": len(samples), "effective_frame_count": count, "warmup_frame_count": onset}
    metrics = {name: float(torch.nanmean(compute_iou(value)).item()) for name, value in confusion.items()}
    return {"host_clean_mIoU": metrics["host"], "ours_clean_mIoU": metrics["ours"], "delta_clean_mIoU": metrics["ours"] - metrics["host"], "effective_frame_count": effective, "sequence_frame_counts": sequence_counts}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/lin/predify/kitti_step")
    parser.add_argument("--output", default=OUTPUT_DEFAULT)
    parser.add_argument("--result-output", default=RESULT_DEFAULT)
    parser.add_argument("--dynamics-checkpoint", default=ROLE_PREDICTOR_CHECKPOINT_DEFAULT)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-check-only", action="store_true")
    args = parser.parse_args(argv)
    if not torch.cuda.is_available(): raise RuntimeError("Clean-task training requires CUDA")
    random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    model, source = load_components(STATIC_CHECKPOINT_DEFAULT, ADAPTER_CHECKPOINT_DEFAULT, args.dynamics_checkpoint, WRITEBACK_CHECKPOINT_DEFAULT)
    payload = torch.load(Path(os.environ.get("PREDIFY_FAST_B_CHECKPOINT", "/home/lin/predify/experiments/kitti_step_semantic_v3_joint_c4_fast_ab_5754714/fast_b_joint_c4_weak_z4/best.pt")), map_location="cpu", weights_only=False)
    model.multi_layer_adapter.output_adapters[3].load_state_dict(payload["c4_output_adapter_state_dict"], strict=True)
    model.host_conditioned_writebacks["3"].load_state_dict(payload["c4_writeback_state_dict"], strict=True)
    predictor = ErrorRegulatedSemanticRestorationPredictor(use_error_temporal_stats=True).cuda()
    predictor.load_dynamics_from_role_separated_state_dict(source.state_dict())
    configure_trainable_path(model, predictor); model.eval(); predictor.eval(); assert_freeze_contract(model, predictor)
    dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "train")
    train_groups = sequence_groups(dataset)
    val_dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "val")
    val_all = sequence_groups(val_dataset)
    val_groups = {key: val_all[key] for key in VALIDATION_SEQUENCES}
    smoke = gradient_smoke(model, predictor, next(iter(train_groups.values())))
    if args.smoke_check_only:
        print(json.dumps({"gradient_smoke": smoke}, sort_keys=True)); return
    if args.smoke:
        train_groups = dict(list(train_groups.items())[:1]); epochs = 1; max_steps = 16
    else:
        epochs = args.epochs; max_steps = 0
    optimizer = torch.optim.AdamW(trainable_parameters(model, predictor), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    output = Path(args.output); result_output = Path(args.result_output); output.mkdir(parents=True, exist_ok=True); result_output.mkdir(parents=True, exist_ok=True)
    history, best = [], None
    for epoch in range(1, epochs + 1):
        predictor.train(); train_metrics = {"steps": 0, "windows": 0, "segmentation_loss": 0.0, "diagnostics": {}}
        for samples in train_groups.values():
            row = train_sequence(model, predictor, samples, optimizer, max_steps)
            train_metrics["steps"] += row["steps"]; train_metrics["windows"] += row["windows"]; train_metrics["segmentation_loss"] += row["segmentation_loss"]; add_diag(train_metrics["diagnostics"], row["diagnostics"])
        for key in train_metrics["diagnostics"]: train_metrics["diagnostics"][key] /= max(train_metrics["steps"], 1)
        if train_metrics["windows"]: train_metrics["segmentation_loss"] /= train_metrics["windows"]
        val = evaluate_clean(model, predictor, val_groups)
        record = {"epoch": epoch, "segmentation_loss": train_metrics["segmentation_loss"], **val, **train_metrics["diagnostics"]}
        history.append(record); print(json.dumps(record, sort_keys=True), flush=True)
        torch.save(checkpoint_payload(model, predictor, args, epoch), output / f"epoch_{epoch:03d}.pt")
        if best is None or (record["ours_clean_mIoU"], -record["segmentation_loss"]) > (best["ours_clean_mIoU"], -best["segmentation_loss"]):
            best = record; torch.save(checkpoint_payload(model, predictor, args, epoch), output / "best.pt")
    summary = {
        "experiment_name": "kitti_step_clean_task_temporal_fast", "git_revision": os.environ.get("PREDIFY_GIT_REVISION"), "checkpoint": str(output / "best.pt"), "epochs": epochs, "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "tbptt_steps": TBPTT_STEPS, "segmentation_supervision_positions": SEGMENTATION_SUPERVISION_POSITIONS, "train_sequence_count": len(train_groups), "validation_sequences": VALIDATION_SEQUENCES, "trainable_parameter_count": sum(p.numel() for p in trainable_parameters(model, predictor)), "backbone_trainable": False, "decoder_trainable": False, "dynamics_trainable": False, "c1_trainable": False, "corruption_training": False, "test_time_adaptation": False, "gradient_smoke": smoke, "decision": "GO" if best["delta_clean_mIoU"] > 0 else "HARMFUL_TEMPORAL_CORRECTION" if best["mean_abs_delta_z4"] > 0 and best["mean_abs_delta_c4"] > 0 else "CORRECTION_COLLAPSE", "epoch_history": history, "best": best,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (result_output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"checkpoint": str(output / "best.pt"), "best": best}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
