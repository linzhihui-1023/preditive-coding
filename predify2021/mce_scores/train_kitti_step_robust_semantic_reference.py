import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from predify2021.datasets.kitti_step import (
    KITTISTEPSegmentationDataset,
    pil_rgb_to_unit_tensor,
    semantic_mask_from_panoptic_png,
)
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import load_image, sequence_groups
from predify2021.mce_scores.evaluate_kitti_step_static_baseline import compute_iou, update_confusion_matrix
from predify2021.mce_scores.kitti_step_cityscapes_c import (
    CITYSCAPES_C_COMMON_CORRUPTIONS,
    CITYSCAPES_C_SEVERITIES,
    apply_cityscapes_c_corruption_uint8,
    corruption_seed,
)
from predify2021.mce_scores.role_separated_dynamic_error_correction import (
    ADAPTER_CHECKPOINT_DEFAULT,
    ROLE_PREDICTOR_CHECKPOINT_DEFAULT,
    STATIC_CHECKPOINT_DEFAULT,
    WRITEBACK_CHECKPOINT_DEFAULT,
    detach_state,
    error_state,
    load_components,
    zero_state,
)
from predify2021.mce_scores.train_kitti_step_semantic_recurrent_predictor import prediction_step
from predify2021.model_factory.deeplabv3plus_resnet50 import HostFeature, UnifiedFeatures

SEED = 0
MAX_EPOCHS = 15
PATIENCE = 3
TBPTT_STEPS = 8
CALIBRATION_CHUNKS = 32
TRAIN_CORRUPTIONS = tuple(
    name for name in CITYSCAPES_C_COMMON_CORRUPTIONS if name != "glass_blur"
)
SEMANTIC_MODULES = ("z4_sem_recurrent", "z4_sem_delta", "z1_sem_recurrent", "z1_sem_delta")
DYNAMICS_MODULES = ("z4_dyn_recurrent", "z4_dyn_delta", "z1_dyn_recurrent", "z1_dyn_delta")


def slice_state(state, index):
    return UnifiedFeatures(*(x[index:index + 1] for x in state.as_tuple()))


def configure_trainable(predictor):
    predictor.requires_grad_(False)
    for name in SEMANTIC_MODULES:
        getattr(predictor, name).requires_grad_(True)
    semantic = [p for name in SEMANTIC_MODULES for p in getattr(predictor, name).parameters() if p.requires_grad]
    if not semantic:
        raise RuntimeError("Semantic Branch trainable parameters must be > 0")
    if any(p.requires_grad for name in DYNAMICS_MODULES for p in getattr(predictor, name).parameters()):
        raise RuntimeError("Dynamics Branch must remain frozen")
    return semantic


def detach_hidden(hidden):
    return tuple(x.detach() if x is not None else None for x in hidden)


def grad_norm(grads):
    total = sum((g.detach().float().pow(2).sum() for g in grads if g is not None), start=torch.tensor(0.0, device="cuda"))
    return float(torch.sqrt(total).item())


def sequence_condition(seed, epoch, sequence_index):
    rng = random.Random(seed + 1000003 * epoch + 7919 * sequence_index)
    return rng.choice(TRAIN_CORRUPTIONS), rng.choice(CITYSCAPES_C_SEVERITIES)


def load_clean_and_corrupted(sample, corruption, severity, seed):
    with Image.open(Path(sample["image_path"])) as opened:
        rgb = opened.convert("RGB")
        clean = pil_rgb_to_unit_tensor(rgb).unsqueeze(0).cuda()
        source = np.asarray(rgb, dtype=np.uint8)
    corrupted = apply_cityscapes_c_corruption_uint8(
        source, corruption, severity, seed=seed
    )
    noisy = (
        torch.from_numpy(corrupted)
        .permute(2, 0, 1)
        .to(device=clean.device, dtype=clean.dtype)
        .div(255.0)
        .unsqueeze(0)
    )
    return clean, noisy


def encode_pair(model, sample, corruption, severity, seed):
    clean, noisy = load_clean_and_corrupted(
        sample, corruption, severity, seed
    )
    images = torch.cat((clean, noisy), dim=0)
    size = tuple(clean.shape[-2:])
    with torch.no_grad():
        raw = model.extract_backbone_features(images)
        states = model.encode_backbone_features(raw)
        target = slice_state(states, 0)
        observation = slice_state(states, 1)
        clean_logits = model.decode_from_host_feature(HostFeature(raw.c4[:1], raw.c1[:1], size))
    return observation, target, clean_logits, size


def losses(model, semantic_reference, clean_target, clean_logits, size):
    reference_logits = model.decode_from_host_feature(prediction_step(model, semantic_reference, size))
    kd = F.kl_div(
        F.log_softmax(reference_logits, dim=1),
        F.softmax(clean_logits.detach(), dim=1),
        reduction="none",
    ).sum(dim=1).mean()
    ref = (
        F.smooth_l1_loss(semantic_reference.z1, clean_target.z1.detach())
        + F.smooth_l1_loss(semantic_reference.z4, clean_target.z4.detach())
    )
    return kd, ref, reference_logits


def corrupt_seed(seed, epoch, sequence_index, frame_index, frame_count, corruption, severity):
    base = seed + 10000019 * epoch + 100003 * sequence_index
    return corruption_seed(base, frame_count, frame_index, corruption, severity)


def calibrate_lambda(model, predictor, groups, semantic_parameters, chunks, tbptt_steps, seed, max_steps=0):
    predictor.train()
    kd_norms, ref_norms = [], []
    for sequence_index, samples in enumerate(groups.values()):
        if len(samples) < 2:
            continue
        corruption, severity = sequence_condition(seed, 0, sequence_index)
        hidden = predictor.initial_state()
        pending = None
        observation, _, _, _ = encode_pair(
            model, samples[0], corruption, severity,
            corrupt_seed(seed, 0, sequence_index, 0, len(samples), corruption, severity),
        )
        kd_chunk, ref_chunk = [], []
        steps = min(len(samples) - 1, max_steps) if max_steps else len(samples) - 1
        for t in range(steps):
            target_index = t + 1
            next_observation, clean_target, clean_logits, size = encode_pair(
                model, samples[target_index], corruption, severity,
                corrupt_seed(seed, 0, sequence_index, target_index, len(samples), corruption, severity),
            )
            prediction_error = zero_state(observation) if pending is None else error_state(observation, pending)
            dynamics, semantic_reference, *hidden = predictor.step(observation, prediction_error, *hidden)
            kd, ref, _ = losses(model, semantic_reference, clean_target, clean_logits, size)
            kd_chunk.append(kd); ref_chunk.append(ref)
            pending, observation = dynamics, next_observation
            if len(kd_chunk) == tbptt_steps or t + 1 == steps:
                kd_mean, ref_mean = torch.stack(kd_chunk).mean(), torch.stack(ref_chunk).mean()
                g_kd = torch.autograd.grad(kd_mean, semantic_parameters, retain_graph=True, allow_unused=True)
                g_ref = torch.autograd.grad(ref_mean, semantic_parameters, allow_unused=True)
                kd_norm, ref_norm = grad_norm(g_kd), grad_norm(g_ref)
                if kd_norm <= 0 or ref_norm <= 0 or not math.isfinite(kd_norm + ref_norm):
                    raise RuntimeError(f"Invalid semantic gradient norms: KD={kd_norm}, ref={ref_norm}")
                kd_norms.append(kd_norm); ref_norms.append(ref_norm)
                hidden = detach_hidden(hidden)
                pending = detach_state(pending)
                kd_chunk, ref_chunk = [], []
                if len(kd_norms) >= chunks:
                    break
        if len(kd_norms) >= chunks:
            break
    if not kd_norms:
        raise RuntimeError("Gradient calibration produced no chunks")
    median_kd = float(torch.tensor(kd_norms).median().item())
    median_ref = float(torch.tensor(ref_norms).median().item())
    return {
        "chunks": len(kd_norms),
        "median_kd_gradient_norm": median_kd,
        "median_ref_gradient_norm": median_ref,
        "lambda_ref": median_kd / median_ref,
    }


def run_sequence(
    model, predictor, samples, sequence_index, condition_epoch, lambda_ref, tbptt_steps,
    seed, optimizer=None, confusion=None, max_steps=0, check_dynamics_grad=False,
    smoke_probe=None,
):
    if len(samples) < 2:
        return 0, 0.0, 0.0, 0.0, check_dynamics_grad
    corruption, severity = sequence_condition(seed, condition_epoch, sequence_index)
    hidden = predictor.initial_state()
    pending = None
    observation, _, _, _ = encode_pair(
        model, samples[0], corruption, severity,
        corrupt_seed(seed, condition_epoch, sequence_index, 0, len(samples), corruption, severity),
    )
    steps = min(len(samples) - 1, max_steps) if max_steps else len(samples) - 1
    kd_sum = ref_sum = total_sum = 0.0
    kd_chunk, ref_chunk = [], []

    for t in range(steps):
        target_index = t + 1
        target_sample = samples[target_index]
        next_observation, clean_target, clean_logits, size = encode_pair(
            model, target_sample, corruption, severity,
            corrupt_seed(seed, condition_epoch, sequence_index, target_index, len(samples), corruption, severity),
        )
        prediction_error = zero_state(observation) if pending is None else error_state(observation, pending)
        dynamics, semantic_reference, *hidden = predictor.step(observation, prediction_error, *hidden)

        # step(O_t, E_t) predicts t+1: both supervision targets are frame t+1.
        kd, ref, reference_logits = losses(model, semantic_reference, clean_target, clean_logits, size)
        total = kd + lambda_ref * ref
        kd_sum += kd.detach().item(); ref_sum += ref.detach().item(); total_sum += total.detach().item()

        if confusion is not None:
            mask = semantic_mask_from_panoptic_png(target_sample["mask_path"])
            update_confusion_matrix(confusion, reference_logits.argmax(1).squeeze(0).detach().cpu(), mask)

        pending, observation = dynamics, next_observation
        if optimizer is not None:
            kd_chunk.append(kd); ref_chunk.append(ref)
            if len(kd_chunk) == tbptt_steps or t + 1 == steps:
                optimizer.zero_grad(set_to_none=True)
                (torch.stack(kd_chunk).mean() + lambda_ref * torch.stack(ref_chunk).mean()).backward()
                if check_dynamics_grad:
                    leaked = [
                        f"{name}.{param_name}"
                        for name in DYNAMICS_MODULES
                        for param_name, p in getattr(predictor, name).named_parameters()
                        if p.grad is not None
                    ]
                    if leaked:
                        raise RuntimeError(f"Dynamics Branch received gradients: {leaked}")
                    check_dynamics_grad = False
                optimizer.step()
                kd_chunk, ref_chunk = [], []

        # TBPTT detaches every 8 prediction steps; state values are never reset inside a sequence.
        if (t + 1) % tbptt_steps == 0 or t + 1 == steps:
            before_detach = hidden
            hidden = detach_hidden(hidden)
            pending = detach_state(pending)
            if smoke_probe is not None and "state_carry_checked" not in smoke_probe:
                if before_detach[1] is None or before_detach[3] is None:
                    raise RuntimeError("Semantic hidden state was reset at TBPTT boundary")
                max_diff = max(
                    float((before.detach() - after).abs().max().item())
                    for before, after in zip(before_detach, hidden)
                    if before is not None
                )
                if max_diff != 0.0:
                    raise RuntimeError(
                        f"TBPTT detach changed hidden-state values: max_diff={max_diff}"
                    )
                smoke_probe["state_carry_checked"] = True
                smoke_probe["tbptt_detach_max_abs_diff"] = max_diff
                smoke_probe["semantic_state_reset_inside_sequence"] = False

    return steps, kd_sum, ref_sum, total_sum, check_dynamics_grad


def run_epoch(
    model, predictor, groups, epoch, lambda_ref, tbptt_steps, seed,
    optimizer=None, validate=False, max_steps=0, smoke_probe=None,
):
    predictor.train(optimizer is not None)
    confusion = torch.zeros((19, 19), dtype=torch.int64) if validate else None
    count = 0; kd_sum = ref_sum = total_sum = 0.0
    check_dynamics_grad = optimizer is not None
    for sequence_index, samples in enumerate(groups.values()):
        n, kd, ref, total, check_dynamics_grad = run_sequence(
            model, predictor, samples, sequence_index, epoch if optimizer is not None else 0,
            lambda_ref, tbptt_steps, seed, optimizer, confusion, max_steps,
            check_dynamics_grad, smoke_probe,
        )
        count += n; kd_sum += kd; ref_sum += ref; total_sum += total
    if count == 0:
        raise RuntimeError("Epoch produced zero prediction steps")
    result = {
        "prediction_count": count,
        "kd_loss": kd_sum / count,
        "feature_loss": ref_sum / count,
        "combined_loss": total_sum / count,
    }
    if validate:
        result["decoded_mIoU"] = float(torch.nanmean(compute_iou(confusion)).item())
    return result


def is_better(val, best):
    if best is None or val["decoded_mIoU"] > best["val"]["decoded_mIoU"]:
        return True
    return (
        val["decoded_mIoU"] == best["val"]["decoded_mIoU"]
        and val["combined_loss"] < best["val"]["combined_loss"]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/lin/predify/kitti_step")
    parser.add_argument("--output", default="/home/lin/predify/experiments/kitti_step_robust_semantic_reference")
    parser.add_argument("--predictor-checkpoint", default=ROLE_PREDICTOR_CHECKPOINT_DEFAULT)
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--tbptt-steps", type=int, default=TBPTT_STEPS)
    parser.add_argument("--calibration-chunks", type=int, default=CALIBRATION_CHUNKS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    model, predictor = load_components(
        STATIC_CHECKPOINT_DEFAULT, ADAPTER_CHECKPOINT_DEFAULT,
        args.predictor_checkpoint, WRITEBACK_CHECKPOINT_DEFAULT,
    )
    model.requires_grad_(False); model.eval()
    semantic_parameters = configure_trainable(predictor)

    train = sequence_groups(KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "train"))
    val = sequence_groups(KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "val"))
    epochs, calibration_chunks, max_steps = args.epochs, args.calibration_chunks, 0
    output = Path(args.output)
    if args.smoke:
        train = dict(list(train.items())[:1]); val = dict(list(val.items())[:1])
        epochs, calibration_chunks, max_steps = 1, min(2, args.calibration_chunks), 16
        output = output / "smoke"
    output.mkdir(parents=True, exist_ok=True)

    calibration = calibrate_lambda(
        model, predictor, train, semantic_parameters, calibration_chunks,
        args.tbptt_steps, args.seed, max_steps,
    )
    lambda_ref = calibration["lambda_ref"]
    print(json.dumps({"gradient_calibration": calibration}, sort_keys=True), flush=True)

    optimizer = torch.optim.AdamW(semantic_parameters, lr=1e-4, weight_decay=0.01)
    history, best, stale = [], None, 0
    smoke_probe = {} if args.smoke else None
    checkpoint = output / "best_robust_semantic_reference.pt"
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model, predictor, train, epoch, lambda_ref, args.tbptt_steps,
            args.seed, optimizer=optimizer, max_steps=max_steps,
            smoke_probe=smoke_probe,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model, predictor, val, epoch, lambda_ref, args.tbptt_steps,
                args.seed, validate=True, max_steps=max_steps,
            )
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row); print(json.dumps(row, sort_keys=True), flush=True)
        if is_better(val_metrics, best):
            best, stale = row, 0
            torch.save({
                "predictor_state_dict": predictor.state_dict(),
                "source_predictor_checkpoint": args.predictor_checkpoint,
                "epoch": epoch,
                "val_metrics": val_metrics,
                "lambda_ref": lambda_ref,
            }, checkpoint)
        else:
            stale += 1
            if stale >= args.patience:
                break

    summary = {
        "experiment": "kitti_step_robust_semantic_reference",
        "checkpoint": str(checkpoint),
        "source_predictor_checkpoint": args.predictor_checkpoint,
        "smoke": args.smoke,
        "trainable_parameter_count": sum(p.numel() for p in semantic_parameters),
        "frozen_dynamics_trainable_parameter_count": 0,
        "config": {
            "max_epochs": epochs,
            "patience": args.patience,
            "tbptt_steps": args.tbptt_steps,
            "semantic_state_lifetime": "full sequence",
            "optimizer": "AdamW",
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "loss": "KD + lambda_ref * SmoothL1(M_sem_t+1, Z_clean_t+1) on z1/z4",
            "target_alignment": "predictor.step(O_t,E_t) -> t+1",
            "augmentation": "Common corruption augmentation on KITTI-STEP train; not a formal Cityscapes-C benchmark",
            "corruption_scope": "type+severity fixed for each sequence; deterministic realization changes per frame",
            "training_corruptions": list(TRAIN_CORRUPTIONS),
            "excluded_training_corruptions": {
                "glass_blur": "excluded from online training augmentation because its full-resolution pixel-swap implementation is a known preprocessing bottleneck; it remains in formal robustness evaluation"
            },
            "severities": list(CITYSCAPES_C_SEVERITIES),
            "validation_selection": "highest decoded mIoU; lower combined loss breaks exact ties",
        },
        "gradient_calibration": calibration,
        "smoke_checks": smoke_probe,
        "dataset": {"train_sequences": len(train), "val_sequences": len(val), "smoke_max_steps": max_steps},
        "history": history,
        "best": best,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
