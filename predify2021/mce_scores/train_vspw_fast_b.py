"""Train the validated FAST-B mainline on clean VSPW-480p clips.

The VSPW path changes only the data protocol, Host class count, and semantic
supervision.  FAST-B's tested modules remain the Semantic Branch, frozen
dynamics, C4 output adapter, and C4 Host-conditioned writeback.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from predify2021.datasets.vspw import (
    VSPWSequentialClipDataset,
    VSPWSegmentationDataset,
    VSPW_IGNORE_LABEL,
    VSPW_NUM_CLASSES,
    collate_vspw_sequential_clips,
)
from predify2021.mce_scores.vspw_fast_b_common import (
    C4_ADAPTER_INDEX,
    C4_WRITEBACK_KEY,
    build_fast_b_predictor,
    configure_fast_b_host,
    corrected_host_feature_with_size,
    encode_vspw_frame,
    load_joint_c4_payload,
    load_vspw_host,
    parameter_report,
    restore_frame,
    reset_temporal_state,
    detach_temporal_state,
)
from predify2021.mce_scores.vspw_fast_b_evaluator import (
    build_validation_loader,
    evaluate_vspw_fast_b,
)
from predify2021.model_factory.deeplabv3plus_resnet50.adapters import (
    BackboneFeatures,
    UnifiedFeatures,
)


DEFAULT_DYNAMICS_CHECKPOINT = (
    "/home/lin/predify/experiments/"
    "kitti_step_role_separated_predictor_3370f78/"
    "best_role_separated_predictor.pt"
)
MAX_EPOCHS = 15
PATIENCE = 3


def parse_args():
    parser = argparse.ArgumentParser(description="Train FAST-B on clean VSPW-480p videos.")
    parser.add_argument("--data-root", default="/home/lin/datasets/VSPW_480p")
    parser.add_argument("--host-checkpoint", required=True, help="best_vspw_host.pt")
    parser.add_argument("--dynamics-checkpoint", default=DEFAULT_DYNAMICS_CHECKPOINT)
    parser.add_argument("--fast-b-checkpoint", default=None, help="Optional validated FAST-B resume payload.")
    parser.add_argument("--output-dir", default="/home/lin/predify/experiments/vspw_fast_b")
    parser.add_argument("--clip-length", type=int, default=16, choices=(16, 32))
    parser.add_argument("--bptt", type=int, default=16, choices=(8, 16, 32))
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--feature-loss-weight", type=float, default=0.25)
    parser.add_argument("--semantic-loss-weight", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _train_loader(dataset, num_workers):
    workers = max(0, int(num_workers))
    kwargs = dict(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        collate_fn=collate_vspw_sequential_clips,
    )
    if workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(
        **kwargs,
    )


def _batched_host_feature(host, records, output_size):
    raw_features = BackboneFeatures(
        *(torch.cat([record[0].as_tuple()[index] for record in records], dim=0)
          for index in range(4))
    )
    observations = UnifiedFeatures(
        *(torch.cat([record[1].as_tuple()[index] for record in records], dim=0)
          for index in range(4))
    )
    restored = UnifiedFeatures(
        *(torch.cat([record[2].as_tuple()[index] for record in records], dim=0)
          for index in range(4))
    )
    return corrected_host_feature_with_size(
        host, raw_features, observations, restored, output_size
    )


def _optimise_window(host, records, feature_losses, optimizer, scaler, args, device):
    corrected = _batched_host_feature(host, records, records[0][3])
    logits = host.decode_from_host_feature(corrected)
    targets = torch.stack([record[4] for record in records], dim=0).to(
        device, non_blocking=True
    )
    feature_loss = torch.stack(feature_losses).mean()
    semantic_loss = F.cross_entropy(
        logits, targets, ignore_index=VSPW_IGNORE_LABEL
    )
    objective = (
        args.feature_loss_weight * feature_loss
        + args.semantic_loss_weight * semantic_loss
    )
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(objective).backward()
    scaler.step(optimizer)
    scaler.update()
    return {
        "feature_loss": float(feature_loss.detach().item()),
        "semantic_loss": float(semantic_loss.detach().item()),
        "total_loss": float(objective.detach().item()),
    }


def train_one_epoch(host, predictor, loader, optimizer, scaler, args, device, epoch):
    predictor.train()
    state = reset_temporal_state()
    current_sequence = None
    records = []
    feature_losses = []
    totals = {"feature_loss": 0.0, "semantic_loss": 0.0, "total_loss": 0.0}
    window_count = 0
    frame_count = 0

    def flush_window():
        nonlocal records, feature_losses, window_count
        if not records:
            return
        result = _optimise_window(
            host, records, feature_losses, optimizer, scaler, args, device
        )
        for key in totals:
            totals[key] += result[key]
        window_count += 1
        records = []
        feature_losses = []
        detach_temporal_state(state)

    for clip_index, clip in enumerate(loader):
        sequence_id = clip["sequence_id"]
        if sequence_id != current_sequence:
            flush_window()
            state = reset_temporal_state()
            current_sequence = sequence_id
        images = clip["images"].to(device, non_blocking=True)
        masks = clip["masks"]
        for frame_index in range(images.shape[0]):
            image = images[frame_index : frame_index + 1]
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                raw, observation = encode_vspw_frame(host, image)
                restored, state, _ = restore_frame(
                    predictor, observation, state, "full"
                )
                feature_loss = F.smooth_l1_loss(restored.z4, observation.z4)
            records.append((raw, observation, restored, tuple(image.shape[-2:]), masks[frame_index]))
            feature_losses.append(feature_loss)
            frame_count += 1
            if len(records) >= args.bptt:
                flush_window()
        if clip["is_sequence_end"]:
            flush_window()
        if args.log_interval and clip_index % args.log_interval == 0:
            print(json.dumps({"epoch": epoch, "clip": clip_index, "frames": frame_count}, sort_keys=True), flush=True)
    flush_window()
    return {
        "frame_count": frame_count,
        "window_count": window_count,
        **{key: value / max(window_count, 1) for key, value in totals.items()},
    }


def _save_checkpoint(path, host, predictor, optimizer, epoch, val, args, report):
    torch.save(
        {
            "model_state_dict": predictor.state_dict(),
            "c4_output_adapter_state_dict": host.multi_layer_adapter.output_adapters[C4_ADAPTER_INDEX].state_dict(),
            "c4_writeback_state_dict": host.host_conditioned_writebacks[C4_WRITEBACK_KEY].state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_metrics": val,
            "joint_c4_training": True,
            "temporal_mode": "full",
            "clip_length": args.clip_length,
            "bptt": args.bptt,
            "num_classes": VSPW_NUM_CLASSES,
            "parameter_report": report,
        },
        path,
    )


def main():
    args = parse_args()
    if args.bptt > args.clip_length:
        raise ValueError("bptt cannot exceed clip_length")
    if args.bptt < 8:
        raise ValueError("VSPW FAST-B BPTT must cover at least 8 frames")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Formal FAST-B training requires CUDA; CPU is reserved for tests.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    host, host_payload = load_vspw_host(args.host_checkpoint, device)
    predictor, dynamics_payload, fast_b_payload = build_fast_b_predictor(
        args.dynamics_checkpoint,
        args.fast_b_checkpoint,
        device,
    )
    if fast_b_payload is not None:
        load_joint_c4_payload(host, fast_b_payload)
    configure_fast_b_host(host, trainable=True)
    report = parameter_report(host, predictor)
    print(json.dumps({"Host Params": report["host_params"], "Frozen Host Params": report["frozen_host_params"], "FAST-B Trainable Params": report["fast_b_trainable_params"], "Additional Inference Params": report["additional_inference_params"], "Total Params": report["total_params"], "Trainable Ratio": report["trainable_ratio"]}, sort_keys=True), flush=True)

    train_dataset = VSPWSequentialClipDataset(
        args.data_root,
        "train",
        clip_length=args.clip_length,
        drop_short_clips=True,
    )
    val_dataset = VSPWSegmentationDataset(args.data_root, "val")
    train_loader = _train_loader(train_dataset, args.num_workers)
    val_loader = build_validation_loader(val_dataset, args.num_workers)
    parameters = [
        parameter for parameter in predictor.semantic_parameters() if parameter.requires_grad
    ]
    parameters += [
        parameter
        for module in (
            host.multi_layer_adapter.output_adapters[C4_ADAPTER_INDEX],
            host.host_conditioned_writebacks[C4_WRITEBACK_KEY],
        )
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda")
    best_key = None
    best_epoch = None
    stale = 0
    history = []
    best_path = output_dir / "best_vspw_fast_b.pt"
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            host, predictor, train_loader, optimizer, scaler, args, device, epoch
        )
        val = evaluate_vspw_fast_b(
            host, predictor, val_loader, device, temporal_mode="full"
        )
        row = {"epoch": epoch, "train": train_metrics, "val": val}
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        key = (val["fast_b"]["mIoU"], -val["fast_b"]["loss"])
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
            stale = 0
            _save_checkpoint(best_path, host, predictor, optimizer, epoch, val, args, report)
        else:
            stale += 1
        if stale >= args.patience:
            break

    summary = {
        "experiment": "vspw_fast_b",
        "host_checkpoint": str(args.host_checkpoint),
        "dynamics_checkpoint": str(args.dynamics_checkpoint),
        "checkpoint": str(best_path),
        "num_classes": VSPW_NUM_CLASSES,
        "clip_length": args.clip_length,
        "bptt": args.bptt,
        "temporal_mode": "full",
        "loss": {
            "feature_smooth_l1_weight": args.feature_loss_weight,
            "ground_truth_semantic_cross_entropy_weight": args.semantic_loss_weight,
            "ignore_index": VSPW_IGNORE_LABEL,
        },
        "parameter_report": report,
        "best_epoch": best_epoch,
        "history": history,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"best_checkpoint": str(best_path), "best_epoch": best_epoch}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
