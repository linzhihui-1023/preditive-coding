"""Gate V2-A: short CUDA forward/backward/state/metric smoke for FAST-B."""

import argparse
import json
from types import SimpleNamespace
from pathlib import Path

import torch
from torch.nn import functional as F

from predify2021.datasets.vspw import (
    VSPWSequentialClipDataset,
    VSPWSegmentationDataset,
    collate_vspw_sequential_clips,
)
from predify2021.mce_scores.train_vspw_fast_b import _optimise_window
from predify2021.mce_scores.vspw_fast_b_common import (
    C4_ADAPTER_INDEX,
    C4_WRITEBACK_KEY,
    build_fast_b_predictor,
    configure_fast_b_host,
    encode_vspw_frame,
    load_joint_c4_payload,
    load_vspw_host,
    parameter_report,
    restore_frame,
    reset_temporal_state,
)
from predify2021.mce_scores.vspw_fast_b_evaluator import (
    build_validation_loader,
    evaluate_vspw_fast_b,
)
from torch.utils.data import DataLoader


DEFAULT_DYNAMICS_CHECKPOINT = (
    "/home/lin/predify/experiments/"
    "kitti_step_role_separated_predictor_3370f78/"
    "best_role_separated_predictor.pt"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the FAST-B VSPW Gate V2-A smoke test.")
    parser.add_argument("--data-root", default="/home/lin/datasets/VSPW_480p")
    parser.add_argument("--host-checkpoint", required=True)
    parser.add_argument("--fast-b-checkpoint", default=None)
    parser.add_argument("--dynamics-checkpoint", default=DEFAULT_DYNAMICS_CHECKPOINT)
    parser.add_argument("--output-dir", default="/home/lin/predify/experiments/vspw_fast_b/gate_v2a")
    parser.add_argument("--max-videos", type=int, default=2)
    parser.add_argument("--max-clips-per-video", type=int, default=1)
    parser.add_argument("--clip-length", type=int, default=16, choices=(16, 32))
    parser.add_argument("--bptt", type=int, default=8, choices=(8, 16))
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def _subset_clips(dataset, max_videos, max_clips_per_video):
    selected_video_ids = tuple(dataset.video_ids[:max_videos])
    video_ids = set(selected_video_ids)
    selected = []
    counts = {}
    for clip in dataset.clips:
        if clip["video_id"] not in video_ids:
            continue
        count = counts.get(clip["video_id"], 0)
        if count >= max_clips_per_video:
            continue
        counts[clip["video_id"]] = count + 1
        selected.append(clip)
    dataset.video_ids = selected_video_ids
    dataset.clips = tuple(selected)
    return dataset


def _clip_loader(dataset, workers):
    workers = max(0, int(workers))
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
    return DataLoader(**kwargs)


def main():
    args = parse_args()
    if args.bptt > args.clip_length:
        raise ValueError("bptt cannot exceed clip_length")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Gate V2-A requires CUDA and must run after Host completion.")
    host, _ = load_vspw_host(args.host_checkpoint, device)
    predictor, _, fast_b_payload = build_fast_b_predictor(
        args.dynamics_checkpoint, args.fast_b_checkpoint, device
    )
    if fast_b_payload is not None:
        load_joint_c4_payload(host, fast_b_payload)
    configure_fast_b_host(host, trainable=True)
    predictor.train()
    smoke_args = SimpleNamespace(
        feature_loss_weight=0.25,
        semantic_loss_weight=1.0,
    )
    optimizer = torch.optim.AdamW(
        [
            parameter for parameter in predictor.semantic_parameters()
            if parameter.requires_grad
        ] + [
            parameter for module in (
                host.multi_layer_adapter.output_adapters[C4_ADAPTER_INDEX],
                host.host_conditioned_writebacks[C4_WRITEBACK_KEY],
            )
            for parameter in module.parameters() if parameter.requires_grad
        ],
        lr=1e-4,
    )
    scaler = torch.amp.GradScaler("cuda")
    clips = _subset_clips(
        VSPWSequentialClipDataset(
            args.data_root, "val", args.clip_length, drop_short_clips=True
        ),
        args.max_videos,
        args.max_clips_per_video,
    )
    loader = _clip_loader(clips, args.num_workers)
    state = reset_temporal_state()
    losses = []
    last_sequence = None
    for clip in loader:
        if clip["sequence_id"] != last_sequence:
            state = reset_temporal_state()
            last_sequence = clip["sequence_id"]
        images = clip["images"].to(device, non_blocking=True)
        records = []
        feature_losses = []
        for index in range(min(images.shape[0], args.bptt)):
            image = images[index:index + 1]
            with torch.amp.autocast("cuda"):
                raw, observation = encode_vspw_frame(host, image)
                restored, state, diagnostics = restore_frame(
                    predictor, observation, state, "full"
                )
                feature_losses.append(F.smooth_l1_loss(restored.z4, observation.z4))
            records.append((raw, observation, restored, tuple(image.shape[-2:]), clip["masks"][index]))
        result = _optimise_window(
            host, records, feature_losses, optimizer, scaler, smoke_args, device
        )
        losses.append(result)
        if not all(torch.isfinite(torch.tensor(value)) for value in result.values()):
            raise RuntimeError(f"Non-finite Gate V2-A loss: {result}")
        if state.pending_prediction is None:
            raise RuntimeError("FAST-B state did not continue after smoke clip.")
        break

    forbidden_gradients = []
    for name, module in (
        ("backbone", host.backbone),
        ("decode_head", host.decode_head),
        ("auxiliary_head", host.auxiliary_head),
    ):
        if any(parameter.grad is not None for parameter in module.parameters()):
            forbidden_gradients.append(name)
    if forbidden_gradients:
        raise RuntimeError(f"Frozen Host received gradients: {forbidden_gradients}")
    configure_fast_b_host(host, trainable=False)
    val_dataset = VSPWSegmentationDataset(args.data_root, "val")
    allowed = set(tuple(clips.video_ids))
    val_dataset.video_ids = tuple(video for video in val_dataset.video_ids if video in allowed)
    val_dataset.samples = tuple(sample for sample in val_dataset.samples if sample["video_id"] in allowed)
    val_loader = build_validation_loader(val_dataset, args.num_workers)
    metrics = {
        mode: evaluate_vspw_fast_b(host, predictor, val_loader, device, mode)
        for mode in ("full", "reset")
    }
    summary = {
        "gate": "V2-A",
        "videos": len(allowed),
        "smoke_losses": losses,
        "metrics": metrics,
        "parameter_report": parameter_report(host, predictor),
        "state_continuity": True,
        "frozen_host_gradients": True,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
