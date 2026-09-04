"""Frozen Error Utility Probe: a strict A/B/C/D/E correction-input test.

FAST-B Host, its C4 adapter/writeback, the Stage-T encoder and predictor, and
the segmentation decoder are frozen.  Each variant trains the same 256->128
1x1 correction probe and decodes ``Z4 + delta`` through the existing frozen
writeback/decoder.  A/B/C/D/E differ only in the packed probe input:
Z, error, [Z,error], [Z,prediction], and [Z,temporally shifted error].
The shifted-error control uses the preceding frame's error, preserving the
same marginal tensor distribution while breaking current-frame alignment.
"""

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset, semantic_mask_from_panoptic_png
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import load_image, sequence_groups
from predify2021.mce_scores.evaluate_kitti_step_static_baseline import compute_iou, update_confusion_matrix
from predify2021.mce_scores.role_separated_dynamic_error_correction import (
    ADAPTER_CHECKPOINT_DEFAULT, ROLE_PREDICTOR_CHECKPOINT_DEFAULT,
    STATIC_CHECKPOINT_DEFAULT, WRITEBACK_CHECKPOINT_DEFAULT,
    load_components, residual_writeback_host_feature,
)
from predify2021.model_factory.deeplabv3plus_resnet50 import (
    AuxiliaryTemporalPredictor, AuxiliaryTemporalStateEncoder, HostFeature,
    UnifiedFeatures,
)


VARIANTS = ("A_Z", "B_error", "C_Z_error", "D_Z_pred", "E_Z_shuffled_error")
FULL9 = ("0002", "0006", "0007", "0008", "0010", "0013", "0014", "0016", "0018")
FAST_B_CHECKPOINT_DEFAULT = (
    "/home/lin/predify/experiments/kitti_step_semantic_v3_joint_c4_fast_ab_5754714/"
    "fast_b_joint_c4_weak_z4/best.pt"
)
STAGE_T_CHECKPOINT_DEFAULT = "/home/lin/predify/experiments/kitti_step_v2_auxiliary_stage_t/best.pt"
TRAIN_EPOCHS = 3
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
NUM_CLASSES = 19
IGNORE_INDEX = 255


class CorrectionProbe(nn.Module):
    """One shared-size probe; absent inputs are zero-padded to 256 channels."""

    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(256, 128, kernel_size=1)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, packed):
        return self.projection(packed)


def load_fast_b_model(path):
    model, _ = load_components(
        STATIC_CHECKPOINT_DEFAULT, ADAPTER_CHECKPOINT_DEFAULT,
        ROLE_PREDICTOR_CHECKPOINT_DEFAULT, WRITEBACK_CHECKPOINT_DEFAULT,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.multi_layer_adapter.output_adapters[3].load_state_dict(
        payload["c4_output_adapter_state_dict"], strict=True
    )
    model.host_conditioned_writebacks["3"].load_state_dict(
        payload["c4_writeback_state_dict"], strict=True
    )
    model.requires_grad_(False).eval()
    return model


def pack_input(kind, state, error, predicted, shuffled_error):
    zeros = torch.zeros_like(state)
    if kind == "A_Z":
        return torch.cat((state, zeros), dim=1)
    if kind == "B_error":
        return torch.cat((zeros, error), dim=1)
    if kind == "C_Z_error":
        return torch.cat((state, error), dim=1)
    if kind == "D_Z_pred":
        return torch.cat((state, predicted), dim=1)
    if kind == "E_Z_shuffled_error":
        return torch.cat((state, shuffled_error), dim=1)
    raise ValueError(kind)


def decode_probe(model, raw, observation, delta, output_size):
    corrected = residual_writeback_host_feature(
        model,
        raw,
        UnifiedFeatures(
            torch.zeros_like(observation.z1), torch.zeros_like(observation.z2),
            torch.zeros_like(observation.z3), delta,
        ),
        output_size,
    )
    return model.decode_from_host_feature(corrected)


def host_logits(model, raw, output_size):
    return model.decode_from_host_feature(HostFeature(raw.c4, raw.c1, output_size))


def make_probe(seed):
    torch.manual_seed(seed)
    probe = CorrectionProbe().cuda()
    return probe


def run_sequence(model, encoder, predictor, probe, kind, samples, optimizer=None, training=False):
    confusion = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
    total_loss = 0.0; frame_count = 0
    pending = None; hidden = None; previous_error = None
    for index, sample in enumerate(samples):
        image = load_image(sample)
        with torch.no_grad():
            raw = model.extract_backbone_features(image)
            observation = model.encode_backbone_features(raw)
            state = encoder(observation.z4)
            if index == 0:
                pending, hidden = predictor.predict_next(state, torch.zeros_like(state), None)
        predicted = state if index == 0 else pending
        error = torch.zeros_like(state) if index == 0 else state - pending
        shuffled_error = error if previous_error is None else previous_error
        packed = pack_input(kind, state, error, predicted, shuffled_error)
        target = semantic_mask_from_panoptic_png(sample["mask_path"]).cuda(non_blocking=True)
        output_size = tuple(image.shape[-2:])
        if training:
            logits = decode_probe(model, raw, observation, probe(packed), output_size)
            loss = F.cross_entropy(logits, target.unsqueeze(0), ignore_index=IGNORE_INDEX)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite probe loss in {kind}")
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            total_loss += float(loss.detach().item())
        else:
            with torch.no_grad():
                logits = decode_probe(model, raw, observation, probe(packed), output_size)
        prediction = logits.argmax(1).squeeze(0).cpu()
        update_confusion_matrix(confusion, prediction, target.cpu())
        if not training:
            total_loss += float(F.cross_entropy(logits, target.unsqueeze(0), ignore_index=IGNORE_INDEX).item())
        frame_count += 1
        previous_error = error.detach()
        if index > 0:
            with torch.no_grad():
                pending, hidden = predictor.predict_next(state, error, hidden)
    iou = compute_iou(confusion)
    return {
        "frames": frame_count,
        "loss": total_loss / max(frame_count, 1),
        "mIoU": float(torch.nanmean(iou).item()),
    }


def run_epoch(model, encoder, predictor, probe, groups, optimizer, training):
    rows = []
    for sequence, samples in groups.items():
        rows.append(run_sequence(model, encoder, predictor, probe, probe.variant, samples, optimizer, training))
    frames = sum(row["frames"] for row in rows)
    return {
        "frames": frames,
        "loss": sum(row["loss"] * row["frames"] for row in rows) / max(frames, 1),
        "mIoU": sum(row["mIoU"] * row["frames"] for row in rows) / max(frames, 1),
    }


@torch.inference_mode()
def evaluate_host(model, groups):
    confusion = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
    frames = 0
    for samples in groups.values():
        for sample in samples:
            image = load_image(sample)
            raw = model.extract_backbone_features(image)
            prediction = host_logits(model, raw, tuple(image.shape[-2:])).argmax(1).squeeze(0).cpu()
            target = semantic_mask_from_panoptic_png(sample["mask_path"])
            update_confusion_matrix(confusion, prediction, target)
            frames += 1
    return {"frames": frames, "mIoU": float(torch.nanmean(compute_iou(confusion)).item())}


def train_variant(model, encoder, predictor, train_groups, val_groups, kind, epochs, lr, weight_decay, seed):
    probe = make_probe(seed); probe.variant = kind
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    history = []; best = None
    for epoch in range(1, epochs + 1):
        probe.train()
        train = run_epoch(model, encoder, predictor, probe, train_groups, optimizer, True)
        probe.eval()
        with torch.no_grad():
            val = run_epoch(model, encoder, predictor, probe, val_groups, optimizer, False)
        row = {"epoch": epoch, "train": train, "val": val}
        history.append(row)
        print(json.dumps({"variant": kind, **row}, sort_keys=True), flush=True)
        if best is None or val["mIoU"] > best["val"]["mIoU"]:
            best = row
    return {"history": history, "best": best, "probe_parameters": sum(p.numel() for p in probe.parameters())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/home/lin/predify/kitti_step")
    parser.add_argument("--fast-b-checkpoint", default=FAST_B_CHECKPOINT_DEFAULT)
    parser.add_argument("--stage-t-checkpoint", default=STAGE_T_CHECKPOINT_DEFAULT)
    parser.add_argument("--epochs", type=int, default=TRAIN_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--result-output", default="results/kitti_step_frozen_error_utility_probe.json")
    args = parser.parse_args(argv)
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    model = load_fast_b_model(args.fast_b_checkpoint)
    payload = torch.load(args.stage_t_checkpoint, map_location="cpu", weights_only=False)
    encoder = AuxiliaryTemporalStateEncoder().cuda(); encoder.load_state_dict(payload["encoder_state_dict"], strict=True)
    predictor = AuxiliaryTemporalPredictor().cuda(); predictor.load_state_dict(payload["predictor_state_dict"], strict=True)
    encoder.requires_grad_(False).eval(); predictor.requires_grad_(False).eval()
    train = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "train")
    val = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "val")
    train_groups = sequence_groups(train); val_groups = sequence_groups(val)
    host = evaluate_host(model, val_groups)
    results = {}
    for index, kind in enumerate(VARIANTS):
        results[kind] = train_variant(model, encoder, predictor, train_groups, val_groups, kind, args.epochs, args.lr, args.weight_decay, args.seed + index)
    output = Path(args.result_output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "experiment": "Frozen Error Utility Probe",
        "variants": VARIANTS, "full9_validation_sequences": FULL9,
        "epochs": args.epochs, "lr": args.lr, "weight_decay": args.weight_decay,
        "probe_parameters": 32896,
        "frozen_host_validation": host,
        "frozen": ["FAST-B Host", "C4 Adapter", "C4 Writeback", "Stage-T Encoder", "Stage-T Predictor", "Decoder"],
        "results": results,
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"experiment": "Frozen Error Utility Probe", "results": results}, sort_keys=True), flush=True)


if __name__ == "__main__": main()
