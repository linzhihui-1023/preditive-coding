"""V2-Staged Stage A: learn a compact semantic representation and decoder."""

import argparse
import json
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset, semantic_mask_from_panoptic_png
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import load_image, sequence_groups
from predify2021.mce_scores.evaluate_kitti_step_static_baseline import compute_iou, update_confusion_matrix
from predify2021.mce_scores.role_separated_dynamic_error_correction import (
    ADAPTER_CHECKPOINT_DEFAULT, ROLE_PREDICTOR_CHECKPOINT_DEFAULT,
    STATIC_CHECKPOINT_DEFAULT, WRITEBACK_CHECKPOINT_DEFAULT, load_components,
    residual_writeback_host_feature,
)
from predify2021.mce_scores.train_kitti_step_z4_only import FAST_B_DEFAULT
from predify2021.model_factory.deeplabv3plus_resnet50 import HostFeature, UnifiedFeatures, Z4PredictiveSemanticEncoder, Z4PredictiveSemanticDecoder

SEED = 0
NUM_CLASSES = 19
IGNORE = 255
MAX_EPOCHS = 10
PATIENCE = 3
MIN_MIOU_IMPROVEMENT = 1e-4
LR = 1e-5
WEIGHT_DECAY = 1e-2
LAMBDA_SEM = 1.0
FULL9 = ("0002", "0006", "0007", "0008", "0010", "0013", "0014", "0016", "0018")
OUTPUT_DEFAULT = "/home/lin/predify/experiments/kitti_step_v2_stage_a_relaxed_gate"
RESULT_DEFAULT = "results/kitti_step_v2_stage_a_relaxed_gate"


def encode_host(model, sample):
    image = load_image(sample)
    with torch.no_grad():
        raw = model.extract_backbone_features(image)
        observation = model.encode_backbone_features(raw)
    return image, observation, raw, tuple(image.shape[-2:])


def decode_with_residual(model, raw, observation, delta_z4, output_size):
    delta = UnifiedFeatures(torch.zeros_like(observation.z1), torch.zeros_like(observation.z2), torch.zeros_like(observation.z3), delta_z4)
    feature = residual_writeback_host_feature(model, raw, delta, output_size)
    return model.decode_from_host_feature(feature)


def load_stage_a(args):
    model, _ = load_components(STATIC_CHECKPOINT_DEFAULT, ADAPTER_CHECKPOINT_DEFAULT, args.dynamics_checkpoint, WRITEBACK_CHECKPOINT_DEFAULT)
    payload = torch.load(args.fast_b_checkpoint, map_location="cpu", weights_only=False)
    model.multi_layer_adapter.output_adapters[3].load_state_dict(payload["c4_output_adapter_state_dict"], strict=True)
    model.host_conditioned_writebacks["3"].load_state_dict(payload["c4_writeback_state_dict"], strict=True)
    model.requires_grad_(False).eval().cuda()
    encoder = Z4PredictiveSemanticEncoder().cuda()
    decoder = Z4PredictiveSemanticDecoder().cuda()
    encoder.train(); decoder.train()
    return model, encoder, decoder


def assert_contract(model, encoder, decoder):
    if any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("Stage A Host must be frozen")
    if not all(p.requires_grad for p in list(encoder.parameters()) + list(decoder.parameters())):
        raise RuntimeError("Stage A encoder/decoder must be trainable")


def train_epoch(model, encoder, decoder, groups, optimizer):
    encoder.train(); decoder.train()
    totals = {"frames": 0, "Lrec": 0.0, "Lsem": 0.0, "total": 0.0, "state_rms": 0.0, "state_std": 0.0, "reconstruction_mse": 0.0}
    for samples in groups.values():
        for sample in samples:
            _, observation, raw, output_size = encode_host(model, sample)
            state = encoder(observation.z4)
            reconstruction = decoder(state)
            delta = reconstruction - observation.z4
            logits = decode_with_residual(model, raw, observation, delta, output_size)
            mask = semantic_mask_from_panoptic_png(sample["mask_path"]).cuda()
            lrec = F.smooth_l1_loss(reconstruction, observation.z4)
            lsem = F.cross_entropy(logits, mask.unsqueeze(0), ignore_index=IGNORE)
            total = lrec + LAMBDA_SEM * lsem
            if not torch.isfinite(total):
                raise FloatingPointError("Non-finite Stage A loss")
            optimizer.zero_grad(set_to_none=True); total.backward(); optimizer.step()
            totals["frames"] += 1
            totals["Lrec"] += float(lrec.detach()); totals["Lsem"] += float(lsem.detach()); totals["total"] += float(total.detach())
            totals["state_rms"] += float(state.detach().square().mean().sqrt()); totals["state_std"] += float(state.detach().std()); totals["reconstruction_mse"] += float(F.mse_loss(reconstruction.detach(), observation.z4).item())
    for key in ("Lrec", "Lsem", "total", "state_rms", "state_std", "reconstruction_mse"):
        totals[key] /= max(totals["frames"], 1)
    return totals


@torch.inference_mode()
def evaluate(model, encoder, decoder, groups):
    encoder.eval(); decoder.eval()
    host_conf = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
    rec_conf = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
    state_rms = state_std = temporal_mse = rec_mse = 0.0; frames = pairs = 0
    per_sequence = {}
    for sequence in FULL9:
        samples = groups[sequence]
        seq_host = torch.zeros_like(host_conf); seq_rec = torch.zeros_like(host_conf)
        prev_state = None; seq_state_rms = seq_state_std = seq_temporal = seq_rec_mse = 0.0
        for sample in samples:
            _, observation, raw, output_size = encode_host(model, sample)
            state = encoder(observation.z4); reconstruction = decoder(state)
            logits = decode_with_residual(model, raw, observation, reconstruction - observation.z4, output_size)
            host_logits = model.decode_from_host_feature(HostFeature(raw.c4, raw.c1, output_size))
            mask = semantic_mask_from_panoptic_png(sample["mask_path"])
            update_confusion_matrix(host_conf, host_logits.argmax(1)[0].cpu(), mask); update_confusion_matrix(seq_host, host_logits.argmax(1)[0].cpu(), mask)
            update_confusion_matrix(rec_conf, logits.argmax(1)[0].cpu(), mask); update_confusion_matrix(seq_rec, logits.argmax(1)[0].cpu(), mask)
            state_rms += float(state.square().mean().sqrt()); state_std += float(state.std()); rec_mse += float(F.mse_loss(reconstruction, observation.z4).item()); frames += 1
            seq_state_rms += float(state.square().mean().sqrt()); seq_state_std += float(state.std()); seq_rec_mse += float(F.mse_loss(reconstruction, observation.z4).item())
            if prev_state is not None:
                value = float(F.mse_loss(state, prev_state).item()); temporal_mse += value; seq_temporal += value; pairs += 1
            prev_state = state
        per_sequence[sequence] = {"mIoU_host": float(torch.nanmean(compute_iou(seq_host)).item()), "mIoU_rec": float(torch.nanmean(compute_iou(seq_rec)).item()), "state_rms": seq_state_rms / max(len(samples), 1), "state_std": seq_state_std / max(len(samples), 1), "state_temporal_mse": seq_temporal / max(len(samples) - 1, 1), "reconstruction_mse": seq_rec_mse / max(len(samples), 1)}
    host_miou = float(torch.nanmean(compute_iou(host_conf)).item()); rec_miou = float(torch.nanmean(compute_iou(rec_conf)).item())
    return {"mIoU_host": host_miou, "mIoU_rec": rec_miou, "delta_mIoU_rec": rec_miou - host_miou, "state_rms": state_rms / max(frames, 1), "state_std": state_std / max(frames, 1), "state_temporal_mse": temporal_mse / max(pairs, 1), "reconstruction_mse": rec_mse / max(frames, 1), "frames": frames, "pairs": pairs, "per_sequence": per_sequence}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/home/lin/predify/kitti_step"); parser.add_argument("--fast-b-checkpoint", default=FAST_B_DEFAULT); parser.add_argument("--dynamics-checkpoint", default=ROLE_PREDICTOR_CHECKPOINT_DEFAULT); parser.add_argument("--output", default=OUTPUT_DEFAULT); parser.add_argument("--result-output", default=RESULT_DEFAULT); parser.add_argument("--epochs", type=int, default=MAX_EPOCHS); parser.add_argument("--patience", type=int, default=PATIENCE)
    args = parser.parse_args(argv)
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    model, encoder, decoder = load_stage_a(args); assert_contract(model, encoder, decoder)
    train = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "train"); val = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "val")
    train_groups, val_groups = sequence_groups(train), sequence_groups(val)
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
    history = []; best = None
    for epoch in range(1, args.epochs + 1):
        train_row = train_epoch(model, encoder, decoder, train_groups, optimizer); val_row = evaluate(model, encoder, decoder, val_groups); row = {"epoch": epoch, "train": train_row, "validation": val_row}; history.append(row); print(json.dumps(row, sort_keys=True), flush=True)
        out = Path(args.output); out.mkdir(parents=True, exist_ok=True); payload = {"experiment": "v2_stage_a", "epoch": epoch, "encoder_state_dict": encoder.state_dict(), "decoder_state_dict": decoder.state_dict(), "metrics": val_row, "config": {"lambda_sem": LAMBDA_SEM, "lr": LR}}
        torch.save(payload, out / f"epoch_{epoch:03d}.pt")
        score = val_row["mIoU_rec"]
        if best is None or score > best["mIoU_rec"] + MIN_MIOU_IMPROVEMENT: best = {"epoch": epoch, "mIoU_rec": score, "delta_mIoU_rec": val_row["delta_mIoU_rec"]}; torch.save(payload, out / "best.pt")
        # Do not judge the first three epochs.  Thereafter stop only when the
        # best representation is still below the hard -0.5 pp boundary and
        # has failed to improve by the numerical tolerance for PATIENCE epochs.
        if epoch >= 3 and epoch - best["epoch"] >= args.patience and best["delta_mIoU_rec"] < -0.005: break
    result = {"experiment": "Predify V2-Staged Stage A (relaxed early gate)", "trainable_modules": ["Z4PredictiveSemanticEncoder", "Z4PredictiveSemanticDecoder"], "frozen_modules": ["Host", "Adapter", "Writeback", "segmentation decoder", "Predictor", "Update Head"], "min_mIoU_improvement": MIN_MIOU_IMPROVEMENT, "no_go_after_epoch": 3, "patience": args.patience, "stop_delta_mIoU_rec": -0.005, "history": history, "best": best}
    result_dir = Path(args.result_output); result_dir.mkdir(parents=True, exist_ok=True); (result_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n"); (result_dir / "README.md").write_text("# V2-Staged Stage A (relaxed gate)\nEpochs 1-3 are never stopped by the mIoU gate. From Epoch 3, stopping requires three epochs without a >=1e-4 mIoU improvement and best delta below -0.5 pp.\n")


if __name__ == "__main__": main()
