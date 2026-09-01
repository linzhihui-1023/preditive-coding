"""Small, diagnostic comparison of old, semantic-only, and decoupled correction.

This entry point intentionally uses only the first few complete sequences.  It is
not a replacement for the formal KITTI-STEP training/evaluation protocols.
"""
import argparse
import csv
import json
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset, semantic_mask_from_panoptic_png
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import sequence_groups
from predify2021.mce_scores.evaluate_kitti_step_static_baseline import compute_iou, update_confusion_matrix
from predify2021.mce_scores.kitti_step_persistent_blur import BLUR_KERNEL_SIZE, BLUR_SIGMA_LEVELS, warmup_frame_count
from predify2021.mce_scores.role_separated_direct_state_correction import load_image, load_role_components, make_paths, next_role_prediction, zero_state
from predify2021.mce_scores.role_separated_dynamic_error_correction import error_state, residual_writeback_host_feature
from predify2021.mce_scores.video_metrics import VideoConsistency, weighted_iou
from predify2021.model_factory.deeplabv3plus_resnet50 import HostFeature, UnifiedFeatures
from predify2021.model_factory.deeplabv3plus_resnet50.semantic_temporal_error_correction import (
    DecoupledSemanticTemporalErrorCorrection,
    ExplicitSemanticCorrection,
    SemanticTemporalErrorCorrection,
)

SEED, EPOCHS, PATIENCE, LR, WEIGHT_DECAY, DISTILL_WEIGHT, BPTT = 0, 5, 2, 1e-4, 0.01, 0.5, 4


def _first_groups(dataset, count):
    groups = sequence_groups(dataset)
    return {key: groups[key] for key in sorted(groups)[:count]}


def _blur(image, frame, total, severity):
    if severity == 0 or frame < warmup_frame_count(total):
        return image
    sigma = BLUR_SIGMA_LEVELS[min(severity - 1, len(BLUR_SIGMA_LEVELS) - 1)]
    import torchvision.transforms.functional as TF
    return TF.gaussian_blur(image, [BLUR_KERNEL_SIZE] * 2, [sigma] * 2)


def _host_logits(model, raw, observation, posterior, size):
    delta = UnifiedFeatures(posterior.z1 - observation.z1, torch.zeros_like(observation.z2), torch.zeros_like(observation.z3), posterior.z4 - observation.z4)
    host = residual_writeback_host_feature(model, raw, delta, size)
    return model.decode_from_host_feature(host)


def _make_models():
    old = torch.nn.ModuleList([SemanticTemporalErrorCorrection(), SemanticTemporalErrorCorrection()]).cuda()
    full = torch.nn.ModuleList([DecoupledSemanticTemporalErrorCorrection(), DecoupledSemanticTemporalErrorCorrection()]).cuda()
    semantic = torch.nn.ModuleList([ExplicitSemanticCorrection(), ExplicitSemanticCorrection()]).cuda()
    for index in range(2):
        semantic[index].load_state_dict(full[index].semantic_correction.state_dict())
    return old, semantic, full


def _step_correction(kind, modules, observation, predicted, reference, hidden):
    if kind == "semantic-only":
        p1, _ = modules[0].forward_semantic_only(observation.z1, reference.z1)
        p4, _ = modules[1].forward_semantic_only(observation.z4, reference.z4)
        return UnifiedFeatures(p1, observation.z2, observation.z3, p4), hidden, None
    if kind == "old":
        out = [modules[i](getattr(observation, f"z{j}"), getattr(predicted, f"z{j}"), getattr(reference, f"z{j}"), hidden[i] if hidden is not None else None) for i, j in enumerate((1, 4))]
        return UnifiedFeatures(observation.z1 + out[0][4], observation.z2, observation.z3, observation.z4 + out[1][4]), (out[0][3], out[1][3]), None
    out = [modules[i](getattr(observation, f"z{j}"), getattr(predicted, f"z{j}"), getattr(reference, f"z{j}"), hidden[i] if hidden is not None else None) for i, j in enumerate((1, 4))]
    values = {"z1": out[0][2], "z4": out[1][2]}
    return UnifiedFeatures(observation.z1 + values["z1"]["delta"], observation.z2, observation.z3, observation.z4 + values["z4"]["delta"]), (out[0][1], out[1][1]), values


def _run(model, predictor, corrections, groups, kind, optimizer=None, severity=1, train=False):
    model.eval(); predictor.eval(); corrections.train(train)
    confusion = torch.zeros((19, 19), dtype=torch.int64) if not train else None
    vc = VideoConsistency(("model",)) if not train else None
    total_loss, frames, temporal_losses, gate_sum, delta_sum = 0.0, 0, 0.0, 0.0, 0.0
    for samples in groups.values():
        hidden = None; pending_prediction = None; predictor_hidden = predictor.initial_state(); pending_semantic = None; pending_dynamics = None; chunk = []
        if vc: vc.reset_sequence()
        for frame, sample in enumerate(samples):
            image = load_image(sample); noisy = _blur(image, frame, len(samples), severity)
            with torch.no_grad():
                raw = model.extract_backbone_features(noisy); observation = model.encode_backbone_features(raw)
                clean_raw = model.extract_backbone_features(image); clean_state = model.encode_backbone_features(clean_raw)
                clean_logits = model.decode_from_host_feature(HostFeature(clean_raw.c4, clean_raw.c1, tuple(image.shape[-2:])))
                if frame == 0:
                    pending_dynamics, pending_semantic, *predictor_hidden = next_role_prediction(predictor, observation, zero_state(observation), predictor_hidden); continue
                error = error_state(observation, pending_dynamics)
                if frame == 1:
                    pending_dynamics, pending_semantic, *predictor_hidden = next_role_prediction(predictor, observation, error, predictor_hidden); continue
            if kind == "full" and pending_prediction is not None:
                temporal_losses += (
                    F.smooth_l1_loss(pending_prediction[0], error.z1.detach())
                    + F.smooth_l1_loss(pending_prediction[1], error.z4.detach())
                ).item()
            posterior, hidden, values = _step_correction(kind, corrections, observation, pending_dynamics, pending_semantic, hidden)
            logits = _host_logits(model, raw, observation, posterior, tuple(image.shape[-2:]))
            mask = semantic_mask_from_panoptic_png(sample["mask_path"]).cuda()
            loss = F.cross_entropy(logits, mask.unsqueeze(0), ignore_index=255) + DISTILL_WEIGHT * F.kl_div(F.log_softmax(logits, 1), F.softmax(clean_logits.detach(), 1), reduction="none").sum(1).mean()
            if kind == "full":
                loss = loss + F.smooth_l1_loss(values["z1"]["predicted_next_task_error"], error.z1.detach())
                pending_prediction = (
                    values["z1"]["predicted_next_task_error"],
                    values["z4"]["predicted_next_task_error"],
                )
                gate_sum += values["z1"]["gate"].mean().item(); delta_sum += values["z1"]["delta"].abs().mean().item()
            if train:
                chunk.append(loss)
                if len(chunk) == BPTT:
                    optimizer.zero_grad(set_to_none=True); torch.stack(chunk).mean().backward(); optimizer.step(); chunk.clear(); hidden = tuple(h.detach() for h in hidden) if hidden is not None else None
            else:
                prediction = logits.argmax(1).squeeze(0).cpu(); update_confusion_matrix(confusion, prediction, mask.cpu()); vc.append(mask.cpu(), {"model": prediction})
            total_loss += loss.detach().item(); frames += 1
            if not train:
                hidden = tuple(value.detach() for value in hidden) if hidden is not None else None
                if pending_prediction is not None:
                    pending_prediction = tuple(value.detach() for value in pending_prediction)
            with torch.no_grad(): pending_dynamics, pending_semantic, *predictor_hidden = next_role_prediction(predictor, observation, error, predictor_hidden)
        if train and chunk:
            optimizer.zero_grad(set_to_none=True); torch.stack(chunk).mean().backward(); optimizer.step()
    result = {"loss": total_loss / max(frames, 1), "frames": frames, "finite": bool(torch.isfinite(torch.tensor(total_loss)))}
    if not train:
        result.update({"mIoU": float(torch.nanmean(compute_iou(confusion)).item()), "wIoU": weighted_iou(confusion), "mVC8": vc.means()[8]["model"], "mVC16": vc.means()[16]["model"]})
    if kind == "full": result.update({"mean_gate": gate_sum / max(frames, 1), "mean_abs_delta": delta_sum / max(frames, 1)})
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--root", default="/home/lin/predify/kitti_step"); parser.add_argument("--output", default="results/kitti_step_convgru_decoupling_quick_c8f1860"); args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    paths = make_paths()
    model, predictor = load_role_components(paths["static"], paths["adapter"], paths["predictor"], paths["writeback"])
    model.requires_grad_(False); predictor.requires_grad_(False)
    train = _first_groups(KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "train"), 4)
    val = _first_groups(KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "val"), 3)
    results = {}; out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    for kind in ("old", "semantic-only", "full"):
        old, semantic, full = _make_models(); modules = {"old": old, "semantic-only": semantic, "full": full}[kind]; optimizer = torch.optim.AdamW(modules.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        best = None; stale = 0
        for epoch in range(EPOCHS):
            _run(model, predictor, modules, train, kind, optimizer, 1, True)
            metrics = _run(model, predictor, modules, val, kind, None, 1, False)
            if best is None or metrics["mIoU"] > best["mIoU"]: best = metrics; stale = 0; torch.save(modules.state_dict(), out / f"{kind.replace('-', '_')}.pt")
            else: stale += 1
            if stale >= PATIENCE: break
        for condition, severity in (("Clean", 0), ("S3", 3), ("S5", 5)):
            results[(kind, condition)] = _run(model, predictor, modules, val, kind, None, severity, False)
    rows = [{"model": k[0], "condition": k[1], **v} for k, v in results.items()]
    with (out / "comparison.csv").open("w", newline="") as stream: writer = csv.DictWriter(stream, fieldnames=sorted({key for row in rows for key in row})); writer.writeheader(); writer.writerows(rows)
    summary = {"config": {"seed": SEED, "epochs": EPOCHS, "lr": LR, "weight_decay": WEIGHT_DECAY, "bptt": BPTT, "distill_weight": DISTILL_WEIGHT}, "train_sequences": list(train), "val_sequences": list(val), "results": rows}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__": main()
