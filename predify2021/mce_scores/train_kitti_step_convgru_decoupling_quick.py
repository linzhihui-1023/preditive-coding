"""Accelerated diagnostic comparison for ConvGRU role decoupling.

Frozen Host/Predictor work is shared once per frame. This is a diagnostic
screening entry point, not a formal KITTI-STEP protocol.
"""
import argparse
import copy
import csv
import json
import random
import time
from pathlib import Path

import torch
from PIL import Image
from torch.nn import functional as F

from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset, pil_rgb_to_unit_tensor, semantic_mask_from_panoptic_png
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import sequence_groups
from predify2021.mce_scores.evaluate_kitti_step_static_baseline import compute_iou, update_confusion_matrix
from predify2021.mce_scores.kitti_step_persistent_blur import BLUR_KERNEL_SIZE, BLUR_SIGMA_LEVELS, persistent_gaussian_blur, warmup_frame_count
from predify2021.mce_scores.role_separated_direct_state_correction import load_role_components, make_paths, next_role_prediction, zero_state
from predify2021.mce_scores.role_separated_dynamic_error_correction import error_state, residual_writeback_host_feature
from predify2021.mce_scores.video_metrics import VideoConsistency, weighted_iou
from predify2021.model_factory.deeplabv3plus_resnet50 import HostFeature, UnifiedFeatures
from predify2021.model_factory.deeplabv3plus_resnet50.semantic_temporal_error_correction import DecoupledSemanticTemporalErrorCorrection, ExplicitSemanticCorrection, SemanticTemporalErrorCorrection

SEED, EPOCHS, PATIENCE, LR, WEIGHT_DECAY, DISTILL_WEIGHT, BPTT = 0, 3, 2, 1e-4, 0.01, 0.5, 4
KINDS = ("old", "semantic-only", "full")


def first_groups(dataset, count):
    groups = sequence_groups(dataset)
    return {key: groups[key] for key in sorted(groups)[:count]}


def preload(groups):
    """Cache only decoded RGB and semantic masks in RAM."""
    cached = {}
    for sequence, samples in groups.items():
        cached[sequence] = []
        for sample in samples:
            with Image.open(sample["image_path"]) as image:
                rgb = (pil_rgb_to_unit_tensor(image.convert("RGB")) * 255).to(torch.uint8)
            cached[sequence].append({"rgb": rgb, "mask": semantic_mask_from_panoptic_png(sample["mask_path"]).to(torch.uint8)})
    return cached


def diagnostic_blur(image, frame, total, condition):
    if condition == "Clean" or frame < warmup_frame_count(total):
        return image
    import torchvision.transforms.functional as TF
    sigma = 2.25 if condition == "Blur-Mid" else 3.0
    return TF.gaussian_blur(image, [BLUR_KERNEL_SIZE] * 2, [sigma] * 2)


def make_models():
    old = torch.nn.ModuleList([SemanticTemporalErrorCorrection(), SemanticTemporalErrorCorrection()]).cuda()
    reference = torch.nn.ModuleList([DecoupledSemanticTemporalErrorCorrection(), DecoupledSemanticTemporalErrorCorrection()]).cuda()
    semantic_init = copy.deepcopy([module.semantic_correction.state_dict() for module in reference])
    semantic = torch.nn.ModuleList([ExplicitSemanticCorrection(), ExplicitSemanticCorrection()]).cuda()
    full = torch.nn.ModuleList([DecoupledSemanticTemporalErrorCorrection(), DecoupledSemanticTemporalErrorCorrection()]).cuda()
    for index in range(2):
        semantic[index].load_state_dict(semantic_init[index], strict=True)
        full[index].semantic_correction.load_state_dict(semantic_init[index], strict=True)
    return {"old": old, "semantic-only": semantic, "full": full}


def correction_step(kind, modules, observation, predicted, reference, hidden):
    if kind == "semantic-only":
        p1, _ = modules[0].forward_semantic_only(observation.z1, reference.z1)
        p4, _ = modules[1].forward_semantic_only(observation.z4, reference.z4)
        return UnifiedFeatures(p1, observation.z2, observation.z3, p4), hidden, None
    outputs = [modules[index](getattr(observation, f"z{stage}"), getattr(predicted, f"z{stage}"), getattr(reference, f"z{stage}"), hidden[index] if hidden is not None else None) for index, stage in enumerate((1, 4))]
    if kind == "old":
        return UnifiedFeatures(observation.z1 + outputs[0][4], observation.z2, observation.z3, observation.z4 + outputs[1][4]), (outputs[0][3], outputs[1][3]), None
    values = {"z1": outputs[0][2], "z4": outputs[1][2]}
    return UnifiedFeatures(observation.z1 + values["z1"]["delta"], observation.z2, observation.z3, observation.z4 + values["z4"]["delta"]), (outputs[0][1], outputs[1][1]), values


def corrected_logits(model, raw, observation, posterior, output_size):
    delta = UnifiedFeatures(posterior.z1 - observation.z1, torch.zeros_like(observation.z2), torch.zeros_like(observation.z3), posterior.z4 - observation.z4)
    return model.decode_from_host_feature(residual_writeback_host_feature(model, raw, delta, output_size))


def flush(states, optimizers, kinds):
    for kind in kinds:
        state = states[kind]
        if state["chunk"]:
            optimizers[kind].zero_grad(set_to_none=True)
            torch.stack(state["chunk"]).mean().backward()
            optimizers[kind].step()
            state["chunk"].clear()
        if state["hidden"] is not None:
            state["hidden"] = tuple(value.detach() for value in state["hidden"])
        if state["pending"] is not None:
            state["pending"] = tuple(value.detach() for value in state["pending"])


def run_shared(model, predictor, modules, cached, train=False, optimizers=None, condition="Clean", bf16=False, log=None):
    for kind in KINDS:
        modules[kind].train(train)
    states = {kind: {"hidden": None, "pending": None, "predictor_hidden": predictor.initial_state(), "pending_dynamics": None, "pending_semantic": None, "chunk": []} for kind in KINDS}
    confusion = {kind: torch.zeros((19, 19), dtype=torch.int64) for kind in KINDS} if not train else None
    vcs = {kind: VideoConsistency(("model",)) for kind in KINDS} if not train else None
    sums = {kind: {name: torch.zeros((), device="cuda") for name in ("loss", "temporal_loss", "gate", "delta")} for kind in KINDS}
    counts = {kind: {name: 0 for name in ("effective_frames", "loss_frames", "metric_frames", "gate_frames")} for kind in KINDS}
    total_sequence_frames = 0; started = time.perf_counter()
    for sequence, frames in cached.items():
        for kind in KINDS:
            states[kind].update(hidden=None, pending=None, predictor_hidden=predictor.initial_state(), pending_dynamics=None, pending_semantic=None, chunk=[])
            if vcs: vcs[kind].reset_sequence()
        for frame, entry in enumerate(frames):
            total_sequence_frames += 1
            image = entry["rgb"].float().div_(255).unsqueeze(0).cuda(non_blocking=True)
            noisy = persistent_gaussian_blur(image, frame, len(frames)) if train else diagnostic_blur(image, frame, len(frames), condition)
            context = torch.enable_grad() if train else torch.no_grad()
            with context, torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16):
                raw = model.extract_backbone_features(noisy); observation = model.encode_backbone_features(raw)
                clean_logits = None
                if train:
                    clean_raw = model.extract_backbone_features(image)
                    clean_logits = model.decode_from_host_feature(HostFeature(clean_raw.c4, clean_raw.c1, tuple(image.shape[-2:])))
                shared = states["full"]
                if frame == 0:
                    dyn, sem, *ph = next_role_prediction(predictor, observation, zero_state(observation), shared["predictor_hidden"])
                    for kind in KINDS: states[kind].update(pending_dynamics=dyn, pending_semantic=sem, predictor_hidden=ph)
                    continue
                current_error = error_state(observation, shared["pending_dynamics"])
                if frame == 1:
                    dyn, sem, *ph = next_role_prediction(predictor, observation, current_error, shared["predictor_hidden"])
                    for kind in KINDS: states[kind].update(pending_dynamics=dyn, pending_semantic=sem, predictor_hidden=ph)
                    continue
                effective = frame >= warmup_frame_count(len(frames))
                for kind in KINDS:
                    state = states[kind]
                    posterior, state["hidden"], values = correction_step(kind, modules[kind], observation, state["pending_dynamics"], state["pending_semantic"], state["hidden"])
                    if kind == "full":
                        state["pending"] = (values["z1"]["predicted_next_task_error"], values["z4"]["predicted_next_task_error"])
                    if not effective:
                        if state["hidden"] is not None: state["hidden"] = tuple(value.detach() for value in state["hidden"])
                        continue
                    counts[kind]["effective_frames"] += 1
                    logits = corrected_logits(model, raw, observation, posterior, tuple(image.shape[-2:])); mask = entry["mask"].long().cuda(non_blocking=True)
                    if train:
                        segmentation = F.cross_entropy(logits, mask.unsqueeze(0), ignore_index=255) + DISTILL_WEIGHT * F.kl_div(F.log_softmax(logits, 1), F.softmax(clean_logits.detach(), 1), reduction="none").sum(1).mean()
                        temporal = torch.zeros((), device="cuda")
                        if kind == "full":
                            current_task = (values["z1"]["task_error"], values["z4"]["task_error"])
                            if state["pending"] is not None: temporal = F.smooth_l1_loss(state["pending"][0], current_task[0].detach()) + F.smooth_l1_loss(state["pending"][1], current_task[1].detach())
                            sums[kind]["temporal_loss"] += temporal.detach(); sums[kind]["gate"] += values["z1"]["gate"].mean().detach(); sums[kind]["delta"] += values["z1"]["delta"].abs().mean().detach(); counts[kind]["gate_frames"] += 1
                        loss = segmentation + temporal; state["chunk"].append(loss); sums[kind]["loss"] += loss.detach(); counts[kind]["loss_frames"] += 1
                        if len(state["chunk"]) == BPTT: flush(states, optimizers, [kind])
                    else:
                        prediction = logits.argmax(1).squeeze(0).cpu(); update_confusion_matrix(confusion[kind], prediction, mask.cpu()); vcs[kind].append(mask.cpu(), {"model": prediction}); counts[kind]["metric_frames"] += 1
                        if kind == "full": sums[kind]["gate"] += values["z1"]["gate"].mean(); sums[kind]["delta"] += values["z1"]["delta"].abs().mean(); counts[kind]["gate_frames"] += 1
                        state["hidden"] = tuple(value.detach() for value in state["hidden"]) if state["hidden"] is not None else None; state["pending"] = tuple(value.detach() for value in state["pending"]) if state["pending"] is not None else None
                dyn, sem, *ph = next_role_prediction(predictor, observation, current_error, shared["predictor_hidden"])
                for kind in KINDS: states[kind].update(pending_dynamics=dyn, pending_semantic=sem, predictor_hidden=ph)
            if log and frame % 25 == 0: log(f"condition={condition} sequence={sequence} frame={frame}/{len(frames)}")
        if train: flush(states, optimizers, KINDS)
    elapsed = max(time.perf_counter() - started, 1e-6)
    result = {kind: {"total_sequence_frames": total_sequence_frames, **counts[kind], "fps": total_sequence_frames / elapsed, "finite": bool(torch.isfinite(sums[kind]["loss"]).item()), "mean_loss": float((sums[kind]["loss"] / max(counts[kind]["loss_frames"], 1)).item()), "mean_temporal_loss": float((sums[kind]["temporal_loss"] / max(counts[kind]["gate_frames"], 1)).item())} for kind in KINDS}
    if not train:
        for kind in KINDS: result[kind].update({"mIoU": float(torch.nanmean(compute_iou(confusion[kind])).item()), "wIoU": weighted_iou(confusion[kind]), "mVC8": vcs[kind].means()[8]["model"], "mVC16": vcs[kind].means()[16]["model"]})
    result["full"].update({"mean_gate": float((sums["full"]["gate"] / max(counts["full"]["gate_frames"], 1)).item()), "mean_abs_delta": float((sums["full"]["delta"] / max(counts["full"]["gate_frames"], 1)).item())})
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--root", default="/home/lin/predify/kitti_step"); parser.add_argument("--output", default="results/kitti_step_convgru_decoupling_quick_61a52e7"); parser.add_argument("--epochs", type=int, default=EPOCHS); parser.add_argument("--fast-bf16", action="store_true"); args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    paths = make_paths(); model, predictor = load_role_components(paths["static"], paths["adapter"], paths["predictor"], paths["writeback"]); model.requires_grad_(False); predictor.requires_grad_(False)
    train = preload(first_groups(KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "train"), 2)); val = preload(first_groups(KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "val"), 2)); modules = make_models(); optimizers = {kind: torch.optim.AdamW(modules[kind].parameters(), lr=LR, weight_decay=WEIGHT_DECAY) for kind in KINDS}; out = Path(args.output); out.mkdir(parents=True, exist_ok=True); log_file = (out / "run.log").open("a")
    def log(message):
        line = f"{time.strftime('%F %T')} {message}"; print(line, flush=True); log_file.write(line + "\n"); log_file.flush()
    best = {kind: {"mIoU": float("-inf"), "epoch": 0} for kind in KINDS}; progress = []
    for epoch in range(1, args.epochs + 1):
        train_result = run_shared(model, predictor, modules, train, True, optimizers, bf16=args.fast_bf16, log=log); val_result = run_shared(model, predictor, modules, val, False, condition="Clean", bf16=args.fast_bf16, log=log)
        for kind in KINDS:
            if val_result[kind]["mIoU"] > best[kind]["mIoU"]: best[kind] = {"mIoU": val_result[kind]["mIoU"], "epoch": epoch}; torch.save(modules[kind].state_dict(), out / {"old": "old.pt", "semantic-only": "semantic_only.pt", "full": "full.pt"}[kind])
        progress.append({"epoch": epoch, "train": train_result, "val": val_result, "best": best}); (out / "progress.json").write_text(json.dumps(progress, indent=2)); log(f"epoch={epoch} best_val_mIoU=" + json.dumps({k: v["mIoU"] for k, v in best.items()}))
    for kind in KINDS: modules[kind].load_state_dict(torch.load(out / {"old": "old.pt", "semantic-only": "semantic_only.pt", "full": "full.pt"}[kind], map_location="cuda", weights_only=True))
    rows = []
    for condition in ("Clean", "Blur-Mid", "Blur-Max"):
        evaluated = run_shared(model, predictor, modules, val, False, condition=condition, bf16=args.fast_bf16, log=log)
        rows.extend({"model": kind, "condition": condition, **evaluated[kind]} for kind in KINDS)
    by = {(row["model"], row["condition"]): row for row in rows}; comparisons = {}
    for other in ("semantic-only", "old"):
        comparisons[f"Full - {other}"] = {f"{condition} ΔmIoU": by[("full", condition)]["mIoU"] - by[(other, condition)]["mIoU"] for condition in ("Clean", "Blur-Mid", "Blur-Max")}; comparisons[f"Full - {other}"].update({f"{condition} ΔmVC16": by[("full", condition)]["mVC16"] - by[(other, condition)]["mVC16"] for condition in ("Blur-Mid", "Blur-Max")})
    with (out / "comparison.csv").open("w", newline="") as stream: writer = csv.DictWriter(stream, fieldnames=sorted({key for row in rows for key in row})); writer.writeheader(); writer.writerows(rows)
    temporal_state = "GO" if ((comparisons["Full - semantic-only"]["Blur-Mid ΔmIoU"] + comparisons["Full - semantic-only"]["Blur-Max ΔmIoU"]) / 2 >= 0.005 or (comparisons["Full - semantic-only"]["Blur-Mid ΔmVC16"] + comparisons["Full - semantic-only"]["Blur-Max ΔmVC16"]) / 2 >= 0.01) else "NO-GO"
    decoupled = "GO" if ((comparisons["Full - old"]["Blur-Mid ΔmIoU"] + comparisons["Full - old"]["Blur-Max ΔmIoU"]) / 2 >= 0.005 or (comparisons["Full - old"]["Blur-Mid ΔmVC16"] + comparisons["Full - old"]["Blur-Max ΔmVC16"]) / 2 >= 0.01) and all(by[("full", condition)]["mIoU"] - by[("old", condition)]["mIoU"] > -0.005 for condition in ("Clean",)) else "NO-GO"
    summary.update({"judgement": {"TEMPORAL STATE": temporal_state, "DECOUPLED ARCHITECTURE": decoupled}}); (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n"); (out / "README.md").write_text("# Quick ConvGRU decoupling screening\n\nBlur-Mid/Blur-Max are diagnostic pressure conditions, not formal corruption severities.\n\n" + json.dumps(comparisons, indent=2) + f"\n\nTEMPORAL STATE: {temporal_state}\nDECOUPLED ARCHITECTURE: {decoupled}\n"); log_file.close()


if __name__ == "__main__": main()
