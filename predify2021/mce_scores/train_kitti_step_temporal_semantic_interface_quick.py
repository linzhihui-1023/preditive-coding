"""Four-way diagnostic screen for the temporal/semantic interface.

The frozen host, predictor, writeback path, data protocol, and one-step
temporal objective are inherited from ``train_kitti_step_convgru_decoupling_quick``.
This entry point only changes the H -> semantic correction interface and writes
to a new result directory.
"""
import argparse
import copy
import csv
import json
import random
import subprocess
import time
from pathlib import Path

import torch

from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset
from predify2021.mce_scores.role_separated_direct_state_correction import load_role_components, make_paths
from predify2021.mce_scores.train_kitti_step_convgru_decoupling_quick import (
    BPTT,
    DISTILL_WEIGHT,
    EPOCHS,
    LR,
    PATIENCE,
    SEED,
    WEIGHT_DECAY,
    first_groups,
    preload,
    run_shared,
)
from predify2021.mce_scores import train_kitti_step_convgru_decoupling_quick as quick
from predify2021.model_factory.deeplabv3plus_resnet50.semantic_temporal_error_correction import (
    DecoupledSemanticTemporalErrorCorrection,
    TemporalConditionedSemanticCorrection,
)

VARIANTS = ("full-scalar", "full-channel-gate", "full-temporal-semantic", "full-channel-temporal")
quick.KINDS = VARIANTS
CHECKPOINT_NAMES = {
    "full-scalar": "full_scalar.pt",
    "full-channel-gate": "full_channel_gate.pt",
    "full-temporal-semantic": "full_temporal_semantic.pt",
    "full-channel-temporal": "full_channel_temporal.pt",
}
DISPLAY = {
    "full-scalar": "F0",
    "full-channel-gate": "F1",
    "full-temporal-semantic": "F2",
    "full-channel-temporal": "F3",
}


def _copy_temporal_branch(source, target):
    for name in ("correlation", "encoder", "error_state", "temporal_prediction"):
        getattr(target, name).load_state_dict(getattr(source, name).state_dict())


@torch.no_grad()
def _copy_scalar_gate(source, target):
    weight = source.gate.gate_head.weight
    bias = source.gate.gate_head.bias
    target.gate.gate_head.weight.copy_(weight.repeat(target.gate.gate_channels, 1, 1, 1))
    target.gate.gate_head.bias.copy_(bias.repeat(target.gate.gate_channels))


def make_variants():
    """Create fair F0--F3 initializations on CUDA."""
    f0 = torch.nn.ModuleList([DecoupledSemanticTemporalErrorCorrection(gate_channels=1) for _ in range(2)]).cuda()
    f1 = torch.nn.ModuleList([DecoupledSemanticTemporalErrorCorrection(gate_channels=128) for _ in range(2)]).cuda()
    f2 = torch.nn.ModuleList([
        DecoupledSemanticTemporalErrorCorrection(
            gate_channels=1,
            semantic_correction=TemporalConditionedSemanticCorrection(channels=128, baseline=f0[index].semantic_correction),
        ) for index in range(2)
    ]).cuda()
    f3 = torch.nn.ModuleList([
        DecoupledSemanticTemporalErrorCorrection(
            gate_channels=128,
            semantic_correction=TemporalConditionedSemanticCorrection(channels=128, baseline=f0[index].semantic_correction),
        ) for index in range(2)
    ]).cuda()
    for index in range(2):
        for target in (f1[index], f2[index], f3[index]):
            _copy_temporal_branch(f0[index], target)
        f1[index].semantic_correction.load_state_dict(f0[index].semantic_correction.state_dict())
        _copy_scalar_gate(f0[index], f1[index]); _copy_scalar_gate(f0[index], f2[index]); _copy_scalar_gate(f0[index], f3[index])
    return {"full-scalar": f0, "full-channel-gate": f1, "full-temporal-semantic": f2, "full-channel-temporal": f3}


def _grad_norm(module, prefixes):
    total = 0.0
    for name, parameter in module.named_parameters():
        if any(name.startswith(prefix) for prefix in prefixes) and parameter.grad is not None:
            total += float(parameter.grad.detach().abs().sum().item())
    return total


def _synthetic_variants():
    """CPU-sized replicas used for deterministic initialization/gradient checks."""
    f0 = DecoupledSemanticTemporalErrorCorrection(channels=8, projection_channels=2, gate_channels=1)
    f1 = DecoupledSemanticTemporalErrorCorrection(channels=8, projection_channels=2, gate_channels=8)
    f2 = DecoupledSemanticTemporalErrorCorrection(
        channels=8,
        projection_channels=2,
        gate_channels=1,
        semantic_correction=TemporalConditionedSemanticCorrection(channels=8, baseline=f0.semantic_correction),
    )
    f3 = DecoupledSemanticTemporalErrorCorrection(
        channels=8,
        projection_channels=2,
        gate_channels=8,
        semantic_correction=TemporalConditionedSemanticCorrection(channels=8, baseline=f0.semantic_correction),
    )
    for target in (f1, f2, f3):
        _copy_temporal_branch(f0, target)
    f1.semantic_correction.load_state_dict(f0.semantic_correction.state_dict())
    _copy_scalar_gate(f0, f1)
    _copy_scalar_gate(f0, f2)
    _copy_scalar_gate(f0, f3)
    return {"full-scalar": f0, "full-channel-gate": f1, "full-temporal-semantic": f2, "full-channel-temporal": f3}


def interface_sanity():
    torch.manual_seed(SEED)
    modules = _synthetic_variants()
    observation = torch.randn(2, 8, 7, 9)
    predicted = torch.randn_like(observation)
    semantic = torch.randn_like(observation)
    hidden = torch.randn_like(observation) + 0.25
    outputs = {name: module(observation, predicted, semantic, hidden) for name, module in modules.items()}
    posterior_ref = outputs["full-scalar"][0]
    logits_head = torch.nn.Conv2d(8, 5, 1)
    logits_ref = logits_head(posterior_ref)
    posterior_diffs = {}
    logits_diffs = {}
    for name, (posterior, _, _) in outputs.items():
        posterior_diffs[name] = float((posterior - posterior_ref).abs().max().item())
        logits_diffs[name] = float((logits_head(posterior) - logits_ref).abs().max().item())
        assert posterior_diffs[name] < 1e-6, (name, posterior_diffs[name])
        assert logits_diffs[name] < 1e-6, (name, logits_diffs[name])
        assert float(modules[name].gate(torch.zeros_like(hidden)).abs().max().detach()) == 0.0

    gradients = {}
    for name, module in modules.items():
        module.zero_grad(set_to_none=True)
        module(observation, predicted, semantic, hidden)[0].mean().backward()
        gradients[name] = {
            "seg_gate": _grad_norm(module, ("gate",)),
            "seg_semantic": _grad_norm(module, ("semantic_correction",)),
            "seg_temporal": _grad_norm(module, ("correlation", "encoder", "error_state", "temporal_prediction")),
        }
        assert gradients[name]["seg_gate"] > 0.0 and gradients[name]["seg_semantic"] > 0.0
        assert gradients[name]["seg_temporal"] == 0.0
        module.zero_grad(set_to_none=True)
        module(observation, predicted, semantic, hidden)[2]["predicted_next_task_error"].mean().backward()
        gradients[name].update({
            "temporal_branch": _grad_norm(module, ("correlation", "encoder", "error_state", "temporal_prediction")),
            "temporal_gate": _grad_norm(module, ("gate",)),
            "temporal_semantic": _grad_norm(module, ("semantic_correction",)),
        })
        assert gradients[name]["temporal_branch"] > 0.0
        assert gradients[name]["temporal_gate"] == 0.0 and gradients[name]["temporal_semantic"] == 0.0

    indexing = {}
    for total in (601, 503):
        warmup = quick.warmup_frame_count(total)
        pairs = [(frame - 1, frame) for frame in range(2, total) if frame >= warmup]
        indexing[str(total)] = {
            "warmup": warmup,
            "first_temporal_pair": [2, 3] if total > 3 and warmup <= 2 else (list(pairs[1]) if len(pairs) > 1 else None),
            "temporal_pairs": len(pairs) - 1,
            "bptt": BPTT,
            "sequence_boundary_reset": True,
        }
        assert warmup < total and indexing[str(total)]["sequence_boundary_reset"]
    return {"posterior_max_abs_diff": posterior_diffs, "logits_max_abs_diff": logits_diffs, "gradients": gradients, "indexing": indexing}


def _load_old_reference():
    path = Path("results/kitti_step_convgru_decoupling_quick_0c83ad0/summary.json")
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return {
        condition: next((row for row in data.get("results", []) if row.get("model") == "old" and row.get("condition") == condition), None)
        for condition in ("Clean", "Blur-Mid", "Blur-Max")
    }


def _write_readme(out, summary, rows, comparisons, sanity):
    values = {(row["model"], row["epoch"], row["condition"]): row for row in rows}
    miou = ["| Model | Epoch | Clean | Blur-Mid | Blur-Max | Blur Mean |", "|---|---:|---:|---:|---:|---:|"]
    mvc = ["| Model | Epoch | Clean | Blur-Mid | Blur-Max | Blur Mean |", "|---|---:|---:|---:|---:|---:|"]
    for epoch in sorted({row["epoch"] for row in rows}):
        for kind in VARIANTS:
            blur_miou = sum(values[(kind, epoch, condition)]["mIoU"] for condition in ("Blur-Mid", "Blur-Max")) / 2
            blur_mvc = sum(values[(kind, epoch, condition)]["mVC16"] for condition in ("Blur-Mid", "Blur-Max")) / 2
            miou.append(f"| {DISPLAY[kind]} | {epoch} | {values[(kind, epoch, 'Clean')]['mIoU']:.6f} | {values[(kind, epoch, 'Blur-Mid')]['mIoU']:.6f} | {values[(kind, epoch, 'Blur-Max')]['mIoU']:.6f} | {blur_miou:.6f} |")
            mvc.append(f"| {DISPLAY[kind]} | {epoch} | {values[(kind, epoch, 'Clean')]['mVC16']:.6f} | {values[(kind, epoch, 'Blur-Mid')]['mVC16']:.6f} | {values[(kind, epoch, 'Blur-Max')]['mVC16']:.6f} | {blur_mvc:.6f} |")
    text = "# Temporal–semantic interface same-epoch quick screen\n\nDiagnostic Blur-Mid/Blur-Max conditions (sigma 2.25/3.0), not formal corruption severities. Every row is evaluated from its own epoch checkpoint.\n\n## mIoU\n\n" + "\n".join(miou) + "\n\n## mVC16\n\n" + "\n".join(mvc) + "\n\n## Deltas and judgement\n\n" + json.dumps(comparisons, indent=2) + "\n\n" + json.dumps(summary["judgement"], indent=2) + "\n\n## Initialization and gradient checks\n\n```json\n" + json.dumps(sanity, indent=2) + "\n```\n"
    (out / "README.md").write_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/lin/predify/kitti_step")
    parser.add_argument("--output", default="results/kitti_step_temporal_semantic_interface_quick_98ef901")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--fast-bf16", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    sanity = interface_sanity()
    paths = make_paths()
    model, predictor = load_role_components(paths["static"], paths["adapter"], paths["predictor"], paths["writeback"])
    model.requires_grad_(False); predictor.requires_grad_(False)
    train_dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "train")
    val_dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "val")
    train = preload(first_groups(train_dataset, 2)); val = preload(first_groups(val_dataset, 2))
    modules = make_variants()
    optimizers = {kind: torch.optim.AdamW(modules[kind].parameters(), lr=LR, weight_decay=WEIGHT_DECAY) for kind in VARIANTS}
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    log_file = (out / "run.log").open("a")
    def log(message):
        line = f"{time.strftime('%F %T')} {message}"; print(line, flush=True); log_file.write(line + "\n"); log_file.flush()
    log("interface_sanity=" + json.dumps(sanity, sort_keys=True))
    best = {kind: {"mIoU": float("-inf"), "epoch": 0} for kind in VARIANTS}
    progress = []
    for epoch in range(1, args.epochs + 1):
        train_result = run_shared(model, predictor, modules, train, True, optimizers, bf16=args.fast_bf16, log=log)
        val_result = run_shared(model, predictor, modules, val, False, condition="Clean", bf16=args.fast_bf16, log=log)
        for kind in VARIANTS:
            # Keep every epoch checkpoint so robustness comparisons use the
            # same training age for every interface variant.
            torch.save(modules[kind].state_dict(), out / f"{DISPLAY[kind]}_epoch{epoch}.pt")
            if val_result[kind]["mIoU"] > best[kind]["mIoU"]:
                best[kind] = {"mIoU": val_result[kind]["mIoU"], "epoch": epoch}
        progress.append({"epoch": epoch, "train": train_result, "val": val_result, "best": best})
        (out / "progress.json").write_text(json.dumps(progress, indent=2))
        log("epoch=" + str(epoch) + " best_val_mIoU=" + json.dumps({k: v["mIoU"] for k, v in best.items()}))
    rows = []
    parameter_counts = {kind: sum(parameter.numel() for parameter in modules[kind].parameters() if parameter.requires_grad) for kind in VARIANTS}
    for epoch in range(1, args.epochs + 1):
        for kind in VARIANTS:
            modules[kind].load_state_dict(torch.load(out / f"{DISPLAY[kind]}_epoch{epoch}.pt", map_location="cuda", weights_only=True))
        for condition in ("Clean", "Blur-Mid", "Blur-Max"):
            evaluated = run_shared(model, predictor, modules, val, False, condition=condition, bf16=args.fast_bf16, log=log)
            for kind in VARIANTS:
                rows.append({"model": kind, "variant": DISPLAY[kind], "epoch": epoch, "condition": condition, "trainable_params": parameter_counts[kind], "additional_params": parameter_counts[kind], **evaluated[kind]})
    by = {(row["model"], row["epoch"], row["condition"]): row for row in rows}
    blur_conditions = ("Blur-Mid", "Blur-Max")
    def blur_mean(metric, kind, epoch):
        return sum(by[(kind, epoch, condition)][metric] for condition in blur_conditions) / 2
    def mean_delta(metric, variant, epoch, baseline="full-scalar"):
        return blur_mean(metric, variant, epoch) - blur_mean(metric, baseline, epoch)
    comparisons = {}
    for kind in VARIANTS[1:]:
        comparisons[f"{DISPLAY[kind]} - F0"] = {}
        for epoch in range(1, args.epochs + 1):
            comparisons[f"{DISPLAY[kind]} - F0"].update({f"epoch{epoch}_blur_mIoU_delta": mean_delta("mIoU", kind, epoch), f"epoch{epoch}_blur_mVC16_delta": mean_delta("mVC16", kind, epoch)})
    final_epoch = args.epochs
    comparisons["same_epoch_final"] = {"F0_blur_mean_mIoU": blur_mean("mIoU", "full-scalar", final_epoch), "F1_blur_mean_mIoU": blur_mean("mIoU", "full-channel-gate", final_epoch), "F2_blur_mean_mIoU": blur_mean("mIoU", "full-temporal-semantic", final_epoch), "F3_blur_mean_mIoU": blur_mean("mIoU", "full-channel-temporal", final_epoch), "F1_minus_F0": mean_delta("mIoU", "full-channel-gate", final_epoch), "F3_minus_F0": mean_delta("mIoU", "full-channel-temporal", final_epoch)}
    f1_final_delta = comparisons["same_epoch_final"]["F1_minus_F0"]
    f3_final_delta = comparisons["same_epoch_final"]["F3_minus_F0"]
    channel_status = "RETEST / SUPPORTED" if f1_final_delta >= -0.005 or f3_final_delta >= -0.005 else "NO-GO"
    combined_go = comparisons["same_epoch_final"]["F3_blur_mean_mIoU"] >= comparisons["same_epoch_final"]["F0_blur_mean_mIoU"]
    old_reference = _load_old_reference()
    ideal = None
    if old_reference and all(old_reference.get(condition) for condition in blur_conditions):
        ideal = blur_mean("mIoU", "full-channel-temporal", final_epoch) >= sum(old_reference[condition]["mIoU"] for condition in blur_conditions) / 2 and all(by[("full-channel-temporal", final_epoch, condition)]["mVC16"] > old_reference[condition]["mVC16"] for condition in blur_conditions)
    judgement = {"CHANNEL-GATE": channel_status, "COMBINED": "GO" if combined_go else "NO-GO; F3 Blur Mean mIoU is below F0 at the same epoch", "IDEAL F3 VS OLD": ideal}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    summary = {"commit": commit, "config": {"seed": SEED, "epochs": args.epochs, "patience": PATIENCE, "lr": LR, "weight_decay": WEIGHT_DECAY, "distill_weight": DISTILL_WEIGHT, "bptt": BPTT, "precision": "bf16" if args.fast_bf16 else "fp32", "train_frames": 601, "val_frames": 503, "conditions": {"Clean": "none", "Blur-Mid": "sigma=2.25", "Blur-Max": "sigma=3.0"}}, "variants": {kind: {"label": DISPLAY[kind], "gate_channels": modules[kind][0].gate.gate_channels, "semantic_interface": "temporal-conditioned" if isinstance(modules[kind][0].semantic_correction, TemporalConditionedSemanticCorrection) else "frame-only", "trainable_params": parameter_counts[kind]} for kind in VARIANTS}, "train_sequences": list(train.keys()), "val_sequences": list(val.keys()), "best": best, "results": rows, "comparisons": comparisons, "judgement": judgement, "old_reference": old_reference, "sanity": sanity}
    with (out / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({key for row in rows for key in row})); writer.writeheader(); writer.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    _write_readme(out, summary, rows, comparisons, sanity)
    log_file.close()


if __name__ == "__main__":
    main()
