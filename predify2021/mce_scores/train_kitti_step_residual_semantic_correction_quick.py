"""Quick diagnostic for a stronger residual semantic correction branch.

This screen changes only ExplicitSemanticCorrection -> ResidualSemanticCorrection.
The frozen host/predictor, temporal error branch, scalar temporal gate, one-step
temporal loss, BPTT, data split, and blur protocol stay unchanged.
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
from predify2021.mce_scores.role_separated_direct_state_correction import (
    load_role_components,
    make_paths,
)
from predify2021.mce_scores import train_kitti_step_convgru_decoupling_quick as quick
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
from predify2021.model_factory.deeplabv3plus_resnet50.semantic_temporal_error_correction import (
    DecoupledSemanticTemporalErrorCorrection,
    ExplicitSemanticCorrection,
    ResidualSemanticCorrection,
    SemanticTemporalErrorCorrection,
)

VARIANTS = ("old", "full-explicit", "full-residual")
quick.KINDS = VARIANTS

CHECKPOINT_NAMES = {
    "old": "old.pt",
    "full-explicit": "full_explicit.pt",
    "full-residual": "full_residual.pt",
}


def _copy_decoupled_nonsemantic(source, target):
    for name in ("correlation", "encoder", "error_state", "gate", "temporal_prediction"):
        getattr(target, name).load_state_dict(getattr(source, name).state_dict(), strict=True)


def make_models():
    """Build a fair old/current/residual comparison."""
    old = torch.nn.ModuleList(
        [SemanticTemporalErrorCorrection(), SemanticTemporalErrorCorrection()]
    ).cuda()

    reference = torch.nn.ModuleList(
        [DecoupledSemanticTemporalErrorCorrection(), DecoupledSemanticTemporalErrorCorrection()]
    ).cuda()
    semantic_init = copy.deepcopy(
        [module.semantic_correction.state_dict() for module in reference]
    )

    # Preserve the original quick-screen RNG creation order.
    _semantic_only_rng_anchor = torch.nn.ModuleList(
        [ExplicitSemanticCorrection(), ExplicitSemanticCorrection()]
    ).cuda()

    explicit = torch.nn.ModuleList(
        [DecoupledSemanticTemporalErrorCorrection(), DecoupledSemanticTemporalErrorCorrection()]
    ).cuda()
    for index in range(2):
        explicit[index].semantic_correction.load_state_dict(
            semantic_init[index], strict=True
        )

    residual_modules = []
    for index in range(2):
        residual_semantic = ResidualSemanticCorrection(
            channels=128,
            bottleneck_channels=64,
            baseline=explicit[index].semantic_correction,
        )
        candidate = DecoupledSemanticTemporalErrorCorrection(
            channels=128,
            projection_channels=32,
            gate_channels=1,
            semantic_correction=residual_semantic,
        )
        _copy_decoupled_nonsemantic(explicit[index], candidate)
        residual_modules.append(candidate)
    residual = torch.nn.ModuleList(residual_modules).cuda()

    del _semantic_only_rng_anchor
    return {
        "old": old,
        "full-explicit": explicit,
        "full-residual": residual,
    }


def _grad_norm(module):
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().abs().sum().item())
    return total


def semantic_refinement_sanity():
    """Check exact initialization equivalence and gradient responsibility."""
    torch.manual_seed(SEED)

    explicit = DecoupledSemanticTemporalErrorCorrection(
        channels=8,
        projection_channels=2,
        gate_channels=1,
    )
    residual_semantic = ResidualSemanticCorrection(
        channels=8,
        bottleneck_channels=4,
        baseline=explicit.semantic_correction,
    )
    residual = DecoupledSemanticTemporalErrorCorrection(
        channels=8,
        projection_channels=2,
        gate_channels=1,
        semantic_correction=residual_semantic,
    )
    _copy_decoupled_nonsemantic(explicit, residual)

    observation = torch.randn(2, 8, 7, 9)
    predicted = torch.randn_like(observation)
    semantic_reference = torch.randn_like(observation)
    hidden = torch.randn_like(observation)

    explicit_out = explicit(observation, predicted, semantic_reference, hidden)
    residual_out = residual(observation, predicted, semantic_reference, hidden)

    posterior_diff = float((explicit_out[0] - residual_out[0]).abs().max().item())
    semantic_diff = float(
        (
            explicit_out[2]["semantic_residual"]
            - residual_out[2]["semantic_residual"]
        )
        .abs()
        .max()
        .item()
    )
    assert posterior_diff < 1e-6, posterior_diff
    assert semantic_diff < 1e-6, semantic_diff

    residual.zero_grad(set_to_none=True)
    residual(
        observation, predicted, semantic_reference, hidden
    )[0].mean().backward()

    seg_semantic_grad = _grad_norm(residual.semantic_correction)
    seg_temporal_grad = sum(
        _grad_norm(getattr(residual, name))
        for name in ("correlation", "encoder", "error_state", "temporal_prediction")
    )
    local_expand_grad = _grad_norm(residual.semantic_correction.local_refine.expand)
    context_expand_grad = _grad_norm(
        residual.semantic_correction.context_refine.expand
    )

    assert seg_semantic_grad > 0.0
    assert local_expand_grad > 0.0
    assert context_expand_grad > 0.0
    assert seg_temporal_grad == 0.0

    residual.zero_grad(set_to_none=True)
    residual(
        observation, predicted, semantic_reference, hidden
    )[2]["predicted_next_task_error"].mean().backward()

    temporal_grad = sum(
        _grad_norm(getattr(residual, name))
        for name in ("correlation", "encoder", "error_state", "temporal_prediction")
    )
    temporal_semantic_grad = _grad_norm(residual.semantic_correction)
    temporal_gate_grad = _grad_norm(residual.gate)

    assert temporal_grad > 0.0
    assert temporal_semantic_grad == 0.0
    assert temporal_gate_grad == 0.0

    return {
        "posterior_max_abs_diff": posterior_diff,
        "semantic_residual_max_abs_diff": semantic_diff,
        "seg_semantic_grad": seg_semantic_grad,
        "seg_temporal_grad": seg_temporal_grad,
        "local_refine_expand_grad": local_expand_grad,
        "context_refine_expand_grad": context_expand_grad,
        "temporal_branch_grad": temporal_grad,
        "temporal_semantic_grad": temporal_semantic_grad,
        "temporal_gate_grad": temporal_gate_grad,
    }


def _write_readme(out, rows, comparisons, best, parameter_counts, sanity):
    by = {(row["model"], row["condition"]): row for row in rows}
    lines = [
        "# Residual semantic correction quick screen",
        "",
        "Diagnostic Blur-Mid/Blur-Max conditions use sigma 2.25/3.0 and are not formal corruption severities.",
        "",
        "## mIoU",
        "",
        "| Model | Clean | Blur-Mid | Blur-Max | Blur Mean |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in VARIANTS:
        blur_mean = (
            by[(model, "Blur-Mid")]["mIoU"]
            + by[(model, "Blur-Max")]["mIoU"]
        ) / 2
        lines.append(
            f"| {model} | {by[(model, 'Clean')]['mIoU']:.6f} | "
            f"{by[(model, 'Blur-Mid')]['mIoU']:.6f} | "
            f"{by[(model, 'Blur-Max')]['mIoU']:.6f} | {blur_mean:.6f} |"
        )

    lines.extend(
        [
            "",
            "## mVC16",
            "",
            "| Model | Clean | Blur-Mid | Blur-Max | Blur Mean |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for model in VARIANTS:
        blur_mean = (
            by[(model, "Blur-Mid")]["mVC16"]
            + by[(model, "Blur-Max")]["mVC16"]
        ) / 2
        lines.append(
            f"| {model} | {by[(model, 'Clean')]['mVC16']:.6f} | "
            f"{by[(model, 'Blur-Mid')]['mVC16']:.6f} | "
            f"{by[(model, 'Blur-Max')]['mVC16']:.6f} | {blur_mean:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Best epoch",
            "",
            json.dumps(best, indent=2),
            "",
            "## Parameters",
            "",
            json.dumps(parameter_counts, indent=2),
            "",
            "## Comparisons",
            "",
            json.dumps(comparisons, indent=2),
            "",
            "## Initialization / gradient sanity",
            "",
            json.dumps(sanity, indent=2),
            "",
        ]
    )
    (out / "README.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/lin/predify/kitti_step")
    parser.add_argument(
        "--output",
        default="results/kitti_step_residual_semantic_correction_quick",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--fast-bf16", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    sanity = semantic_refinement_sanity()

    paths = make_paths()
    model, predictor = load_role_components(
        paths["static"],
        paths["adapter"],
        paths["predictor"],
        paths["writeback"],
    )
    model.requires_grad_(False)
    predictor.requires_grad_(False)

    train_dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(
        Path(args.root), "train"
    )
    val_dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(
        Path(args.root), "val"
    )
    train = preload(first_groups(train_dataset, 2))
    val = preload(first_groups(val_dataset, 2))

    modules = make_models()
    optimizers = {
        kind: torch.optim.AdamW(
            modules[kind].parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )
        for kind in VARIANTS
    }

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    log_file = (out / "run.log").open("a")

    def log(message):
        line = f"{time.strftime('%F %T')} {message}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    log("semantic_refinement_sanity=" + json.dumps(sanity, sort_keys=True))

    best = {
        kind: {"mIoU": float("-inf"), "epoch": 0}
        for kind in VARIANTS
    }
    progress = []

    for epoch in range(1, args.epochs + 1):
        train_result = run_shared(
            model,
            predictor,
            modules,
            train,
            True,
            optimizers,
            bf16=args.fast_bf16,
            log=log,
        )
        val_result = run_shared(
            model,
            predictor,
            modules,
            val,
            False,
            condition="Clean",
            bf16=args.fast_bf16,
            log=log,
        )

        for kind in VARIANTS:
            if val_result[kind]["mIoU"] > best[kind]["mIoU"]:
                best[kind] = {
                    "mIoU": val_result[kind]["mIoU"],
                    "epoch": epoch,
                }
                torch.save(
                    modules[kind].state_dict(),
                    out / CHECKPOINT_NAMES[kind],
                )

        progress.append(
            {
                "epoch": epoch,
                "train": train_result,
                "val": val_result,
                "best": copy.deepcopy(best),
            }
        )
        (out / "progress.json").write_text(
            json.dumps(progress, indent=2)
        )
        log(
            "epoch="
            + str(epoch)
            + " best_val_mIoU="
            + json.dumps({k: v["mIoU"] for k, v in best.items()})
        )

    for kind in VARIANTS:
        modules[kind].load_state_dict(
            torch.load(
                out / CHECKPOINT_NAMES[kind],
                map_location="cuda",
                weights_only=True,
            )
        )

    parameter_counts = {
        kind: sum(
            parameter.numel()
            for parameter in modules[kind].parameters()
            if parameter.requires_grad
        )
        for kind in VARIANTS
    }
    parameter_counts["residual_semantic_extra_vs_explicit"] = (
        parameter_counts["full-residual"]
        - parameter_counts["full-explicit"]
    )

    rows = []
    for condition in ("Clean", "Blur-Mid", "Blur-Max"):
        evaluated = run_shared(
            model,
            predictor,
            modules,
            val,
            False,
            condition=condition,
            bf16=args.fast_bf16,
            log=log,
        )
        for kind in VARIANTS:
            rows.append(
                {
                    "model": kind,
                    "condition": condition,
                    "trainable_params": parameter_counts[kind],
                    **evaluated[kind],
                }
            )

    by = {(row["model"], row["condition"]): row for row in rows}
    blur_conditions = ("Blur-Mid", "Blur-Max")

    def mean_delta(metric, candidate, baseline):
        return sum(
            by[(candidate, condition)][metric]
            - by[(baseline, condition)][metric]
            for condition in blur_conditions
        ) / len(blur_conditions)

    comparisons = {
        "Residual - Explicit": {
            "Clean mIoU delta": (
                by[("full-residual", "Clean")]["mIoU"]
                - by[("full-explicit", "Clean")]["mIoU"]
            ),
            "Mean Blur mIoU delta": mean_delta(
                "mIoU", "full-residual", "full-explicit"
            ),
            "Mean Blur mVC16 delta": mean_delta(
                "mVC16", "full-residual", "full-explicit"
            ),
        },
        "Residual - Old": {
            "Clean mIoU delta": (
                by[("full-residual", "Clean")]["mIoU"]
                - by[("old", "Clean")]["mIoU"]
            ),
            "Mean Blur mIoU delta": mean_delta(
                "mIoU", "full-residual", "old"
            ),
            "Mean Blur mVC16 delta": mean_delta(
                "mVC16", "full-residual", "old"
            ),
        },
    }

    residual_gain = comparisons["Residual - Explicit"]["Mean Blur mIoU delta"]
    residual_mvc = comparisons["Residual - Explicit"]["Mean Blur mVC16 delta"]
    if residual_gain >= 0.01 and residual_mvc > -0.01:
        judgement = "STRONGLY SUPPORTED"
    elif residual_gain >= 0.005 and residual_mvc > -0.01:
        judgement = "SUPPORTED"
    else:
        judgement = "NOT SUPPORTED"

    comparisons["SEMANTIC REFINEMENT"] = judgement

    with (out / "comparison.csv").open("w", newline="") as stream:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    summary = {
        "commit": commit,
        "config": {
            "seed": SEED,
            "epochs": args.epochs,
            "patience": PATIENCE,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "distill_weight": DISTILL_WEIGHT,
            "bptt": BPTT,
            "precision": "bf16" if args.fast_bf16 else "fp32",
            "train_frames": 601,
            "val_frames": 503,
            "conditions": {
                "Clean": "none",
                "Blur-Mid": "sigma=2.25",
                "Blur-Max": "sigma=3.0",
            },
        },
        "architecture": {
            "baseline": "ExplicitSemanticCorrection: 1x1 -> GELU -> 3x3",
            "candidate": (
                "ExplicitSemanticCorrection + bottleneck residual refine "
                "(dilation=1) + bottleneck residual refine (dilation=2)"
            ),
            "gate": "unchanged scalar temporal gate",
            "temporal_branch": "unchanged",
        },
        "train_sequences": list(train.keys()),
        "val_sequences": list(val.keys()),
        "best": best,
        "parameter_counts": parameter_counts,
        "results": rows,
        "comparisons": comparisons,
        "sanity": sanity,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n"
    )
    _write_readme(
        out,
        rows,
        comparisons,
        best,
        parameter_counts,
        sanity,
    )
    log_file.close()


if __name__ == "__main__":
    main()
