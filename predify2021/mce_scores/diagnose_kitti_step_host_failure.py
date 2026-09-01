"""Direct Host Failure Diagnosis on KITTI-STEP under Blur-Mid/Blur-Max.

This is a diagnostic experiment only. It does not train or update parameters.

Questions answered:
1. Which Host backbone/unified feature stages shift most under blur?
2. How much final segmentation can be recovered by replacing blurred C1/C4
   with their clean oracle counterparts before the frozen DeepLabV3+ decoder?

The blur conditions exactly reuse the current quick-screen definitions:
  Blur-Mid: sigma=2.25
  Blur-Max: sigma=3.0
and evaluate only post-warmup frames, matching the current pressure-condition
evaluation window.
"""

import argparse
import csv
import json
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from predify2021.datasets.kitti_step import (
    KITTISTEPSegmentationDataset,
    semantic_mask_from_panoptic_png,
)
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import (
    load_image,
    sequence_groups,
)
from predify2021.mce_scores.evaluate_kitti_step_static_baseline import (
    CITYSCAPES_CLASSES,
    compute_iou,
    update_confusion_matrix,
)
from predify2021.mce_scores.kitti_step_persistent_blur import warmup_frame_count
from predify2021.mce_scores.role_separated_dynamic_error_correction import (
    ADAPTER_CHECKPOINT_DEFAULT,
    STATIC_CHECKPOINT_DEFAULT,
)
from predify2021.mce_scores.train_kitti_step_convgru_decoupling_quick import (
    diagnostic_blur,
)
from predify2021.mce_scores.train_kitti_step_state_predictor import (
    load_static_kitti_checkpoint,
)
from predify2021.model_factory.deeplabv3plus_resnet50 import (
    HostFeature,
    build_deeplabv3plus_resnet50_host,
)

SEED = 0
CONDITIONS = ("Blur-Mid", "Blur-Max")
RAW_STAGES = ("c1", "c2", "c3", "c4")
UNIFIED_STAGES = ("z1", "z2", "z3", "z4")
VARIANTS = ("clean", "blur", "clean_c1", "clean_c4", "clean_c1_c4")


def load_host():
    model = build_deeplabv3plus_resnet50_host().cuda()
    load_static_kitti_checkpoint(model, Path(STATIC_CHECKPOINT_DEFAULT))
    adapter_payload = torch.load(
        ADAPTER_CHECKPOINT_DEFAULT,
        map_location="cpu",
        weights_only=False,
    )
    model.multi_layer_adapter.load_state_dict(
        adapter_payload["adapter_state_dict"],
        strict=True,
    )
    model.requires_grad_(False)
    model.eval()
    return model


def new_shift_aggregate():
    return {"count": 0, "mse": 0.0, "relative_mse": 0.0, "cosine": 0.0}


def add_shift(aggregate, clean, corrupted):
    mse = F.mse_loss(corrupted, clean).item()
    clean_energy = clean.detach().float().pow(2).mean().item()
    cosine = (
        F.cosine_similarity(
            corrupted.float(),
            clean.float(),
            dim=1,
            eps=1e-8,
        )
        .mean()
        .item()
    )
    aggregate["count"] += 1
    aggregate["mse"] += mse
    aggregate["relative_mse"] += mse / max(clean_energy, 1e-12)
    aggregate["cosine"] += cosine


def finalize_shift(aggregate):
    count = max(aggregate["count"], 1)
    return {
        "count": aggregate["count"],
        "mse": aggregate["mse"] / count,
        "relative_mse": aggregate["relative_mse"] / count,
        "cosine": aggregate["cosine"] / count,
    }


def decode_variants(model, clean_raw, blur_raw, output_size):
    features = {
        "clean": HostFeature(clean_raw.c4, clean_raw.c1, output_size),
        "blur": HostFeature(blur_raw.c4, blur_raw.c1, output_size),
        "clean_c1": HostFeature(blur_raw.c4, clean_raw.c1, output_size),
        "clean_c4": HostFeature(clean_raw.c4, blur_raw.c1, output_size),
        "clean_c1_c4": HostFeature(clean_raw.c4, clean_raw.c1, output_size),
    }
    return {
        name: model.decode_from_host_feature(feature).argmax(1).squeeze(0).cpu()
        for name, feature in features.items()
    }


def recovery_fraction(blur, oracle, clean):
    gap = clean - blur
    if abs(gap) < 1e-12:
        return 0.0
    return (oracle - blur) / gap


def evaluate_condition(model, groups, condition, max_effective_frames=0):
    confusion = {
        name: torch.zeros((19, 19), dtype=torch.int64)
        for name in VARIANTS
    }
    raw_shift = {stage: new_shift_aggregate() for stage in RAW_STAGES}
    unified_shift = {
        stage: new_shift_aggregate() for stage in UNIFIED_STAGES
    }
    effective_frames = 0

    with torch.inference_mode():
        for samples in groups.values():
            onset = warmup_frame_count(len(samples))
            for frame_index, sample in enumerate(samples):
                if frame_index < onset:
                    continue
                if max_effective_frames and effective_frames >= max_effective_frames:
                    break

                clean_image = load_image(sample)
                blur_image = diagnostic_blur(
                    clean_image,
                    frame_index,
                    len(samples),
                    condition,
                )
                output_size = tuple(clean_image.shape[-2:])

                clean_raw = model.extract_backbone_features(clean_image)
                blur_raw = model.extract_backbone_features(blur_image)
                clean_state = model.encode_backbone_features(clean_raw)
                blur_state = model.encode_backbone_features(blur_raw)

                for stage in RAW_STAGES:
                    add_shift(
                        raw_shift[stage],
                        getattr(clean_raw, stage),
                        getattr(blur_raw, stage),
                    )
                for stage in UNIFIED_STAGES:
                    add_shift(
                        unified_shift[stage],
                        getattr(clean_state, stage),
                        getattr(blur_state, stage),
                    )

                predictions = decode_variants(
                    model,
                    clean_raw,
                    blur_raw,
                    output_size,
                )
                mask = semantic_mask_from_panoptic_png(sample["mask_path"])
                for name in VARIANTS:
                    update_confusion_matrix(
                        confusion[name],
                        predictions[name],
                        mask,
                    )
                effective_frames += 1

            if max_effective_frames and effective_frames >= max_effective_frames:
                break

    metrics = {}
    for name in VARIANTS:
        iou = compute_iou(confusion[name])
        metrics[name] = {
            "mIoU": float(torch.nanmean(iou).item()),
            "per_class_iou": {
                class_name: (
                    None if torch.isnan(iou[index]) else float(iou[index].item())
                )
                for index, class_name in enumerate(CITYSCAPES_CLASSES)
            },
        }

    blur_miou = metrics["blur"]["mIoU"]
    clean_miou = metrics["clean"]["mIoU"]
    for name in ("clean_c1", "clean_c4", "clean_c1_c4"):
        metrics[name]["absolute_gain_vs_blur"] = (
            metrics[name]["mIoU"] - blur_miou
        )
        metrics[name]["recovery_fraction_of_clean_gap"] = recovery_fraction(
            blur_miou,
            metrics[name]["mIoU"],
            clean_miou,
        )

    return {
        "condition": condition,
        "effective_frame_count": effective_frames,
        "feature_shift": {
            "backbone": {
                stage: finalize_shift(raw_shift[stage])
                for stage in RAW_STAGES
            },
            "unified": {
                stage: finalize_shift(unified_shift[stage])
                for stage in UNIFIED_STAGES
            },
        },
        "segmentation": metrics,
    }


def infer_host_failure(summary):
    decisions = {}
    for condition, result in summary["conditions"].items():
        segmentation = result["segmentation"]
        blur = segmentation["blur"]["mIoU"]
        clean = segmentation["clean"]["mIoU"]
        c1 = segmentation["clean_c1"]["mIoU"]
        c4 = segmentation["clean_c4"]["mIoU"]
        both = segmentation["clean_c1_c4"]["mIoU"]

        c1_gain = c1 - blur
        c4_gain = c4 - blur
        if c4_gain > c1_gain:
            dominant = "C4 high-level semantic degradation"
        elif c1_gain > c4_gain:
            dominant = "C1 low-level spatial/boundary degradation"
        else:
            dominant = "C1/C4 contributions are equal"

        decisions[condition] = {
            "clean_host_mIoU": clean,
            "blur_host_mIoU": blur,
            "blur_drop": clean - blur,
            "clean_c1_gain": c1_gain,
            "clean_c4_gain": c4_gain,
            "clean_c1_c4_gain": both - blur,
            "clean_c1_recovery_fraction": recovery_fraction(blur, c1, clean),
            "clean_c4_recovery_fraction": recovery_fraction(blur, c4, clean),
            "clean_c1_c4_recovery_fraction": recovery_fraction(
                blur, both, clean
            ),
            "dominant_recoverable_stage": dominant,
            "decoder_check": (
                "PASS"
                if abs(both - clean) <= 1e-6
                else "CHECK"
            ),
        }
    return decisions


def write_feature_csv(path, condition_results):
    rows = []
    for condition, result in condition_results.items():
        for space in ("backbone", "unified"):
            for stage, metrics in result["feature_shift"][space].items():
                rows.append(
                    {
                        "condition": condition,
                        "space": space,
                        "stage": stage,
                        **metrics,
                    }
                )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_oracle_csv(path, condition_results):
    rows = []
    for condition, result in condition_results.items():
        clean = result["segmentation"]["clean"]["mIoU"]
        blur = result["segmentation"]["blur"]["mIoU"]
        for variant in VARIANTS:
            miou = result["segmentation"][variant]["mIoU"]
            rows.append(
                {
                    "condition": condition,
                    "variant": variant,
                    "mIoU": miou,
                    "gain_vs_blur": miou - blur,
                    "recovery_fraction_of_clean_gap": (
                        recovery_fraction(blur, miou, clean)
                        if variant != "blur"
                        else 0.0
                    ),
                }
            )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="/home/lin/predify/kitti_step",
    )
    parser.add_argument(
        "--output",
        default="results/kitti_step_host_failure_diagnosis",
    )
    parser.add_argument(
        "--sequence-limit",
        type=int,
        default=0,
        help="0 evaluates all validation sequences.",
    )
    parser.add_argument(
        "--max-effective-frames",
        type=int,
        default=0,
        help="0 evaluates every post-warmup frame.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="One sequence, at most 32 post-warmup frames.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Host Failure Diagnosis requires CUDA.")

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    model = load_host()
    dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(
        Path(args.root),
        "val",
    )
    groups = sequence_groups(dataset)

    sequence_limit = args.sequence_limit
    max_effective_frames = args.max_effective_frames
    if args.smoke:
        sequence_limit = 1
        max_effective_frames = 32
    if sequence_limit:
        groups = dict(list(groups.items())[:sequence_limit])

    output = Path(args.output)
    if args.smoke:
        output = output / "smoke"
    output.mkdir(parents=True, exist_ok=True)

    condition_results = {}
    for condition in CONDITIONS:
        result = evaluate_condition(
            model,
            groups,
            condition,
            max_effective_frames=max_effective_frames,
        )
        condition_results[condition] = result
        print(
            json.dumps(
                {
                    "condition": condition,
                    "effective_frames": result["effective_frame_count"],
                    "mIoU": {
                        name: values["mIoU"]
                        for name, values in result["segmentation"].items()
                    },
                },
                sort_keys=True,
            ),
            flush=True,
        )

    summary = {
        "experiment": "kitti_step_host_failure_diagnosis",
        "diagnostic_only": True,
        "parameters_updated": False,
        "split": "val",
        "sequence_count": len(groups),
        "conditions_definition": {
            "Blur-Mid": "Gaussian blur sigma=2.25 after 10% sequence warmup",
            "Blur-Max": "Gaussian blur sigma=3.0 after 10% sequence warmup",
        },
        "evaluated_window": "post-warmup frames only",
        "feature_metrics": {
            "mse": "mean squared error to same-frame clean feature",
            "relative_mse": "MSE divided by same-frame clean feature energy",
            "cosine": "channel-wise cosine similarity to same-frame clean feature",
        },
        "oracle_variants": {
            "clean": "clean C1 + clean C4",
            "blur": "blur C1 + blur C4",
            "clean_c1": "clean C1 + blur C4",
            "clean_c4": "blur C1 + clean C4",
            "clean_c1_c4": "clean C1 + clean C4",
        },
        "conditions": condition_results,
    }
    summary["judgement"] = infer_host_failure(summary)

    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    write_feature_csv(output / "feature_shift.csv", condition_results)
    write_oracle_csv(output / "oracle_recovery.csv", condition_results)

    print(json.dumps({"judgement": summary["judgement"]}, indent=2), flush=True)
    print(f"results={output}", flush=True)


if __name__ == "__main__":
    main()
