import json
import os
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from predify2021.datasets.kitti_step import (
    KITTISTEPSegmentationDataset,
    semantic_mask_from_panoptic_png,
)
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import sequence_groups
from predify2021.mce_scores.evaluate_kitti_step_semantic_temporal_error_correction import (
    host_from_delta,
    load_new,
    metric_from_confusion,
)
from predify2021.mce_scores.evaluate_kitti_step_static_baseline import update_confusion_matrix
from predify2021.mce_scores.kitti_step_cityscapes_c import (
    CITYSCAPES_C_COMMON_CORRUPTIONS,
    CITYSCAPES_C_SEVERITIES,
    apply_cityscapes_c_corruption_uint8,
    corruption_cache_path,
    corruption_seed,
)
from predify2021.mce_scores.role_separated_direct_state_correction import (
    error_state,
    load_role_components,
    make_paths,
    next_role_prediction,
    zero_state,
)
from predify2021.mce_scores.semantic_temporal_error_step import (
    detach_error_state,
    semantic_temporal_error_step,
    zero_semantic_temporal_state,
)
from predify2021.mce_scores.video_metrics import VideoConsistency
from predify2021.model_factory.deeplabv3plus_resnet50 import HostFeature, UnifiedFeatures


MODEL_NAMES = ("host", "ours")


def parse_csv(value, allowed, cast=str):
    if not value:
        return tuple(allowed)
    selected = tuple(cast(item.strip()) for item in value.split(",") if item.strip())
    unknown = tuple(item for item in selected if item not in allowed)
    if unknown:
        raise ValueError(f"Unsupported values: {unknown}")
    return selected


def parse_optional_csv(value, allowed):
    if not value:
        return ()
    return parse_csv(value, allowed)


class CorruptedKITTIStepSuite(Dataset):
    def __init__(
        self,
        frame_records,
        conditions,
        seed,
        cache_root=None,
        cached_corruptions=(),
    ):
        self.frame_records = tuple(frame_records)
        self.conditions = tuple(conditions)
        self.seed = int(seed)
        self.frames_per_condition = len(self.frame_records)
        self.cache_root = Path(cache_root) if cache_root else None
        self.cached_corruptions = frozenset(cached_corruptions)

    def __len__(self):
        return self.frames_per_condition * len(self.conditions)

    def __getitem__(self, index):
        condition_index, frame_index = divmod(index, self.frames_per_condition)
        condition_key, corruption, severity = self.conditions[condition_index]
        record = self.frame_records[frame_index]

        if corruption in self.cached_corruptions:
            cached_path = corruption_cache_path(
                self.cache_root,
                corruption,
                severity,
                record["sequence_id"],
                record["image_path"],
            )
            with Image.open(cached_path) as image:
                image_uint8 = np.array(image.convert("RGB"), dtype=np.uint8)
        else:
            with Image.open(record["image_path"]) as image:
                image_uint8 = np.array(image.convert("RGB"), dtype=np.uint8)

            if corruption:
                sample_seed = corruption_seed(
                    self.seed,
                    self.frames_per_condition,
                    frame_index,
                    corruption,
                    severity,
                )
                image_uint8 = apply_cityscapes_c_corruption_uint8(
                    image_uint8,
                    corruption,
                    severity,
                    seed=sample_seed,
                )

        image_tensor = torch.from_numpy(image_uint8).permute(2, 0, 1).contiguous()
        mask = semantic_mask_from_panoptic_png(record["mask_path"]).to(torch.uint8)
        return {
            "image": image_tensor,
            "mask": mask,
            "condition_key": condition_key,
            "corruption": corruption or "",
            "severity": int(severity or 0),
            "sequence_id": record["sequence_id"],
            "sequence_frame_index": int(record["sequence_frame_index"]),
        }


def build_frame_records(groups):
    records = []
    for sequence_id, samples in groups.items():
        for sequence_frame_index, sample in enumerate(samples):
            records.append(
                {
                    "image_path": str(sample["image_path"]),
                    "mask_path": str(sample["mask_path"]),
                    "sequence_id": str(sequence_id),
                    "sequence_frame_index": sequence_frame_index,
                }
            )
    return records


def condition_list(corruptions, severities):
    return (
        ("clean", None, None),
        *tuple(
            (f"{corruption}:S{severity}", corruption, severity)
            for corruption in corruptions
            for severity in severities
        ),
    )


def make_loader(dataset, num_workers, prefetch_factor):
    kwargs = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if num_workers > 0:
        kwargs.update(
            {
                "prefetch_factor": prefetch_factor,
                "persistent_workers": True,
            }
        )
    return DataLoader(dataset, **kwargs)


def empty_condition_state():
    return {
        "confusion": {
            name: torch.zeros((19, 19), dtype=torch.int64)
            for name in MODEL_NAMES
        },
        "video_consistency": VideoConsistency(MODEL_NAMES),
        "frame_count": 0,
    }


def finalize_condition(state):
    mvc = state["video_consistency"].means()
    metrics = {
        name: {
            **metric_from_confusion(state["confusion"][name]),
            "mvc8": mvc[8][name],
            "mvc16": mvc[16][name],
        }
        for name in MODEL_NAMES
    }
    return {
        "metrics": metrics,
        "effective_frame_count": state["frame_count"],
        "mvc_window_counts": state["video_consistency"].window_counts(),
    }


def evaluate_suite(model, predictor, corrections, loader):
    clean = None
    results = {}
    timings = {}

    current_condition = None
    current_corruption = None
    current_severity = 0
    current_sequence = None
    state = None
    condition_start = None

    hidden = None
    predictor_hidden = None
    pending_dynamics = None
    pending_semantic = None

    def finish_current():
        nonlocal clean, state, condition_start
        if current_condition is None:
            return
        item = finalize_condition(state)
        timings[current_condition] = time.perf_counter() - condition_start
        if current_condition == "clean":
            clean = item
        else:
            results.setdefault(current_corruption, {})[f"S{current_severity}"] = item
        print(
            f"Completed {current_condition} in {timings[current_condition]:.1f}s",
            flush=True,
        )

    with torch.inference_mode():
        for batch in loader:
            condition_key = batch["condition_key"][0]
            corruption = batch["corruption"][0]
            severity = int(batch["severity"].item())
            sequence_id = batch["sequence_id"][0]
            sequence_frame_index = int(batch["sequence_frame_index"].item())

            if condition_key != current_condition:
                finish_current()
                current_condition = condition_key
                current_corruption = corruption
                current_severity = severity
                current_sequence = None
                state = empty_condition_state()
                condition_start = time.perf_counter()
                print(f"Evaluating {condition_key}", flush=True)

            if sequence_id != current_sequence:
                current_sequence = sequence_id
                state["video_consistency"].reset_sequence()
                hidden = None
                predictor_hidden = predictor.initial_state()
                pending_dynamics = None
                pending_semantic = None

            image = batch["image"].cuda(non_blocking=True).float().div_(255.0)
            raw = model.extract_backbone_features(image)
            observation = model.encode_backbone_features(raw)
            output_size = tuple(image.shape[-2:])

            if hidden is None:
                hidden = zero_semantic_temporal_state(observation)

            if sequence_frame_index == 0:
                pending_dynamics, pending_semantic, *predictor_hidden = next_role_prediction(
                    predictor,
                    observation,
                    zero_state(observation),
                    predictor_hidden,
                )
                continue

            error = error_state(observation, pending_dynamics)
            if sequence_frame_index == 1:
                pending_dynamics, pending_semantic, *predictor_hidden = next_role_prediction(
                    predictor,
                    observation,
                    error,
                    predictor_hidden,
                )
                continue

            posterior, hidden, _ = semantic_temporal_error_step(
                corrections,
                observation,
                pending_dynamics,
                pending_semantic,
                hidden,
            )
            zero = zero_state(observation)
            host_features = {
                "host": HostFeature(raw.c4, raw.c1, output_size),
                "ours": host_from_delta(
                    model,
                    raw,
                    UnifiedFeatures(
                        posterior.z1 - observation.z1,
                        zero.z2,
                        zero.z3,
                        posterior.z4 - observation.z4,
                    ),
                    output_size,
                ),
            }
            predictions = {
                name: model.decode_from_host_feature(features).argmax(1).squeeze(0).cpu()
                for name, features in host_features.items()
            }
            mask = batch["mask"].squeeze(0).to(torch.int64)
            for name, prediction in predictions.items():
                update_confusion_matrix(
                    state["confusion"][name],
                    prediction.to(torch.int64),
                    mask,
                )
            state["video_consistency"].append(mask, predictions)
            state["frame_count"] += 1

            pending_dynamics, pending_semantic, *predictor_hidden = next_role_prediction(
                predictor,
                observation,
                error,
                predictor_hidden,
            )
            hidden = detach_error_state(hidden)

    finish_current()
    return clean, results, timings


def per_corruption_summary(clean, severity_results):
    output = {}
    for name in MODEL_NAMES:
        clean_miou = clean["metrics"][name]["miou"]
        host_values = [
            severity_results[f"S{severity}"]["metrics"]["host"]["miou"]
            for severity in CITYSCAPES_C_SEVERITIES
        ]
        values = [
            severity_results[f"S{severity}"]["metrics"][name]["miou"]
            for severity in CITYSCAPES_C_SEVERITIES
        ]
        degradation = [1.0 - value for value in values]
        host_degradation = [1.0 - value for value in host_values]
        clean_error = 1.0 - clean_miou
        host_clean_error = 1.0 - clean["metrics"]["host"]["miou"]
        cd = sum(degradation) / sum(host_degradation)
        rcd_denominator = sum(host_degradation) - host_clean_error
        rcd = (sum(degradation) - clean_error) / rcd_denominator

        output[name] = {
            "severity_miou": values,
            "mean_corruption_miou": sum(values) / len(values),
            "mean_corruption_wiou": sum(
                severity_results[f"S{severity}"]["metrics"][name]["wiou"]
                for severity in CITYSCAPES_C_SEVERITIES
            )
            / len(CITYSCAPES_C_SEVERITIES),
            "mean_corruption_mvc8": sum(
                severity_results[f"S{severity}"]["metrics"][name]["mvc8"]
                for severity in CITYSCAPES_C_SEVERITIES
            )
            / len(CITYSCAPES_C_SEVERITIES),
            "mean_corruption_mvc16": sum(
                severity_results[f"S{severity}"]["metrics"][name]["mvc16"]
                for severity in CITYSCAPES_C_SEVERITIES
            )
            / len(CITYSCAPES_C_SEVERITIES),
            "cd_vs_host_reference": cd,
            "rcd_vs_host_reference": rcd,
        }
    return output


def aggregate_suite(clean, results, corruptions):
    per_corruption = {
        corruption: per_corruption_summary(clean, results[corruption])
        for corruption in corruptions
    }
    aggregate = {}
    condition_count = len(corruptions) * len(CITYSCAPES_C_SEVERITIES)

    for name in MODEL_NAMES:
        aggregate[name] = {
            metric: sum(
                results[corruption][f"S{severity}"]["metrics"][name][metric]
                for corruption in corruptions
                for severity in CITYSCAPES_C_SEVERITIES
            )
            / condition_count
            for metric in ("miou", "wiou", "mvc8", "mvc16")
        }
        aggregate[name]["mean_cd_vs_host_reference"] = sum(
            per_corruption[corruption][name]["cd_vs_host_reference"]
            for corruption in corruptions
        ) / len(corruptions)
        aggregate[name]["mean_rcd_vs_host_reference"] = sum(
            per_corruption[corruption][name]["rcd_vs_host_reference"]
            for corruption in corruptions
        ) / len(corruptions)

    aggregate["ours_minus_host"] = {
        metric: aggregate["ours"][metric] - aggregate["host"][metric]
        for metric in ("miou", "wiou", "mvc8", "mvc16")
    }
    aggregate["severity"] = {
        f"S{severity}": {
            name: {
                metric: sum(
                    results[corruption][f"S{severity}"]["metrics"][name][metric]
                    for corruption in corruptions
                )
                / len(corruptions)
                for metric in ("miou", "wiou", "mvc8", "mvc16")
            }
            for name in MODEL_NAMES
        }
        for severity in CITYSCAPES_C_SEVERITIES
    }
    return per_corruption, aggregate


def write_readme(result, path):
    clean = result["clean"]["metrics"]
    aggregate = result["aggregate"]
    lines = [
        "# KITTI-STEP Common Corruption Evaluation",
        "",
        "| Model | Clean mIoU | Mean Corruption mIoU | Corruption Drop |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, label in (("host", "Baseline Host"), ("ours", "Ours")):
        clean_miou = clean[name]["miou"]
        corruption_miou = aggregate[name]["miou"]
        lines.append(
            f"| {label} | {clean_miou:.7f} | {corruption_miou:.7f} | "
            f"{clean_miou - corruption_miou:.7f} |"
        )
    lines.extend(
        [
            "",
            f"Ours - Baseline Clean: {clean['ours']['miou'] - clean['host']['miou']:.7f}",
            f"Ours - Baseline Corruption: {aggregate['ours_minus_host']['miou']:.7f}",
            "",
            "| Model | Clean mVC8 | Corruption mVC8 | Clean mVC16 | Corruption mVC16 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, label in (("host", "Baseline"), ("ours", "Ours")):
        lines.append(
            f"| {label} | {clean[name]['mvc8']:.7f} | {aggregate[name]['mvc8']:.7f} | "
            f"{clean[name]['mvc16']:.7f} | {aggregate[name]['mvc16']:.7f} |"
        )
    lines.extend(
        [
            "",
            "## Severity Trend",
            "",
            "| Severity | Baseline | Ours | Delta |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for severity in CITYSCAPES_C_SEVERITIES:
        baseline = aggregate["severity"][f"S{severity}"]["host"]["miou"]
        ours = aggregate["severity"][f"S{severity}"]["ours"]["miou"]
        lines.append(
            f"| S{severity} | {baseline:.7f} | {ours:.7f} | {ours - baseline:.7f} |"
        )
    wins = sum(
        result["conditions"][corruption][f"S{severity}"]["metrics"]["ours"]["miou"]
        > result["conditions"][corruption][f"S{severity}"]["metrics"]["host"]["miou"]
        for corruption in result["conditions"]
        for severity in CITYSCAPES_C_SEVERITIES
    )
    total = len(result["conditions"]) * len(CITYSCAPES_C_SEVERITIES)
    lines.extend(
        [
            "",
            f"Ours wins {wins}/{total} corruption conditions by mIoU.",
            "CD/rCD in summary.json are reported against the internal Host reference, not official Cityscapes-C metrics.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("KITTI-STEP common-corruption evaluation requires CUDA")

    seed = int(os.environ.get("PREDIFY_SEED", "0"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    root = Path(os.environ.get("PREDIFY_KITTI_STEP_ROOT", "/home/lin/predify/kitti_step"))
    checkpoint = Path(os.environ["PREDIFY_SEMANTIC_TEMPORAL_ERROR_CHECKPOINT"])
    output = Path(
        os.environ.get(
            "PREDIFY_CITYSCAPES_C_OUTPUT_DIR",
            "results/kitti_step_cityscapes_c",
        )
    )
    corruptions = parse_csv(
        os.environ.get("PREDIFY_CITYSCAPES_C_CORRUPTIONS"),
        CITYSCAPES_C_COMMON_CORRUPTIONS,
    )
    severities = parse_csv(
        os.environ.get("PREDIFY_CITYSCAPES_C_SEVERITIES"),
        CITYSCAPES_C_SEVERITIES,
        int,
    )
    cached_corruptions = parse_optional_csv(
        os.environ.get("PREDIFY_CITYSCAPES_C_CACHED_CORRUPTIONS"),
        CITYSCAPES_C_COMMON_CORRUPTIONS,
    )
    cache_root_value = os.environ.get("PREDIFY_CITYSCAPES_C_CACHE_ROOT")
    cache_root = Path(cache_root_value) if cache_root_value else None
    if cached_corruptions and cache_root is None:
        raise ValueError("PREDIFY_CITYSCAPES_C_CACHE_ROOT is required for cached corruptions")

    max_sequences = int(os.environ.get("PREDIFY_CITYSCAPES_C_MAX_SEQUENCES", "0"))
    num_workers = int(os.environ.get("PREDIFY_CITYSCAPES_C_NUM_WORKERS", "8"))
    prefetch_factor = int(os.environ.get("PREDIFY_CITYSCAPES_C_PREFETCH_FACTOR", "4"))

    paths = make_paths()
    model, predictor = load_role_components(
        paths["static"],
        paths["adapter"],
        paths["predictor"],
        paths["writeback"],
    )
    corrections, _ = load_new(checkpoint)

    dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(root, "val")
    groups = sequence_groups(dataset)
    if max_sequences > 0:
        groups = dict(list(groups.items())[:max_sequences])

    conditions = condition_list(corruptions, severities)
    suite_dataset = CorruptedKITTIStepSuite(
        build_frame_records(groups),
        conditions,
        seed,
        cache_root=cache_root,
        cached_corruptions=cached_corruptions,
    )
    loader = make_loader(suite_dataset, num_workers, prefetch_factor)

    suite_start = time.perf_counter()
    clean, results, timings = evaluate_suite(model, predictor, corrections, loader)
    total_wall_seconds = time.perf_counter() - suite_start

    if severities == CITYSCAPES_C_SEVERITIES:
        per_corruption, aggregate = aggregate_suite(clean, results, corruptions)
    else:
        per_corruption, aggregate = {}, {}

    result = {
        "experiment": "kitti_step_cityscapes_c_common_corruption",
        "git_revision": os.environ.get("PREDIFY_GIT_REVISION"),
        "checkpoint": str(checkpoint),
        "dataset": {
            "split": "val",
            "sequence_count": len(groups),
            "total_frame_count": sum(len(samples) for samples in groups.values()),
            "effective_frame_count_per_condition": clean["effective_frame_count"],
        },
        "protocol": {
            "corruption_implementation": "bethgelab/imagecorruptions",
            "corruption_family": "ImageNet-C common 15 as used for Cityscapes-C style evaluation",
            "corruptions": corruptions,
            "severities": severities,
            "fixed_corruption_and_severity_within_sequence": True,
            "corruption_from_first_frame": True,
            "clean_warmup_frames": 0,
            "model_state_initialization_frames_excluded_from_metrics": 2,
            "training": False,
            "checkpoint_selection": False,
            "correction_feedback_to_predictor": False,
            "cd_reference_model": "host",
            "corruption_seed_policy": "canonical_condition_ordinal_plus_global_frame_index",
        },
        "pipeline": {
            "corruption_device": "cpu",
            "cpu_image_dtype": "uint8",
            "num_workers": num_workers,
            "prefetch_factor": prefetch_factor if num_workers > 0 else 0,
            "pin_memory": True,
            "non_blocking_cuda_copy": True,
            "shared_backbone_for_host_and_ours": True,
            "single_loader_for_clean_and_all_conditions": True,
            "cached_corruptions": cached_corruptions,
            "cache_root": str(cache_root) if cache_root is not None else None,
            "cache_format": "lossless_png",
        },
        "timing": {
            "total_wall_seconds": total_wall_seconds,
            "per_condition_wall_seconds": timings,
        },
        "clean": clean,
        "conditions": results,
        "per_corruption_summary": per_corruption,
        "aggregate": aggregate,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if aggregate:
        write_readme(result, output / "README.md")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
