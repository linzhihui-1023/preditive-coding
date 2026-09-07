"""Prediction-only KITTI-STEP Full9 PC + VEC8/VEC16 evaluation.

中文：只读取已经存在的 Host / 当前模型逐帧预测，重新计算：
- PC: Perceptual Consistency（感知一致性），按 WACV 2022 作者公开 SPC 实现；
- VEC8 / VEC16: Video Effective Consistency（视频有效一致性），按 ICCV 2025
  DTERN Eq.17，并显式记录 KITTI-STEP 的 ignore / empty-class 处理约定。

No segmentation model, checkpoint loader, RAFT, or training module is imported.
RGB frames are read only because PC requires the published ImageNet ResNet-18
perceptual features; GT is read only for VEC.  Existing prediction files are
never regenerated.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from predify2021.datasets.kitti_step import (
    KITTISTEPSegmentationDataset,
    semantic_mask_from_panoptic_png,
)
from predify2021.mce_scores.perceptual_consistency import (
    build_resnet18_perceptual_extractor,
    extract_perceptual_feature,
    perceptual_consistency_from_correlation,
    perceptual_correlation,
    resize_semantic_to_feature,
)
from predify2021.mce_scores.video_metrics import VideoEffectiveConsistency


FULL9 = ("0002", "0006", "0007", "0008", "0010", "0013", "0014", "0016", "0018")
NAMES = ("host", "current_model")
NUM_CLASSES = 19
IGNORE_LABEL = 255
DEFAULT_OUTPUT = "results/kitti_step_full9_pc_vec_from_predictions.json"
SUPPORTED_SUFFIXES = (".png", ".npy", ".npz", ".pt", ".pth")


def sequence_groups(dataset):
    """Group KITTI-STEP samples in true per-sequence frame order."""
    groups = {}
    for sample in dataset.samples:
        groups.setdefault(sample["sequence_id"], []).append(sample)
    for samples in groups.values():
        samples.sort(key=lambda sample: int(sample["frame_id"]))
    return groups


class PredictionStore:
    """Index existing per-frame semantic predictions without model inference."""

    def __init__(self, root):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"prediction root not found: {self.root}")
        self.by_sequence_frame = {}
        self.by_frame = {}
        files = [
            path
            for path in self.root.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        ]
        for path in files:
            frame_id = path.stem
            sequence_id = path.parent.name
            self.by_sequence_frame.setdefault((sequence_id, frame_id), []).append(path)
            self.by_frame.setdefault(frame_id, []).append(path)
        if not files:
            raise RuntimeError(
                f"no prediction files with {SUPPORTED_SUFFIXES} under {self.root}"
            )

    def _resolve(self, sequence_id, frame_id):
        sequence_id = str(sequence_id)
        frame_id = str(frame_id)
        exact = self.by_sequence_frame.get((sequence_id, frame_id), [])
        if len(exact) == 1:
            return exact[0]
        if len(exact) > 1:
            raise RuntimeError(
                f"ambiguous prediction for {sequence_id}/{frame_id}: {exact}"
            )

        flat = self.by_frame.get(frame_id, [])
        if len(flat) == 1:
            return flat[0]
        if len(flat) > 1:
            matches = [path for path in flat if sequence_id in path.parts]
            if len(matches) == 1:
                return matches[0]
        raise FileNotFoundError(
            f"prediction not found for sequence={sequence_id}, frame={frame_id} "
            f"under {self.root}"
        )

    @staticmethod
    def _unwrap(value):
        if isinstance(value, dict):
            for key in (
                "prediction",
                "pred",
                "mask",
                "labels",
                "semantic",
                "logits",
                "probability",
                "probabilities",
            ):
                if key in value:
                    return value[key]
            raise ValueError(
                "prediction dict must contain a recognized hard-mask or score key"
            )
        return value

    @staticmethod
    def _as_tensor(value):
        value = PredictionStore._unwrap(value)
        if isinstance(value, np.ndarray):
            tensor = torch.from_numpy(np.array(value, copy=True))
        else:
            tensor = torch.as_tensor(value)
        tensor = tensor.detach().cpu()
        if tensor.ndim == 4:
            if tensor.shape[0] != 1:
                raise ValueError(
                    f"prediction batch dimension must be 1, got {tuple(tensor.shape)}"
                )
            tensor = tensor[0]
        return tensor

    @staticmethod
    def _hard_mask_from_tensor(tensor, expected_hw):
        """Convert stored hard labels or stored scores to the final HxW mask.

        Score tensors are resized before argmax, matching the usual semantic
        inference order.  This avoids the invalid argmax->nearest-resize path.
        """
        expected_hw = tuple(int(v) for v in expected_hw)
        source_kind = None

        if tensor.ndim == 2:
            labels = tensor.long()
            source_kind = "hard_mask"
        elif tensor.ndim == 3 and tensor.shape[0] == 1:
            labels = tensor[0].long()
            source_kind = "hard_mask"
        elif tensor.ndim == 3 and tensor.shape[-1] == 1:
            labels = tensor[..., 0].long()
            source_kind = "hard_mask"
        elif tensor.ndim == 3 and tensor.shape[0] == NUM_CLASSES:
            scores = tensor.float().unsqueeze(0)
            if not bool(torch.isfinite(scores).all()):
                raise ValueError("stored prediction scores contain NaN/Inf")
            if tuple(scores.shape[-2:]) != expected_hw:
                scores = F.interpolate(
                    scores,
                    size=expected_hw,
                    mode="bilinear",
                    align_corners=False,
                )
            labels = scores.argmax(dim=1)[0].long()
            source_kind = "class_scores_chw"
        elif tensor.ndim == 3 and tensor.shape[-1] == NUM_CLASSES:
            scores = tensor.permute(2, 0, 1).contiguous().float().unsqueeze(0)
            if not bool(torch.isfinite(scores).all()):
                raise ValueError("stored prediction scores contain NaN/Inf")
            if tuple(scores.shape[-2:]) != expected_hw:
                scores = F.interpolate(
                    scores,
                    size=expected_hw,
                    mode="bilinear",
                    align_corners=False,
                )
            labels = scores.argmax(dim=1)[0].long()
            source_kind = "class_scores_hwc"
        elif tensor.ndim == 3 and tensor.shape[-1] in (3, 4):
            rgb = tensor[..., :3]
            if not bool((rgb == rgb[..., :1]).all()):
                raise ValueError(
                    "RGB prediction uses a color palette; class-color decoding is undefined"
                )
            labels = rgb[..., 0].long()
            source_kind = "replicated_rgb_hard_mask"
        else:
            raise ValueError(
                f"unsupported stored prediction shape: {tuple(tensor.shape)}"
            )

        if source_kind in {"hard_mask", "replicated_rgb_hard_mask"}:
            if tuple(labels.shape) != expected_hw:
                labels = F.interpolate(
                    labels[None, None].float(),
                    size=expected_hw,
                    mode="nearest",
                )[0, 0].long()

        if tuple(labels.shape) != expected_hw:
            raise RuntimeError(
                f"prediction shape conversion failed: {tuple(labels.shape)} != {expected_hw}"
            )
        invalid = (labels < 0) | (labels >= NUM_CLASSES)
        if bool(invalid.any()):
            values = torch.unique(labels[invalid])[:16].tolist()
            raise ValueError(
                "prediction masks must contain semantic class IDs 0..18 only; "
                f"found {values}. Ignore label 255 belongs to GT, not prediction."
            )
        return labels, source_kind

    def load(self, sequence_id, frame_id, expected_hw):
        path = self._resolve(sequence_id, frame_id)
        suffix = path.suffix.lower()
        if suffix == ".png":
            with Image.open(path) as image:
                value = np.array(image, copy=True)
        elif suffix == ".npy":
            value = np.load(path, allow_pickle=False)
        elif suffix == ".npz":
            with np.load(path, allow_pickle=False) as payload:
                if len(payload.files) != 1:
                    raise ValueError(f"NPZ prediction must contain one array: {path}")
                value = np.array(payload[payload.files[0]], copy=True)
        else:
            value = torch.load(path, map_location="cpu", weights_only=False)

        tensor = self._as_tensor(value)
        prediction, source_kind = self._hard_mask_from_tensor(tensor, expected_hw)
        return prediction, path, source_kind


def _mean(values):
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else float("nan")


def _evaluate_sequence(sequence_id, samples, stores, pc_extractor):
    vec = VideoEffectiveConsistency(
        NAMES,
        num_classes=NUM_CLASSES,
        ignore_label=IGNORE_LABEL,
    )
    pc_values = {name: [] for name in NAMES}
    prediction_paths = {name: [] for name in NAMES}
    source_kinds = {name: {} for name in NAMES}

    previous_feature = None
    previous_predictions = None

    for sample in samples:
        gt = semantic_mask_from_panoptic_png(sample["mask_path"])
        predictions = {}
        for name in NAMES:
            prediction, prediction_path, source_kind = stores[name].load(
                sequence_id,
                sample["frame_id"],
                gt.shape,
            )
            predictions[name] = prediction
            prediction_paths[name].append(str(prediction_path))
            source_kinds[name][source_kind] = source_kinds[name].get(source_kind, 0) + 1
        vec.append(gt, predictions)

        feature = extract_perceptual_feature(pc_extractor, sample["image_path"])
        if previous_feature is not None:
            # Perceptual image correlation is model-independent and is reused by
            # Host and current-model segmentation scores.
            correlation = perceptual_correlation(previous_feature, feature)
            for name in NAMES:
                labels_a = resize_semantic_to_feature(
                    previous_predictions[name],
                    previous_feature.shape[-2:],
                    device=correlation.device,
                )
                labels_b = resize_semantic_to_feature(
                    predictions[name],
                    feature.shape[-2:],
                    device=correlation.device,
                )
                pc_row = perceptual_consistency_from_correlation(
                    correlation,
                    labels_a,
                    labels_b,
                )
                if math.isfinite(pc_row["pc"]):
                    pc_values[name].append(pc_row["pc"])
            del correlation

        previous_feature = feature
        previous_predictions = predictions

    vec_means = vec.means()
    return {
        "frames": len(samples),
        "pc_pair_count": {name: len(pc_values[name]) for name in NAMES},
        "metrics": {
            name: {
                "PC": _mean(pc_values[name]),
                "VEC8": vec_means[8][name],
                "VEC16": vec_means[16][name],
            }
            for name in NAMES
        },
        "vec_window_counts": vec.window_counts(),
        "vec_mean_valid_class_counts": vec.mean_valid_class_counts(),
        "prediction_source_kinds": source_kinds,
        "prediction_first_last": {
            name: (
                [prediction_paths[name][0], prediction_paths[name][-1]]
                if prediction_paths[name]
                else []
            )
            for name in NAMES
        },
        "_pc_values": pc_values,
        "_vec_values": vec.values,
        "_vec_class_counts": vec.valid_class_counts,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/home/lin/predify/kitti_step")
    parser.add_argument("--host-pred-root", required=True)
    parser.add_argument("--model-pred-root", required=True)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("PC evaluation requested CUDA but CUDA is unavailable")

    dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "val")
    all_groups = sequence_groups(dataset)
    missing = [sequence for sequence in FULL9 if sequence not in all_groups]
    if missing:
        raise RuntimeError(f"missing KITTI-STEP Full9 sequences: {missing}")
    groups = {sequence: all_groups[sequence] for sequence in FULL9}

    stores = {
        "host": PredictionStore(args.host_pred_root),
        "current_model": PredictionStore(args.model_pred_root),
    }
    pc_extractor = build_resnet18_perceptual_extractor(args.device)

    global_pc = {name: [] for name in NAMES}
    global_vec = {
        8: {name: [] for name in NAMES},
        16: {name: [] for name in NAMES},
    }
    global_vec_classes = {
        8: {name: [] for name in NAMES},
        16: {name: [] for name in NAMES},
    }
    per_sequence = {}

    for sequence_id, samples in groups.items():
        row = _evaluate_sequence(sequence_id, samples, stores, pc_extractor)
        pc_internal = row.pop("_pc_values")
        vec_internal = row.pop("_vec_values")
        vec_classes_internal = row.pop("_vec_class_counts")
        for name in NAMES:
            global_pc[name].extend(pc_internal[name])
            for length in (8, 16):
                global_vec[length][name].extend(vec_internal[length][name])
                global_vec_classes[length][name].extend(
                    vec_classes_internal[length][name]
                )
        per_sequence[sequence_id] = row
        print(
            json.dumps(
                {
                    "sequence": sequence_id,
                    "frames": row["frames"],
                    "metrics": row["metrics"],
                }
            ),
            flush=True,
        )

    # Primary Full9 aggregation follows the existing project convention:
    # all valid frame-pairs/windows are pooled across the nine sequences.
    metrics = {
        name: {
            "PC": _mean(global_pc[name]),
            "VEC8": _mean(global_vec[8][name]),
            "VEC16": _mean(global_vec[16][name]),
        }
        for name in NAMES
    }
    # Also report a sequence-macro view because PC Eq.4 is defined per video;
    # the papers do not prescribe how multiple independent KITTI sequences
    # should be combined into one Full9 number.
    metrics_sequence_macro = {
        name: {
            key: _mean(
                per_sequence[sequence]["metrics"][name][key]
                for sequence in FULL9
            )
            for key in ("PC", "VEC8", "VEC16")
        }
        for name in NAMES
    }
    delta = {
        key: metrics["current_model"][key] - metrics["host"][key]
        for key in ("PC", "VEC8", "VEC16")
    }
    delta_sequence_macro = {
        key: (
            metrics_sequence_macro["current_model"][key]
            - metrics_sequence_macro["host"][key]
        )
        for key in ("PC", "VEC8", "VEC16")
    }

    result = {
        "experiment": "KITTI-STEP Full9 prediction-only PC + VEC evaluation",
        "full9": list(FULL9),
        "prediction_sources": {
            "host": str(Path(args.host_pred_root)),
            "current_model": str(Path(args.model_pred_root)),
        },
        "metrics": metrics,
        "metrics_sequence_macro": metrics_sequence_macro,
        "delta_current_model_vs_host": delta,
        "delta_sequence_macro_current_model_vs_host": delta_sequence_macro,
        "counts": {
            "pc_consecutive_pairs": {
                name: len(global_pc[name]) for name in NAMES
            },
            "vec_windows": {
                str(length): {
                    name: len(global_vec[length][name]) for name in NAMES
                }
                for length in (8, 16)
            },
            "vec_mean_valid_classes_per_window": {
                str(length): {
                    name: _mean(global_vec_classes[length][name]) for name in NAMES
                }
                for length in (8, 16)
            },
        },
        "protocol": {
            "PC": {
                "paper": (
                    "Zhang et al., Perceptual Consistency in Video Segmentation, WACV 2022"
                ),
                "reference_code": "yizhezhang2000/SPC/example.py",
                "feature_extractor": (
                    "ImageNet pretrained ResNet-18 children()[:-4] (through layer2)"
                ),
                "feature_stride": 8,
                "segmentation_resize": "nearest to perceptual feature HxW",
                "pair_definition": "consecutive frames",
                "pair_score": "paper Eq.3, bidirectional minimum",
                "video_score": "paper Eq.4 mean over consecutive pairs",
                "full9_primary_aggregation": (
                    "project convention: pair-weighted over all Full9 sequences"
                ),
                "full9_secondary_aggregation": "macro mean of nine per-sequence PC scores",
            },
            "VEC": {
                "paper": "Xu et al., DTERN, ICCV 2025",
                "definition": (
                    "paper Eq.17: class-wise IoU of GT-persistent and "
                    "prediction-persistent C-frame regions"
                ),
                "lengths": [8, 16],
                "window_stride": 1,
                "window_alignment": (
                    "same sliding complete windows as existing mVC8/mVC16"
                ),
                "empty_class_convention": (
                    "classes with empty persistent GT/pred union are excluded from the "
                    "clip macro mean; Eq.17 does not specify the 0/0 implementation case"
                ),
                "ignore_handling": (
                    "KITTI-STEP convention: positions with GT ignore in any frame of a "
                    "clip are excluded from that clip"
                ),
                "full9_primary_aggregation": (
                    "project convention: window-weighted over all Full9 sequences"
                ),
                "full9_secondary_aggregation": (
                    "macro mean of nine per-sequence VEC scores"
                ),
                "official_code_status": (
                    "DTERN repository exposes VC_perclip.py; no dedicated VEC evaluator "
                    "was used as a source for empty-class/ignore edge cases"
                ),
            },
        },
        "stored_prediction_policy": {
            "hard_masks": "nearest resize only if stored hard mask resolution differs",
            "class_scores": "bilinear resize scores first, then argmax",
            "prediction_ignore_label_allowed": False,
            "segmentation_model_inference": False,
        },
        "no_segmentation_model_inference": True,
        "per_sequence": per_sequence,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({"primary": metrics, "sequence_macro": metrics_sequence_macro}, indent=2), flush=True)
    print(f"wrote: {output}", flush=True)
    return result


if __name__ == "__main__":
    main()
