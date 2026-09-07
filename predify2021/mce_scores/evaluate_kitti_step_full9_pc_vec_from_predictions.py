"""Prediction-only KITTI-STEP Full9 PC + VEC8/VEC16 evaluation.

中文：只读取已经存在的 Host / 当前模型逐帧语义预测，重新计算：
- PC: Perceptual Consistency（感知一致性）, WACV 2022 SPC official protocol;
- VEC8 / VEC16: Video Effective Consistency（视频有效一致性）, ICCV 2025 Eq.17.

This script deliberately imports no segmentation model/checkpoint loader and
performs no Host/current-model inference.  RGB frames are read only because PC
requires ImageNet ResNet-18 perceptual features; GT is read only for VEC.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from predify2021.datasets.kitti_step import (
    KITTISTEPSegmentationDataset,
    semantic_mask_from_panoptic_png,
)
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import (
    sequence_groups,
)
from predify2021.mce_scores.perceptual_consistency import (
    build_resnet18_perceptual_extractor,
    extract_perceptual_feature,
    perceptual_consistency_from_correlation,
    perceptual_correlation,
    resize_semantic_to_feature,
)
from predify2021.mce_scores.train_kitti_step_task_space_prior_c_only import FULL9
from predify2021.mce_scores.video_metrics import VideoEffectiveConsistency


NAMES = ("host", "current_model")
NUM_CLASSES = 19
IGNORE_LABEL = 255
DEFAULT_OUTPUT = "results/kitti_step_full9_pc_vec_from_predictions.json"
SUPPORTED_SUFFIXES = (".png", ".npy", ".npz", ".pt", ".pth")


class PredictionStore:
    """Index existing per-frame semantic predictions without rerunning a model."""

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
        exact = self.by_sequence_frame.get((str(sequence_id), str(frame_id)), [])
        if len(exact) == 1:
            return exact[0]
        if len(exact) > 1:
            raise RuntimeError(
                f"ambiguous prediction for {sequence_id}/{frame_id}: {exact}"
            )
        flat = self.by_frame.get(str(frame_id), [])
        if len(flat) == 1:
            return flat[0]
        if len(flat) > 1:
            matches = [p for p in flat if str(sequence_id) in p.parts]
            if len(matches) == 1:
                return matches[0]
        raise FileNotFoundError(
            f"prediction not found for sequence={sequence_id}, frame={frame_id} "
            f"under {self.root}"
        )

    @staticmethod
    def _tensor_from_loaded(value):
        if isinstance(value, dict):
            for key in ("prediction", "pred", "mask", "labels", "semantic"):
                if key in value:
                    value = value[key]
                    break
            else:
                raise ValueError(
                    "prediction checkpoint dict must contain prediction/pred/mask/labels/semantic"
                )
        if isinstance(value, np.ndarray):
            tensor = torch.from_numpy(value)
        else:
            tensor = torch.as_tensor(value)
        tensor = tensor.detach().cpu()
        while tensor.ndim > 2 and tensor.shape[0] == 1:
            tensor = tensor[0]
        if tensor.ndim == 3:
            if tensor.shape[0] == NUM_CLASSES:
                tensor = tensor.argmax(dim=0)
            elif tensor.shape[-1] == NUM_CLASSES:
                tensor = tensor.argmax(dim=-1)
            elif tensor.shape[-1] in (3, 4):
                channels = tensor[..., :3]
                if not bool((channels == channels[..., :1]).all()):
                    raise ValueError(
                        "RGB prediction is not an index mask; class-color decoding is undefined"
                    )
                tensor = channels[..., 0]
            else:
                raise ValueError(f"unsupported prediction tensor shape: {tuple(tensor.shape)}")
        if tensor.ndim != 2:
            raise ValueError(f"prediction must reduce to HxW, got {tuple(tensor.shape)}")
        return tensor.long()

    def load(self, sequence_id, frame_id, expected_hw):
        path = self._resolve(sequence_id, frame_id)
        suffix = path.suffix.lower()
        if suffix == ".png":
            with Image.open(path) as image:
                array = np.asarray(image)
            tensor = self._tensor_from_loaded(array)
        elif suffix == ".npy":
            tensor = self._tensor_from_loaded(np.load(path, allow_pickle=False))
        elif suffix == ".npz":
            payload = np.load(path, allow_pickle=False)
            if len(payload.files) != 1:
                raise ValueError(f"NPZ prediction must contain one array: {path}")
            tensor = self._tensor_from_loaded(payload[payload.files[0]])
        else:
            tensor = self._tensor_from_loaded(torch.load(path, map_location="cpu"))

        if tuple(tensor.shape) != tuple(expected_hw):
            tensor = torch.nn.functional.interpolate(
                tensor[None, None].float(),
                size=tuple(expected_hw),
                mode="nearest",
            )[0, 0].long()
        invalid = ~(((tensor >= 0) & (tensor < NUM_CLASSES)) | (tensor == IGNORE_LABEL))
        if bool(invalid.any()):
            values = torch.unique(tensor[invalid])[:16].tolist()
            raise ValueError(f"prediction has invalid class ids {values}: {path}")
        return tensor, path


def _mean(values):
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return sum(finite) / len(finite) if finite else float("nan")


def _evaluate_sequence(
    sequence_id,
    samples,
    stores,
    pc_extractor,
):
    vec = VideoEffectiveConsistency(NAMES, num_classes=NUM_CLASSES, ignore_label=IGNORE_LABEL)
    pc_values = {name: [] for name in NAMES}
    prediction_paths = {name: [] for name in NAMES}

    previous_feature = None
    previous_predictions = None
    previous_frame_id = None

    for sample in samples:
        gt = semantic_mask_from_panoptic_png(sample["mask_path"])
        predictions = {}
        for name in NAMES:
            pred, pred_path = stores[name].load(
                sequence_id,
                sample["frame_id"],
                gt.shape,
            )
            predictions[name] = pred
            prediction_paths[name].append(str(pred_path))
        vec.append(gt, predictions)

        feature = extract_perceptual_feature(pc_extractor, sample["image_path"])
        if previous_feature is not None:
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
                row = perceptual_consistency_from_correlation(
                    correlation,
                    labels_a,
                    labels_b,
                )
                if not math.isfinite(row["pc"]):
                    raise FloatingPointError(
                        f"non-finite PC for {name} {sequence_id}:"
                        f"{previous_frame_id}->{sample['frame_id']}"
                    )
                pc_values[name].append(row["pc"])
            del correlation

        previous_feature = feature
        previous_predictions = predictions
        previous_frame_id = sample["frame_id"]

    vec_means = vec.means()
    return {
        "frames": len(samples),
        "pc_pair_count": len(pc_values[NAMES[0]]),
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
    groups = sequence_groups(dataset)
    missing = [sequence for sequence in FULL9 if sequence not in groups]
    if missing:
        raise RuntimeError(f"missing KITTI-STEP Full9 sequences: {missing}")
    groups = {sequence: groups[sequence] for sequence in FULL9}

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
        row = _evaluate_sequence(
            sequence_id,
            samples,
            stores,
            pc_extractor,
        )
        for name in NAMES:
            global_pc[name].extend(row.pop("_pc_values")[name] if False else [])
        # Pop internal arrays after copying them explicitly.
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

    metrics = {
        name: {
            "PC": _mean(global_pc[name]),
            "VEC8": _mean(global_vec[8][name]),
            "VEC16": _mean(global_vec[16][name]),
        }
        for name in NAMES
    }
    delta = {
        key: metrics["current_model"][key] - metrics["host"][key]
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
        "delta_current_model_vs_host": delta,
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
                "paper": "Zhang et al., Perceptual Consistency in Video Segmentation, WACV 2022",
                "reference_code": "yizhezhang2000/SPC/example.py",
                "feature_extractor": "ImageNet pretrained ResNet-18 children()[:-4] (through layer2)",
                "feature_stride": 8,
                "segmentation_resize": "nearest to perceptual feature HxW",
                "pair_definition": "consecutive frames",
                "pair_score": "paper Eq.3, bidirectional minimum",
                "video_aggregation": "paper Eq.4; Full9 primary score is frame-pair-weighted over all sequences",
            },
            "VEC": {
                "paper": "Xu et al., DTERN, ICCV 2025",
                "definition": "paper Eq.17: class-wise IoU of GT-persistent and prediction-persistent C-frame regions",
                "lengths": [8, 16],
                "window_stride": 1,
                "window_alignment": "same sliding complete windows as existing mVC8/mVC16",
                "class_aggregation": "macro over classes with non-empty persistent GT/pred union; empty-union 0/0 classes excluded",
                "ignore_handling": "positions with GT ignore in any frame of the clip are excluded",
                "full9_aggregation": "window-weighted over all Full9 sequences",
            },
        },
        "no_model_inference": True,
        "per_sequence": per_sequence,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result["metrics"], indent=2), flush=True)
    print(f"wrote: {output}", flush=True)
    return result


if __name__ == "__main__":
    main()
