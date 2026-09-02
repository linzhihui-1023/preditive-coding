"""Full clean ISS->VSS evaluation matching the DiTTA-style core table.

No training and no corruption are used. The complete KITTI-STEP validation
split is evaluated with the frame-wise DeepLabV3+ Host and the trained Predify
FAST-B checkpoint. Metrics are mIoU, mVC8 and mVC16.

mVC follows the VSPW definition: for each n-frame sliding window, evaluate the
fraction of pixels whose ground-truth semantic label is unchanged throughout
the window and whose predictions remain correct throughout the same window.
VC_n is averaged over windows inside each video, then averaged over videos.
"""

import argparse
import json
from collections import deque
from pathlib import Path

import torch

from predify2021.datasets.kitti_step import (
    KITTISTEPSegmentationDataset,
    semantic_mask_from_panoptic_png,
)
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import (
    load_image,
    sequence_groups,
)
from predify2021.mce_scores.evaluate_kitti_step_static_baseline import (
    compute_iou,
    update_confusion_matrix,
)
from predify2021.mce_scores.role_separated_dynamic_error_correction import (
    ADAPTER_CHECKPOINT_DEFAULT,
    ROLE_PREDICTOR_CHECKPOINT_DEFAULT,
    STATIC_CHECKPOINT_DEFAULT,
    WRITEBACK_CHECKPOINT_DEFAULT,
    error_state,
    load_components,
    residual_writeback_host_feature,
    zero_state,
)
from predify2021.model_factory.deeplabv3plus_resnet50 import (
    ErrorRegulatedSemanticRestorationPredictor,
    HostFeature,
    UnifiedFeatures,
)


FAST_B_CHECKPOINT_DEFAULT = (
    "/home/lin/predify/experiments/"
    "kitti_step_semantic_v3_joint_c4_fast_ab_5754714/"
    "fast_b_joint_c4_weak_z4/best.pt"
)
IGNORE_LABEL = 255
NUM_CLASSES = 19


class VideoConsistency:
    """Streaming VSPW-style VC8/VC16 accumulator for one video."""

    def __init__(self, lengths=(8, 16)):
        self.lengths = tuple(lengths)
        self.max_length = max(self.lengths)
        self.gt = deque(maxlen=self.max_length)
        self.pred = deque(maxlen=self.max_length)
        self.sums = {length: 0.0 for length in self.lengths}
        self.counts = {length: 0 for length in self.lengths}

    @staticmethod
    def _window_score(gt_window, pred_window):
        gt_stack = torch.stack(tuple(gt_window), dim=0)
        pred_stack = torch.stack(tuple(pred_window), dim=0)
        reference = gt_stack[0]

        stable_gt = reference != IGNORE_LABEL
        stable_gt &= torch.all(gt_stack == reference.unsqueeze(0), dim=0)
        denominator = int(stable_gt.sum().item())
        if denominator == 0:
            return None

        consistently_correct = stable_gt & torch.all(
            pred_stack == reference.unsqueeze(0), dim=0
        )
        numerator = int(consistently_correct.sum().item())
        return numerator / denominator

    def update(self, ground_truth, prediction):
        self.gt.append(ground_truth.to(torch.int16).cpu())
        self.pred.append(prediction.to(torch.int16).cpu())
        for length in self.lengths:
            if len(self.gt) < length:
                continue
            gt_window = list(self.gt)[-length:]
            pred_window = list(self.pred)[-length:]
            score = self._window_score(gt_window, pred_window)
            if score is not None:
                self.sums[length] += score
                self.counts[length] += 1

    def values(self):
        return {
            length: self.sums[length] / self.counts[length]
            if self.counts[length] else float("nan")
            for length in self.lengths
        }


def slice_state(state, index):
    return UnifiedFeatures(*(value[index:index + 1] for value in state.as_tuple()))


def encode_clean(model, sample):
    image = load_image(sample)
    raw = model.extract_backbone_features(image)
    observation = model.encode_backbone_features(raw)
    return image, observation, raw, tuple(image.shape[-2:])


def load_fast_b(args):
    model, _ = load_components(
        STATIC_CHECKPOINT_DEFAULT,
        ADAPTER_CHECKPOINT_DEFAULT,
        ROLE_PREDICTOR_CHECKPOINT_DEFAULT,
        WRITEBACK_CHECKPOINT_DEFAULT,
    )
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    required = {
        "model_state_dict",
        "c4_output_adapter_state_dict",
        "c4_writeback_state_dict",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise RuntimeError(
            "FAST-B checkpoint does not contain the complete joint-C4 state: "
            f"missing={missing}"
        )

    predictor = ErrorRegulatedSemanticRestorationPredictor(
        use_error_temporal_stats=True
    ).cuda()
    predictor.load_state_dict(payload["model_state_dict"], strict=True)
    model.multi_layer_adapter.output_adapters[3].load_state_dict(
        payload["c4_output_adapter_state_dict"], strict=True
    )
    model.host_conditioned_writebacks["3"].load_state_dict(
        payload["c4_writeback_state_dict"], strict=True
    )

    model.requires_grad_(False)
    predictor.requires_grad_(False)
    model.eval()
    predictor.eval()
    return model, predictor, payload


def predify_logits(model, raw, observation, restored, output_size):
    zero = zero_state(observation)
    delta = UnifiedFeatures(
        zero.z1,
        zero.z2,
        zero.z3,
        restored.z4 - observation.z4,
    )
    host_feature = residual_writeback_host_feature(
        model, raw, delta, output_size
    )
    return model.decode_from_host_feature(host_feature)


def update_prediction_metrics(confusion, consistency, prediction, mask, sequence_confusion=None):
    prediction = prediction.squeeze(0).cpu().to(torch.int64)
    update_confusion_matrix(confusion, prediction, mask)
    if sequence_confusion is not None:
        update_confusion_matrix(sequence_confusion, prediction, mask)
    consistency.update(mask, prediction)


def evaluate(args):
    model, predictor, payload = load_fast_b(args)
    dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(
        Path(args.root), "val"
    )
    all_groups = sequence_groups(dataset)
    groups = dict(sorted(all_groups.items()))
    if len(groups) != 9:
        raise RuntimeError(
            "Full KITTI-STEP validation protocol expects 9 sequences, "
            f"got {len(groups)}: {list(groups)}"
        )

    confusion = {
        "host": torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64),
        "predify": torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64),
    }
    per_sequence = {}

    with torch.inference_mode():
        for sequence, samples in groups.items():
            if not samples:
                continue

            host_vc = VideoConsistency()
            predify_vc = VideoConsistency()
            sequence_confusion = {
                "host": torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64),
                "predify": torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64),
            }

            # Frame 0: Predify has no temporal history, so its causal output is
            # exactly the frame-wise Host output.  The frame is still included
            # for a fair all-frame evaluation.
            _, observation, raw, output_size = encode_clean(model, samples[0])
            mask = semantic_mask_from_panoptic_png(samples[0]["mask_path"])
            host_logits = model.decode_from_host_feature(
                HostFeature(raw.c4, raw.c1, output_size)
            )
            host_prediction = host_logits.argmax(dim=1)
            update_prediction_metrics(
                confusion["host"], host_vc, host_prediction, mask,
                sequence_confusion["host"],
            )
            update_prediction_metrics(
                confusion["predify"], predify_vc, host_prediction, mask,
                sequence_confusion["predify"],
            )

            pending, h4, h1 = predictor.predict_next(
                observation, zero_state(observation), None, None
            )
            h_sem = predictor.initial_semantic_state(observation)
            error_stats = predictor.initial_error_temporal_statistics()

            for sample in samples[1:]:
                _, observation, raw, output_size = encode_clean(model, sample)
                mask = semantic_mask_from_panoptic_png(sample["mask_path"])

                host_logits = model.decode_from_host_feature(
                    HostFeature(raw.c4, raw.c1, output_size)
                )
                host_prediction = host_logits.argmax(dim=1)

                prediction_error = error_state(observation, pending)
                restored, h_sem, diagnostics = predictor.restore_current(
                    observation,
                    pending,
                    h_sem,
                    error_temporal_state=error_stats,
                )
                error_stats = diagnostics["error_temporal_state"]
                restored_logits = predify_logits(
                    model, raw, observation, restored, output_size
                )
                restored_prediction = restored_logits.argmax(dim=1)

                update_prediction_metrics(
                    confusion["host"], host_vc, host_prediction, mask,
                    sequence_confusion["host"],
                )
                update_prediction_metrics(
                    confusion["predify"], predify_vc, restored_prediction, mask,
                    sequence_confusion["predify"],
                )

                pending, h4, h1 = predictor.predict_next(
                    observation, prediction_error, h4, h1
                )

            host_values = host_vc.values()
            predify_values = predify_vc.values()
            sequence_host_iou = compute_iou(sequence_confusion["host"])
            sequence_predify_iou = compute_iou(sequence_confusion["predify"])
            sequence_host_miou = float(torch.nanmean(sequence_host_iou).item())
            sequence_predify_miou = float(torch.nanmean(sequence_predify_iou).item())
            per_sequence[sequence] = {
                "frame_count": len(samples),
                "host": {
                    "mIoU": sequence_host_miou,
                    "mVC8": host_values[8],
                    "mVC16": host_values[16],
                },
                "predify": {
                    "mIoU": sequence_predify_miou,
                    "mVC8": predify_values[8],
                    "mVC16": predify_values[16],
                },
                "delta": {
                    "mIoU": sequence_predify_miou - sequence_host_miou,
                    "mVC8": predify_values[8] - host_values[8],
                    "mVC16": predify_values[16] - host_values[16],
                },
            }

    host_iou = compute_iou(confusion["host"])
    predify_iou = compute_iou(confusion["predify"])
    host_miou = float(torch.nanmean(host_iou).item())
    predify_miou = float(torch.nanmean(predify_iou).item())

    def mean_sequence_metric(model_name, metric):
        values = [
            row[model_name][metric]
            for row in per_sequence.values()
            if row[model_name][metric] == row[model_name][metric]
        ]
        return sum(values) / len(values)

    host_mvc8 = mean_sequence_metric("host", "mVC8")
    host_mvc16 = mean_sequence_metric("host", "mVC16")
    predify_mvc8 = mean_sequence_metric("predify", "mVC8")
    predify_mvc16 = mean_sequence_metric("predify", "mVC16")

    result = {
        "experiment": "kitti_step_clean_full_val_iss_to_vss",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": payload.get("epoch"),
        "sequences": list(groups),
        "corruption": None,
        "test_time_parameter_updates": False,
        "metrics": {
            "host_iss": {
                "mIoU": host_miou,
                "mVC8": host_mvc8,
                "mVC16": host_mvc16,
            },
            "predify_fast_b": {
                "mIoU": predify_miou,
                "mVC8": predify_mvc8,
                "mVC16": predify_mvc16,
            },
            "delta": {
                "mIoU": predify_miou - host_miou,
                "mVC8": predify_mvc8 - host_mvc8,
                "mVC16": predify_mvc16 - host_mvc16,
            },
        },
        "per_sequence": per_sequence,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    # DiTTA-style display in percentage points.
    h = result["metrics"]["host_iss"]
    p = result["metrics"]["predify_fast_b"]
    d = result["metrics"]["delta"]
    print("| Model | mIoU | mVC8 | mVC16 |")
    print("|---|---:|---:|---:|")
    print(
        f"| DeepLabV3+ ISS Host | {100*h['mIoU']:.2f} | "
        f"{100*h['mVC8']:.2f} | {100*h['mVC16']:.2f} |"
    )
    print(
        f"| Predify FAST-B | {100*p['mIoU']:.2f} | "
        f"{100*p['mVC8']:.2f} | {100*p['mVC16']:.2f} |"
    )
    print(
        f"| Delta | {100*d['mIoU']:+.2f} | "
        f"{100*d['mVC8']:+.2f} | {100*d['mVC16']:+.2f} |"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="/home/lin/predify/kitti_step"
    )
    parser.add_argument(
        "--checkpoint", default=FAST_B_CHECKPOINT_DEFAULT
    )
    parser.add_argument(
        "--output",
        default="results/kitti_step_clean_full_val_iss_to_vss.json",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    evaluate(args)


if __name__ == "__main__":
    main()
