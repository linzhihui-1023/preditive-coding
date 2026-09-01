"""Diagnose Semantic V2 on the same Blur-Mid/Blur-Max pressure conditions.

No parameters are updated. The test compares no-history and continuous semantic
state while sharing the same frozen Dynamics Branch, then measures whether the
restored Z4 moves toward clean Z4 and whether the current writeback can convert
that restoration into segmentation mIoU.
"""

import argparse
import csv
import json
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
    compute_iou,
    update_confusion_matrix,
)
from predify2021.mce_scores.kitti_step_persistent_blur import warmup_frame_count
from predify2021.mce_scores.role_separated_dynamic_error_correction import (
    ADAPTER_CHECKPOINT_DEFAULT,
    STATIC_CHECKPOINT_DEFAULT,
    WRITEBACK_CHECKPOINT_DEFAULT,
    error_state,
    load_components,
    residual_writeback_host_feature,
    zero_state,
)
from predify2021.mce_scores.train_kitti_step_convgru_decoupling_quick import (
    diagnostic_blur,
)
from predify2021.model_factory.deeplabv3plus_resnet50 import (
    ErrorGuidedSemanticRestorationPredictor,
    HostFeature,
    UnifiedFeatures,
)

CONDITIONS = ("Blur-Mid", "Blur-Max")
VARIANTS = (
    "blur_host",
    "semantic_no_history",
    "semantic_continuous",
    "oracle_z4_writeback",
    "clean_c4_oracle",
    "clean_host",
)
CHECKPOINT_DEFAULT = (
    "/home/lin/predify/experiments/kitti_step_error_guided_semantic_restoration/"
    "best_error_guided_semantic_restoration.pt"
)


def slice_state(state, index):
    return UnifiedFeatures(*(value[index:index + 1] for value in state.as_tuple()))


def zero_z4_delta(reference, z4_delta):
    zero = zero_state(reference)
    return UnifiedFeatures(zero.z1, zero.z2, zero.z3, z4_delta)


def decode_host(model, raw, output_size):
    return model.decode_from_host_feature(HostFeature(raw.c4, raw.c1, output_size))


def decode_clean_c4_oracle(model, clean_raw, blur_raw, output_size):
    return model.decode_from_host_feature(
        HostFeature(clean_raw.c4, blur_raw.c1, output_size)
    )


def decode_z4_writeback(model, blur_raw, observation, target_z4, output_size):
    delta = zero_z4_delta(observation, target_z4 - observation.z4)
    host_feature = residual_writeback_host_feature(model, blur_raw, delta, output_size)
    return model.decode_from_host_feature(host_feature)


def new_feature_aggregate():
    return {
        "element_count": 0,
        "observation_sse": 0.0,
        "semantic_sse": 0.0,
        "direction_cosine_sum": 0.0,
        "direction_cosine_count": 0,
    }


def add_feature_metrics(aggregate, observation_z4, semantic_z4, clean_z4):
    observation_error = observation_z4.float() - clean_z4.float()
    semantic_error = semantic_z4.float() - clean_z4.float()
    aggregate["element_count"] += observation_error.numel()
    aggregate["observation_sse"] += float(observation_error.square().sum().item())
    aggregate["semantic_sse"] += float(semantic_error.square().sum().item())

    predicted_residual = semantic_z4.float() - observation_z4.float()
    oracle_residual = clean_z4.float() - observation_z4.float()
    cosine = F.cosine_similarity(
        predicted_residual,
        oracle_residual,
        dim=1,
        eps=1e-8,
    )
    aggregate["direction_cosine_sum"] += float(cosine.sum().item())
    aggregate["direction_cosine_count"] += cosine.numel()


def finalize_feature_metrics(aggregate):
    elements = max(aggregate["element_count"], 1)
    observation_mse = aggregate["observation_sse"] / elements
    semantic_mse = aggregate["semantic_sse"] / elements
    cosine_count = max(aggregate["direction_cosine_count"], 1)
    return {
        "observation_mse_to_clean_z4": observation_mse,
        "semantic_mse_to_clean_z4": semantic_mse,
        "feature_recovery_fraction": 1.0 - semantic_mse / max(observation_mse, 1e-12),
        "direction_cosine_to_oracle_residual": (
            aggregate["direction_cosine_sum"] / cosine_count
        ),
    }


def recovery_fraction(baseline, value, upper):
    gap = upper - baseline
    return 0.0 if abs(gap) < 1e-12 else (value - baseline) / gap


def evaluate_condition(model, predictor, groups, condition, max_effective_frames=0):
    confusion = {
        name: torch.zeros((19, 19), dtype=torch.int64)
        for name in VARIANTS
    }
    feature = {
        "semantic_no_history": new_feature_aggregate(),
        "semantic_continuous": new_feature_aggregate(),
    }
    effective_frames = 0

    predictor.eval()
    with torch.inference_mode():
        for samples in groups.values():
            if len(samples) < 2:
                continue
            onset = warmup_frame_count(len(samples))

            first_clean = load_image(samples[0])
            first_blur = diagnostic_blur(first_clean, 0, len(samples), condition)
            first_raw = model.extract_backbone_features(first_blur)
            first_observation = model.encode_backbone_features(first_raw)
            zero = zero_state(first_observation)
            pending_prediction, h4_dyn, h1_dyn = predictor.predict_next(
                first_observation,
                zero,
                None,
                None,
            )
            continuous_hidden = None

            for frame_index in range(1, len(samples)):
                if max_effective_frames and effective_frames >= max_effective_frames:
                    break

                sample = samples[frame_index]
                clean_image = load_image(sample)
                blur_image = diagnostic_blur(
                    clean_image,
                    frame_index,
                    len(samples),
                    condition,
                )
                output_size = tuple(clean_image.shape[-2:])
                images = torch.cat((clean_image, blur_image), dim=0)
                raw = model.extract_backbone_features(images)
                states = model.encode_backbone_features(raw)
                clean_state = slice_state(states, 0)
                observation = slice_state(states, 1)
                clean_raw = type(raw)(*(value[0:1] for value in raw.as_tuple()))
                blur_raw = type(raw)(*(value[1:2] for value in raw.as_tuple()))

                prediction_error = error_state(observation, pending_prediction)
                no_history, _, _ = predictor.restore_current(
                    observation,
                    pending_prediction,
                    None,
                )
                continuous, continuous_hidden, _ = predictor.restore_current(
                    observation,
                    pending_prediction,
                    continuous_hidden,
                )

                if frame_index >= onset:
                    mask = semantic_mask_from_panoptic_png(sample["mask_path"])
                    add_feature_metrics(
                        feature["semantic_no_history"],
                        observation.z4,
                        no_history.z4,
                        clean_state.z4,
                    )
                    add_feature_metrics(
                        feature["semantic_continuous"],
                        observation.z4,
                        continuous.z4,
                        clean_state.z4,
                    )

                    logits = {
                        "blur_host": decode_host(model, blur_raw, output_size),
                        "semantic_no_history": decode_z4_writeback(
                            model, blur_raw, observation, no_history.z4, output_size
                        ),
                        "semantic_continuous": decode_z4_writeback(
                            model, blur_raw, observation, continuous.z4, output_size
                        ),
                        "oracle_z4_writeback": decode_z4_writeback(
                            model, blur_raw, observation, clean_state.z4, output_size
                        ),
                        "clean_c4_oracle": decode_clean_c4_oracle(
                            model, clean_raw, blur_raw, output_size
                        ),
                        "clean_host": decode_host(model, clean_raw, output_size),
                    }
                    for name, value in logits.items():
                        update_confusion_matrix(
                            confusion[name],
                            value.argmax(1).squeeze(0).cpu(),
                            mask,
                        )
                    effective_frames += 1

                pending_prediction, h4_dyn, h1_dyn = predictor.predict_next(
                    observation,
                    prediction_error,
                    h4_dyn,
                    h1_dyn,
                )

            if max_effective_frames and effective_frames >= max_effective_frames:
                break

    miou = {
        name: float(torch.nanmean(compute_iou(matrix)).item())
        for name, matrix in confusion.items()
    }
    feature_metrics = {
        name: finalize_feature_metrics(values)
        for name, values in feature.items()
    }
    blur = miou["blur_host"]
    continuous = miou["semantic_continuous"]
    no_history = miou["semantic_no_history"]
    oracle_z4 = miou["oracle_z4_writeback"]
    c4_oracle = miou["clean_c4_oracle"]

    task = {
        "semantic_no_history_gain": no_history - blur,
        "semantic_continuous_gain": continuous - blur,
        "temporal_memory_mIoU_gain": continuous - no_history,
        "semantic_continuous_fraction_of_host_c4_gap": recovery_fraction(
            blur, continuous, c4_oracle
        ),
        "semantic_continuous_fraction_of_writeback_capacity": recovery_fraction(
            blur, continuous, oracle_z4
        ),
        "oracle_z4_writeback_fraction_of_host_c4_gap": recovery_fraction(
            blur, oracle_z4, c4_oracle
        ),
    }
    judgement = {
        "feature_restoration": (
            "SUPPORTED"
            if feature_metrics["semantic_continuous"]["feature_recovery_fraction"] > 0.0
            else "NOT_SUPPORTED"
        ),
        "temporal_semantic_state": (
            "SUPPORTED" if continuous > no_history else "NOT_SUPPORTED"
        ),
        "task_recovery": (
            "SUPPORTED" if continuous > blur else "NOT_SUPPORTED"
        ),
        "STAGE_A": (
            "GO"
            if (
                feature_metrics["semantic_continuous"]["feature_recovery_fraction"] > 0.0
                and continuous > no_history
                and continuous > blur
            )
            else "NO-GO"
        ),
    }
    return {
        "condition": condition,
        "effective_frame_count": effective_frames,
        "mIoU": miou,
        "feature": feature_metrics,
        "task_recovery": task,
        "judgement": judgement,
    }



def evaluate_clean_identity(model, predictor, groups, max_effective_frames=0):
    confusion = {
        name: torch.zeros((19, 19), dtype=torch.int64)
        for name in ("clean_host", "semantic_no_history", "semantic_continuous")
    }
    no_history_sse = 0.0
    continuous_sse = 0.0
    element_count = 0
    effective_frames = 0

    predictor.eval()
    with torch.inference_mode():
        for samples in groups.values():
            if len(samples) < 2:
                continue

            first_image = load_image(samples[0])
            first_raw = model.extract_backbone_features(first_image)
            first_observation = model.encode_backbone_features(first_raw)
            zero = zero_state(first_observation)
            pending_prediction, h4_dyn, h1_dyn = predictor.predict_next(
                first_observation, zero, None, None
            )
            continuous_hidden = None

            for frame_index in range(1, len(samples)):
                if max_effective_frames and effective_frames >= max_effective_frames:
                    break

                sample = samples[frame_index]
                image = load_image(sample)
                output_size = tuple(image.shape[-2:])
                raw = model.extract_backbone_features(image)
                observation = model.encode_backbone_features(raw)
                prediction_error = error_state(observation, pending_prediction)

                no_history, _, _ = predictor.restore_current(
                    observation, pending_prediction, None
                )
                continuous, continuous_hidden, _ = predictor.restore_current(
                    observation, pending_prediction, continuous_hidden
                )

                no_history_error = no_history.z4.float() - observation.z4.float()
                continuous_error = continuous.z4.float() - observation.z4.float()
                no_history_sse += float(no_history_error.square().sum().item())
                continuous_sse += float(continuous_error.square().sum().item())
                element_count += continuous_error.numel()

                mask = semantic_mask_from_panoptic_png(sample["mask_path"])
                logits = {
                    "clean_host": decode_host(model, raw, output_size),
                    "semantic_no_history": decode_z4_writeback(
                        model, raw, observation, no_history.z4, output_size
                    ),
                    "semantic_continuous": decode_z4_writeback(
                        model, raw, observation, continuous.z4, output_size
                    ),
                }
                for name, value in logits.items():
                    update_confusion_matrix(
                        confusion[name], value.argmax(1).squeeze(0).cpu(), mask
                    )
                effective_frames += 1

                pending_prediction, h4_dyn, h1_dyn = predictor.predict_next(
                    observation, prediction_error, h4_dyn, h1_dyn
                )

            if max_effective_frames and effective_frames >= max_effective_frames:
                break

    elements = max(element_count, 1)
    miou = {
        name: float(torch.nanmean(compute_iou(matrix)).item())
        for name, matrix in confusion.items()
    }
    return {
        "effective_frame_count": effective_frames,
        "mIoU": miou,
        "no_history_z4_change_mse": no_history_sse / elements,
        "continuous_z4_change_mse": continuous_sse / elements,
        "no_history_mIoU_gain_vs_clean": (
            miou["semantic_no_history"] - miou["clean_host"]
        ),
        "continuous_mIoU_gain_vs_clean": (
            miou["semantic_continuous"] - miou["clean_host"]
        ),
    }

def write_csv(path, results):
    rows = []
    for condition, result in results.items():
        for variant in VARIANTS:
            row = {
                "condition": condition,
                "variant": variant,
                "mIoU": result["mIoU"][variant],
                "gain_vs_blur": result["mIoU"][variant] - result["mIoU"]["blur_host"],
            }
            if variant in result["feature"]:
                row.update(result["feature"][variant])
            rows.append(row)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/lin/predify/kitti_step")
    parser.add_argument("--checkpoint", default=CHECKPOINT_DEFAULT)
    parser.add_argument(
        "--output",
        default="results/kitti_step_error_guided_semantic_restoration_diagnostic",
    )
    parser.add_argument("--sequence-limit", type=int, default=0)
    parser.add_argument("--max-effective-frames", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Semantic V2 diagnostic requires CUDA")

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    dynamics_checkpoint = payload["source_dynamics_checkpoint"]

    model, _ = load_components(
        STATIC_CHECKPOINT_DEFAULT,
        ADAPTER_CHECKPOINT_DEFAULT,
        dynamics_checkpoint,
        WRITEBACK_CHECKPOINT_DEFAULT,
    )
    predictor = ErrorGuidedSemanticRestorationPredictor().cuda()
    predictor.load_state_dict(payload["model_state_dict"], strict=True)
    predictor.requires_grad_(False)
    predictor.eval()
    model.requires_grad_(False)
    model.eval()

    groups = sequence_groups(
        KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "val")
    )
    sequence_limit = args.sequence_limit
    max_effective_frames = args.max_effective_frames
    output = Path(args.output)
    if args.smoke:
        sequence_limit = 1
        max_effective_frames = 32
        output = output / "smoke"
    if sequence_limit:
        groups = dict(list(groups.items())[:sequence_limit])
    output.mkdir(parents=True, exist_ok=True)

    results = {}
    for condition in CONDITIONS:
        result = evaluate_condition(
            model,
            predictor,
            groups,
            condition,
            max_effective_frames=max_effective_frames,
        )
        results[condition] = result
        print(json.dumps(result, sort_keys=True), flush=True)

    clean_identity = evaluate_clean_identity(
        model, predictor, groups, max_effective_frames=max_effective_frames
    )
    print(json.dumps({"Clean-Identity": clean_identity}, sort_keys=True), flush=True)

    pressure_go = all(
        result["judgement"]["STAGE_A"] == "GO" for result in results.values()
    )
    clean_go = clean_identity["continuous_mIoU_gain_vs_clean"] >= -0.005
    overall_go = pressure_go and clean_go
    summary = {
        "experiment": "kitti_step_error_guided_semantic_restoration_diagnostic",
        "diagnostic_only": True,
        "parameters_updated": False,
        "checkpoint": str(checkpoint),
        "split": "val",
        "sequence_count": len(groups),
        "conditions": {
            "Blur-Mid": "Gaussian blur sigma=2.25 after existing 10% warmup",
            "Blur-Max": "Gaussian blur sigma=3.0 after existing 10% warmup",
        },
        "results": results,
        "clean_identity": clean_identity,
        "decision": {
            "STAGE_A": "GO" if overall_go else "NO-GO",
            "criterion": (
                "Blur-Mid and Blur-Max both require feature recovery > 0, "
                "continuous mIoU > no-history mIoU, and continuous mIoU > Blur Host; "
                "Clean continuous mIoU loss must be <= 0.005 (diagnostic tolerance)"
            ),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    write_csv(output / "comparison.csv", results)
    print(json.dumps({"decision": summary["decision"]}, indent=2), flush=True)
    print(f"results={output}", flush=True)


if __name__ == "__main__":
    main()
