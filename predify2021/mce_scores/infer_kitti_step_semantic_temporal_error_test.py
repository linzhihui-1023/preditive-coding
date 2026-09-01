import json
import os
from pathlib import Path

import torch
from PIL import Image

from predify2021.mce_scores.role_separated_direct_state_correction import (
    error_state,
    load_image,
    load_role_components,
    make_paths,
    next_role_prediction,
    zero_state,
)
from predify2021.mce_scores.role_separated_dynamic_error_correction import (
    residual_writeback_host_feature,
)
from predify2021.mce_scores.semantic_temporal_error_step import (
    detach_error_state,
    semantic_temporal_error_step,
    zero_semantic_temporal_state,
)
from predify2021.model_factory.deeplabv3plus_resnet50 import HostFeature, UnifiedFeatures
from predify2021.model_factory.deeplabv3plus_resnet50.semantic_temporal_error_correction import (
    build_semantic_temporal_corrections,
)


def test_sequence_groups(root):
    image_root = Path(root) / "testing" / "image_02"
    groups = {}
    for image_path in sorted(image_root.glob("*/*.png")):
        sample = {
            "sequence_id": image_path.parent.name,
            "frame_id": image_path.stem,
            "image_path": image_path,
        }
        groups.setdefault(sample["sequence_id"], []).append(sample)
    for samples in groups.values():
        samples.sort(key=lambda sample: int(sample["frame_id"]))
    return groups


def load_corrections(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    dynamic_enabled = any(
        key.startswith("0.dynamic_encoder.") for key in payload["corrections"]
    )
    temporal_prediction_enabled = any(
        key.startswith("0.temporal_prediction.") for key in payload["corrections"]
    )
    corrections = build_semantic_temporal_corrections(
        use_dynamic_error=dynamic_enabled,
        use_temporal_prediction=temporal_prediction_enabled,
    )
    corrections.load_state_dict(payload["corrections"], strict=True)
    corrections.requires_grad_(False)
    corrections.eval()
    return corrections


def corrected_host_feature(model, raw_features, observation, posterior, output_size):
    delta = UnifiedFeatures(
        posterior.z1 - observation.z1,
        torch.zeros_like(observation.z2),
        torch.zeros_like(observation.z3),
        posterior.z4 - observation.z4,
    )
    return residual_writeback_host_feature(model, raw_features, delta, output_size)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("KITTI-STEP test inference requires CUDA")

    root = Path(os.environ.get("PREDIFY_KITTI_STEP_ROOT", "/home/lin/predify/kitti_step"))
    checkpoint = Path(os.environ["PREDIFY_SEMANTIC_TEMPORAL_ERROR_CHECKPOINT"])
    output = Path(
        os.environ.get(
            "PREDIFY_KITTI_STEP_TEST_OUTPUT_DIR",
            "results/kitti_step_semantic_temporal_error_test",
        )
    )
    prediction_root = output / "semantic_predictions"

    paths = make_paths()
    model, predictor = load_role_components(
        paths["static"],
        paths["adapter"],
        paths["predictor"],
        paths["writeback"],
    )
    corrections = load_corrections(checkpoint)
    groups = test_sequence_groups(root)
    if len(groups) != 29:
        raise RuntimeError(
            f"KITTI-STEP official test protocol expects 29 sequences, got {len(groups)}"
        )

    frame_count = 0
    with torch.inference_mode():
        for sequence_id, samples in groups.items():
            hidden = None
            predictor_hidden = predictor.initial_state()
            pending_dynamics = None
            pending_semantic = None
            sequence_output = prediction_root / sequence_id
            sequence_output.mkdir(parents=True, exist_ok=True)

            for frame_index, sample in enumerate(samples):
                image = load_image(sample)
                raw = model.extract_backbone_features(image)
                observation = model.encode_backbone_features(raw)
                output_size = tuple(image.shape[-2:])
                if hidden is None:
                    hidden = zero_semantic_temporal_state(observation)

                if frame_index == 0:
                    prediction_error = zero_state(observation)
                    pending_dynamics, pending_semantic, *predictor_hidden = next_role_prediction(
                        predictor,
                        observation,
                        prediction_error,
                        predictor_hidden,
                    )
                    posterior = observation
                else:
                    prediction_error = error_state(observation, pending_dynamics)
                    if frame_index == 1:
                        posterior = observation
                    else:
                        posterior, hidden, _ = semantic_temporal_error_step(
                            corrections,
                            observation,
                            pending_dynamics,
                            pending_semantic,
                            hidden,
                        )
                    pending_dynamics, pending_semantic, *predictor_hidden = next_role_prediction(
                        predictor,
                        observation,
                        prediction_error,
                        predictor_hidden,
                    )

                corrected = corrected_host_feature(
                    model,
                    raw,
                    observation,
                    posterior,
                    output_size,
                )
                logits = model.decode_from_host_feature(corrected)
                prediction = logits.argmax(dim=1).squeeze(0).cpu().to(torch.uint8)
                Image.fromarray(prediction.numpy(), mode="L").save(
                    sequence_output / f"{sample['frame_id']}.png"
                )
                hidden = detach_error_state(hidden)
                frame_count += 1

    result = {
        "experiment": "kitti_step_semantic_temporal_error_test_inference",
        "git_revision": os.environ.get("PREDIFY_GIT_REVISION"),
        "checkpoint": str(checkpoint),
        "dataset": {
            "split": "test",
            "sequence_count": len(groups),
            "frame_count": frame_count,
            "ground_truth_available_locally": False,
        },
        "prediction_error_definition": "observation_minus_prediction",
        "dynamic_error_enabled": corrections[0].use_dynamic_error,
        "dynamic_error_usage": (
            "gate_modulation_and_H_next_error_prediction"
            if corrections[0].use_dynamic_error
            else "tracked_only_not_connected_to_correction"
        ),
        "metrics_computed": False,
        "reason_metrics_not_computed": "official KITTI-STEP test ground truth is not publicly provided",
        "semantic_prediction_dir": str(prediction_root),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
