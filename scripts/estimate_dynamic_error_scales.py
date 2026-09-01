#!/usr/bin/env python3
"""Estimate frozen Z1/Z4 RMS scales for the Dynamics Error branch.

The statistic is computed from the raw prediction error e_t = O_t - Zhat_t
using the frozen Host/Predictor path and the same persistent-blur protocol as
the training script.  It intentionally does not instantiate correction
modules or update any parameters.
"""

import json
import os
from pathlib import Path

import torch

from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import sequence_groups
from predify2021.mce_scores.role_separated_direct_state_correction import (
    error_state,
    load_role_components,
    make_paths,
    next_role_prediction,
    zero_state,
)
from predify2021.mce_scores.train_kitti_step_semantic_temporal_error_correction import (
    encode_frozen_sequence_chunk,
    limit_sequence_groups,
    make_sequence_chunk_loader,
    select_unified_features,
)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Dynamic error scale estimation requires CUDA")
    root = Path(os.environ.get("PREDIFY_KITTI_STEP_ROOT", "/home/lin/predify/kitti_step"))
    output = Path(
        os.environ.get(
            "PREDIFY_DYNAMIC_ERROR_SCALE_OUTPUT",
            "results/dynamic_error_scales.json",
        )
    )
    dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(root, "train")
    groups = limit_sequence_groups(
        sequence_groups(dataset),
        sequence_limit=int(os.environ.get("PREDIFY_SCALE_SEQUENCE_LIMIT", "0")),
        frame_limit=int(os.environ.get("PREDIFY_SCALE_FRAMES_PER_SEQUENCE", "0")),
    )
    paths = make_paths()
    model, predictor = load_role_components(
        paths["static"], paths["adapter"], paths["predictor"], paths["writeback"]
    )
    model.requires_grad_(False)
    predictor.requires_grad_(False)
    loader = make_sequence_chunk_loader(groups)
    sum_squared = [0.0, 0.0]
    sample_count = [0, 0]
    previous_sequence = None
    predictor_hidden = None
    pending_dynamics = None
    for chunk in loader:
        if chunk["sequence_id"] != previous_sequence:
            previous_sequence = chunk["sequence_id"]
            predictor_hidden = predictor.initial_state()
            pending_dynamics = None
        images = chunk["images"].cuda(non_blocking=True)
        frame_indices = range(chunk["start"], chunk["start"] + len(images))
        with torch.no_grad():
            _, _, _, observations, _ = encode_frozen_sequence_chunk(
                model, images, frame_indices, chunk["total_frames"]
            )
            for index, frame_index in enumerate(frame_indices):
                observation = select_unified_features(observations, index)
                if frame_index == 0:
                    pending_dynamics, _, *predictor_hidden = next_role_prediction(
                        predictor, observation, zero_state(observation), predictor_hidden
                    )
                    continue
                error = error_state(observation, pending_dynamics)
                for level, value in enumerate((error.z1, error.z4)):
                    sum_squared[level] += value.double().square().sum().item()
                    sample_count[level] += value.numel()
                pending_dynamics, _, *predictor_hidden = next_role_prediction(
                    predictor, observation, error, predictor_hidden
                )
    scales = {
        "z1": (sum_squared[0] / sample_count[0]) ** 0.5,
        "z4": (sum_squared[1] / sample_count[1]) ** 0.5,
    }
    result = {
        "scales": scales,
        "definition": "RMS(e_t), e_t=observation-prediction",
        "sequence_count": len(groups),
        "sample_count": {"z1": sample_count[0], "z4": sample_count[1]},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
