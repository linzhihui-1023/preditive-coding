import json
import os
import random
from pathlib import Path

import torch

from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import (
    ADAPTER_CHECKPOINT_DEFAULT,
    PREDICTOR_CHECKPOINT_DEFAULT,
    STATIC_CHECKPOINT_DEFAULT,
    sequence_groups,
)
from predify2021.mce_scores.evaluate_kitti_step_raft_error_memory import (
    CLOSED_LOOP_CORRECTION_CHECKPOINT_DEFAULT,
    evaluate_unaligned_baselines,
)
from predify2021.mce_scores.train_kitti_step_state_predictor import (
    load_static_kitti_checkpoint,
)
from predify2021.model_factory.deeplabv3plus_resnet50 import (
    MultiLayerPredictor,
    build_deeplabv3plus_resnet50_host,
)
from predify2021.model_factory.deeplabv3plus_resnet50.corrections import (
    ErrorGainCorrection,
)


REFERENCE_NOISY = 0.2940764905
REFERENCE_UNALIGNED = 0.3011305694


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Baseline reproduction expects GPU 0.")
    seed = 0
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    root = Path(os.environ.get("PREDIFY_KITTI_STEP_ROOT", "/home/lin/predify/kitti_step"))
    static_checkpoint = Path(
        os.environ.get("PREDIFY_KITTI_STEP_STATIC_CHECKPOINT", STATIC_CHECKPOINT_DEFAULT)
    )
    adapter_checkpoint = Path(
        os.environ.get("PREDIFY_KITTI_STEP_ADAPTER_CHECKPOINT", ADAPTER_CHECKPOINT_DEFAULT)
    )
    predictor_checkpoint = Path(
        os.environ.get("PREDIFY_KITTI_STEP_PREDICTOR_CHECKPOINT", PREDICTOR_CHECKPOINT_DEFAULT)
    )
    correction_checkpoint = Path(
        os.environ.get(
            "PREDIFY_KITTI_STEP_CLOSED_LOOP_CORRECTION_CHECKPOINT",
            CLOSED_LOOP_CORRECTION_CHECKPOINT_DEFAULT,
        )
    )
    output_dir = Path(
        os.environ.get(
            "PREDIFY_KITTI_STEP_BASELINE_REPRO_OUTPUT_DIR",
            "/home/lin/predify/experiments/kitti_step_baseline_repro",
        )
    )
    sigma = 0.10
    model = build_deeplabv3plus_resnet50_host(
        checkpoint_path=os.environ.get("PREDIFY_DEEPLABV3PLUS_CITYSCAPES_CHECKPOINT"),
    ).cuda()
    load_static_kitti_checkpoint(model, static_checkpoint)
    adapter_payload = torch.load(adapter_checkpoint, map_location="cpu", weights_only=False)
    model.multi_layer_adapter.load_state_dict(adapter_payload["adapter_state_dict"], strict=True)
    predictor = MultiLayerPredictor().cuda()
    predictor_payload = torch.load(predictor_checkpoint, map_location="cpu", weights_only=False)
    predictor.load_state_dict(predictor_payload["predictor_state_dict"], strict=True)
    corrections = torch.nn.ModuleList(
        [ErrorGainCorrection(), ErrorGainCorrection()]
    ).cuda()
    correction_payload = torch.load(
        correction_checkpoint, map_location="cpu", weights_only=False
    )
    corrections.load_state_dict(
        correction_payload["correction_state_dict"], strict=True
    )
    model.requires_grad_(False)
    predictor.requires_grad_(False)
    corrections.requires_grad_(False)
    dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(root, "val")
    groups = sequence_groups(dataset)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    metrics, finite, evaluated_frame_count = evaluate_unaligned_baselines(
        model, predictor, corrections, groups, sigma
    )
    noisy_delta = abs(metrics["noisy"] - REFERENCE_NOISY)
    unaligned_delta = abs(metrics["unaligned"] - REFERENCE_UNALIGNED)
    decision = (
        "BASELINE REPRODUCTION PASS"
        if finite and noisy_delta < 1e-4 and unaligned_delta < 1e-4
        else "BASELINE REPRODUCTION FAIL"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment": "kitti_step_baseline_reproduction",
        "git_revision": os.environ.get("PREDIFY_GIT_REVISION"),
        "seed": seed,
        "gaussian_noise_sigma": sigma,
        "dataset": {
            "name": "KITTI-STEP",
            "split": "val",
            "sequence_count": len(groups),
            "frame_count": len(dataset.samples),
            "evaluated_frame_count": evaluated_frame_count,
        },
        "checkpoints": {
            "static_host": str(static_checkpoint),
            "fixed_adapter": str(adapter_checkpoint),
            "fixed_predictor": str(predictor_checkpoint),
            "fixed_closed_loop_correction": str(correction_checkpoint),
        },
        "protocol": {
            "raft_ran": False,
            "alignment_ran": False,
            "optimizer": None,
            "same_noisy_frame_for_noisy_and_unaligned": True,
            "dynamic_error_gamma": 0.207,
            "dynamic_error_history_weight": 0.793,
        },
        "frozen": {
            "host": True,
            "adapter": True,
            "predictor": True,
            "correction": True,
        },
        "finite": finite,
        "metrics": {
            "mIoU_noisy_static": metrics["noisy"],
            "mIoU_unaligned_closed_loop": metrics["unaligned"],
        },
        "reference": {
            "mIoU_noisy_static": REFERENCE_NOISY,
            "mIoU_unaligned_closed_loop": REFERENCE_UNALIGNED,
            "noisy_absolute_difference": noisy_delta,
            "unaligned_absolute_difference": unaligned_delta,
        },
        "decision": decision,
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
