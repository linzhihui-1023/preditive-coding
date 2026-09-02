from pathlib import Path

import torch

from predify2021.datasets.vspw import (
    VSPWSequentialClipDataset,
    VSPW_NUM_CLASSES,
)
from predify2021.mce_scores.vspw_fast_b_common import (
    configure_fast_b_host,
    restore_frame,
    reset_temporal_state,
)
from predify2021.mce_scores.vspw_fast_b_evaluator import VSPWMetricAccumulator
from predify2021.model_factory.deeplabv3plus_resnet50 import (
    ErrorRegulatedSemanticRestorationPredictor,
    UnifiedFeatures,
)


DATA_ROOT = "/home/lin/datasets/VSPW_480p"


def test_sequential_clips_stay_in_one_video_and_run():
    dataset = VSPWSequentialClipDataset(DATA_ROOT, "train", clip_length=16)
    assert len(dataset) > 0
    for clip in dataset.clips[:100]:
        assert len(clip["frame_ids"]) <= 16
        assert len(set(sample["video_id"] for sample in clip["samples"])) == 1
        frame_ids = [int(frame_id) for frame_id in clip["frame_ids"]]
        assert frame_ids == list(range(frame_ids[0], frame_ids[0] + len(frame_ids)))


def test_sequential_dataset_does_not_cross_a_real_frame_gap():
    dataset = VSPWSequentialClipDataset(DATA_ROOT, "val", clip_length=16)
    for clip in dataset.clips:
        frame_ids = [int(frame_id) for frame_id in clip["frame_ids"]]
        assert all(right == left + 1 for left, right in zip(frame_ids, frame_ids[1:]))
    for left, right in zip(dataset.clips, dataset.clips[1:]):
        if left["sequence_id"] == right["sequence_id"]:
            assert int(right["frame_ids"][0]) == int(left["frame_ids"][-1]) + 1
        else:
            assert left["is_sequence_end"]


def test_fast_b_host_freeze_interface():
    from predify2021.model_factory.deeplabv3plus_resnet50 import (
        build_deeplabv3plus_resnet50_host,
    )

    host = build_deeplabv3plus_resnet50_host(
        num_classes=VSPW_NUM_CLASSES, load_cityscapes_checkpoint=False
    )
    configure_fast_b_host(host, trainable=True)
    assert all(not parameter.requires_grad for parameter in host.backbone.parameters())
    assert all(not parameter.requires_grad for parameter in host.decode_head.parameters())
    assert all(not parameter.requires_grad for parameter in host.auxiliary_head.parameters())
    assert all(parameter.requires_grad for parameter in host.multi_layer_adapter.output_adapters[3].parameters())
    assert all(parameter.requires_grad for parameter in host.host_conditioned_writebacks["3"].parameters())


def _latent(batch=1, height=4, width=5):
    return UnifiedFeatures(
        torch.randn(batch, 128, height * 4, width * 4),
        torch.randn(batch, 128, height * 2, width * 2),
        torch.randn(batch, 128, height, width),
        torch.randn(batch, 128, height, width),
    )


def test_full_state_is_carried_and_reset_state_is_empty():
    predictor = ErrorRegulatedSemanticRestorationPredictor(
        use_error_temporal_stats=True
    )
    predictor.eval()
    with torch.no_grad():
        predictor.semantic_state_cell.candidate[-1].weight.normal_(0.0, 0.02)
    first = _latent()
    full = reset_temporal_state()
    restored1, full, _ = restore_frame(predictor, first, full, "full")
    semantic_state1 = full.semantic_hidden.clone()
    restored2, full, _ = restore_frame(predictor, first, full, "full")
    assert semantic_state1.abs().sum() > 0
    assert not torch.equal(full.semantic_hidden, torch.zeros_like(full.semantic_hidden))
    assert full.pending_prediction is not None
    reset = reset_temporal_state()
    _, reset, _ = restore_frame(predictor, first, reset, "reset")
    assert reset.pending_prediction is None
    assert reset.h4_dynamics is None
    assert reset.semantic_hidden is None


def test_bptt_unroll_keeps_gradient_through_sixteen_steps():
    predictor = ErrorRegulatedSemanticRestorationPredictor(
        use_error_temporal_stats=False
    )
    predictor.train()
    hidden0 = torch.randn(1, 128, 4, 5, requires_grad=True)
    hidden = hidden0
    observation = torch.randn(1, 128, 4, 5)
    error = torch.randn(1, 128, 4, 5)
    for _ in range(16):
        hidden, _ = predictor.semantic_state_cell(observation, error, hidden)
    hidden.square().mean().backward()
    assert hidden0.grad is not None
    assert torch.isfinite(hidden0.grad).all()
    assert hidden0.grad.abs().sum() > 0


def test_unified_vspw_accumulator_uses_stage_v1_window_counts():
    accumulator = VSPWMetricAccumulator()
    target = torch.zeros(3, 4, dtype=torch.int64)
    prediction = torch.zeros_like(target)
    for _ in range(20):
        accumulator.add("video", target, prediction, prediction)
    result = accumulator.finish()
    assert result["host"]["mIoU"] == 1.0
    assert result["host"]["mVC8_window_count"] == 12
    assert result["host"]["mVC16_window_count"] == 4
    assert result["fast_b"]["mVC8"] == 1.0


def test_vspw_fast_b_path_has_no_legacy_kitti_corruption_dependencies():
    root = Path(__file__).parents[1]
    source = "\n".join(
        (root / "predify2021/mce_scores" / name).read_text()
        for name in (
            "train_vspw_fast_b.py",
            "evaluate_vspw_fast_b.py",
            "vspw_fast_b_common.py",
        )
    )
    for forbidden in (
        "persistent_gaussian_blur",
        "BLUR_SIGMA",
        "BLUR_WARMUP",
        "DISTILL_WEIGHT",
        "kl_div",
    ):
        assert forbidden not in source
    assert f"VSPW_NUM_CLASSES = {VSPW_NUM_CLASSES}" in (
        root / "predify2021/mce_scores/vspw_fast_b_common.py"
    ).read_text()
