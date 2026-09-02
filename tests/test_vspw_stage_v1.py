import numpy as np
import pytest
import torch

from predify2021.datasets.vspw import (
    VSPW_NUM_CLASSES,
    VSPWSegmentationDataset,
    semantic_mask_from_vspw_png,
)
from predify2021.mce_scores.vspw_metrics import official_vc_window_scores
from predify2021.model_factory.deeplabv3plus_resnet50 import build_deeplabv3plus_resnet50_host


def test_vspw_label_mapping():
    from PIL import Image

    path = "/tmp/vspw_label_mapping_test.png"
    Image.fromarray(np.arange(125, dtype=np.uint8).reshape(1, -1), mode="L").save(path)
    mapped = semantic_mask_from_vspw_png(path).tolist()[0]
    assert mapped == [255, *range(124)]


def test_vspw_split_and_frame_counts():
    dataset_root = "/home/lin/datasets/VSPW_480p"
    dataset = VSPWSegmentationDataset(dataset_root, "train")
    val = VSPWSegmentationDataset(dataset_root, "val")
    test = VSPWSegmentationDataset(dataset_root, "test")
    assert (len(dataset.video_ids), len(val.video_ids), len(test.video_ids)) == (2806, 343, 387)
    assert (len(dataset), len(val), len(test)) == (197253, 24392, 28767)


def test_vspw_model_shape_and_124_class_heads():
    model = build_deeplabv3plus_resnet50_host(num_classes=VSPW_NUM_CLASSES, load_cityscapes_checkpoint=False)
    model.eval()
    with torch.inference_mode():
        output = model(torch.rand(1, 3, 64, 96))
    assert output.shape == (1, VSPW_NUM_CLASSES, 64, 96)
    assert model.decode_head.conv_seg.out_channels == VSPW_NUM_CLASSES
    assert model.auxiliary_head.conv_seg.out_channels == VSPW_NUM_CLASSES


def _official_reference(gt_frames, prediction_frames, clip_size):
    scores = []
    for i in range(len(gt_frames) - clip_size):
        global_common = np.ones(gt_frames[0].shape, dtype=bool)
        predglobal_common = np.ones(gt_frames[0].shape, dtype=bool)
        for j in range(1, clip_size):
            global_common = np.logical_and(global_common, gt_frames[i] == gt_frames[i + j])
            predglobal_common = np.logical_and(predglobal_common, prediction_frames[i] == prediction_frames[i + j])
        denominator = global_common.sum()
        if denominator:
            scores.append(float((predglobal_common * global_common).sum() / denominator))
    return scores


@pytest.mark.parametrize("clip_size", [8, 16])
def test_vspw_official_vc_equivalence(clip_size):
    rng = np.random.default_rng(1234 + clip_size)
    gt = [rng.integers(0, 125, size=(7, 9), dtype=np.uint8) for _ in range(clip_size + 4)]
    prediction = [rng.integers(0, 124, size=(7, 9), dtype=np.uint8) for _ in gt]
    assert np.allclose(
        official_vc_window_scores(gt, prediction, clip_size),
        _official_reference(gt, prediction, clip_size),
        atol=1e-12,
    )
