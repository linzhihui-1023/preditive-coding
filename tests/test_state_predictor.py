import torch

from predify2021.model_factory.deeplabv3plus_resnet50 import (
    MultiLayerPredictor,
    UnifiedFeatures,
)


def test_multilayer_predictor_preserves_each_state_shape():
    previous = UnifiedFeatures(
        torch.randn(2, 128, 16, 24),
        torch.randn(2, 128, 8, 12),
        torch.randn(2, 128, 8, 12),
        torch.randn(2, 128, 8, 12),
    )
    delta = UnifiedFeatures(*(value * 0.1 for value in previous.as_tuple()))
    predicted_delta = MultiLayerPredictor()(previous, delta)
    predicted = UnifiedFeatures(
        *(state + change for state, change in zip(previous.as_tuple(), predicted_delta.as_tuple()))
    )

    assert [value.shape for value in predicted_delta.as_tuple()] == [
        value.shape for value in previous.as_tuple()
    ]
    assert [value.shape for value in predicted.as_tuple()] == [
        value.shape for value in previous.as_tuple()
    ]
