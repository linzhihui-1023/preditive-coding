import torch

from predify2021.model_factory.deeplabv3plus_resnet50 import (
    BackboneFeatures,
    MultiLayerAdapter,
    UnifiedFeatures,
)


def test_multilayer_adapter_preserves_spatial_shapes_and_maps_channels():
    features = BackboneFeatures(
        torch.randn(2, 256, 32, 48),
        torch.randn(2, 512, 16, 24),
        torch.randn(2, 1024, 8, 12),
        torch.randn(2, 2048, 4, 6),
    )
    adapter = MultiLayerAdapter()

    unified = adapter.encode(features)
    assert [value.shape for value in unified.as_tuple()] == [
        (2, 128, 32, 48),
        (2, 128, 16, 24),
        (2, 128, 8, 12),
        (2, 128, 4, 6),
    ]

    deltas = adapter.decode_deltas(
        UnifiedFeatures(*(torch.randn_like(value) for value in unified.as_tuple()))
    )
    assert [value.shape for value in deltas.as_tuple()] == [
        value.shape for value in features.as_tuple()
    ]


def test_zero_gain_adapter_does_not_change_host_features():
    features = BackboneFeatures(
        torch.randn(1, 256, 8, 8),
        torch.randn(1, 512, 4, 4),
        torch.randn(1, 1024, 2, 2),
        torch.randn(1, 2048, 1, 1),
    )
    adapter = MultiLayerAdapter()
    unified = adapter.encode(features)
    deltas = UnifiedFeatures(*(torch.randn_like(value) for value in unified.as_tuple()))

    updated = adapter.apply_deltas(features, deltas)

    assert torch.equal(adapter.gain, torch.zeros(4))
    assert all(
        torch.equal(before, after)
        for before, after in zip(features.as_tuple(), updated.as_tuple())
    )
