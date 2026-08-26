import torch
from torch import nn

from predify2021.mce_scores.train_kitti_step_writeback_delta import (
    EXPECTED_TRAINABLE_PARAMETERS,
    configure_writeback_only,
    writeback_loss,
)
from predify2021.model_factory.deeplabv3plus_resnet50 import (
    BackboneFeatures,
    MultiLayerAdapter,
    UnifiedFeatures,
)


class AdapterHost(nn.Module):
    def __init__(self):
        super().__init__()
        self.host_weight = nn.Parameter(torch.ones(1))
        self.multi_layer_adapter = MultiLayerAdapter()


def test_configure_writeback_only_selects_exactly_d1_and_d4():
    model = AdapterHost()
    adapters, parameters, details = configure_writeback_only(model)

    assert adapters == [
        model.multi_layer_adapter.output_adapters[0],
        model.multi_layer_adapter.output_adapters[3],
    ]
    assert sum(parameter.numel() for parameter in parameters) == (
        EXPECTED_TRAINABLE_PARAMETERS
    )
    assert [detail["parameter_count"] for detail in details] == [32_768, 262_144]
    selected_ids = {id(parameter) for parameter in parameters}
    assert all(parameter.requires_grad for parameter in parameters)
    assert all(
        not parameter.requires_grad
        for parameter in model.parameters()
        if id(parameter) not in selected_ids
    )


def test_writeback_loss_is_exact_mean_of_two_mse_terms():
    model = AdapterHost()
    adapters, _, _ = configure_writeback_only(model)
    clean_features = BackboneFeatures(
        torch.randn(1, 256, 2, 2),
        torch.empty(1, 512, 1, 1),
        torch.empty(1, 1024, 1, 1),
        torch.randn(1, 2048, 1, 1),
    )
    noisy_features = BackboneFeatures(
        torch.randn_like(clean_features.c1),
        torch.empty_like(clean_features.c2),
        torch.empty_like(clean_features.c3),
        torch.randn_like(clean_features.c4),
    )
    clean_state = UnifiedFeatures(
        torch.randn(1, 128, 2, 2),
        torch.empty(1, 128, 1, 1),
        torch.empty(1, 128, 1, 1),
        torch.randn(1, 128, 1, 1),
    )
    noisy_state = UnifiedFeatures(
        torch.randn_like(clean_state.z1),
        torch.empty_like(clean_state.z2),
        torch.empty_like(clean_state.z3),
        torch.randn_like(clean_state.z4),
    )

    loss, loss_z1_c1, loss_z4_c4 = writeback_loss(
        adapters,
        clean_features,
        noisy_features,
        clean_state,
        noisy_state,
    )

    assert torch.equal(loss, 0.5 * (loss_z1_c1 + loss_z4_c4))
    loss.backward()
    assert all(
        parameter.grad is not None
        for adapter in adapters
        for parameter in adapter.parameters()
    )
