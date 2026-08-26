import inspect

import torch
from torch import nn

from predify2021.mce_scores.evaluate_kitti_step_error_correction import (
    corrected_host_feature,
)
from predify2021.mce_scores.evaluate_kitti_step_oracle_upper_bound import (
    load_writeback_checkpoint,
)
from predify2021.mce_scores.train_kitti_step_writeback_delta import (
    EXPECTED_TRAINABLE_PARAMETERS,
    WRITEBACK_INDICES,
    configure_writeback_only,
    writeback_loss,
)
from predify2021.model_factory.deeplabv3plus_resnet50 import (
    BackboneFeatures,
    HostConditionedResidualWriteback,
    MultiLayerAdapter,
    UnifiedFeatures,
)


class AdapterHost(nn.Module):
    def __init__(self):
        super().__init__()
        self.host_weight = nn.Parameter(torch.ones(1))
        self.multi_layer_adapter = MultiLayerAdapter()
        self.host_conditioned_writebacks = nn.ModuleDict(
            {
                "0": HostConditionedResidualWriteback(256),
                "3": HostConditionedResidualWriteback(2048),
            }
        )
        self.host_conditioned_writeback_enabled = False

    def decode_conditioned_adapter_deltas(self, features, deltas):
        base = self.multi_layer_adapter.decode_deltas(deltas)
        if not self.host_conditioned_writeback_enabled:
            return base
        return BackboneFeatures(
            base.c1 + self.host_conditioned_writebacks["0"](features.c1, deltas.z1),
            base.c2,
            base.c3,
            base.c4 + self.host_conditioned_writebacks["3"](features.c4, deltas.z4),
        )


def sample_pair():
    clean_features = BackboneFeatures(
        torch.randn(1, 256, 2, 3),
        torch.empty(1, 512, 1, 1),
        torch.empty(1, 1024, 1, 1),
        torch.randn(1, 2048, 1, 2),
    )
    noisy_features = BackboneFeatures(
        torch.randn_like(clean_features.c1),
        torch.empty_like(clean_features.c2),
        torch.empty_like(clean_features.c3),
        torch.randn_like(clean_features.c4),
    )
    clean_state = UnifiedFeatures(
        torch.randn(1, 128, 2, 3),
        torch.empty(1, 128, 1, 1),
        torch.empty(1, 128, 1, 1),
        torch.randn(1, 128, 1, 2),
    )
    noisy_state = UnifiedFeatures(
        torch.randn_like(clean_state.z1),
        torch.empty_like(clean_state.z2),
        torch.empty_like(clean_state.z3),
        torch.randn_like(clean_state.z4),
    )
    return clean_features, noisy_features, clean_state, noisy_state


def predicted_delta(model, host_c1, host_c4, delta_z1, delta_z4):
    return (
        model.multi_layer_adapter.output_adapters[0](delta_z1)
        + model.host_conditioned_writebacks["0"](host_c1, delta_z1),
        model.multi_layer_adapter.output_adapters[3](delta_z4)
        + model.host_conditioned_writebacks["3"](host_c4, delta_z4),
    )


def test_shapes_initial_equivalence_and_zero_correction():
    model = AdapterHost()
    host_c1 = torch.randn(1, 256, 5, 7)
    host_c4 = torch.randn(1, 2048, 3, 4)
    delta_z1 = torch.randn(1, 128, 5, 7)
    delta_z4 = torch.randn(1, 128, 3, 4)

    residual_c1 = model.host_conditioned_writebacks["0"](host_c1, delta_z1)
    residual_c4 = model.host_conditioned_writebacks["3"](host_c4, delta_z4)
    assert residual_c1.shape == host_c1.shape
    assert residual_c4.shape == host_c4.shape
    assert torch.count_nonzero(residual_c1) == 0
    assert torch.count_nonzero(residual_c4) == 0

    combined_c1, combined_c4 = predicted_delta(
        model, host_c1, host_c4, delta_z1, delta_z4
    )
    assert torch.equal(
        combined_c1, model.multi_layer_adapter.output_adapters[0](delta_z1)
    )
    assert torch.equal(
        combined_c4, model.multi_layer_adapter.output_adapters[3](delta_z4)
    )
    assert torch.equal(
        host_c1 + combined_c1,
        host_c1 + model.multi_layer_adapter.output_adapters[0](delta_z1),
    )
    assert torch.equal(
        host_c4 + combined_c4,
        host_c4 + model.multi_layer_adapter.output_adapters[3](delta_z4),
    )

    for writeback in model.host_conditioned_writebacks.values():
        nn.init.normal_(writeback.output_projection.weight)
    assert torch.count_nonzero(
        model.host_conditioned_writebacks["0"](host_c1, torch.zeros_like(delta_z1))
    ) == 0
    assert torch.count_nonzero(
        model.host_conditioned_writebacks["3"](host_c4, torch.zeros_like(delta_z4))
    ) == 0


def test_configure_and_one_step_change_only_conditioned_writeback():
    model = AdapterHost()
    writebacks, output_adapters, parameters, details = configure_writeback_only(model)
    assert writebacks == [
        model.host_conditioned_writebacks["0"],
        model.host_conditioned_writebacks["3"],
    ]
    assert output_adapters == [
        model.multi_layer_adapter.output_adapters[0],
        model.multi_layer_adapter.output_adapters[3],
    ]
    assert sum(parameter.numel() for parameter in parameters) == (
        EXPECTED_TRAINABLE_PARAMETERS
    )
    assert [detail["parameter_count"] for detail in details] == [
        32_768,
        147_456,
        32_768,
        262_144,
        147_456,
        262_144,
    ]
    selected_ids = {id(parameter) for parameter in parameters}
    before = {id(parameter): parameter.detach().clone() for parameter in model.parameters()}
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=0.01)
    clean_features, noisy_features, clean_state, noisy_state = sample_pair()
    loss, _, _ = writeback_loss(
        writebacks,
        output_adapters,
        clean_features,
        noisy_features,
        clean_state,
        noisy_state,
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    changed_ids = {
        id(parameter)
        for parameter in model.parameters()
        if not torch.equal(parameter.detach(), before[id(parameter)])
    }
    assert changed_ids
    assert changed_ids <= selected_ids
    assert all(
        parameter.grad is None
        for parameter in model.parameters()
        if id(parameter) not in selected_ids
    )


def test_loss_is_exact_equal_mean_of_c1_and_c4_target_delta_mse():
    model = AdapterHost()
    writebacks, output_adapters, _, _ = configure_writeback_only(model)
    clean_features, noisy_features, clean_state, noisy_state = sample_pair()

    loss, loss_c1, loss_c4 = writeback_loss(
        writebacks,
        output_adapters,
        clean_features,
        noisy_features,
        clean_state,
        noisy_state,
    )
    delta_z1 = clean_state.z1 - noisy_state.z1
    delta_z4 = clean_state.z4 - noisy_state.z4
    predicted_c1, predicted_c4 = predicted_delta(
        model, noisy_features.c1, noisy_features.c4, delta_z1, delta_z4
    )
    expected_c1 = torch.nn.functional.mse_loss(
        predicted_c1, clean_features.c1 - noisy_features.c1
    )
    expected_c4 = torch.nn.functional.mse_loss(
        predicted_c4, clean_features.c4 - noisy_features.c4
    )
    assert torch.equal(loss_c1, expected_c1)
    assert torch.equal(loss_c4, expected_c4)
    assert torch.equal(loss, 0.5 * (expected_c1 + expected_c4))


def test_checkpoint_round_trip_and_old_checkpoint_compatibility(tmp_path):
    model = AdapterHost()
    for writeback in model.host_conditioned_writebacks.values():
        nn.init.normal_(writeback.output_projection.weight)
    host_c1 = torch.randn(1, 256, 2, 2)
    host_c4 = torch.randn(1, 2048, 1, 1)
    delta_z1 = torch.randn(1, 128, 2, 2)
    delta_z4 = torch.randn(1, 128, 1, 1)
    expected = predicted_delta(model, host_c1, host_c4, delta_z1, delta_z4)
    output_adapters = {
        str(index): model.multi_layer_adapter.output_adapters[index].state_dict()
        for index in WRITEBACK_INDICES
    }
    checkpoint = tmp_path / "conditioned.pt"
    torch.save(
        {
            "output_adapters": output_adapters,
            "host_conditioned_writebacks": {
                str(index): model.host_conditioned_writebacks[str(index)].state_dict()
                for index in WRITEBACK_INDICES
            },
        },
        checkpoint,
    )
    restored = AdapterHost()
    load_writeback_checkpoint(restored, checkpoint)
    actual = predicted_delta(restored, host_c1, host_c4, delta_z1, delta_z4)
    assert restored.host_conditioned_writeback_enabled
    assert all(torch.equal(left, right) for left, right in zip(expected, actual))

    old_checkpoint = tmp_path / "old.pt"
    torch.save({"output_adapters": output_adapters}, old_checkpoint)
    old_restored = AdapterHost()
    load_writeback_checkpoint(old_restored, old_checkpoint)
    assert not old_restored.host_conditioned_writeback_enabled


def test_inference_boundary_uses_only_noisy_host_and_delta_z():
    signature = inspect.signature(corrected_host_feature)
    assert tuple(signature.parameters) == (
        "model",
        "raw_features",
        "noisy_state",
        "target_state",
        "output_size",
    )
    signature = inspect.signature(HostConditionedResidualWriteback.forward)
    assert tuple(signature.parameters) == ("self", "host_feature", "delta")

    model = AdapterHost()
    model.host_conditioned_writeback_enabled = True
    raw_features = BackboneFeatures(
        torch.randn(1, 256, 2, 2),
        torch.randn(1, 512, 1, 1),
        torch.randn(1, 1024, 1, 1),
        torch.randn(1, 2048, 1, 1),
    )
    noisy_state = UnifiedFeatures(
        torch.randn(1, 128, 2, 2),
        torch.randn(1, 128, 1, 1),
        torch.randn(1, 128, 1, 1),
        torch.randn(1, 128, 1, 1),
    )
    target_state = UnifiedFeatures(
        *(torch.randn_like(value) for value in noisy_state.as_tuple())
    )
    host_feature = corrected_host_feature(
        model, raw_features, noisy_state, target_state, (8, 8)
    )
    assert host_feature.low_level.shape == raw_features.c1.shape
    assert host_feature.tensor.shape == raw_features.c4.shape
    assert host_feature.output_size == (8, 8)
