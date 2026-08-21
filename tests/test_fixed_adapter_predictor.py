from predify2021.mce_scores.train_kitti_step_fixed_adapter_predictor import (
    configure_predictor_only,
)
from predify2021.model_factory.deeplabv3plus_resnet50 import (
    MultiLayerPredictor,
    build_deeplabv3plus_resnet50_host,
)


def test_fixed_adapter_training_only_enables_predictor_parameters():
    model = build_deeplabv3plus_resnet50_host(load_cityscapes_checkpoint=False)
    predictor = MultiLayerPredictor()

    trainable_parameters = configure_predictor_only(model, predictor)

    assert not any(parameter.requires_grad for parameter in model.parameters())
    assert all(parameter.requires_grad for parameter in predictor.parameters())
    assert sum(parameter.numel() for parameter in trainable_parameters) == sum(
        parameter.numel() for parameter in predictor.parameters()
    )
