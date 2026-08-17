# DeepLabV3+ ResNet-50 Cityscapes Host

This directory is independent from the VGG16 Predify/TargetFlow host.

The host implements the OpenMMLab MMSegmentation model
`deeplabv3plus_r50-d8_4xb2-80k_cityscapes-512x1024` and loads the official
Cityscapes 80k checkpoint:

```text
deeplabv3plus_r50-d8_512x1024_80k_cityscapes_20200606_114049-f9fb496d.pth
```

The static host exposes a segmentation-forward boundary:

```python
host_feature = model.extract_host_feature(images)
logits = model.decode_from_host_feature(host_feature)
logits = model.decode_from_host_feature(host_feature.replace(modified_tensor))
```

`HostFeature.tensor` is the high-level spatial feature intended for later
adapter modules. `HostFeature.low_level` and `HostFeature.output_size` are the
decoder context required by DeepLabV3+ and should be preserved when replacing
the feature tensor.

No predictor, prediction-error path, state correction, or Predify feedback
recurrence is implemented in this host.
