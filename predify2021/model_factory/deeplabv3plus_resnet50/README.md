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

External images must be RGB PyTorch tensors with shape `B x 3 x H x W` and
values in `[0, 1]`. Before the backbone, the host converts them to the
OpenMMLab Cityscapes normalization space:

```text
mean = [123.675, 116.28, 103.53]
std  = [58.395, 57.12, 57.375]
```

`HostFeature.tensor` is the high-level spatial feature intended for later
adapter modules. `HostFeature.low_level` and `HostFeature.output_size` are the
decoder context required by DeepLabV3+ and should be preserved when replacing
the feature tensor.

The backbone also exposes all four spatial stage features without changing the
static decoder path:

```python
stages = model.extract_backbone_features(images)
z = model.encode_backbone_features(stages)
delta_c = model.decode_adapter_deltas(z)
updated = model.apply_adapter_deltas(stages, z)
```

`stages.c1` through `stages.c4` have 256, 512, 1024, and 2048 channels. The
adapter outputs `z1` through `z4` all have 128 channels and keep their own
spatial resolution. The four output mappings restore the original channels.
Their per-layer gains start at zero, and these adapter mappings are not wired
into the static segmentation decoder.

No predictor, prediction-error path, state correction, or Predify feedback
recurrence is implemented in this host.
