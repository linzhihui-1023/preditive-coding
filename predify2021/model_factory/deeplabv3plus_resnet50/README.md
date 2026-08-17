# DeepLabV3+ ResNet-50 Host

This directory is independent from the VGG16 Predify/TargetFlow host.

The static host exposes a segmentation-forward boundary:

```python
host_feature = model.extract_host_feature(images)
logits = model.decode_from_host_feature(host_feature)
logits = model.decode_from_host_feature(host_feature.replace(modified_tensor))
```

`HostFeature.tensor` is the high-level spatial feature intended for later
adapter modules. `HostFeature.low_level` and `HostFeature.output_size` are
decoder context required by DeepLabV3+ and should be preserved when replacing
the feature tensor.

No predictor, prediction-error path, state correction, or Predify feedback
recurrence is implemented in this host.
