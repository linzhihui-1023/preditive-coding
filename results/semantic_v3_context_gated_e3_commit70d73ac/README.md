# Semantic V3 Context-Gated Restoration Fast Validation

Code under test: `70d73ac71028c3bb13c8259061b78da38a7b36d8` (the context-gated
restoration predictor introduced after `dd0c4cfdb00a750ea67a768be6a645b920b597d3`).
Protocol is unchanged: 3 Epoch, sequences `0002/0010/0018`, Blur-Mid
(`sigma=2.25`) and Blur-Max (`sigma=3.0`). No old baseline was retrained.
Best checkpoint by validation mIoU is **Epoch 1**.

## mIoU (best Epoch 1)

| condition | Corrupted Host | Restored | Delta mIoU |
|---|---:|---:|---:|
| Blur-Mid | 0.437277 | 0.441826 | +0.004550 |
| Blur-Max | 0.349050 | 0.365959 | +0.016909 |

Mean restored mIoU is `0.403893`; mean Delta mIoU is **+0.010729**.

## Training loss

| epoch | feature_loss | segmentation_loss | total_loss |
|---:|---:|---:|---:|
| 1 | 5.40152e-5 | 0.176854 | 1.07029e-4 |
| 2 | 6.51047e-5 | 0.191241 | 1.22431e-4 |
| 3 | 5.13104e-5 | 0.174095 | 1.03478e-4 |

## Internal diagnostics (best Epoch 1)

| condition | Feature Recovery | Temporal Gain | State Recovery | Hidden Growth |
|---|---:|---:|---:|---:|
| Blur-Mid | 11.326% | +0.069776 | -1.038114 | 0.99871 |
| Blur-Max | 11.791% | +0.080772 | -0.609149 | 0.99989 |

Both blur conditions improve over the corrupted Host mIoU. Temporal Gain is
positive in both conditions; State Recovery remains negative and Hidden Growth
is stable.
