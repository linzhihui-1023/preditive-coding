# Semantic V3-TemporalStats Task-Aware Fast Validation (commit 4903b031)

This run uses the pushed commit `4903b0315543605d826c9518842d8d1c9f038971`
without further code changes. Protocol: 3 Epoch, fixed sequences
`0002/0010/0018`, Blur-Mid (`sigma=2.25`) and Blur-Max (`sigma=3.0`), with the
existing training assignment, TBPTT=8 and validation frame range. The commit's
actual loss settings are `lambda_feature=1.0`, `lambda_segmentation=3e-4` and
segmentation supervision at window positions 2/4/6/8. The old baseline was not
rerun. Best checkpoint by validation mIoU is **Epoch 3**.

## Result 1 — mIoU

| condition | Corrupted Host | Task-Aware Restored | Delta mIoU |
|---|---:|---:|---:|
| Blur-Mid | 0.437277 | 0.448065 | +0.010788 |
| Blur-Max | 0.349050 | 0.362269 | +0.013219 |

Mean Delta mIoU: **+0.012003**, above the previous approximately `+0.00035`
mean gain. Both conditions improve over the corrupted host.

## Result 2 — training loss

| epoch | feature_loss (`L_Z4`) | segmentation_loss (`L_seg`) | total_loss |
|---:|---:|---:|---:|
| 1 | 5.53981e-5 | 0.179121 | 1.09090e-4 |
| 2 | 6.72339e-5 | 0.191782 | 1.24717e-4 |
| 3 | 5.32155e-5 | 0.174082 | 1.05368e-4 |

Each epoch contains 5,015 feature steps, 632 TBPTT windows and 2,510
segmentation-supervised frames.

## Result 3 — internal diagnostics (best Epoch 3)

| condition | Feature Recovery | Temporal Gain | State Recovery | Hidden Growth |
|---|---:|---:|---:|---:|
| Blur-Mid | 5.812% | -0.017473 | -1.896751 | 0.98918 |
| Blur-Max | 6.980% | +0.000247 | -1.172548 | 0.98930 |

## Final answers

1. Blur-Mid mIoU improved: **Yes**, `+0.010788`.
2. Blur-Max mIoU improved: **Yes**, `+0.013219`.
3. Mean mIoU gain: **+0.012003**.
4. Task-Aware Loss solved the “Z4 MSE improves but mIoU does not” issue in this
   fast validation: **Yes**, mIoU improved in both blur conditions.
