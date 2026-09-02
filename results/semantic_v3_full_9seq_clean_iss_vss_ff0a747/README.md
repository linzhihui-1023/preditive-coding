# Full KITTI-STEP 9-Sequence Clean ISS→VSS Evaluation

Evaluated the latest full-validation head (`70fa616`, with the subsequent
official mVC correction in `ff0a747`) using the existing FAST-B best checkpoint
(Epoch 3). No training or test-time parameter updates were performed. All nine
KITTI-STEP validation sequences were evaluated frame-wise.

## Global metrics

| Model | mIoU | mVC8 | mVC16 |
|---|---:|---:|---:|
| DeepLabV3+ ISS Host | 65.41% | 93.79% | 92.90% |
| Predify FAST-B | 65.08% | 94.12% | 93.55% |
| Delta | -0.33 pp | +0.33 pp | +0.65 pp |

Exact deltas: mIoU `-0.0033373`, mVC8 `+0.0032509`, mVC16 `+0.0065376`.

## Per-sequence mIoU delta

| sequence | ΔmIoU |
|---|---:|
| 0002 | +0.012645 |
| 0006 | -0.001459 |
| 0007 | -0.009439 |
| 0008 | +0.019569 |
| 0010 | -0.011638 |
| 0013 | -0.012148 |
| 0014 | -0.010502 |
| 0016 | -0.061190 |
| 0018 | -0.000757 |

The full clean evaluation improves temporal consistency (mVC8/mVC16) but does
not improve global semantic accuracy (mIoU).
