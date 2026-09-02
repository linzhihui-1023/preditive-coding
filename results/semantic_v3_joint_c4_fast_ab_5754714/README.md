# Joint C4 FAST-A/B Validation (head 5754714)

Code head: `575471417f3bbcc450f714ec4dd0d84b1db0a018`. Protocol: 3 Epoch,
fixed sequences `0002/0010/0018`, Blur-Mid (`sigma=2.25`) and Blur-Max
(`sigma=3.0`). No 9-sequence or 15-epoch run was performed. Each variant has
5,015 feature steps, 632 windows and 2,510 segmentation-supervised frames per
epoch.

## FAST-A — joint C4, full Z4 weight (best Epoch 2)

| condition | Corrupted Host | Restored | Delta mIoU |
|---|---:|---:|---:|
| Blur-Mid | 0.437277 | 0.448883 | +0.011607 |
| Blur-Max | 0.349050 | 0.380405 | +0.031355 |

Mean restored mIoU: `0.414644`; mean Delta: **+0.021481**.

Loss trajectory (`feature_loss`, `segmentation_loss`, `total_loss`):

| epoch | feature | segmentation | total |
|---:|---:|---:|---:|
| 1 | 5.37978e-5 | 0.170562 | 1.04917e-4 |
| 2 | 6.37635e-5 | 0.171507 | 1.15173e-4 |
| 3 | 4.98352e-5 | 0.151438 | 9.52001e-5 |

## FAST-B — joint C4, weak Z4 weight (best Epoch 3)

| condition | Corrupted Host | Restored | Delta mIoU |
|---|---:|---:|---:|
| Blur-Mid | 0.437277 | 0.466610 | +0.029333 |
| Blur-Max | 0.349050 | 0.405978 | +0.056928 |

Mean restored mIoU: `0.436294`; mean Delta: **+0.043131**.

Loss trajectory (`feature_loss`, `segmentation_loss`, `total_loss`):

| epoch | feature | segmentation | total |
|---:|---:|---:|---:|
| 1 | 6.12583e-5 | 0.168679 | 6.59202e-5 |
| 2 | 7.81596e-5 | 0.175952 | 7.23062e-5 |
| 3 | 5.63737e-5 | 0.149446 | 5.89150e-5 |

FAST-B gives the larger mIoU gain in both blur conditions. Joint C4 training
updated the Semantic Branch, C4 Output Adapter and C4 host-conditioned
Writeback; Backbone, Dynamics, C1 path and Decoder remained frozen.
