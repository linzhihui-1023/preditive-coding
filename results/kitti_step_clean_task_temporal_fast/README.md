# FAST-B Clean-Task Temporal Training Quick Validation

This independent experiment trained the existing FAST-B semantic/error/C4 path
on clean KITTI-STEP video frames only.  Backbone, decoder, dynamics, and C1
path were frozen.  The objective was segmentation cross-entropy only
(`lambda_feature=0`), with TBPTT=8 and supervision positions 2/4/6/8.

Validation sequences: `0002 / 0010 / 0018`  
Train split: complete KITTI-STEP train sequences  
Epochs: 3  
Corruption training: no  
Test-time adaptation: no

| Epoch | Host clean mIoU | Ours clean mIoU | ΔmIoU | Mean abs ΔZ4 | Mean abs ΔC4 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.5847409576401138 | 0.5567868951351275 | -0.027954062504986332 | 0.005187365004620101 | 0.03958486782104624 |
| 2 | 0.5847409576401138 | 0.5478548236864192 | -0.036886133953694555 | 0.010768887794233956 | 0.08221860482178235 |
| 3 | 0.5847409576401138 | 0.5543019584217715 | -0.030438999218342255 | 0.009619313746259909 | 0.069163368939462 |

Best checkpoint by validation clean mIoU: **Epoch 1**.

Decision: **HARMFUL_TEMPORAL_CORRECTION**.  The learned correction is clearly
nonzero, but it lowers clean segmentation mIoU in every epoch.  Per the task
gate, the corruption stress-test phase was not run.
