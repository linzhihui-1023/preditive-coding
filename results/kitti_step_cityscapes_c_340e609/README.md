# KITTI-STEP Common Corruption Evaluation

| Model | Clean mIoU | Mean Corruption mIoU | Corruption Drop |
| --- | ---: | ---: | ---: |
| Baseline Host | 0.6552115 | 0.3487591 | 0.3064524 |
| Ours | 0.6144192 | 0.3547722 | 0.2596470 |

Ours - Baseline Clean: -0.0407923
Ours - Baseline Corruption: 0.0060130

| Model | Clean mVC8 | Corruption mVC8 | Clean mVC16 | Corruption mVC16 |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 0.9426642 | 0.7434247 | 0.9345354 | 0.6877364 |
| Ours | 0.9408265 | 0.7822238 | 0.9348334 | 0.7463339 |

## Severity Trend

| Severity | Baseline | Ours | Delta |
| --- | ---: | ---: | ---: |
| S1 | 0.5052382 | 0.4879800 | -0.0172582 |
| S2 | 0.4226733 | 0.4183503 | -0.0043230 |
| S3 | 0.3380919 | 0.3451379 | 0.0070461 |
| S4 | 0.2627276 | 0.2824649 | 0.0197372 |
| S5 | 0.2150647 | 0.2399278 | 0.0248631 |

Ours wins 41/75 corruption conditions by mIoU.
CD/rCD in summary.json are reported against the internal Host reference, not official Cityscapes-C metrics.
