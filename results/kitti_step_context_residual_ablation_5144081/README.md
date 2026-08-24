# Context Residual Input Ablation

固定 KITTI-STEP train/val、seed `0`、Gaussian noise `sigma=0.10`，四个 Variant 使用相同的 `640 -> 128 -> 128 -> 128` ContextResidualCorrection、492,288 个参数、AdamW `1e-4`、weight decay `0.01`，各训练 3 epochs。每个 Variant 的初始化 Legacy Equivalence Gate 均通过。

| Variant | Context mIoU | Best val posterior MSE |
|---|---:|---:|
| GAIN_ONLY | 0.3131999369 | 0.0007438136 |
| GAIN_DYNAMIC | 0.3131225352 | 0.0007402573 |
| GAIN_DYNAMIC_INSTANT | 0.3157625378 | 0.0006977812 |
| FULL_CONTEXT | 0.3099579653 | 0.0006829157 |

本次 paired Legacy mIoU 为 `0.3011305694`；共同参考路径为 Clean `0.6552125562`、Noisy `0.2940764905`、Prediction-only `0.2897704727`。

嵌套增量：

- Legacy -> GAIN_ONLY: `+0.0120693675`, **MEANINGFUL CONTRIBUTION**
- GAIN_ONLY -> GAIN_DYNAMIC: `-0.0000774016`, **NO CONTRIBUTION**
- GAIN_DYNAMIC -> GAIN_DYNAMIC_INSTANT: `+0.0026400025`, **WEAK CONTRIBUTION**
- GAIN_DYNAMIC_INSTANT -> FULL_CONTEXT: `-0.0058045725`, **NO CONTRIBUTION**

Full Context 自身达到 `0.3099579653`，相对 Legacy `+0.0088273960`，且 posterior MSE 优于 Legacy；因此 Full Context reproduction 通过。结果说明本协议下主要收益来自 Legacy Gain 的 residual recalibration，Dynamic Error 的直接输入没有额外收益，Instantaneous Error 有弱增量，绝对 Prediction/Observation context 没有额外收益。
