# Legacy-Gain Context Residual Correction

本实验在冻结 Legacy Dynamic-Error Correction 的基础上，只训练 Z1/Z4 的 Context Residual Correction。初始 zero-residual 等价 Gate 通过：Legacy mIoU 为 `0.3011305694`，zero-residual mIoU 为 `0.3011243643`，差值为 `6.21e-6`；最大增益差为 `1.00e-6`。

训练 3 个 epoch 的 train/val posterior MSE 分别为：

| Epoch | Train | Val |
|---:|---:|---:|
| 1 | 0.0008061974 | 0.0007266849 |
| 2 | 0.0007575801 | 0.0007081649 |
| 3 | 0.0007327714 | 0.0006841931 |

最佳 epoch 为 3。Val mIoU：Clean `0.6552125562`，Noisy `0.2940764905`，Prediction-only `0.2897704727`，Legacy `0.3011305694`，Context Residual `0.3089081760`。相对 Legacy 提升 `+0.0077776066`，posterior MSE 由 `0.0008552177` 降至 `0.0006834500`。

结论：`CONTEXT RESIDUAL CORRECTION GO`。
