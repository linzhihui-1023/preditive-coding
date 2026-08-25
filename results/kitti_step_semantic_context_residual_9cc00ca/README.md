# Semantic-supervised GAIN_DYNAMIC_INSTANT Correction

本实验保持 `GAIN_DYNAMIC_INSTANT` 网络、closed-loop recurrence、noise 和 optimizer 配置不变，只将训练目标从 posterior feature MSE 改为 `CrossEntropyLoss(ignore_index=255)`。Host、Adapter、Predictor、Legacy Correction 均冻结，只有 492,288 个 ContextResidualCorrection 参数可训练。

| Epoch | Train CrossEntropy | Val CrossEntropy |
|---:|---:|---:|
| 1 | 7.7161171755e21 | 7.3006016984e9 |
| 2 | 7.4977162838e12 | 4.8548475364e9 |
| 3 | 1.2876065268e15 | 2.4983703984e8 |

最佳 checkpoint 按 Val CrossEntropy 选择 epoch 3。正式 Val mIoU 为 `0.1911031280`，相对 Feature-MSE GAIN_DYNAMIC_INSTANT `0.3157625378` 下降 `-0.1246594098`，相对 Legacy `0.3011305694` 下降 `-0.1100274414`。

Trainable Parameter、Gradient Path、Closed-loop 和 Finite Gate 均通过，但 loss、logit residual 与 posterior MSE 的数值尺度发生明显放大。按预先定义的判断规则，结论为 `SEMANTIC CORRECTION NO-GO`；本轮未进行调参、稳定化处理或其他 loss 尝试。
