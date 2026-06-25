# Dynamic Error and KITTI Stream Notes

本文档记录当前阶段对 Predify/PVGG16 的修改、固定时间参数的含义、KITTI 时间流测试方式，以及后续如何证明修改有效。

## 1. 当前目标

当前不是重写整个 Predify 架构，而是先做第一层级修改：

```text
只改误差机制，不改 VGG16 主干，不改原有前向预测流，不改反馈权重结构。
```

也就是说：

```text
backbone: torchvision VGG16
feedback module: 仍使用已有 PVGG16 feedback weights
PCoder 状态更新: 仍保留 beta, lambda, alpha 形式
error 计算: 从即时 MSE 改成带时间状态的 dynamic error
```

## 2. 原始误差机制

原始 `PCoderN` 中的误差是即时 MSE：

```text
prediction_error = mse(prediction, target)
gradient = d(prediction_error) / d(rep)
```

对应代码在：

```text
/home/lin/predify/predify/modules/base.py
```

原始更新逻辑可以概括为：

```text
rep_next =
    beta * feedforward
  + lambda * feedback
  + (1 - beta - lambda) * rep_old
  - alpha * scaled_gradient
```

其中 `scaled_gradient` 来自即时预测误差。

## 3. 新的 dynamic error 机制

现在新增了 `DynamicErrorPCoderN`，核心公式是：

```text
epsilon_{k+1}
  = (Ts / tau) * residual_k
  + (1 - Ts / tau) * epsilon_k

residual_k = target_k - prediction_k
```

代码中：

```text
gamma = Ts / tau
dynamic_error = gamma * residual + (1 - gamma) * previous_error
prediction_error = mean(dynamic_error ** 2)
gradient = d(prediction_error) / d(rep)
```

所以这次不是直接把 MSE 数值替换成另一个静态误差，而是给误差本身增加了跨帧状态。

## 4. 固定时间 Ts 的含义

固定时间不是模型实际推理耗时。

固定进公式里的时间由环境变量决定：

```text
PREDIFY_ERROR_TS
```

例如这次 KITTI 命令里：

```text
PREDIFY_ERROR_TS=0.1
PREDIFY_ERROR_TAU=0.1
```

含义是：

```text
Ts = 0.1 s = 100 ms
tau = 0.1 s = 100 ms
gamma = Ts / tau = 1
```

实际模型推理耗时约 `10.85 ms`，只说明模型能在 `100 ms/frame` 的 KITTI 帧周期内跑完。它不会自动替代公式里的 `Ts`。

重要结论：

```text
如果 Ts = tau，则 dynamic_error = residual，历史误差权重为 0。
如果想让历史误差真正起作用，需要 tau > Ts。
```

建议后续参数：

```text
Ts = 0.1, tau = 0.2  -> 历史误差权重 0.5
Ts = 0.1, tau = 0.5  -> 历史误差权重 0.8
Ts = 0.1, tau = 1.0  -> 历史误差权重 0.9
```

KITTI Raw 相机大约是 `10 Hz`，所以使用：

```text
Ts = 0.1 s
```

是合理的。

## 5. 代码改动位置

### predify 核心库

文件：

```text
/home/lin/predify/predify/modules/base.py
```

主要改动：

```text
1. 增加 _is_zero_multiplier，用于 alpha=0 时跳过不必要的梯度计算。
2. 增加 DynamicErrorPCoderN。
3. DynamicErrorPCoderN 保存 dynamic_error 状态。
4. reset() 时清空 dynamic_error。
5. prediction_error 从 mse(prediction, target) 改成 mean(dynamic_error ** 2)。
```

文件：

```text
/home/lin/predify/predify/networks/base.py
```

主要改动：

```text
1. 给 PNetSameHP 增加 step_frame(x)。
2. 给 PNetSeparateHP 增加 step_frame(x)。
3. step_frame(x) 不调用 reset()，用于连续时间流。
4. forward(x) 保留原行为，输入新图片时仍会 reset()。
```

### predify2021 实验层

文件：

```text
/home/lin/predify2021/predify2021/model_factory/pvgg16_shared.py
```

主要改动：

```text
1. 引入 DynamicErrorPCoderN。
2. 增加 _make_pcoder(...)。
3. 当 dynamic_error=True 时使用 DynamicErrorPCoderN。
4. 当 dynamic_error=False 时继续使用原始 PCoderN。
```

文件：

```text
/home/lin/predify2021/predify2021/model_factory/get_model.py
```

主要改动：

```text
1. get_model(...) 增加 dynamic_error 参数。
2. get_model(...) 增加 error_sample_time 参数，对应 Ts。
3. get_model(...) 增加 error_tau 参数，对应 tau。
4. PVGG16 模型构造时把这些参数传入 pvgg16_shared.py。
```

文件：

```text
/home/lin/predify2021/predify2021/mce_scores/calculate_gaussian_noise_scores.py
```

主要改动：

```text
1. 增加 PREDIFY_DYNAMIC_ERROR。
2. 增加 PREDIFY_ERROR_TS。
3. 增加 PREDIFY_ERROR_TAU。
4. 仍然是旧的同一图片多 timestep 测试，不适合作为最终时间流实验。
```

文件：

```text
/home/lin/predify2021/predify2021/mce_scores/calculate_gaussian_noise_stream_scores.py
```

作用：

```text
把 ImageNet 图片当成 stream 做 smoke test。
不适合作为论文实验，因为 ImageNet val 图片之间没有真实时间连续性。
```

文件：

```text
/home/lin/predify2021/predify2021/mce_scores/calculate_kitti_stream_scores.py
```

作用：

```text
1. 读取 KITTI Raw 连续帧。
2. 序列开始时 net.reset()。
3. 每帧调用一次 net.step_frame(frame)。
4. 不使用 timestep loop。
5. 输出时间流诊断指标。
```

## 6. KITTI 数据

当前下载并解压的小序列：

```text
/home/lin/predify/kitti_raw/2011_09_26/2011_09_26_drive_0005_sync/image_02/data
```

内容：

```text
drive: 2011_09_26_drive_0005_sync
camera: image_02, left color camera
frames: 154
frame range: 0000000000.png to 0000000153.png
approx frame rate: 10 Hz
```

## 7. KITTI 运行命令

完整 154 帧测试：

```bash
PREDIFY_KITTI_ROOT=/home/lin/predify/kitti_raw \
PREDIFY_KITTI_DRIVE=2011_09_26/2011_09_26_drive_0005_sync \
PREDIFY_KITTI_CAMERA=image_02 \
PREDIFY_HP_PRESET=fixed \
PREDIFY_ALPHA_VALUES=0.01 \
PREDIFY_DYNAMIC_ERROR=1 \
PREDIFY_ERROR_TS=0.1 \
PREDIFY_ERROR_TAU=0.1 \
PREDIFY_SIGMAS=0 \
PREDIFY_MAX_FRAMES=0 \
PYTHONPATH=/home/lin/predify2021:/home/lin/predify \
TORCH_HOME=/home/lin/predify/.torch \
python -u -m predify2021.mce_scores.calculate_kitti_stream_scores
```

如果要让历史误差起作用，建议改成：

```bash
PREDIFY_ERROR_TS=0.1
PREDIFY_ERROR_TAU=0.5
```

## 8. 当前 KITTI 结果解释

当前一次运行结果：

```text
Frames: 154
device: cuda:0
Mean frame time: 10.8454 ms
P95 frame time: 9.6792 ms
Top-1 change rate: 22.2222%
Mean top-1 confidence: 0.257842
Mean PCoder error:
  pcoder1: 0.0682391541
  pcoder2: 0.2041813881
  pcoder3: 1.0808315246
  pcoder4: 2.3994173121
  pcoder5: 0.6628097040
```

解释：

```text
1. 154 帧真实连续图像已经跑通。
2. 模型平均每帧约 10.85 ms，小于 KITTI 100 ms 帧周期。
3. dynamic_error 状态在各 PCoder 层都有输出。
4. 这不是 accuracy，因为 KITTI Raw 没有 ImageNet 分类标签。
```

注意：

```text
PREDIFY_ERROR_TS=0.1 且 PREDIFY_ERROR_TAU=0.1 时，历史误差没有保留。
这次运行主要证明代码路径正确，不足以证明动态误差机制有效。
```

## 9. 为什么不能直接用 accuracy

KITTI Raw 是车载时间流数据，没有 ImageNet 1000 类分类标签。

所以当前不能说：

```text
KITTI accuracy = ...
```

只能说：

```text
stream diagnostics
```

也就是：

```text
1. 连续帧输出是否稳定。
2. 加噪前后输出是否一致。
3. PCoder 误差是否按预期变化。
4. 模型是否能在真实帧周期内运行。
```

## 10. 后续如何证明模型有用

单独跑 `dynamic_error=1` 只能证明代码能运行，不能证明模型有效。

必须做对照实验。

最低限度对照：

```text
baseline: dynamic_error=0, 原始 MSE error
ours:     dynamic_error=1, 新的 dynamic error
```

建议实验一：clean-vs-noisy consistency

```text
同一段 KITTI stream
同一帧分别输入 clean 和 noisy
比较输出是否一致
```

指标：

```text
top-1 agreement
KL divergence between logits
cosine similarity between logits or features
```

这个最适合证明动态误差机制是否提高时间流鲁棒性。

建议实验二：temporal smoothness

```text
比较相邻帧输出变化
```

指标：

```text
mean ||logits_t - logits_{t-1}||
top-1 change rate
mean cosine(logits_t, logits_{t-1})
```

建议实验三：negative control

```text
continuous_frames: 正常 KITTI 顺序帧
reset_each_frame: 每帧都 reset，破坏时间状态
shuffled_frames: 打乱帧顺序，破坏时间连续性
```

如果新机制真的依赖时间流，则应该看到：

```text
continuous_frames 表现最好
reset_each_frame 优势下降
shuffled_frames 优势下降
```

## 11. 怎么证明代码修改准确

代码准确性需要单元测试，而不是只看实验结果。

建议增加一个小测试：

```text
给定 target, prediction, epsilon_old, Ts, tau
手算 epsilon_new
检查 DynamicErrorPCoderN.dynamic_error 是否等于手算结果
```

公式：

```text
expected =
    (Ts / tau) * (target - prediction)
  + (1 - Ts / tau) * epsilon_old
```

如果代码输出和 `expected` 一致，说明误差状态更新是准确的。

实验有效性则需要第 10 节的对照实验来证明。

## 12. 当前结论

目前已经完成：

```text
1. Predify 核心中加入 dynamic error PCoder。
2. 网络 wrapper 支持 step_frame，不再每帧 reset。
3. PVGG16 可以通过参数切换原始 MSE error 和 dynamic error。
4. KITTI Raw 小序列已下载。
5. KITTI 连续帧 stream 已经跑通。
6. CUDA 上 154 帧平均每帧约 10.85 ms。
```

目前还没有完成：

```text
1. dynamic_error=0 vs dynamic_error=1 的系统对照。
2. clean-vs-noisy consistency 指标。
3. shuffled/reset negative control。
4. 多条 KITTI drive 验证。
5. 单元测试验证公式实现。
```

因此当前阶段结论应表述为：

```text
我们已经实现并跑通了基于真实时间流的 dynamic error 机制。
下一步需要通过对照实验验证它是否带来鲁棒性或稳定性提升。
```

## 13. README 更新规则

从现在开始，每次做以下两类事情都要同步更新本文档：

```text
1. 代码改动：新增脚本、修改模型、修改公式实现、修改实验入口。
2. 实验记录：运行条件、关键参数、核心指标、结论和限制。
```

记录原则：

```text
1. 不只记录好结果，也记录没有改善的结果。
2. 明确区分“代码跑通”“内部误差变化”“任务指标改善”。
3. 每次实验都写清楚 Ts、tau、sigma、数据集、模型版本和指标含义。
```

## 14. 实验日志：2026-05-09 clean-vs-noisy consistency

### 14.1 新增代码

新增文件：

```text
/home/lin/predify2021/predify2021/mce_scores/calculate_kitti_clean_noisy_consistency.py
```

用途：

```text
1. 同一段 KITTI 时间流，同时构造 clean stream 和 noisy stream。
2. 两条流使用两份独立模型，避免状态互相污染。
3. clean 模型输入原始帧。
4. noisy 模型输入同一帧加 Gaussian noise 后的版本。
5. 每帧分别调用 step_frame(frame)，不使用 timestep loop。
6. 比较同一时刻 clean/noisy 输出的一致性。
```

输出指标：

```text
top1_agreement:
  clean 和 noisy 同一帧 top-1 类别一致的比例，越高越好。

mean_logit_cosine:
  clean logits 和 noisy logits 的余弦相似度，越高越好。

mean_kl_noisy_to_clean:
  noisy 输出分布到 clean 输出分布的 KL divergence，越低越好。

mean_logit_l2:
  clean logits 和 noisy logits 的 L2 距离，越低越好。

clean_top1_change_rate:
  clean stream 相邻帧 top-1 类别变化率。

noisy_top1_change_rate:
  noisy stream 相邻帧 top-1 类别变化率。
```

### 14.2 smoke test

运行条件：

```text
frames = 8
sigma = 0.25
dynamic_error = 1
Ts = 0.1
tau = 0.5
device = CPU in Codex sandbox
```

结果：

```text
Clean-vs-noisy top-1 agreement: 12.5000%
Mean logit cosine: 0.920028
Mean KL noisy->clean: 0.350574
Mean logit L2: 23.919210
Clean top-1 change rate: 42.8571%
Noisy top-1 change rate: 14.2857%
Mean clean confidence: 0.107880
Mean noisy confidence: 0.091247
```

结论：

```text
脚本可以正常运行，clean/noisy 双模型状态流可以正常比较。
8 帧结果只用于检查代码路径，不作为实验结论。
```

### 14.3 完整 clean-vs-noisy 对照

共同条件：

```text
dataset: KITTI Raw 2011_09_26_drive_0005_sync image_02
frames: 154
model: PVGG16
alpha: 0.01
beta: 0.8
lambda: 0.1
Ts: 0.1 s
tau: 0.5 s
history weight: 1 - Ts/tau = 0.8
sigmas: 0.25, 0.5
device: CPU in Codex sandbox
```

注意：

```text
这里的速度不能和用户终端 CUDA 速度比较。
本组实验只比较输出一致性指标。
```

#### sigma = 0.25

| 方法 | top-1 agreement | mean logit cosine | KL noisy->clean | logit L2 | clean change | noisy change |
|---|---:|---:|---:|---:|---:|---:|
| MSE baseline | 24.0260% | 0.873571 | 1.057365 | 31.098308 | 22.2222% | 31.3725% |
| Dynamic error | 24.0260% | 0.873582 | 1.058051 | 31.121028 | 22.2222% | 31.3725% |

#### sigma = 0.5

| 方法 | top-1 agreement | mean logit cosine | KL noisy->clean | logit L2 | clean change | noisy change |
|---|---:|---:|---:|---:|---:|---:|
| MSE baseline | 0.6494% | 0.687810 | 2.305165 | 48.983782 | 22.2222% | 29.4118% |
| Dynamic error | 0.6494% | 0.687618 | 2.308561 | 49.037933 | 22.2222% | 29.4118% |

### 14.4 本组结论

这组结果不支持“当前 dynamic error 提高输出层抗噪一致性”。

具体结论：

```text
1. Dynamic error 与 MSE baseline 的 top-1 agreement 完全相同。
2. Dynamic error 与 MSE baseline 的 cosine/KL/L2 差异极小。
3. 之前实验显示 Dynamic error 会显著降低内部 PCoder error。
4. 但降低内部 PCoder error 没有传导成 logits 层面的鲁棒性提升。
```

当前可以严谨表述为：

```text
Dynamic error 已经改变内部误差动力学，并显著降低 PCoder prediction error。
但在当前 PVGG16 + KITTI/ImageNet 分类输出设置下，
它没有改善 clean-vs-noisy 输出一致性。
```

### 14.5 下一步建议

不建议继续只跑分类 top-1 或 logits consistency。

更合理的下一步：

```text
1. 记录中间层 feature consistency，而不是只看最终 logits。
2. 比较 pcoder1~pcoder5 的 rep/prd/dynamic_error 随时间变化。
3. 做 reset_each_frame 和 shuffled_frames negative control。
4. 如果目标是自动驾驶时间流，应换成 KITTI 任务指标，例如 detection/tracking/segmentation，而不是 ImageNet classification。
```

## 15. 实验日志：2026-05-11 dynamic error 公式层验证

### 15.1 新增代码

新增文件：

```text
/home/lin/predify/tools/check_dynamic_error_formula.py
```

用途：

```text
1. 直接对 DynamicErrorPCoderN 做公式级检查。
2. 不看最终分类输出，只检查 dynamic_error 和 prediction_error 是否等于手算值。
3. 覆盖第一次更新、第二次更新、reset() 清空状态、Ts=tau 特例。
```

### 15.2 检查内容

检查 1：两步递推是否严格符合公式

```text
epsilon_{k+1}
= (Ts / tau) * (target - prediction)
+ (1 - Ts / tau) * epsilon_k
```

脚本中：

```text
step 1:
expected_e1 = gamma * residual1

step 2:
expected_e2 = gamma * residual2 + (1 - gamma) * expected_e1
```

并同时检查：

```text
prediction_error = mean(dynamic_error^2)
```

检查 2：reset() 后状态是否清空

```text
pcoder.reset()
assert pcoder.dynamic_error is None
```

检查 3：Ts = tau 时是否退化成 residual

```text
如果 Ts = tau
则 gamma = 1
dynamic_error = target - prediction
prediction_error = mean((target - prediction)^2)
```

### 15.3 运行命令

```bash
/home/lin/anaconda3/envs/predifyproject/bin/python /home/lin/predify/tools/check_dynamic_error_formula.py
```

### 15.4 运行结果

```text
[ok] step1.dynamic_error
[ok] step1.prediction_error
[ok] step2.dynamic_error
[ok] step2.prediction_error
[ok] reset.dynamic_error_is_none
[ok] ts_equals_tau.dynamic_error
[ok] ts_equals_tau.prediction_error
dynamic error formula checks passed
```

### 15.5 本组结论

这组检查可以支持以下结论：

```text
1. DynamicErrorPCoderN 中的 dynamic_error 更新公式实现正确。
2. prediction_error 确实等于 mean(dynamic_error^2)。
3. reset() 会清空误差状态。
4. 我们之前关于 “Ts=tau 时历史误差权重为 0” 的解释是正确的。
```

但这组检查不能支持以下结论：

```text
1. 误差机制已经改善了中间表征。
2. 误差机制已经改善了最终任务表现。
```

因此现在可以更准确地分层表述：

```text
公式层：已经验证正确。
内部误差层：已经确认 dynamic error 改变并降低了 PCoder error。
输出行为层：目前仍未看到 clean-vs-noisy robustness 提升。
```

## 16. 实验日志：2026-05-11 KITTI 中间层 feature consistency

### 16.1 新增代码

新增文件：

```text
/home/lin/predify2021/predify2021/mce_scores/calculate_kitti_feature_consistency.py
```

用途：

```text
1. 在同一段 KITTI 时间流上分别跑 clean stream 和 noisy stream。
2. 不再只看最终 logits，而是直接比较 pcoder1~pcoder5 的中间状态。
3. 对每一层同时记录：
   - rep_cosine
   - rep_rmse
   - prd_cosine
   - prd_rmse
4. 用来判断 dynamic error 是否已经体现在中间表征的一致性上。
```

### 16.2 指标定义

对同一帧的 clean/noisy 两路模型状态，分别比较：

```text
rep_cosine:
pcoder.rep 的 cosine similarity，越高越好

rep_rmse:
pcoder.rep 的 RMSE，越低越好

prd_cosine:
pcoder.prd 的 cosine similarity，越高越好

prd_rmse:
pcoder.prd 的 RMSE，越低越好
```

如果 dynamic error 有用，那么在相同 sigma 下应该看到：

```text
1. rep_cosine / prd_cosine 上升
2. rep_rmse / prd_rmse 下降
3. 尤其在深层 pcoder3~pcoder5 上更明显
```

### 16.3 运行命令

baseline:

```bash
PREDIFY_KITTI_ROOT=/home/lin/predify/kitti_raw \
PREDIFY_KITTI_DRIVE=2011_09_26/2011_09_26_drive_0005_sync \
PREDIFY_KITTI_CAMERA=image_02 \
PREDIFY_HP_PRESET=fixed \
PREDIFY_ALPHA_VALUES=0.01 \
PREDIFY_DYNAMIC_ERROR=0 \
PREDIFY_ERROR_TS=0.1 \
PREDIFY_ERROR_TAU=0.5 \
PREDIFY_SIGMAS=0.25,0.5 \
PREDIFY_MAX_FRAMES=0 \
PYTHONPATH=/home/lin/predify2021:/home/lin/predify \
TORCH_HOME=/home/lin/predify/.torch \
/home/lin/anaconda3/envs/predifyproject/bin/python -u -m predify2021.mce_scores.calculate_kitti_feature_consistency
```

dynamic error:

```bash
PREDIFY_KITTI_ROOT=/home/lin/predify/kitti_raw \
PREDIFY_KITTI_DRIVE=2011_09_26/2011_09_26_drive_0005_sync \
PREDIFY_KITTI_CAMERA=image_02 \
PREDIFY_HP_PRESET=fixed \
PREDIFY_ALPHA_VALUES=0.01 \
PREDIFY_DYNAMIC_ERROR=1 \
PREDIFY_ERROR_TS=0.1 \
PREDIFY_ERROR_TAU=0.5 \
PREDIFY_SIGMAS=0.25,0.5 \
PREDIFY_MAX_FRAMES=0 \
PYTHONPATH=/home/lin/predify2021:/home/lin/predify \
TORCH_HOME=/home/lin/predify/.torch \
/home/lin/anaconda3/envs/predifyproject/bin/python -u -m predify2021.mce_scores.calculate_kitti_feature_consistency
```

### 16.4 结果文件

```text
/home/lin/predify/kitti_feature_consistency_pvgg_alpha_0p01.p
/home/lin/predify/kitti_feature_consistency_pvgg_alpha_0p01_dynerr.p
/home/lin/predify/kitti_feature_consistency_pvgg_comparison.p
```

### 16.5 运行结果

sigma = 0.25:

```text
baseline
pcoder1: rep_cosine=0.850589, rep_rmse=0.660522, prd_cosine=0.979545, prd_rmse=0.251827
pcoder2: rep_cosine=0.739366, rep_rmse=1.346748, prd_cosine=0.861468, prd_rmse=0.478597
pcoder3: rep_cosine=0.743433, rep_rmse=1.487771, prd_cosine=0.837236, prd_rmse=0.656382
pcoder4: rep_cosine=0.716105, rep_rmse=0.713887, prd_cosine=0.834580, prd_rmse=0.763880
pcoder5: rep_cosine=0.633233, rep_rmse=0.348028, prd_cosine=0.768536, prd_rmse=0.392822

dynamic error
pcoder1: rep_cosine=0.850583, rep_rmse=0.660521, prd_cosine=0.979485, prd_rmse=0.252064
pcoder2: rep_cosine=0.739359, rep_rmse=1.346692, prd_cosine=0.861419, prd_rmse=0.478598
pcoder3: rep_cosine=0.743422, rep_rmse=1.487611, prd_cosine=0.837253, prd_rmse=0.656038
pcoder4: rep_cosine=0.716099, rep_rmse=0.713845, prd_cosine=0.834293, prd_rmse=0.763004
pcoder5: rep_cosine=0.633202, rep_rmse=0.348331, prd_cosine=0.767813, prd_rmse=0.393657
```

sigma = 0.5:

```text
baseline
pcoder1: rep_cosine=0.694639, rep_rmse=1.163331, prd_cosine=0.944764, prd_rmse=0.427437
pcoder2: rep_cosine=0.570828, rep_rmse=2.101175, prd_cosine=0.723585, prd_rmse=0.806862
pcoder3: rep_cosine=0.571043, rep_rmse=1.918833, prd_cosine=0.713805, prd_rmse=0.938305
pcoder4: rep_cosine=0.517067, rep_rmse=0.896586, prd_cosine=0.710998, prd_rmse=0.982017
pcoder5: rep_cosine=0.382144, rep_rmse=0.443278, prd_cosine=0.585824, prd_rmse=0.510078

dynamic error
pcoder1: rep_cosine=0.694633, rep_rmse=1.163326, prd_cosine=0.944645, prd_rmse=0.427700
pcoder2: rep_cosine=0.570820, rep_rmse=2.101059, prd_cosine=0.723546, prd_rmse=0.806788
pcoder3: rep_cosine=0.571030, rep_rmse=1.918597, prd_cosine=0.713863, prd_rmse=0.937478
pcoder4: rep_cosine=0.517041, rep_rmse=0.896542, prd_cosine=0.710613, prd_rmse=0.980687
pcoder5: rep_cosine=0.381989, rep_rmse=0.443732, prd_cosine=0.584847, prd_rmse=0.511072
```

### 16.6 本组结论

这组实验说明：

```text
1. dynamic error 没有把中间层 feature consistency 明显抬高。
2. baseline 和 dynamic error 在 pcoder1~pcoder5 的 rep/prd 指标上几乎重合。
3. 这不仅是“最终 logits 没有改善”，而且连中间表征的一致性也没有明显改善。
```

更具体地说：

```text
1. sigma=0.25 时，四个指标在每一层上的差异都非常小，属于数值噪声范围。
2. sigma=0.5 时，深层一致性明显下降，但 dynamic error 也没有把这条衰减曲线抬起来。
3. 因此目前没有证据支持 “dynamic error 已经改善了中间表征鲁棒性”。
```

### 16.7 到目前为止的分层判断

```text
公式层：
已经验证正确。

内部误差层：
dynamic error 会改变并降低我们定义的内部误差能量。

中间表征层：
目前没有看到 rep/prd clean-vs-noisy consistency 改善。

输出行为层：
目前没有看到 logits/top-1 clean-vs-noisy consistency 改善。
```

### 16.8 下一步建议

现在不建议继续只围绕同一个 dynamic error 公式做小幅调参。

更合理的下一步是：

```text
1. 做 negative control：
   - reset_each_frame
   - shuffled_frames

2. 检查 dynamic error 是否只是被 alpha 梯度路径“弱耦合”到状态更新里。

3. 如果目标是时间流任务，应尽快定义真正的 KITTI 任务指标，
   而不是继续依赖 ImageNet 分类输出的代理指标。

4. 如果想继续沿这条路走，需要开始改动力学耦合位置，而不只是改误差定义。
```

## 17. 实验日志：2026-05-11 alpha 梯度路径耦合强度检查

### 17.1 目的

前面的实验已经说明：

```text
1. dynamic error 改变了内部 prediction error。
2. 但它没有明显改变中间表征一致性。
3. 也没有明显改变最终 logits / top-1 一致性。
```

因此需要直接检查：

```text
alpha * grad 这条路径在 rep 更新中到底有多大权重。
```

如果这条路径本身就非常小，那么“误差机制改了但模型行为几乎不变”就是预期结果。

### 17.2 新增代码

修改文件：

```text
/home/lin/predify/predify/modules/base.py
```

新增诊断内容：

```text
1. 在 PCoderN / DynamicErrorPCoderN 中记录每次状态更新的分量统计。
2. 使用 delta 形式拆解 rep 更新：

   rep_new = rep_old + ff_delta + fb_delta - alpha_term

   其中：
   ff_delta = beta * (ff - rep_old)
   fb_delta = lambda * (fb - rep_old)
   alpha_term = alpha * error_scale * grad

3. 记录每层每步的：
   - ff_delta_rms
   - fb_delta_rms
   - drive_delta_rms
   - alpha_delta_rms
   - total_delta_rms
   - grad_rms
   - source_prediction_error
   - alpha_to_drive_ratio
   - alpha_to_total_ratio
```

新增脚本：

```text
/home/lin/predify2021/predify2021/mce_scores/calculate_kitti_alpha_coupling.py
```

用途：

```text
1. 在 KITTI 序列上逐帧跑模型。
2. 汇总每个 PCoder 层中 alpha 项相对 drive 项和 total update 的占比。
3. 直接判断 alpha 路径对状态更新是否有实质耦合。
```

### 17.3 运行命令

baseline:

```bash
PREDIFY_KITTI_ROOT=/home/lin/predify/kitti_raw \
PREDIFY_KITTI_DRIVE=2011_09_26/2011_09_26_drive_0005_sync \
PREDIFY_KITTI_CAMERA=image_02 \
PREDIFY_HP_PRESET=fixed \
PREDIFY_ALPHA_VALUES=0.01 \
PREDIFY_DYNAMIC_ERROR=0 \
PREDIFY_ERROR_TS=0.1 \
PREDIFY_ERROR_TAU=0.5 \
PREDIFY_SIGMAS=0.5 \
PREDIFY_MAX_FRAMES=0 \
PYTHONPATH=/home/lin/predify2021:/home/lin/predify \
TORCH_HOME=/home/lin/predify/.torch \
/home/lin/anaconda3/envs/predifyproject/bin/python -u -m predify2021.mce_scores.calculate_kitti_alpha_coupling
```

dynamic error:

```bash
PREDIFY_KITTI_ROOT=/home/lin/predify/kitti_raw \
PREDIFY_KITTI_DRIVE=2011_09_26/2011_09_26_drive_0005_sync \
PREDIFY_KITTI_CAMERA=image_02 \
PREDIFY_HP_PRESET=fixed \
PREDIFY_ALPHA_VALUES=0.01 \
PREDIFY_DYNAMIC_ERROR=1 \
PREDIFY_ERROR_TS=0.1 \
PREDIFY_ERROR_TAU=0.5 \
PREDIFY_SIGMAS=0.5 \
PREDIFY_MAX_FRAMES=0 \
PYTHONPATH=/home/lin/predify2021:/home/lin/predify \
TORCH_HOME=/home/lin/predify/.torch \
/home/lin/anaconda3/envs/predifyproject/bin/python -u -m predify2021.mce_scores.calculate_kitti_alpha_coupling
```

### 17.4 结果文件

```text
/home/lin/predify/kitti_alpha_coupling_pvgg_alpha_0p01.p
/home/lin/predify/kitti_alpha_coupling_pvgg_alpha_0p01_dynerr.p
/home/lin/predify/kitti_alpha_coupling_pvgg_comparison.p
```

### 17.5 关键结果

sigma = 0.5, 154 帧：

baseline:

```text
pcoder1: alpha_rms=0.000338, drive_rms=1.600395, alpha/drive mean=0.000211
pcoder2: alpha_rms=0.000776, drive_rms=2.180159, alpha/drive mean=0.000356
pcoder3: alpha_rms=0.000950, drive_rms=1.532176, alpha/drive mean=0.000627
pcoder4: alpha_rms=0.001032, drive_rms=0.466181, alpha/drive mean=0.002216
pcoder5: alpha_rms=0.000430, drive_rms=0.155372, alpha/drive mean=0.002812
```

dynamic error:

```text
pcoder1: alpha_rms=0.000031, drive_rms=1.600402, alpha/drive mean=0.000019
pcoder2: alpha_rms=0.000068, drive_rms=2.180167, alpha/drive mean=0.000031
pcoder3: alpha_rms=0.000140, drive_rms=1.532228, alpha/drive mean=0.000092
pcoder4: alpha_rms=0.000115, drive_rms=0.466235, alpha/drive mean=0.000249
pcoder5: alpha_rms=0.000055, drive_rms=0.155479, alpha/drive mean=0.000361
```

更完整的极值信息：

```text
baseline, alpha_to_drive_ratio p95 / max
pcoder1: 0.000230 / 0.000235
pcoder2: 0.000372 / 0.000378
pcoder3: 0.000717 / 0.000741
pcoder4: 0.002574 / 0.002960
pcoder5: 0.003727 / 0.004073

dynamic error, alpha_to_drive_ratio p95 / max
pcoder1: 0.000022 / 0.000026
pcoder2: 0.000034 / 0.000035
pcoder3: 0.000108 / 0.000114
pcoder4: 0.000320 / 0.000355
pcoder5: 0.000522 / 0.000647
```

### 17.6 直接结论

这组实验给出的结论非常直接：

```text
1. alpha 梯度路径对 rep 更新的耦合极弱。
2. 即使在噪声 sigma=0.5 下，alpha 项相对 drive 项的平均占比也只有：
   - baseline: 0.0211% ~ 0.2812%
   - dynamic error: 0.0019% ~ 0.0361%
3. dynamic error 不但没有增强 alpha 路径耦合，反而进一步减弱了这条路径的量级。
```

这意味着：

```text
当前 Predify 设定下，rep 更新几乎完全由 feedforward / feedback drive 主导，
alpha * grad 只是一个非常小的微扰项。
```

因此前面看到的现象现在可以统一解释：

```text
1. 你改了 error 的定义。
2. error 的数值本身确实变了。
3. 但 error 只通过一个非常弱的 alpha*grad 路径进入状态更新。
4. 所以中间表征和最终输出几乎不变。
```

### 17.7 当前最重要的判断

到这一步，可以比较明确地说：

```text
问题不在于 dynamic error 公式没有接进去。
问题在于当前 Predify 动力学里，alpha 梯度项的耦合强度太弱，
弱到不足以显著影响状态轨迹。
```

### 17.8 后续方向

如果后面还想继续沿这条线做研究，优先级应该改成：

```text
1. 不再只改 error 定义。
2. 直接改动力学耦合位置或耦合强度。
3. 重点考虑：
   - 增大 alpha 路径作用
   - 让时间状态直接进入 rep 更新，而不是只先进入 error 再走弱梯度路径
   - 改写 forward / feedback / memory / error 的组合方式
```

## 18. 架构分支：targetflow-arch 最小整体架构骨架

### 18.1 目的

在 baseline 分支已经完成“只改误差机制”之后，单独在 `targetflow-arch` worktree 上开始“层级 3：改整体架构”。

目标不是马上复现完整新模型，而是先搭一个最小可运行骨架：

```text
1. 不污染 baseline / ablation 分支。
2. 不把新架构硬塞进旧 PCoder 语义。
3. 先有独立的 forward / target / error 状态。
4. 先保证模型可实例化、可前向、状态 shape 正确。
```

### 18.2 worktree 状态

使用分支：

```text
/home/lin/predify2021_targetflow
branch: targetflow-arch
```

开始时该分支是干净的，没有已有 target-flow 改动。

### 18.3 新增代码

新增文件：

```text
/home/lin/predify2021_targetflow/predify2021/model_factory/targetflow/__init__.py
/home/lin/predify2021_targetflow/predify2021/model_factory/targetflow/core.py
/home/lin/predify2021_targetflow/predify2021/model_factory/pvgg16_targetflow.py
```

修改文件：

```text
/home/lin/predify2021_targetflow/predify2021/model_factory/get_model.py
```

### 18.4 当前骨架内容

当前 `pvgg_tf` 做到的事情：

```text
1. 保留 VGG16 backbone 的前向识别主干。
2. 按 PVGG16 的 5 个 stage 切分 forward stages。
3. 新建独立 feedback_modules，作为 backward target flow。
4. 对每一层记录：
   - forward_input
   - forward_output
   - target_output
   - error = target_output - forward_output
5. 暴露统一入口：
   get_model('pvgg_tf')
```

这一步还没有做的内容：

```text
1. 还没有把 learn flow 加进去。
2. 还没有把新 target/error dynamics 代入状态更新。
3. 还没有把 temporal state 接进 target-flow 架构。
4. 还没有做训练/评测脚本适配。
```

### 18.5 smoke test

运行命令：

```bash
PYTHONPATH=/home/lin/predify2021_targetflow:/home/lin/predify \
TORCH_HOME=/home/lin/predify/.torch \
/home/lin/anaconda3/envs/predifyproject/bin/python - <<'PY'
import torch
from predify2021.model_factory import get_model
net = get_model('pvgg_tf', pretrained=False)
x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    y = net(x)
print('logits_shape', tuple(y.shape))
print('num_states', len(net.layer_states))
for state in net.layer_states:
    print(state.layer_index, tuple(state.forward_output.shape), tuple(state.target_output.shape), tuple(state.error.shape))
PY
```

运行结果：

```text
logits_shape (1, 1000)
num_states 5
1 (1, 64, 224, 224) (1, 64, 224, 224) (1, 64, 224, 224)
2 (1, 128, 112, 112) (1, 128, 112, 112) (1, 128, 112, 112)
3 (1, 256, 56, 56) (1, 256, 56, 56) (1, 256, 56, 56)
4 (1, 512, 28, 28) (1, 512, 28, 28) (1, 512, 28, 28)
5 (1, 512, 14, 14) (1, 512, 14, 14) (1, 512, 14, 14)
```

### 18.6 当前结论

这一步说明：

```text
1. 层级 3 的整体架构改动已经开始，并且不再依赖旧 PCoder 状态容器。
2. 新的 pvgg_tf 骨架已经可运行。
3. forward / target / error 三条流已经有独立状态表示。
4. 后面可以直接在这个骨架上继续加 dynamics / learn flow，而不必继续魔改旧 PVGG16 hooks。
```

### 18.7 target flow 正式写入

在最小骨架基础上，又把 target flow 从“占位 target tensor”升级成了正式递推。

新增内容：

```text
1. TargetFlowFeedbackModule
   - 显式区分 projector 和 activation
   - 可以分别得到：
     - target_preactivation
     - target_output

2. TargetFlowLayerState 扩展字段
   - target_source
   - target_preactivation
   - target_output
   - error
   - learn_signal（占位，供后续 learn flow 使用）

3. run_backward_target_flow(...)
   - 支持 top_target
   - 支持 mode:
     - quasi_steady
     - recursive
```

当前默认模式：

```text
target_flow_mode = quasi_steady
```

也就是：

```text
低层 target 由“上一层 forward output”通过 feedback module 投影得到，
而不是由上一层 target 继续递推得到。
```

这更接近前面讨论时的 quasi-steady target flow 版本。

### 18.8 当前状态定义

现在每层已经明确包含：

```text
forward_input
forward_output
target_source
target_preactivation
target_output
error = target_output - forward_output
```

top layer 的当前处理方式：

```text
如果没有外部 top_target
则使用 top layer 自己的 forward_output 作为 target_output
```

这意味着：

```text
当前 top layer error 仍然是占位式的；
真正的监督 target / task target 还没有接入。
```

### 18.9 预训练反馈模块兼容

由于 target-flow 现在把反馈模块拆成了：

```text
projector + activation
```

因此额外修改了 `get_model.py` 中的反馈权重加载逻辑：

```text
旧 PCoder checkpoint 的 pmodule 权重，
现在会加载到新 target-flow feedback module 的 projector 上。
```

这保证现有 `weights_pvgg16_imagenet` 仍然可用于 `pvgg_tf`。

### 18.10 smoke tests

测试 1：不加载反馈权重

```text
plain_logits_shape (1, 1000)
plain_num_states 5
plain 1 (1, 64, 224, 224) (1, 64, 224, 224) (1, 64, 224, 224) (1, 64, 224, 224)
plain 2 (1, 128, 112, 112) (1, 128, 112, 112) (1, 128, 112, 112) (1, 128, 112, 112)
plain 3 (1, 256, 56, 56) (1, 256, 56, 56) (1, 256, 56, 56) (1, 256, 56, 56)
plain 4 (1, 512, 28, 28) (1, 512, 28, 28) (1, 512, 28, 28) (1, 512, 28, 28)
plain 5 (1, 512, 14, 14) (1, 512, 14, 14) (1, 512, 14, 14) (1, 512, 14, 14)
```

测试 2：加载现有反馈权重

```text
pretrained_logits_shape (1, 1000)
pretrained_num_states 5
pretrained 1 (1, 64, 224, 224) (1, 64, 224, 224) (1, 64, 224, 224) (1, 64, 224, 224)
pretrained 2 (1, 128, 112, 112) (1, 128, 112, 112) (1, 128, 112, 112) (1, 128, 112, 112)
pretrained 3 (1, 256, 56, 56) (1, 256, 56, 56) (1, 256, 56, 56) (1, 256, 56, 56)
pretrained 4 (1, 512, 28, 28) (1, 512, 28, 28) (1, 512, 28, 28) (1, 512, 28, 28)
pretrained 5 (1, 512, 14, 14) (1, 512, 14, 14) (1, 512, 14, 14) (1, 512, 14, 14)
```

### 18.11 这一步的意义

到这里可以明确说：

```text
1. pvgg_tf 不再只是“有个 target 张量”的占位版。
2. 它已经有正式的 backward target flow 状态和递推。
3. 现有反馈权重仍然可以被复用。
4. 下一步继续做 learn flow 时，就不需要再回头重写 target state 结构。
```

### 18.12 learn flow 已接入

在正式 target flow 基础上，继续把第四条流补进了 `pvgg_tf`。

本次新增内容：

```text
1. 每层新增 learn flow 状态：
   - learn_signal
   - local_loss
   - parameter_grad_stats

2. learn_signal 定义：
   learn_signal = target_output.detach() - forward_output

3. local_loss 定义：
   local_loss = mean((target_output.detach() - forward_output)^2)

4. 可选的局部参数梯度统计：
   如果 compute_local_param_grads=True 且开启 autograd，
   则对每个 forward stage 计算：
   grad(local_loss, stage.parameters())
   并汇总：
   - rms
   - mean_abs
   - max_abs
   - num_parameters

5. 新增接口：
   net.collect_learn_flow_losses()
   返回：
   - per_layer local losses
   - total local loss
```

### 18.13 代码改动

修改文件：

```text
/home/lin/predify2021_targetflow/predify2021/model_factory/targetflow/core.py
/home/lin/predify2021_targetflow/predify2021/model_factory/targetflow/__init__.py
/home/lin/predify2021_targetflow/predify2021/model_factory/pvgg16_targetflow.py
/home/lin/predify2021_targetflow/predify2021/model_factory/get_model.py
```

其中 `get_model()` 现在支持：

```text
target_flow_mode="quasi_steady"
compute_local_param_grads=False
```

### 18.14 smoke tests

测试 1：无梯度前向，检查 learn_signal / local_loss 存在

```text
nograd_logits_shape (1, 1000)
nograd 1 (1, 64, 224, 224) 2.7646117210388184 None
nograd 2 (1, 128, 112, 112) 11.991255760192871 None
nograd 3 (1, 256, 56, 56) 4.744359970092773 None
nograd 4 (1, 512, 28, 28) 0.4205807149410248 None
nograd 5 (1, 512, 14, 14) 0.0 None
```

含义：

```text
1. 5 层都有 learn_signal 和 local_loss。
2. 因为没开 local param grads，所以 parameter_grad_stats 为 None。
3. 第 5 层 local_loss = 0 是因为当前 top target 默认就是 top forward output。
```

测试 2：开启局部参数梯度统计

```text
grad_logits_shape (1, 1000)
num_local_losses 5 total_loss 19.630788803100586
grad 1 (1, 64, 224, 224) 2.7096803188323975 {'rms': 0.00970701314508915, 'mean_abs': 0.005206055007874966, 'max_abs': 0.14341996610164642, 'num_parameters': 38720}
grad 2 (1, 128, 112, 112) 11.848226547241211 {'rms': 0.09885997325181961, 'mean_abs': 0.04258139804005623, 'max_abs': 1.1360586881637573, 'num_parameters': 221440}
grad 3 (1, 256, 56, 56) 4.692220211029053 {'rms': 0.04767125844955444, 'mean_abs': 0.014303234405815601, 'max_abs': 2.140191078186035, 'num_parameters': 1475328}
grad 4 (1, 512, 28, 28) 0.3806605041027069 {'rms': 0.0019019186729565263, 'mean_abs': 0.0004195749934297055, 'max_abs': 0.15110920369625092, 'num_parameters': 5899776}
grad 5 (1, 512, 14, 14) 0.0 {'rms': 0.0, 'mean_abs': 0.0, 'max_abs': 0.0, 'num_parameters': 7079424}
```

含义：

```text
1. learn flow 已经不只是状态占位，而是能生成局部训练信号。
2. 前 4 层都已有非零 local loss 和非零参数梯度统计。
3. 第 5 层仍然是零，因为当前没有把外部任务 target 接入 top layer。
```

### 18.15 当前结论

到这里可以明确说：

```text
1. forward flow：有
2. target flow：有
3. error flow：有
4. learn flow：有
```

但同时也要清楚当前边界：

```text
1. learn flow 目前是“局部学习信号与局部梯度统计”层面，
   还没有接入完整训练循环。

2. top layer 还没有外部监督 target，
   所以第 5 层 learn flow 当前为零。

3. 如果下一步要进入真正训练，
   就要决定 top target 来自：
   - 分类标签 / classifier loss
   - 外部 task target
   - 或其他自定义 top target 规则
```


### 18.16 时间型 top target：x_t 逼近 x_{t+1} 的顶层表示

在确认研究方向是“时序 + 误差累计”之后，`pvgg_tf` 进一步改成显式双帧接口：

```text
当前帧 x_t：正常 forward / target / error / learn
下一帧 x_{t+1}：只提取顶层 forward feature
顶层 target：top_target_t = stopgrad(h_top(x_{t+1}))
```

这一步的含义是：

```text
模型在时刻 t 的顶层表示
去逼近时刻 t+1 的顶层表示
```

这比“分类标签作为 top target”更符合当前 KITTI Raw + 时序误差累计方向。

### 18.17 代码改动

文件：

```text
/home/lin/predify2021_targetflow/predify2021/model_factory/pvgg16_targetflow.py
```

新增能力：

```text
1. _run_forward_stages(x)
   - 统一抽取 5 个 stage 的 forward input / output。

2. extract_top_forward_feature(x, detach=True)
   - 单独提取任意输入帧的顶层前向表示。

3. _resolve_top_target(top_target=None, next_x=None)
   - 如果给定 next_x，则自动构造：
     top_target = stopgrad(h_top(next_x))

4. forward(x, top_target=None, next_x=None)
   - 普通入口现在支持显式 next_x。

5. forward_with_next_target(x, next_x)
   - 明确表达“x_t 用 x_{t+1} 做 top target”。

6. step_pair(x, next_x)
   - stream / 时序评测时的双帧入口。
```

### 18.18 这一步解决了什么

之前 top layer 的默认逻辑是：

```text
top_target = top_forward_output
```

所以第 5 层：

```text
error = 0
local_loss = 0
learn_signal = 0
```

现在如果传入 `next_x`，则变成：

```text
top_target_t = stopgrad(h_top(x_{t+1}))
```

于是第 5 层的 target / error / learn flow 都会变成非零，这样顶层也真正进入了时序学习闭环。


### 18.19 smoke test

命令：

```bash
PYTHONPATH=/home/lin/predify2021_targetflow:/home/lin/predify TORCH_HOME=/home/lin/predify/.torch /home/lin/anaconda3/envs/predifyproject/bin/python
```

测试逻辑：

```text
1. 随机生成 x_t 和 x_{t+1}
2. 调用 net.forward_with_next_target(x_t, x_{t+1})
3. 检查 top layer 的 local_loss / error / parameter_grad_stats 是否非零
```

结果：

```text
logits_shape   = (1, 1000)
num_states     = 5
top_local_loss = 0.033918
top_error_rms  = 0.184168
top_grad_stats = {
  rms: 9.264e-05,
  mean_abs: 2.096e-05,
  max_abs: 6.003e-03,
  num_parameters: 7079424,
}
total_local_loss = 34.967381
```

解释：

```text
1. 双帧接口已经跑通。
2. 第 5 层不再是 local_loss = 0。
3. 第 5 层不再是 parameter_grad_stats = 0。
4. 说明 top target = h_top(x_{t+1}) 已经真实进入 top layer 的 learn flow。
```

### 18.20 时间对齐方式：KITTI 相邻帧配对

时序目标现在明确固定为：

```text
(x_t, x_{t+1})
```

也就是 KITTI drive 中的相邻帧配对，而不是跨多步的：

```text
(x_t, x_{t+2}), (x_t, x_{t+3}), ...
```

当前定义：

```text
sample 0: (frame_0000000000, frame_0000000001)
sample 1: (frame_0000000001, frame_0000000002)
...
sample N: (frame_n, frame_{n+1})
```

这使得“顶层表示逼近下一时刻顶层表示”的定义和数据入口保持一致。

### 18.21 新增 pair dataset / smoke script

新增文件：

```text
/home/lin/predify2021_targetflow/predify2021/mce_scores/kitti_pairs.py
/home/lin/predify2021_targetflow/predify2021/mce_scores/calculate_kitti_targetflow_pair_smoke.py
```

作用：

```text
1. KITTINextFramePairDataset
   - 从同一个 KITTI Raw 序列构造相邻帧 pair。

2. build_kitti_pair_dataloader(...)
   - 统一构造 pair dataloader。

3. calculate_kitti_targetflow_pair_smoke.py
   - 直接跑 pvgg_tf(x_t, x_{t+1})
   - 检查 top layer local loss / error / grad 是否非零
   - 作为后续训练前的最小入口
```

### 18.22 get_model('pvgg_tf', pretrained=False) 语义修正

之前 `targetflow-arch` 中存在一个不一致点：

```text
get_model('pvgg_tf', pretrained=False)
```

仍然会构造 torchvision 预训练 VGG16。

现在已经修正为：

```text
pretrained=False -> backbone weights = None
pretrained=True  -> backbone weights = VGG16_Weights.IMAGENET1K_V1
```

也就是说，`pvgg_tf` 的 backbone 初始化现在和 `pretrained` 参数字面含义一致。

### 18.23 下一步：相邻帧时序训练入口

在相邻帧 pair 数据入口稳定之后，下一步不再是评测脚本，而是最小训练循环。

新增文件：

```text
/home/lin/predify2021_targetflow/predify2021/mce_scores/train_kitti_targetflow_adjacent_pairs.py
```

第一版训练规则：

```text
1. 数据：KITTI 相邻帧 pair (x_t, x_{t+1})
2. top target：EMA teacher 的 h_top(x_{t+1})
3. student：用 x_t 前向，top_target = stopgrad(teacher_top_feature)
4. loss：5 层 local_loss 加权求和
5. optimizer：只更新 student.forward_stages
6. feedback_modules：当前不更新
```

默认层权重：

```text
(0.1, 0.1, 0.2, 0.2, 1.0)
```

目的：

```text
1. 先验证 loss 能否下降
2. 先监控 top layer 是否真的参与学习
3. 先检查 top feature std 是否塌缩
```

训练和验证划分：

```text
同一条 KITTI drive 按时间顺序切分：
前 TRAIN_FRACTION 部分做 train
后半段做 val
```

### 18.24 训练 smoke test

命令（最小版本）：

```bash
PREDIFY_KITTI_ROOT=/home/lin/predify/kitti_raw \
PREDIFY_KITTI_DRIVE=2011_09_26/2011_09_26_drive_0005_sync \
PREDIFY_KITTI_CAMERA=image_02 \
PREDIFY_MAX_PAIRS=8 \
PREDIFY_BATCHSIZE=1 \
PREDIFY_NUM_WORKERS=0 \
PREDIFY_EPOCHS=1 \
PREDIFY_PRETRAINED=0 \
PREDIFY_LR=1e-4 \
PREDIFY_EMA_DECAY=0.99 \
PYTHONPATH=/home/lin/predify2021_targetflow:/home/lin/predify \
TORCH_HOME=/home/lin/predify/.torch \
python -u -m predify2021.mce_scores.train_kitti_targetflow_adjacent_pairs
```

结果：

```text
train_pairs = 6
val_pairs   = 2

Epoch 001
train_weighted = 0.005316
train_top      = 0.000902
train_top_std  = 0.043719

val_weighted   = 0.003535
val_top        = 0.000514
val_top_std    = 0.041895
```

输出文件：

```text
kitti_targetflow_adjacent_pairs_train.p
```

### 18.25 这一步说明什么

```text
1. 相邻帧 pair dataset 能直接进入训练循环。
2. EMA teacher 路径是通的。
3. top target = h_top(x_{t+1}) 能参与反向传播到 student.forward_stages。
4. 训练和验证的 weighted local loss 都能正常计算并记录。
5. top feature std 目前非零，至少在这个 smoke test 里没有立刻塌缩成常数特征。
```

### 18.26 正式训练结果：pretrained student + feedback weights，10 epochs

命令：

```bash
PREDIFY_KITTI_ROOT=/home/lin/predify/kitti_raw \
PREDIFY_KITTI_DRIVE=2011_09_26/2011_09_26_drive_0005_sync \
PREDIFY_KITTI_CAMERA=image_02 \
PREDIFY_MAX_PAIRS=0 \
PREDIFY_BATCHSIZE=4 \
PREDIFY_NUM_WORKERS=4 \
PREDIFY_EPOCHS=10 \
PREDIFY_PRETRAINED=1 \
PREDIFY_PCODER_WEIGHTS=/home/lin/predify/weights_pvgg16_imagenet \
PREDIFY_LR=1e-4 \
PREDIFY_WEIGHT_DECAY=0 \
PREDIFY_EMA_DECAY=0.99 \
PREDIFY_TRAIN_FRACTION=0.8 \
PREDIFY_LAYER_LOSS_WEIGHTS=0.1,0.1,0.2,0.2,1.0 \
PREDIFY_OUTPUT_PATH=kitti_targetflow_adjacent_pairs_train_pretrained_e10.p \
PYTHONPATH=/home/lin/predify2021_targetflow:/home/lin/predify \
TORCH_HOME=/home/lin/predify/.torch \
python -u -m predify2021.mce_scores.train_kitti_targetflow_adjacent_pairs
```

数据规模：

```text
total_pairs = 153
train_pairs = 122
val_pairs   = 31
device      = cuda:0
```

首尾结果对比：

```text
Epoch 001
train_weighted = 0.697198
train_top      = 0.171199
train_top_std  = 0.371257
val_weighted   = 0.327252
val_top        = 0.124603
val_top_std    = 0.401672

Epoch 010
train_weighted = 0.171315
train_top      = 0.016656
train_top_std  = 0.682195
val_weighted   = 0.148445
val_top        = 0.023084
val_top_std    = 0.523367
```

最终 epoch 详细统计：

```text
train:
mean_weighted_loss    = 0.171315
mean_total_local_loss = 1.070750
mean_top_local_loss   = 0.016656
mean_top_feature_std  = 0.682195
mean_per_layer_local_loss =
  (0.413832, 0.147764, 0.277871, 0.214627, 0.016656)

val:
mean_weighted_loss    = 0.148445
mean_total_local_loss = 0.844350
mean_top_local_loss   = 0.023084
mean_top_feature_std  = 0.523367
mean_per_layer_local_loss =
  (0.259897, 0.129018, 0.263684, 0.168667, 0.023084)
```

输出文件：

```text
/home/lin/predify/kitti_targetflow_adjacent_pairs_train_pretrained_e10.p
```

### 18.27 这一步的解读

```text
1. 训练是有效的。
   train_weighted 和 val_weighted 都持续下降。

2. 顶层时序目标也在起作用。
   train_top 从 0.171199 降到 0.016656，
   val_top 从 0.124603 降到 0.023084。

3. 目前没有看到明显塌缩。
   top_feature_std 没有掉到 0，反而维持在较高非零水平。

4. 这说明：
   “x_t 的顶层表示去逼近 x_{t+1} 的顶层表示”
   这个训练目标已经可以在当前 targetflow-arch 上稳定优化。

5. 但这还只是训练信号层面的成功，
   还没有证明这种时序目标带来了更好的下游行为或更强鲁棒性。
```

### 18.28 第二条 KITTI drive：cross-drive validation 数据

新增下载并解压：

```text
/home/lin/predify/kitti_raw/2011_09_26/2011_09_26_drive_0011_sync
```

关键数据：

```text
drive: 2011_09_26_drive_0011_sync
camera: image_02
frames: 233
range: 0000000000.png -> 0000000232.png
size: 约 900M
zip: /home/lin/predify/kitti_raw/2011_09_26_drive_0011_sync.zip
```

到这里有两条可用序列：

```text
train candidate: 2011_09_26_drive_0005_sync (154 frames)
val candidate:   2011_09_26_drive_0011_sync (233 frames)
```

### 18.29 训练脚本支持 cross-drive

文件：

```text
/home/lin/predify2021_targetflow/predify2021/mce_scores/train_kitti_targetflow_adjacent_pairs.py
/home/lin/predify2021_targetflow/predify2021/mce_scores/kitti_pairs.py
```

新增能力：

```text
1. PREDIFY_TRAIN_DRIVES
   - 逗号分隔的训练 drive 列表

2. PREDIFY_VAL_DRIVES
   - 逗号分隔的验证 drive 列表

3. PREDIFY_MAX_TRAIN_PAIRS
   - 训练集 pair 上限

4. PREDIFY_MAX_VAL_PAIRS
   - 验证集 pair 上限

5. 多 drive 数据通过 ConcatDataset 拼接
```

如果不设置 `PREDIFY_VAL_DRIVES`，脚本仍保持旧行为：

```text
同一条 drive 内部按 TRAIN_FRACTION 切分 train / val
```

如果设置了 `PREDIFY_VAL_DRIVES`，脚本会进入真正的 cross-drive 模式：

```text
train = TRAIN_DRIVES
val   = VAL_DRIVES
```

### 18.30 cross-drive smoke test

命令：

```bash
PREDIFY_KITTI_ROOT=/home/lin/predify/kitti_raw \
PREDIFY_TRAIN_DRIVES=2011_09_26/2011_09_26_drive_0005_sync \
PREDIFY_VAL_DRIVES=2011_09_26/2011_09_26_drive_0011_sync \
PREDIFY_KITTI_CAMERA=image_02 \
PREDIFY_MAX_TRAIN_PAIRS=8 \
PREDIFY_MAX_VAL_PAIRS=8 \
PREDIFY_BATCHSIZE=1 \
PREDIFY_NUM_WORKERS=0 \
PREDIFY_EPOCHS=1 \
PREDIFY_PRETRAINED=0 \
PREDIFY_LR=1e-4 \
PREDIFY_EMA_DECAY=0.99 \
PREDIFY_OUTPUT_PATH=kitti_targetflow_cross_drive_smoke.p \
PYTHONPATH=/home/lin/predify2021_targetflow:/home/lin/predify \
TORCH_HOME=/home/lin/predify/.torch \
python -u -m predify2021.mce_scores.train_kitti_targetflow_adjacent_pairs
```

结果：

```text
train_drives = (2011_09_26/2011_09_26_drive_0005_sync,)
val_drives   = (2011_09_26/2011_09_26_drive_0011_sync,)
train_pairs  = 8
val_pairs    = 8

Epoch 001
train_weighted = 0.004391
train_top      = 0.000713
train_top_std  = 0.056393
val_weighted   = 0.009132
val_top        = 0.003997
val_top_std    = 0.143853
```

输出文件：

```text
/home/lin/predify/kitti_targetflow_cross_drive_smoke.p
```

### 18.31 这一步的意义

```text
1. 训练入口已经不再局限于“单条 drive 内部切分”。
2. 现在可以直接做：
   train = 0005
   val   = 0011

3. 这类验证比同一条序列的前后切分更严格，
   因为验证场景已经换了。

4. 如果 cross-drive 下 loss 仍然稳定下降，
   才更能说明模型学到的是可迁移的时序规律，而不是只记住一条视频。
```

### 18.32 正式 cross-drive 结果：train=0005, val=0011, pretrained, 10 epochs

命令：

```bash
PREDIFY_KITTI_ROOT=/home/lin/predify/kitti_raw \
PREDIFY_TRAIN_DRIVES=2011_09_26/2011_09_26_drive_0005_sync \
PREDIFY_VAL_DRIVES=2011_09_26/2011_09_26_drive_0011_sync \
PREDIFY_KITTI_CAMERA=image_02 \
PREDIFY_MAX_TRAIN_PAIRS=0 \
PREDIFY_MAX_VAL_PAIRS=0 \
PREDIFY_BATCHSIZE=4 \
PREDIFY_NUM_WORKERS=4 \
PREDIFY_EPOCHS=10 \
PREDIFY_PRETRAINED=1 \
PREDIFY_PCODER_WEIGHTS=/home/lin/predify/weights_pvgg16_imagenet \
PREDIFY_LR=1e-4 \
PREDIFY_WEIGHT_DECAY=0 \
PREDIFY_EMA_DECAY=0.99 \
PREDIFY_LAYER_LOSS_WEIGHTS=0.1,0.1,0.2,0.2,1.0 \
PREDIFY_OUTPUT_PATH=kitti_targetflow_cross_drive_pretrained_e10.p \
PYTHONPATH=/home/lin/predify2021_targetflow:/home/lin/predify \
TORCH_HOME=/home/lin/predify/.torch \
python -u -m predify2021.mce_scores.train_kitti_targetflow_adjacent_pairs
```

数据规模：

```text
train_drives = (2011_09_26/2011_09_26_drive_0005_sync,)
val_drives   = (2011_09_26/2011_09_26_drive_0011_sync,)
train_pairs  = 153
val_pairs    = 232
device       = cuda:0
```

首尾结果对比：

```text
Epoch 001
train_weighted = 0.611331
train_top      = 0.155465
train_top_std  = 0.361878
val_weighted   = 0.310316
val_top        = 0.092194
val_top_std    = 0.353723

Epoch 010
train_weighted = 0.136855
train_top      = 0.014199
train_top_std  = 0.625573
val_weighted   = 0.123090
val_top        = 0.015779
val_top_std    = 0.420760
```

输出文件：

```text
/home/lin/predify/kitti_targetflow_cross_drive_pretrained_e10.p
```

### 18.33 cross-drive 结果解读

```text
1. 这次结果比“同一条 drive 内部切分”更有说服力，
   因为验证集已经换成另一条独立序列。

2. train_weighted 和 val_weighted 都持续下降，
   说明时序目标在 cross-drive 条件下仍然可优化。

3. train_top 和 val_top 都持续下降，
   说明“h_top(x_t) 逼近 h_top(x_{t+1})”这个顶层时序目标
   没有只局限在训练 drive 上成立。

4. top_feature_std 始终保持非零，
   当前没有看到明显的表示塌缩。

5. 这说明：
   当前 targetflow-arch 至少已经学到了一部分可迁移的短时序规律，
   而不只是记住单条视频。

6. 但这里仍然要保持边界意识：
   这证明的是“跨 drive 的时序训练目标有效”，
   还没有证明它能改善具体下游任务或视觉鲁棒性。
```

### 18.34 EMA teacher ablation 支持

文件：

```text
/home/lin/predify2021_targetflow/predify2021/mce_scores/train_kitti_targetflow_adjacent_pairs.py
```

新增参数：

```text
PREDIFY_TOP_TARGET_SOURCE
```

支持两种模式：

```text
1. ema_teacher
   top_target = stopgrad(EMA_teacher(h_top(x_{t+1})))

2. student_self
   top_target = stopgrad(student(h_top(x_{t+1})))
```

这样可以做最直接的 ablation：

```text
只改 top target 的来源
其余数据、网络、超参数保持一致
```

目的：

```text
判断 EMA teacher 是否真的提供了更稳定、更可迁移的时序训练信号。
```

### 18.35 student_self cross-drive smoke test

命令：

```bash
PREDIFY_KITTI_ROOT=/home/lin/predify/kitti_raw \
PREDIFY_TRAIN_DRIVES=2011_09_26/2011_09_26_drive_0005_sync \
PREDIFY_VAL_DRIVES=2011_09_26/2011_09_26_drive_0011_sync \
PREDIFY_KITTI_CAMERA=image_02 \
PREDIFY_MAX_TRAIN_PAIRS=8 \
PREDIFY_MAX_VAL_PAIRS=8 \
PREDIFY_BATCHSIZE=1 \
PREDIFY_NUM_WORKERS=0 \
PREDIFY_EPOCHS=1 \
PREDIFY_PRETRAINED=0 \
PREDIFY_LR=1e-4 \
PREDIFY_EMA_DECAY=0.99 \
PREDIFY_TOP_TARGET_SOURCE=student_self \
PREDIFY_OUTPUT_PATH=kitti_targetflow_cross_drive_student_self_smoke.p \
PYTHONPATH=/home/lin/predify2021_targetflow:/home/lin/predify \
TORCH_HOME=/home/lin/predify/.torch \
python -u -m predify2021.mce_scores.train_kitti_targetflow_adjacent_pairs
```

结果：

```text
train_pairs = 8
val_pairs   = 8

Epoch 001
train_weighted = 0.009310
train_top      = 0.000073
train_top_std  = 0.035112
val_weighted   = 0.008206
val_top        = 0.000010
val_top_std    = 0.008820
```

输出文件：

```text
/home/lin/predify/kitti_targetflow_cross_drive_student_self_smoke.p
```

### 18.36 下一步的正式 ablation

为了和已经完成的 `ema_teacher` 正式实验严格可比，下一条要跑的是：

```text
只把：
PREDIFY_TOP_TARGET_SOURCE=ema_teacher

改成：
PREDIFY_TOP_TARGET_SOURCE=student_self

其余超参数保持完全一致
```

也就是：

```text
train = 0005
val   = 0011
pretrained = 1
batchsize  = 4
epochs     = 10
lr         = 1e-4
ema_decay  = 0.99
layer_loss_weights = (0.1, 0.1, 0.2, 0.2, 1.0)
```

### 18.37 正式 ablation 结果：student_self，cross-drive，pretrained，10 epochs

命令：

```bash
PREDIFY_KITTI_ROOT=/home/lin/predify/kitti_raw \
PREDIFY_TRAIN_DRIVES=2011_09_26/2011_09_26_drive_0005_sync \
PREDIFY_VAL_DRIVES=2011_09_26/2011_09_26_drive_0011_sync \
PREDIFY_KITTI_CAMERA=image_02 \
PREDIFY_MAX_TRAIN_PAIRS=0 \
PREDIFY_MAX_VAL_PAIRS=0 \
PREDIFY_BATCHSIZE=4 \
PREDIFY_NUM_WORKERS=4 \
PREDIFY_EPOCHS=10 \
PREDIFY_PRETRAINED=1 \
PREDIFY_PCODER_WEIGHTS=/home/lin/predify/weights_pvgg16_imagenet \
PREDIFY_LR=1e-4 \
PREDIFY_WEIGHT_DECAY=0 \
PREDIFY_EMA_DECAY=0.99 \
PREDIFY_TOP_TARGET_SOURCE=student_self \
PREDIFY_LAYER_LOSS_WEIGHTS=0.1,0.1,0.2,0.2,1.0 \
PREDIFY_OUTPUT_PATH=kitti_targetflow_cross_drive_student_self_pretrained_e10.p \
PYTHONPATH=/home/lin/predify2021_targetflow:/home/lin/predify \
TORCH_HOME=/home/lin/predify/.torch \
python -u -m predify2021.mce_scores.train_kitti_targetflow_adjacent_pairs
```

首尾结果对比：

```text
Epoch 001
train_weighted = 0.431126
train_top      = 0.006036
train_top_std  = 0.091290
val_weighted   = 0.176035
val_top        = 0.000020
val_top_std    = 0.055051

Epoch 010
train_weighted = 0.044652
train_top      = 0.000001
train_top_std  = 0.060131
val_weighted   = 0.045861
val_top        = 0.000001
val_top_std    = 0.055634
```

最终 epoch 详细统计：

```text
train:
mean_weighted_loss    = 0.044652
mean_total_local_loss = 0.402072
mean_top_local_loss   = 0.00000108
mean_top_feature_std  = 0.060131
mean_per_layer_local_loss =
  (0.330452, 0.027179, 0.035934, 0.008506, 0.000001)

val:
mean_weighted_loss    = 0.045861
mean_total_local_loss = 0.416725
mean_top_local_loss   = 0.00000084
mean_top_feature_std  = 0.055634
mean_per_layer_local_loss =
  (0.339218, 0.035626, 0.034239, 0.007641, 0.000001)
```

输出文件：

```text
/home/lin/predify/kitti_targetflow_cross_drive_student_self_pretrained_e10.p
```

### 18.38 ema_teacher vs student_self：正式对比结论

最终 epoch 对比：

```text
ema_teacher:
train_weighted   = 0.145713
val_weighted     = 0.130330
train_top        = 0.015692
val_top          = 0.015379
train_top_std    = 0.650256
val_top_std      = 0.445971

student_self:
train_weighted   = 0.044652
val_weighted     = 0.045861
train_top        = 0.000001
val_top          = 0.000001
train_top_std    = 0.060131
val_top_std      = 0.055634
```

正确解读：

```text
1. student_self 的 loss 更低，但这不自动代表更好。

2. 关键问题在于 top_feature_std：
   ema_teacher 维持在 0.65 / 0.45 左右，
   student_self 只有 0.06 / 0.056 左右。

3. 这说明 student_self 更倾向于把顶层表示压到很小、很容易彼此接近的区域，
   从而把 top loss 降得几乎为 0。

4. 换句话说：
   student_self 更像是在追求“自洽地变小”，
   ema_teacher 更像是在保持非平凡表示的同时做时序对齐。

5. 因此从方法角度看，
   EMA teacher 是更可信的 top target 来源。

6. 现在最合理的结论不是
   “student_self 更好”，
   而是
   “student_self 更容易得到近乎零的 top loss，但伴随明显更低的特征方差；
    EMA teacher 提供了更强、更非塌缩的训练信号。”
```

### 18.39 新增：训练脚本自动保存 student / teacher checkpoint

文件：

```text
/home/lin/predify2021_targetflow/predify2021/mce_scores/train_kitti_targetflow_adjacent_pairs.py
```

新增行为：

```text
1. 训练 history 仍然保存到 PREDIFY_OUTPUT_PATH 指定的 .p 文件。
2. 训练结束后，脚本会自动再保存：
   - <output_stem>_student.pt
   - <output_stem>_teacher.pt   （仅当 top_target_source=ema_teacher 时）
3. checkpoint 内容包括：
   - state_dict
   - config
   - final_epoch
```

默认路径示例：

```text
PREDIFY_OUTPUT_PATH=kitti_targetflow_cross_drive_pretrained_e10.p
-> /home/lin/predify/kitti_targetflow_cross_drive_pretrained_e10_student.pt
-> /home/lin/predify/kitti_targetflow_cross_drive_pretrained_e10_teacher.pt

PREDIFY_OUTPUT_PATH=kitti_targetflow_cross_drive_student_self_pretrained_e10.p
-> /home/lin/predify/kitti_targetflow_cross_drive_student_self_pretrained_e10_student.pt
```

注意：

```text
这个自动保存逻辑是在后续加入的。
早于这次改动的历史训练结果只有 .p，没有 .pt checkpoint。
如果需要对旧实验做 frozen-feature probe，需要重新跑一遍对应训练命令。
```

### 18.40 新增：KITTI ego-motion pair 数据集

文件：

```text
/home/lin/predify2021_targetflow/predify2021/mce_scores/kitti_pairs.py
```

新增类：

```text
KITTIEgoMotionPairDataset
```

样本定义：

```text
sample i = (frame_i, frame_{i+1}, target_i)
```

其中：

```text
target_i = [delta_forward_m, delta_yaw_rad]
```

具体计算方式：

```text
delta_time_s   = t_{i+1} - t_i
mean_forward_v = 0.5 * (vf_i + vf_{i+1})
delta_forward_m = mean_forward_v * delta_time_s
delta_yaw_rad   = wrap(yaw_{i+1} - yaw_i)
```

说明：

```text
1. OXTS 时间戳使用 timestamps.txt。
2. 由于 KITTI 时间戳包含纳秒，小数部分在读取时截断/补零到 6 位微秒，便于 datetime 解析。
3. yaw 使用 wrap_angle 规范到 [-pi, pi)。
```

### 18.41 新增：冻结特征的 ego-motion linear probe

文件：

```text
/home/lin/predify2021_targetflow/predify2021/mce_scores/calculate_kitti_ego_motion_linear_probe.py
```

目标：

```text
冻结已经训练好的 pvgg_tf 特征，
只训练一个 closed-form ridge probe，
检验顶层时序特征是否包含可线性读取的自车运动信息。
```

当前 probe 设置：

```text
特征：
  current_top = h_top(x_t)
  next_top    = h_top(x_{t+1})
  z           = mean_HW(next_top) - mean_HW(current_top)

目标：
  y = [delta_forward_m, delta_yaw_rad]

回归器：
  ridge regression
  W = (X^T X + lambda I)^(-1) X^T Y
```

评估指标：

```text
mae_delta_forward_m
rmse_delta_forward_m
mae_delta_yaw_rad
rmse_delta_yaw_rad
```

### 18.42 ego-motion probe：CPU smoke test 结果

使用：

```text
train drive = 0005
val drive   = 0011
max_train_pairs = 4
max_val_pairs   = 4
checkpoint = /tmp/kitti_probe_train_smoke_student.pt
model_label = smoke_student
```

结果：

```text
Train pairs: 4
Val pairs:   4
Feature dim: 512

Train metrics:
mae_delta_forward_m  = 3.343843e-08
mae_delta_yaw_rad    = 2.584227e-09
rmse_delta_forward_m = 3.737655e-08
rmse_delta_yaw_rad   = 3.138859e-09

Val metrics:
mae_delta_forward_m  = 0.855428
mae_delta_yaw_rad    = 0.018156
rmse_delta_forward_m = 0.856990
rmse_delta_yaw_rad   = 0.018163
```

解读：

```text
1. 这只是一个 4/4 pair 的 CPU smoke test，只用于确认数据读取、特征提取、ridge 拟合和指标输出全链路正常。
2. 这个结果不能拿来做方法结论。
3. 正式结论必须使用重新保存出的 ema_teacher / student_self checkpoint，再做完整 cross-drive probe。
```

### 18.43 正式 ego-motion probe 的执行前提

当前状态：

```text
/home/lin/predify/kitti_targetflow_cross_drive_pretrained_e10_student.pt
/home/lin/predify/kitti_targetflow_cross_drive_pretrained_e10_teacher.pt
/home/lin/predify/kitti_targetflow_cross_drive_student_self_pretrained_e10_student.pt
```

在本次检查时仍不存在。

原因：

```text
这些正式训练是在自动保存 checkpoint 逻辑加入之前完成的，
因此只有 .p history，没有 .pt model checkpoint。
```

因此正式 probe 的下一步是：

```text
1. 重新跑一遍 ema_teacher 的正式 cross-drive 训练；
2. 重新跑一遍 student_self 的正式 cross-drive 训练；
3. 让新脚本自动产出对应的 _student.pt / _teacher.pt；
4. 再用 frozen-feature ego-motion linear probe 做正式对比。
```

### 18.44 正式 rerun：ema_teacher，cross-drive，已产出 checkpoint

命令：

```bash
PREDIFY_KITTI_ROOT=/home/lin/predify/kitti_raw \
PREDIFY_TRAIN_DRIVES=2011_09_26/2011_09_26_drive_0005_sync \
PREDIFY_VAL_DRIVES=2011_09_26/2011_09_26_drive_0011_sync \
PREDIFY_KITTI_CAMERA=image_02 \
PREDIFY_MAX_TRAIN_PAIRS=0 \
PREDIFY_MAX_VAL_PAIRS=0 \
PREDIFY_BATCHSIZE=4 \
PREDIFY_NUM_WORKERS=4 \
PREDIFY_EPOCHS=10 \
PREDIFY_PRETRAINED=1 \
PREDIFY_PCODER_WEIGHTS=/home/lin/predify/weights_pvgg16_imagenet \
PREDIFY_LR=1e-4 \
PREDIFY_WEIGHT_DECAY=0 \
PREDIFY_EMA_DECAY=0.99 \
PREDIFY_TOP_TARGET_SOURCE=ema_teacher \
PREDIFY_LAYER_LOSS_WEIGHTS=0.1,0.1,0.2,0.2,1.0 \
PREDIFY_OUTPUT_PATH=kitti_targetflow_cross_drive_pretrained_e10.p \
PYTHONPATH=/home/lin/predify2021_targetflow:/home/lin/predify \
TORCH_HOME=/home/lin/predify/.torch \
python -u -m predify2021.mce_scores.train_kitti_targetflow_adjacent_pairs
```

最终结果：

```text
Epoch 010
train_weighted = 0.145901
train_top      = 0.016155
train_top_std  = 0.676293
val_weighted   = 0.126257
val_top        = 0.015384
val_top_std    = 0.448606
```

新增产出文件：

```text
/home/lin/predify/kitti_targetflow_cross_drive_pretrained_e10.p
/home/lin/predify/kitti_targetflow_cross_drive_pretrained_e10_student.pt
/home/lin/predify/kitti_targetflow_cross_drive_pretrained_e10_teacher.pt
```

### 18.45 正式 rerun：student_self，cross-drive，已产出 checkpoint

命令：

```bash
PREDIFY_KITTI_ROOT=/home/lin/predify/kitti_raw \
PREDIFY_TRAIN_DRIVES=2011_09_26/2011_09_26_drive_0005_sync \
PREDIFY_VAL_DRIVES=2011_09_26/2011_09_26_drive_0011_sync \
PREDIFY_KITTI_CAMERA=image_02 \
PREDIFY_MAX_TRAIN_PAIRS=0 \
PREDIFY_MAX_VAL_PAIRS=0 \
PREDIFY_BATCHSIZE=4 \
PREDIFY_NUM_WORKERS=4 \
PREDIFY_EPOCHS=10 \
PREDIFY_PRETRAINED=1 \
PREDIFY_PCODER_WEIGHTS=/home/lin/predify/weights_pvgg16_imagenet \
PREDIFY_LR=1e-4 \
PREDIFY_WEIGHT_DECAY=0 \
PREDIFY_EMA_DECAY=0.99 \
PREDIFY_TOP_TARGET_SOURCE=student_self \
PREDIFY_LAYER_LOSS_WEIGHTS=0.1,0.1,0.2,0.2,1.0 \
PREDIFY_OUTPUT_PATH=kitti_targetflow_cross_drive_student_self_pretrained_e10.p \
PYTHONPATH=/home/lin/predify2021_targetflow:/home/lin/predify \
TORCH_HOME=/home/lin/predify/.torch \
python -u -m predify2021.mce_scores.train_kitti_targetflow_adjacent_pairs
```

最终结果：

```text
Epoch 010
train_weighted = 0.051178
train_top      = 0.000002
train_top_std  = 0.071114
val_weighted   = 0.055408
val_top        = 0.000001
val_top_std    = 0.079227
```

新增产出文件：

```text
/home/lin/predify/kitti_targetflow_cross_drive_student_self_pretrained_e10.p
/home/lin/predify/kitti_targetflow_cross_drive_student_self_pretrained_e10_student.pt
```

### 18.46 正式 ego-motion linear probe：ema_teacher student checkpoint

命令：

```bash
PREDIFY_KITTI_ROOT=/home/lin/predify/kitti_raw \
PREDIFY_TRAIN_DRIVES=2011_09_26/2011_09_26_drive_0005_sync \
PREDIFY_VAL_DRIVES=2011_09_26/2011_09_26_drive_0011_sync \
PREDIFY_KITTI_CAMERA=image_02 \
PREDIFY_MAX_TRAIN_PAIRS=0 \
PREDIFY_MAX_VAL_PAIRS=0 \
PREDIFY_BATCHSIZE=16 \
PREDIFY_NUM_WORKERS=4 \
PREDIFY_PROBE_RIDGE=1e-3 \
PREDIFY_PRETRAINED=1 \
PREDIFY_PCODER_WEIGHTS=/home/lin/predify/weights_pvgg16_imagenet \
PREDIFY_MODEL_CHECKPOINT=/home/lin/predify/kitti_targetflow_cross_drive_pretrained_e10_student.pt \
PREDIFY_MODEL_LABEL=ema_teacher \
PREDIFY_OUTPUT_PATH=kitti_ego_motion_probe_ema_teacher.p \
PYTHONPATH=/home/lin/predify2021_targetflow:/home/lin/predify \
TORCH_HOME=/home/lin/predify/.torch \
python -u -m predify2021.mce_scores.calculate_kitti_ego_motion_linear_probe
```

结果：

```text
Train pairs: 153
Val pairs:   232
Feature dim: 512

Train metrics:
mae_delta_forward_m  = 9.607551e-07
mae_delta_yaw_rad    = 1.159842e-07
rmse_delta_forward_m = 1.194289e-06
rmse_delta_yaw_rad   = 1.445361e-07

Val metrics:
mae_delta_forward_m  = 0.475708
mae_delta_yaw_rad    = 0.013949
rmse_delta_forward_m = 0.537695
rmse_delta_yaw_rad   = 0.021383

Train feature stats:
mean_abs   = 0.009103
std_mean   = 0.013805
std_global = 0.021582

Val feature stats:
mean_abs   = 0.005382
std_mean   = 0.008361
std_global = 0.014558
```

当前解读：

```text
1. 现在已经得到第一组正式的“冻结特征 -> 自车运动线性探针”结果。
2. 这说明 ema_teacher 训练后的顶层特征，确实包含可线性读取的 ego-motion 信息。
3. 但当前还不能下“更好/更差”的结论，因为 baseline 和 student_self probe 还没跑。
4. 下一步必须补齐：
   - baseline probe
   - student_self probe
```

### 18.47 正式 ego-motion linear probe：baseline pretrained

命令：

```bash
PREDIFY_KITTI_ROOT=/home/lin/predify/kitti_raw \
PREDIFY_TRAIN_DRIVES=2011_09_26/2011_09_26_drive_0005_sync \
PREDIFY_VAL_DRIVES=2011_09_26/2011_09_26_drive_0011_sync \
PREDIFY_KITTI_CAMERA=image_02 \
PREDIFY_MAX_TRAIN_PAIRS=0 \
PREDIFY_MAX_VAL_PAIRS=0 \
PREDIFY_BATCHSIZE=16 \
PREDIFY_NUM_WORKERS=4 \
PREDIFY_PROBE_RIDGE=1e-3 \
PREDIFY_PRETRAINED=1 \
PREDIFY_PCODER_WEIGHTS=/home/lin/predify/weights_pvgg16_imagenet \
PREDIFY_MODEL_LABEL=baseline_pretrained \
PREDIFY_OUTPUT_PATH=kitti_ego_motion_probe_baseline_pretrained.p \
PYTHONPATH=/home/lin/predify2021_targetflow:/home/lin/predify \
python -u -m predify2021.mce_scores.calculate_kitti_ego_motion_linear_probe
```

结果：

```text
Train pairs: 153
Val pairs:   232
Feature dim: 512

Train metrics:
mae_delta_forward_m  = 3.784126e-06
mae_delta_yaw_rad    = 5.278006e-07
rmse_delta_forward_m = 4.439529e-06
rmse_delta_yaw_rad   = 6.045175e-07

Val metrics:
mae_delta_forward_m  = 0.510362
mae_delta_yaw_rad    = 0.028417
rmse_delta_forward_m = 0.598583
rmse_delta_yaw_rad   = 0.043002

Train feature stats:
mean_abs   = 0.032767
std_mean   = 0.048879
std_global = 0.060115

Val feature stats:
mean_abs   = 0.019310
std_mean   = 0.032736
std_global = 0.042789
```

### 18.48 baseline_pretrained vs ema_teacher：当前 probe 对比

验证集指标对比：

```text
baseline_pretrained:
mae_delta_forward_m  = 0.510362
mae_delta_yaw_rad    = 0.028417
rmse_delta_forward_m = 0.598583
rmse_delta_yaw_rad   = 0.043002

ema_teacher:
mae_delta_forward_m  = 0.475708
mae_delta_yaw_rad    = 0.013949
rmse_delta_forward_m = 0.537695
rmse_delta_yaw_rad   = 0.021383
```

当前结论：

```text
1. ema_teacher 在 4 个 ego-motion probe 指标上都优于 baseline_pretrained。
2. 尤其是 yaw 相关指标改善更明显：
   - mae_delta_yaw_rad:  0.028417 -> 0.013949
   - rmse_delta_yaw_rad: 0.043002 -> 0.021383
3. 这说明 target-flow + ema_teacher 学到的顶层时序特征，
   相比原始 pretrained 特征，包含了更可线性读取的自车运动信息。
4. 但最终结论仍需补齐 student_self probe，才能形成完整三组对比：
   baseline_pretrained vs ema_teacher vs student_self
```

### 18.49 正式 ego-motion linear probe：student_self student checkpoint

命令：

```bash
PREDIFY_KITTI_ROOT=/home/lin/predify/kitti_raw \
PREDIFY_TRAIN_DRIVES=2011_09_26/2011_09_26_drive_0005_sync \
PREDIFY_VAL_DRIVES=2011_09_26/2011_09_26_drive_0011_sync \
PREDIFY_KITTI_CAMERA=image_02 \
PREDIFY_MAX_TRAIN_PAIRS=0 \
PREDIFY_MAX_VAL_PAIRS=0 \
PREDIFY_BATCHSIZE=16 \
PREDIFY_NUM_WORKERS=0 \
PREDIFY_PROBE_RIDGE=1e-3 \
PREDIFY_PRETRAINED=1 \
PREDIFY_PCODER_WEIGHTS=/home/lin/predify/weights_pvgg16_imagenet \
PREDIFY_MODEL_CHECKPOINT=/home/lin/predify/kitti_targetflow_cross_drive_student_self_pretrained_e10_student.pt \
PREDIFY_MODEL_LABEL=student_self \
PREDIFY_OUTPUT_PATH=/tmp/kitti_ego_motion_probe_student_self_rerun.p \
PYTHONPATH=/home/lin/predify2021_targetflow:/home/lin/predify \
TORCH_HOME=/home/lin/predify/.torch \
/home/lin/anaconda3/envs/predifyproject/bin/python -u -m predify2021.mce_scores.calculate_kitti_ego_motion_linear_probe
```

结果：

```text
Train pairs: 153
Val pairs:   232
Feature dim: 512

Train metrics:
mae_delta_forward_m  = 0.091665
mae_delta_yaw_rad    = 0.015517
rmse_delta_forward_m = 0.105943
rmse_delta_yaw_rad   = 0.016647

Val metrics:
mae_delta_forward_m  = 0.475929
mae_delta_yaw_rad    = 0.003136
rmse_delta_forward_m = 0.513674
rmse_delta_yaw_rad   = 0.004119

Train feature stats:
mean_abs   = 0.000037
std_mean   = 0.000048
std_global = 0.000349

Val feature stats:
mean_abs   = 0.000025
std_mean   = 0.000039
std_global = 0.000273
```

说明：

```text
1. 本次直接使用 /home/lin/anaconda3/envs/predifyproject/bin/python 运行，
   等价于在 predifyproject conda 环境中执行，不需要额外 activate。
2. 这次 rerun 与根目录中已有的 kitti_ego_motion_probe_student_self.p 一致，
   说明 student_self probe 入口当前是可重现的。
3. 但 student_self 的特征方差极低，说明它学到的是一个非常压缩的顶层表示。
```

### 18.50 baseline_pretrained vs ema_teacher vs student_self：完整 probe 对比

验证集指标对比：

```text
baseline_pretrained:
mae_delta_forward_m  = 0.510362
mae_delta_yaw_rad    = 0.028417
rmse_delta_forward_m = 0.598583
rmse_delta_yaw_rad   = 0.043002

ema_teacher:
mae_delta_forward_m  = 0.475708
mae_delta_yaw_rad    = 0.013949
rmse_delta_forward_m = 0.537695
rmse_delta_yaw_rad   = 0.021383

student_self:
mae_delta_forward_m  = 0.475929
mae_delta_yaw_rad    = 0.003136
rmse_delta_forward_m = 0.513674
rmse_delta_yaw_rad   = 0.004119
```

顶层特征尺度对比（验证集）：

```text
baseline_pretrained:
std_global = 0.042789

ema_teacher:
std_global = 0.014558

student_self:
std_global = 0.000273
```

当前完整结论：

```text
1. 三组对比已经补齐，ego-motion probe 这条验证链现在是闭环的。
2. 相比 baseline_pretrained，ema_teacher 和 student_self 都学到了更可线性读取的自车运动信息。
3. 从 probe 指标本身看，student_self 在 yaw 预测上最好，
   在 forward 指标上与 ema_teacher 基本同一量级。
4. 但 student_self 的顶层特征方差极低，远低于 baseline 和 ema_teacher，
   说明它更像是把表示压到一个很小的运动编码子空间里。
5. 因此如果目标是“做一个 narrow ego-motion probe”，student_self 很强；
   但如果目标是“保留更丰富、更不易塌缩的通用时序表示”，ema_teacher 更稳妥。
6. 这也说明下一步如果继续 targetflow，不应该只盯 probe 分数，
   还要同时监控特征方差、表示塌缩风险，以及更广的下游任务表现。
```

### 18.51 新增：student_self 的最小 anti-collapse 正则

改动文件：

```text
/home/lin/predify2021_targetflow/predify2021/mce_scores/train_kitti_targetflow_adjacent_pairs.py
```

改动目的：

```text
student_self 在 ego-motion probe 上表现很好，但顶层特征方差极低。
因此先加入一个最小的 top-feature variance regularizer，
测试是否能在不改整体 targetflow 结构的前提下，抑制表示塌缩。
```

新增环境变量：

```text
PREDIFY_TOP_VARIANCE_WEIGHT
  方差正则权重，默认 0.0，表示关闭。

PREDIFY_TOP_VARIANCE_TARGET
  目标顶层 pooled 特征标准差下界，默认 0.01。

PREDIFY_TOP_VARIANCE_EPS
  数值稳定项，默认 1e-4。
```

实现方式：

```text
1. 取顶层 forward_output。
2. 对空间维做平均池化，得到 pooled top feature。
3. 计算每个通道在 batch 维上的 std。
4. 使用：
   variance_loss = mean(relu(target_std - std_per_dim))
5. 训练目标变为：
   optimized_loss = weighted_loss + weight * variance_loss
```

日志新增：

```text
train_objective / val_objective
train_top_poolstd / val_top_poolstd
train_varloss / val_varloss
```

### 18.52 anti-collapse smoke：student_self，无方差正则

命令：

```bash
PREDIFY_KITTI_ROOT=/home/lin/predify/kitti_raw \
PREDIFY_TRAIN_DRIVES=2011_09_26/2011_09_26_drive_0005_sync \
PREDIFY_VAL_DRIVES=2011_09_26/2011_09_26_drive_0011_sync \
PREDIFY_KITTI_CAMERA=image_02 \
PREDIFY_MAX_TRAIN_PAIRS=32 \
PREDIFY_MAX_VAL_PAIRS=32 \
PREDIFY_BATCHSIZE=4 \
PREDIFY_NUM_WORKERS=0 \
PREDIFY_EPOCHS=2 \
PREDIFY_PRETRAINED=1 \
PREDIFY_PCODER_WEIGHTS=/home/lin/predify/weights_pvgg16_imagenet \
PREDIFY_LR=1e-4 \
PREDIFY_WEIGHT_DECAY=0 \
PREDIFY_EMA_DECAY=0.99 \
PREDIFY_TOP_TARGET_SOURCE=student_self \
PREDIFY_LAYER_LOSS_WEIGHTS=0.1,0.1,0.2,0.2,1.0 \
PREDIFY_TOP_VARIANCE_WEIGHT=0.0 \
PREDIFY_TOP_VARIANCE_TARGET=0.01 \
PREDIFY_OUTPUT_PATH=/tmp/kitti_targetflow_student_self_var0_smoke.p \
PYTHONPATH=/home/lin/predify2021_targetflow:/home/lin/predify \
TORCH_HOME=/home/lin/predify/.torch \
/home/lin/anaconda3/envs/predifyproject/bin/python -u -m predify2021.mce_scores.train_kitti_targetflow_adjacent_pairs
```

第 2 轮结果：

```text
train_weighted   = 0.344267
train_top        = 0.000774
train_top_std    = 0.063885
train_top_poolstd= 0.011040
train_varloss    = 0.000000

val_weighted     = 0.422897
val_top          = 0.000198
val_top_std      = 0.051337
val_top_poolstd  = 0.010170
val_varloss      = 0.000000
```

说明：

```text
1. 在这个小设置下，student_self 会快速把 pooled top std 压到 0.01 左右。
2. 因为 target_std=0.01，所以正则实际上没有被触发。
3. 这说明如果要真正顶住塌缩，target_std 需要设得更高。
```

### 18.53 anti-collapse smoke：student_self，开启方差正则

命令：

```bash
PREDIFY_KITTI_ROOT=/home/lin/predify/kitti_raw \
PREDIFY_TRAIN_DRIVES=2011_09_26/2011_09_26_drive_0005_sync \
PREDIFY_VAL_DRIVES=2011_09_26/2011_09_26_drive_0011_sync \
PREDIFY_KITTI_CAMERA=image_02 \
PREDIFY_MAX_TRAIN_PAIRS=32 \
PREDIFY_MAX_VAL_PAIRS=32 \
PREDIFY_BATCHSIZE=4 \
PREDIFY_NUM_WORKERS=0 \
PREDIFY_EPOCHS=2 \
PREDIFY_PRETRAINED=1 \
PREDIFY_PCODER_WEIGHTS=/home/lin/predify/weights_pvgg16_imagenet \
PREDIFY_LR=1e-4 \
PREDIFY_WEIGHT_DECAY=0 \
PREDIFY_EMA_DECAY=0.99 \
PREDIFY_TOP_TARGET_SOURCE=student_self \
PREDIFY_LAYER_LOSS_WEIGHTS=0.1,0.1,0.2,0.2,1.0 \
PREDIFY_TOP_VARIANCE_WEIGHT=1.0 \
PREDIFY_TOP_VARIANCE_TARGET=0.02 \
PREDIFY_OUTPUT_PATH=/tmp/kitti_targetflow_student_self_var1_smoke.p \
PYTHONPATH=/home/lin/predify2021_targetflow:/home/lin/predify \
TORCH_HOME=/home/lin/predify/.torch \
/home/lin/anaconda3/envs/predifyproject/bin/python -u -m predify2021.mce_scores.train_kitti_targetflow_adjacent_pairs
```

第 2 轮结果：

```text
train_weighted    = 0.342825
train_objective   = 0.351527
train_top         = 0.001165
train_top_std     = 0.079352
train_top_poolstd = 0.011903
train_varloss     = 0.008702

val_weighted      = 0.413923
val_objective     = 0.423473
val_top           = 0.000529
val_top_std       = 0.076345
val_top_poolstd   = 0.010477
val_varloss       = 0.009550
```

### 18.54 当前解读：第一步 anti-collapse 是有效的，但还不够强

```text
1. 方差正则已经真正进入训练目标：
   train_objective > train_weighted，且 train/val_varloss 都大于 0。
2. 开启后，pooled top std 的确被往上推了一点：
   train: 0.011040 -> 0.011903
   val:   0.010170 -> 0.010477
3. 这说明方向是对的：正则不是摆设，模型已经在回应这个约束。
4. 但提升幅度还很小，说明当前 (weight=1.0, target=0.02) 仍然偏弱。
5. 因此下一步最合理的是继续做一个小 sweep：
   - 提高 weight
   - 或提高 target
   - 看 pooled top std、probe 指标、以及是否出现训练不稳定
```

### 18.55 anti-collapse sweep：更强的 variance regularizer

继续测试两组更强设置：

```text
A) weight=5.0, target=0.02
B) weight=5.0, target=0.03
```

运行条件保持不变：

```text
train_pairs = 32
val_pairs   = 32
epochs      = 2
top_target_source = student_self
```

第 2 轮结果汇总：

```text
no_reg (weight=0, target=0.01 implicit):
train_top_poolstd = 0.011040
val_top_poolstd   = 0.010170
val_weighted      = 0.422897

weight=1.0, target=0.02:
train_top_poolstd = 0.011903
val_top_poolstd   = 0.010477
val_weighted      = 0.413923

weight=5.0, target=0.02:
train_top_poolstd = 0.025994
val_top_poolstd   = 0.017120
val_weighted      = 0.443615

weight=5.0, target=0.03:
train_top_poolstd = 0.042272
val_top_poolstd   = 0.022901
val_weighted      = 0.426949
```

当前解读：

```text
1. 更强的正则确实能显著抬高 pooled top std。
2. 最明显的是 weight=5.0, target=0.03：
   - train_top_poolstd: 0.011040 -> 0.042272
   - val_top_poolstd:   0.010170 -> 0.022901
3. 从 weighted loss 看，weight=5.0, target=0.03 没有明显把训练打坏；
   val_weighted 仍在和无正则同量级。
4. 因此在当前小 sweep 中，
   weight=5.0, target=0.03 是最值得继续往下验证的一组。
```

### 18.56 小训练泛化检查：无正则 vs 最佳 anti-collapse smoke

为了确认“方差抬起来”不是空指标，继续做了一个非常直接的检查：

```text
把 2 epoch / 32 pair 训练得到的 checkpoint，
分别接到 frozen-feature ego-motion linear probe 上，
看 cross-drive 验证集效果。
```

比较对象：

```text
1. student_self_var0_smoke
   - 无方差正则
   - checkpoint: /tmp/kitti_targetflow_student_self_var0_smoke_student.pt

2. student_self_var5_t03_smoke
   - weight=5.0, target=0.03
   - checkpoint: /tmp/kitti_targetflow_student_self_var5_t03_smoke_student.pt
```

probe 验证集结果：

```text
student_self_var0_smoke:
mae_delta_forward_m  = 2.147193
mae_delta_yaw_rad    = 0.201866
rmse_delta_forward_m = 5.052003
rmse_delta_yaw_rad   = 0.407693
val_feature_std_global = 0.002115

student_self_var5_t03_smoke:
mae_delta_forward_m  = 0.604543
mae_delta_yaw_rad    = 0.051942
rmse_delta_forward_m = 0.762106
rmse_delta_yaw_rad   = 0.070713
val_feature_std_global = 0.024141
```

这组结果的意义：

```text
1. 在极小训练设置下，无正则 student_self 虽然能快速收敛局部目标，
   但 frozen-feature probe 的跨 drive 泛化非常差。
2. 加上 anti-collapse 正则后，验证集 probe 显著改善：
   - mae_delta_forward_m: 2.147 -> 0.605
   - mae_delta_yaw_rad:   0.202 -> 0.052
3. 同时特征方差也明显提高：
   std_global: 0.002115 -> 0.024141
4. 这说明 anti-collapse 不是只把方差指标“做漂亮”，
   而是在当前小设置下，确实提高了时序特征的可泛化性。
5. 当然，这仍然只是 smoke 级别验证；
   它说明方向正确，但不能直接替代正式 10 epoch / full pair 对比。
```

### 18.57 当前最合理的下一步

```text
1. 采用当前最优 smoke 设置：
   PREDIFY_TOP_VARIANCE_WEIGHT=5.0
   PREDIFY_TOP_VARIANCE_TARGET=0.03

2. 在 full cross-drive 训练上正式跑一版：
   train=0005, val=0011, pretrained, student_self + anti-collapse

3. 再做正式 frozen-feature ego-motion probe，
   与旧的 student_self / ema_teacher / baseline_pretrained 做完整对比。
```

### 18.58 这一步正式 full cross-drive 的目的

```text
为什么现在要正式跑一版 student_self + anti-collapse 的 full cross-drive？

目的 1：
把 smoke 级发现升级成正式实验结论。
目前我们已经知道在 32 pair / 2 epoch 的小设置下，
anti-collapse 能提高 top feature variance，并改善小规模 probe 泛化。
但这还不足以证明它在正式训练规模下也成立。

目的 2：
验证 anti-collapse 是否真的改变了模型学到的时序表示，
而不是只在小样本上碰巧有效。
如果 full cross-drive 下也能维持更高的 top pooled std，
同时 probe 不明显变差或进一步变好，
就能说明这个约束确实在塑造更健康的时序表示。

目的 3：
把新模型线补成一个完整对比闭环。
正式跑完之后，我们就能比较：
baseline_pretrained
ema_teacher
student_self
student_self + anti-collapse

目的 4：
为后续“继续改整个模型”提供一个稳定起点。
如果这一步成功，后续再改 target definition、learn flow、temporal objective，
就不是在一个明显塌缩的 student_self 基线上继续堆东西，
而是在一个已经被验证更稳的版本上继续推进。
```

### 18.59 正式 full cross-drive：student_self + anti-collapse 定稿结果

正式训练配置：

```text
train drive: 2011_09_26_drive_0005_sync
val drive:   2011_09_26_drive_0011_sync
pretrained:  True
top_target_source: student_self
top_variance_weight: 5.0
top_variance_target: 0.03
epochs: 10
output:
/home/lin/predify/kitti_targetflow_cross_drive_student_self_var5_t03_pretrained_e10.p
/home/lin/predify/kitti_targetflow_cross_drive_student_self_var5_t03_pretrained_e10_student.pt
```

第 10 epoch 训练终点：

```text
train_weighted   = 0.104442
train_objective  = 0.108041
train_top        = 0.002973
train_top_std    = 0.537844
train_top_poolstd= 0.090340
train_varloss    = 0.000720

val_weighted     = 0.069283
val_objective    = 0.156437
val_top          = 0.001225
val_top_std      = 0.463838
val_top_poolstd  = 0.012589
val_varloss      = 0.017431
```

随后对该 checkpoint 做正式 frozen-feature ego-motion linear probe：

```text
output:
/home/lin/predify/kitti_ego_motion_probe_student_self_var5_t03_pretrained_e10.p
```

验证集 probe 结果：

```text
mae_delta_forward_m  = 0.514171
mae_delta_yaw_rad    = 0.029783
rmse_delta_forward_m = 0.613119
rmse_delta_yaw_rad   = 0.040938
val_feature_std_global = 0.010863
```

与已有三组正式结果对比：

```text
baseline_pretrained:
mae_forward = 0.510362
mae_yaw     = 0.028417
rmse_forward= 0.598583
rmse_yaw    = 0.043002
std_global  = 0.042789

ema_teacher:
mae_forward = 0.475708
mae_yaw     = 0.013949
rmse_forward= 0.537695
rmse_yaw    = 0.021383
std_global  = 0.014558

student_self:
mae_forward = 0.475929
mae_yaw     = 0.003136
rmse_forward= 0.513674
rmse_yaw    = 0.004119
std_global  = 0.000273

student_self + anti-collapse:
mae_forward = 0.514171
mae_yaw     = 0.029783
rmse_forward= 0.613119
rmse_yaw    = 0.040938
std_global  = 0.010863
```

当前结论：

```text
1. anti-collapse 正则确实把 student_self 从严重塌缩状态里拉出来了：
   std_global 从 0.000273 提高到 0.010863。

2. 但在这次正式 full cross-drive probe 上，
   它没有超过 baseline_pretrained，也没有超过 ema_teacher。

3. 这说明：
   - “只用 student_self 顶层目标”本身仍然不够稳；
   - anti-collapse 解决了塌缩问题的一部分，
     但还没有把 target flow 变成更强的时序表示学习方案。

4. 因而这一轮的最重要收获不是“性能最好”，
   而是：
   - 我们已经拿到了一个不再明显塌缩的 student_self 版本；
   - 并且用 formal probe 证明了它的可泛化性回到了 baseline 同量级；
   - 可以作为后续继续改 target definition / learn flow / temporal objective 的新起点。
```

### 18.60 新一轮结构改动：显式 temporal predictor

这一轮不再只是讨论 target flow 本身，而是把“根据前面误差去预测下一时刻信息”正式落成代码。

改动文件：

```text
/home/lin/predify2021_targetflow/predify2021/model_factory/pvgg16_targetflow.py
/home/lin/predify2021_targetflow/predify2021/mce_scores/train_kitti_targetflow_adjacent_pairs.py
/home/lin/predify2021_targetflow/predify2021/mce_scores/calculate_kitti_targetflow_pair_smoke.py
```

核心改动：

```text
1. 在 pvgg_tf 中新增显式 temporal predictor。
2. 输入不是只看当前顶层 feature，
   而是把：
   - 当前顶层 forward pooled feature
   - 5 层的 pooled target-flow error
   拼成一个 temporal context。
3. 用一个两层 MLP 直接预测“下一时刻顶层 pooled representation”。
4. 训练目标中新增 temporal prediction loss：
   MSE(predicted_next_top, target_next_top)
5. 训练日志新增：
   - temporal loss
   - temporal mae
   - temporal cosine
6. pair smoke 也同步输出 temporal 指标，
   用来快速检查这条新链路是否真实接通。
```

这一步的意义：

```text
以前的 pvgg_tf 更像是：
当前帧通过 backward target flow 去对齐下一帧目标。

现在的 pvgg_tf 额外变成：
当前帧内部各层误差状态 -> 显式预测下一时刻顶层表示。

也就是说，
“根据前面信息之间的误差去预测下一时刻的信息”
已经不只是概念，而是被写成了模型里的一个可训练分支。
```

### 18.61 pair smoke：确认 temporal 链路接通

运行设置：

```text
KITTI drive: 2011_09_26_drive_0005_sync
pairs: 8
pretrained: True
target_flow_mode: quasi_steady
```

结果：

```text
Mean top local loss  = 0.079323
Mean top error rms   = 0.280712
Mean total local loss= 7.511069
Mean temporal loss   = 0.041134
Mean temporal mae    = 0.095783
Mean temporal cosine = -0.013812
```

解释：

```text
1. local loss 正常存在，说明旧的 target flow 没被破坏。
2. temporal loss / temporal mae / temporal cosine 都已经能计算，
   说明“误差上下文 -> 下一时刻顶层表示”的新链路确实接通了。
3. 初始化状态下 temporal cosine 接近 0，
   这是正常的，说明它还没学。
```

### 18.62 小规模 temporal smoke：看它会不会学

运行设置：

```text
drive: 2011_09_26_drive_0005_sync
max_pairs: 64
train/val split: 51 / 13
epochs: 3
top_target_source: ema_teacher
top_variance_weight: 0.0
temporal_prediction_weight: 1.0
output:
/tmp/kitti_targetflow_temporal_smoke_e3.p
```

3 个 epoch 的关键指标：

```text
Epoch 001:
train_temploss = 0.044785
train_tempcos  = 0.456855
val_temploss   = 0.031200
val_tempcos    = 0.680508

Epoch 002:
train_temploss = 0.018421
train_tempcos  = 0.838840
val_temploss   = 0.017317
val_tempcos    = 0.827199

Epoch 003:
train_temploss = 0.009661
train_tempcos  = 0.924986
val_temploss   = 0.015334
val_tempcos    = 0.857650
```

这一步给出的结论：

```text
1. temporal loss 明显下降。
2. temporal cosine 明显上升，而且验证集也同步变好。
3. 这说明新加的 temporal predictor 不是“接上了但没用”，
   而是真的开始学习：
   误差上下文 -> 下一时刻顶层表示。
```

### 18.63 正式 full cross-drive：temporal_ema_teacher_e10

正式运行设置：

```text
train drive: 2011_09_26_drive_0005_sync
val drive:   2011_09_26_drive_0011_sync
epochs: 10
pretrained: True
top_target_source: ema_teacher
top_variance_weight: 0.0
temporal_prediction_weight: 1.0

output:
/home/lin/predify/kitti_targetflow_cross_drive_temporal_ema_teacher_e10.p
/home/lin/predify/kitti_targetflow_cross_drive_temporal_ema_teacher_e10_student.pt
/home/lin/predify/kitti_targetflow_cross_drive_temporal_ema_teacher_e10_teacher.pt
```

第 10 epoch 结果：

```text
train_weighted   = 0.157469
train_objective  = 0.159136
train_top        = 0.018351
train_top_std    = 0.683248
train_top_poolstd= 0.067398
train_temploss   = 0.001667
train_tempmae    = 0.024367
train_tempcos    = 0.992855

val_weighted     = 0.131826
val_objective    = 0.152363
val_top          = 0.017449
val_top_std      = 0.452395
val_top_poolstd  = 0.012661
val_temploss     = 0.020537
val_tempmae      = 0.068096
val_tempcos      = 0.877319
```

正式结论：

```text
1. 这版模型已经明确满足“模型能跑”。
2. 它也明确满足“能学习时序信息”：
   因为 temporal loss 稳定下降，temporal cosine 稳定升高。
3. 它还明确满足“能根据前面信息之间的误差去预测下一时刻的信息”：
   因为 temporal predictor 的输入就是 pooled forward top +
   5 层 pooled error，而输出是 next-top representation。
4. 从 formal cross-drive 看，
   验证集 val_tempcos = 0.877319，
   这是一个很强的信号，
   说明这种时序预测关系并没有只停留在训练集里。
```

### 18.64 frozen-feature ego-motion probe：temporal_ema_teacher_e10

probe 输出：

```text
/home/lin/predify/kitti_ego_motion_probe_temporal_ema_teacher_e10.p
```

验证集结果：

```text
mae_delta_forward_m  = 1.687091
mae_delta_yaw_rad    = 0.028218
rmse_delta_forward_m = 5.669855
rmse_delta_yaw_rad   = 0.064424
val_feature_std_global = 0.014623
```

和旧的 ema_teacher probe 对比：

```text
旧 ema_teacher:
mae_forward = 0.475708
mae_yaw     = 0.013949
rmse_forward= 0.537695
rmse_yaw    = 0.021383
std_global  = 0.014558

新 temporal_ema_teacher_e10:
mae_forward = 1.687091
mae_yaw     = 0.028218
rmse_forward= 5.669855
rmse_yaw    = 0.064424
std_global  = 0.014623
```

这说明什么：

```text
1. 新模型在线性 probe 的“ego-motion 回归”这个下游上，
   目前还没有超过旧的 ema_teacher 版本。
2. 但这不否定本轮改动的成功，
   因为本轮的首要目标不是“提高 probe 分数”，
   而是把：
   误差上下文 -> 下一时刻表示预测
   这件事真正写进模型并验证它能学起来。
3. 当前结果说明：
   - 这条新时序预测机制已经成功实现；
   - 它学到的是强 next-top alignment；
   - 但这种表示还不等于最适合 ego-motion linear probe 的表示。
4. 因而下一步研究重点不应该是“怀疑这次改动有没有写进去”，
   而应该是：
   如何把当前已学到的 temporal prediction，
   再变成对下游时序任务更有用的表征。
```

### 18.65 为什么下一步先改 target definition

当前 temporal_ema_teacher_e10 已经证明了一件事：

```text
模型可以很稳定地学到
“误差上下文 -> 下一时刻顶层表示”
这个映射。
```

但 probe 结果也说明了另一件事：

```text
“会对齐 next-top representation”
不等于
“这种表示最适合下游时序任务”。
```

所以现在最合理的下一步不是继续重复同一种 target，
而是改 temporal target 的定义本身。

三种候选方向：

```text
1. motion-aware target
   让目标更直接描述“变化/运动”，
   而不只是下一帧的绝对表示。

2. multi-horizon target
   让模型同时看短期和稍长期未来，
   避免只学到 1-step 的局部贴合。

3. task-aligned target
   让时序目标更贴近我们真正关心的下游任务，
   例如 ego-motion / scene dynamics。
```

为什么先做 motion-aware：

```text
1. 改动最小。
2. 不需要先把整个框架变成显式监督任务。
3. 最符合“根据前面误差去预测下一时刻变化”这个目标。
4. 最容易快速验证是否比 next-top 更合理。
```

因此本轮优先实现：

```text
temporal_target_mode = delta_top

其中：
delta_top = pooled(next_top) - pooled(current_top)
```

也就是让 temporal predictor 不再预测“下一帧顶层绝对表示”，
而是预测“下一帧相对当前帧的顶层变化量”。

### 18.66 delta_top 版本代码改动

改动文件：

```text
/home/lin/predify2021_targetflow/predify2021/model_factory/pvgg16_targetflow.py
/home/lin/predify2021_targetflow/predify2021/model_factory/get_model.py
/home/lin/predify2021_targetflow/predify2021/mce_scores/train_kitti_targetflow_adjacent_pairs.py
/home/lin/predify2021_targetflow/predify2021/mce_scores/calculate_kitti_targetflow_pair_smoke.py
```

改动内容：

```text
1. 给 pvgg_tf 新增 temporal_target_mode:
   - next_top
   - delta_top

2. 当 temporal_target_mode=delta_top 时：
   temporal_target = pooled(next_top) - pooled(current_top)

3. get_model 增加 temporal_target_mode 透传。

4. 训练脚本新增环境变量：
   PREDIFY_TEMPORAL_TARGET_MODE

5. pair smoke 输出中保留 temporal 指标，
   方便快速比较 next_top / delta_top。
```

### 18.67 delta_top pair smoke

运行设置：

```text
KITTI drive: 2011_09_26_drive_0005_sync
pairs: 8
pretrained: True
target_flow_mode: quasi_steady
temporal_target_mode: delta_top
```

结果：

```text
Mean temporal loss   = 0.004101
Mean temporal mae    = 0.045815
Mean temporal cosine = -0.003184
```

和旧 next_top pair smoke 对比：

```text
next_top:
temporal_loss   = 0.041134
temporal_mae    = 0.095783
temporal_cosine = -0.013812

delta_top:
temporal_loss   = 0.004101
temporal_mae    = 0.045815
temporal_cosine = -0.003184
```

解释：

```text
1. delta_top 的初始 loss 明显更小。
2. 这是合理的，因为变化量目标本来就比绝对 next-top 更小。
3. 它说明：
   delta_top 至少在尺度上是一个更“紧”的目标。
```

补充诊断：

```text
在 64 pair 上测得：
mean_next_norm  ≈ 5.5567
mean_delta_norm ≈ 1.4735

mean_next_abs   ≈ 0.1104
mean_delta_abs  ≈ 0.0361
```

这说明 delta_top 的目标幅值确实显著更小，
所以它的 loss 和 cosine 需要结合起来看，
不能直接拿 next_top 的数值直觉硬比。

### 18.68 delta_top 小规模 temporal smoke

运行设置：

```text
drive: 2011_09_26_drive_0005_sync
max_pairs: 64
train/val split: 51 / 13
epochs: 3
top_target_source: ema_teacher
temporal_target_mode: delta_top
temporal_prediction_weight: 1.0
output:
/tmp/kitti_targetflow_temporal_delta_smoke_e3.p
```

结果：

```text
Epoch 001:
train_temploss = 0.012077
train_tempcos  = 0.211762
val_temploss   = 0.016995
val_tempcos    = 0.200418

Epoch 002:
train_temploss = 0.006841
train_tempcos  = 0.133534
val_temploss   = 0.008452
val_tempcos    = 0.162386

Epoch 003:
train_temploss = 0.003764
train_tempcos  = 0.108352
val_temploss   = 0.004808
val_tempcos    = 0.164688
```

当前解读：

```text
1. delta_top 明显能学：
   temporal loss 持续下降。

2. 但它和 next_top 的学习形态不同：
   - next_top: cosine 会快速冲高
   - delta_top: loss 降得快，但 cosine 不高

3. 这很可能是因为 delta_top 目标本身幅值更小，
   方向更容易受小噪声影响；
   因而对 delta_top，更适合优先看：
   - temporal loss
   - temporal mae
   而不是只盯着 cosine。
```

### 18.69 当前状态

```text
1. delta_top 版本已经完成代码接入。
2. smoke 验证已经完成。
3. 正在启动正式 full cross-drive 版本：
   /home/lin/predify/kitti_targetflow_cross_drive_temporal_delta_ema_teacher_e10.p
```

### 18.70 delta_top 正式 full cross-drive：temporal_delta_ema_teacher_e10

正式运行设置：

```text
train drive: 2011_09_26_drive_0005_sync
val drive:   2011_09_26_drive_0011_sync
epochs: 10
pretrained: True
top_target_source: ema_teacher
temporal_target_mode: delta_top
top_variance_weight: 0.0
temporal_prediction_weight: 1.0

output:
/home/lin/predify/kitti_targetflow_cross_drive_temporal_delta_ema_teacher_e10.p
/home/lin/predify/kitti_targetflow_cross_drive_temporal_delta_ema_teacher_e10_student.pt
/home/lin/predify/kitti_targetflow_cross_drive_temporal_delta_ema_teacher_e10_teacher.pt
```

第 10 epoch 结果：

```text
train_weighted    = 0.143661
train_objective   = 0.145029
train_top         = 0.017687
train_top_std     = 0.646898
train_top_poolstd = 0.063175
train_temploss    = 0.001369
train_tempmae     = 0.017096
train_tempcos     = 0.204215

val_weighted      = 0.126478
val_objective     = 0.128610
val_top           = 0.019597
val_top_std       = 0.421129
val_top_poolstd   = 0.012329
val_temploss      = 0.002132
val_tempmae       = 0.023855
val_tempcos       = 0.123434
```

如何解读：

```text
1. delta_top 目标在 formal cross-drive 上同样能稳定学起来。
2. 它的 temporal loss / temporal mae 明显很低，
   说明模型确实学到了“下一时刻变化量”的幅值结构。
3. 但 temporal cosine 不像 next_top 那样高，
   这不代表没学到，
   更像是 delta 目标本身幅值小、方向更敏感。
4. 因而：
   - next_top 更像“绝对下一状态对齐”
   - delta_top 更像“变化量拟合”
```

### 18.71 frozen-feature ego-motion probe：temporal_delta_ema_teacher_e10

probe 输出：

```text
/home/lin/predify/kitti_ego_motion_probe_temporal_delta_ema_teacher_e10.p
```

验证集结果：

```text
mae_delta_forward_m  = 0.511407
mae_delta_yaw_rad    = 0.011728
rmse_delta_forward_m = 0.680875
rmse_delta_yaw_rad   = 0.017006
val_feature_std_global = 0.013559
```

和 next_top probe 对比：

```text
next_top temporal_ema_teacher_e10:
mae_forward = 1.687091
mae_yaw     = 0.028218
rmse_forward= 5.669855
rmse_yaw    = 0.064424
std_global  = 0.014623

delta_top temporal_delta_ema_teacher_e10:
mae_forward = 0.511407
mae_yaw     = 0.011728
rmse_forward= 0.680875
rmse_yaw    = 0.017006
std_global  = 0.013559
```

和更早的旧 ema_teacher probe 对比：

```text
旧 ema_teacher:
mae_forward = 0.475708
mae_yaw     = 0.013949
rmse_forward= 0.537695
rmse_yaw    = 0.021383
std_global  = 0.014558
```

当前结论：

```text
1. delta_top 比 next_top 明显更有下游时序可用性。
2. 它把 probe 表现从“明显失真”拉回到了“接近旧 baseline”的水平。
3. 在 yaw 方向上，delta_top 甚至优于旧 ema_teacher。
4. 在 forward 误差上，delta_top 仍略差于旧 ema_teacher，
   说明 target definition 方向是对的，但还没有完全到最优。
```

### 18.72 到目前为止的结论与下一步

```text
1. pvgg_tf 已经不是只会跑的骨架了，
   而是一个能真正学习“误差上下文 -> 未来表示/变化量”的正式研究模型。

2. 在 target definition 上：
   - next_top 证明了“未来绝对表示预测”可以学到
   - delta_top 证明了“未来变化量预测”更接近我们真正想要的时序信息

3. 当前最合理的下一步，
   不再是回头怀疑 targetflow 有没有写进去，
   而是继续沿着 target definition 往前推：
   - multi-horizon target
   - task-aligned target

4. 如果只选一个最小下一步，
   应该优先做 multi-horizon delta target：
   让模型同时预测 t+1 和 t+2 的变化量，
   这样更贴近“时序动态学”而不是单步配对。
```

### 18.73 multi-horizon delta target 代码接入

这一步的目标：

```text
不再只让 temporal head 预测一个 horizon，
而是最小化地扩成：
同时预测 t+1 和 t+2 的变化量目标。
```

改动文件：

```text
/home/lin/predify2021_targetflow/predify2021/model_factory/pvgg16_targetflow.py
/home/lin/predify2021_targetflow/predify2021/model_factory/get_model.py
/home/lin/predify2021_targetflow/predify2021/mce_scores/kitti_pairs.py
/home/lin/predify2021_targetflow/predify2021/mce_scores/train_kitti_targetflow_adjacent_pairs.py
/home/lin/predify2021_targetflow/predify2021/mce_scores/calculate_kitti_targetflow_pair_smoke.py
```

核心改动：

```text
1. pvgg_tf 新增 temporal_horizons 参数。
   例如：
   temporal_horizons = (1, 2)

2. temporal_predictor 输出维度从：
   512
   扩成：
   2 * 512
   并 reshape 成 [batch, horizons, feature_dim]。

3. temporal target 也扩成多 horizon：
   - next_top 模式：分别预测 t+1 / t+2 的 pooled top
   - delta_top 模式：分别预测
     pooled(top_{t+1}) - pooled(top_t)
     pooled(top_{t+2}) - pooled(top_t)

4. KITTI 数据从 pair 扩成 multi-horizon sample：
   sample_t = (x_t, [x_{t+1}, x_{t+2}])

5. local target-flow 主干没有推翻：
   layer-wise target / error / learn flow 仍保持现有结构，
   multi-horizon 只作用在 temporal head 这条分支上。
```

这样做的意义：

```text
1. 这是对现有 target definition 的自然延伸。
2. 它比单步 delta 更像真正的时序动态学约束。
3. 但它仍是“最小改动”，
   没有把整个框架一下子改成全新的序列模型。
```

### 18.74 multi-horizon delta target smoke

#### 18.74.1 前向 smoke

运行设置：

```text
drive: 2011_09_26_drive_0005_sync
samples: 8
pretrained: True
temporal_target_mode: delta_top
temporal_horizons: (1, 2)
```

结果：

```text
Mean top local loss = 0.079323
Mean top error rms  = 0.280712
Mean total local loss = 7.511069
Mean temporal loss  = 0.005055
Mean temporal mae   = 0.050349
Mean temporal cosine= -0.004815
```

解读：

```text
1. multi-horizon 版本已经能完整前向通过。
2. local target-flow 与 temporal head 可以同时工作。
3. t+1 / t+2 两个 future target 已经真实接入，
   不是只加了一个配置参数。
```

#### 18.74.2 2-epoch 小训练 smoke

运行设置：

```text
drive: 2011_09_26_drive_0005_sync
samples: 32
split: train 25 / val 7
epochs: 2
pretrained: True
top_target_source: ema_teacher
temporal_target_mode: delta_top
temporal_horizons: (1, 2)
temporal_prediction_weight: 1.0
output:
/tmp/kitti_targetflow_temporal_delta_h12_smoke_e2.p
```

结果：

```text
Epoch 001:
train_temploss = 0.011606
train_tempmae  = 0.059940
train_tempcos  = 0.153516
val_temploss   = 0.022176
val_tempmae    = 0.080904
val_tempcos    = 0.377883

Epoch 002:
train_temploss = 0.006292
train_tempmae  = 0.046883
train_tempcos  = 0.284834
val_temploss   = 0.011383
val_tempmae    = 0.064777
val_tempcos    = 0.261519
```

当前结论：

```text
1. multi-horizon delta target 不只是“能算”，而且“能学”。
2. 只用 2 个 epoch，
   train / val temporal loss 就都明显下降了。
3. 这说明：
   我们现在已经从单步时序目标，
   成功推进到了最小可运行的多步时序目标。
4. 因而下一步可以继续做正式 cross-drive 版本，
   或者进一步改成更 task-aligned 的 temporal target。
```

### 18.75 multi-horizon delta 正式 cross-drive 与 probe

正式训练设置：

```text
train drive: 2011_09_26_drive_0005_sync
val drive:   2011_09_26_drive_0011_sync
epochs: 10
pretrained: True
top_target_source: ema_teacher
temporal_target_mode: delta_top
temporal_horizons: (1, 2)
temporal_prediction_weight: 1.0
output:
/home/lin/predify/kitti_targetflow_cross_drive_temporal_delta_h12_ema_teacher_e10.p
/home/lin/predify/kitti_targetflow_cross_drive_temporal_delta_h12_ema_teacher_e10_student.pt
/home/lin/predify/kitti_targetflow_cross_drive_temporal_delta_h12_ema_teacher_e10_teacher.pt
```

第 10 epoch 结果：

```text
train_weighted    = 0.127159
train_objective   = 0.128210
train_top         = 0.012193
train_top_std     = 0.580893
train_top_poolstd = 0.058419
train_temploss    = 0.001051
train_tempmae     = 0.015361
train_tempcos     = 0.188695

val_weighted      = 0.113638
val_objective     = 0.114792
val_top           = 0.011986
val_top_std       = 0.410872
val_top_poolstd   = 0.012328
val_temploss      = 0.001154
val_tempmae       = 0.018899
val_tempcos       = -0.008547
```

如何解读：

```text
1. multi-horizon delta 在 temporal loss / mae 上继续变强了。
2. 尤其 val_temploss = 0.001154，已经低于单步 delta 的 0.002132。
3. 这说明：
   让模型同时拟合 t+1 和 t+2 的变化量，
   从“目标拟合”角度看是可行的。
4. 但这还不能直接说明它对下游更有用，
   所以必须继续接 ego-motion probe。
```

probe 输出：

```text
/home/lin/predify/kitti_ego_motion_probe_temporal_delta_h12_ema_teacher_e10.p
```

验证集结果：

```text
mae_delta_forward_m  = 0.544449
mae_delta_yaw_rad    = 0.023129
rmse_delta_forward_m = 0.677680
rmse_delta_yaw_rad   = 0.044099
val_feature_std_global = 0.013474
```

### 18.76 单步 delta vs 多步 delta

对比：

```text
single-horizon delta_top:
mae_forward = 0.511407
mae_yaw     = 0.011728
rmse_forward= 0.680875
rmse_yaw    = 0.017006
val_temploss= 0.002132

multi-horizon delta_top (1,2):
mae_forward = 0.544449
mae_yaw     = 0.023129
rmse_forward= 0.677680
rmse_yaw    = 0.044099
val_temploss= 0.001154
```

阶段性结论：

```text
1. multi-horizon 比 single-horizon 更会拟合 temporal target。
2. 但在当前的 ego-motion linear probe 上，
   multi-horizon 还没有超过 single-horizon delta_top。
3. 这说明目前多步目标更像是在“优化时序拟合”，
   但还没有自动转化成更好的下游运动表征。
4. 因而最合理的下一步，
   不再是继续堆更多 horizon，
   而是开始做 task-aligned temporal target：
   让时序目标更直接对齐运动/动态任务本身。
```

### 18.77 task-aligned temporal target：直接预测 ego-motion

这一步不再让 temporal head 去逼近抽象的 future feature，
而是直接预测下一时刻的运动量。

用更直白的话说：

```text
以前：
给当前帧 x_t，
让模型去猜“下一帧特征会变成什么样”。

现在：
给当前帧 x_t，
让模型直接去猜“下一时刻车会怎么动”。
```

当前采用的最小 task-aligned 目标是：

```text
delta_forward_m
delta_yaw_rad
```

也就是：

```text
1. 前向位移变化量
2. 偏航角变化量
```

代码改动点：

```text
1. kitti_pairs.py
   - 新增 KITTIEgoMotionMultiHorizonDataset
   - 直接返回 current frame / future frame / ego-motion target

2. pvgg16_targetflow.py
   - temporal_target_mode 新增 ego_motion
   - temporal predictor 输出维度从 feature dim 改成 motion dim
   - forward 支持 temporal_target_override

3. train_kitti_targetflow_adjacent_pairs.py
   - 新增 PREDIFY_TASK_ALIGNED_TARGET=ego_motion
   - 训练时把真实 ego-motion target 直接喂给 temporal 分支

4. calculate_kitti_targetflow_pair_smoke.py
   - pair smoke 同样支持 ego_motion target
```

#### 18.77.1 pair smoke

设置：

```text
temporal_target_mode: ego_motion
task_aligned_target: ego_motion
temporal_horizons: (1,)
max_pairs: 8
```

结果：

```text
Mean top local loss  = 0.079316
Mean top error rms   = 0.280700
Mean total local loss= 7.511060
Mean temporal loss   = 0.085292
Mean temporal mae    = 0.214873
Mean temporal cosine = -0.991758
```

解读：

```text
1. 这说明代码链路已经打通了。
2. temporal 分支现在确实在对真实 motion target 做预测。
3. 初始 cosine 很差是正常的，
   因为刚接上时模型还不会猜运动量。
```

#### 18.77.2 2-epoch 小训练 smoke

结果：

```text
Epoch 001:
train_temploss = 0.011583
train_tempmae  = 0.059462
train_tempcos  = 0.840407
val_temploss   = 0.010231
val_tempmae    = 0.074676
val_tempcos    = 0.999836

Epoch 002:
train_temploss = 0.001258
train_tempmae  = 0.022860
train_tempcos  = 0.999527
val_temploss   = 0.003691
val_tempmae    = 0.050615
val_tempcos    = 0.998772
```

解读：

```text
1. 它不是只能“跑通”，而是真的能学。
2. 只训练 2 个 epoch，
   temporal 分支就能快速贴近 motion target。
3. 这说明：
   task-aligned target 比 abstract feature target 更贴近我们的目标。
```

#### 18.77.3 正式 cross-drive：ego_motion

正式训练设置：

```text
train drive: 2011_09_26_drive_0005_sync
val drive:   2011_09_26_drive_0011_sync
epochs: 10
pretrained: True
top_target_source: ema_teacher
temporal_target_mode: ego_motion
task_aligned_target: ego_motion
temporal_horizons: (1,)
output:
/home/lin/predify/kitti_targetflow_cross_drive_ego_motion_ema_teacher_e10.p
/home/lin/predify/kitti_targetflow_cross_drive_ego_motion_ema_teacher_e10_student.pt
/home/lin/predify/kitti_targetflow_cross_drive_ego_motion_ema_teacher_e10_teacher.pt
```

第 10 epoch 结果：

```text
train_weighted    = 0.134323
train_objective   = 0.134984
train_top         = 0.014179
train_top_std     = 0.634072
train_top_poolstd = 0.066157
train_temploss    = 0.000661
train_tempmae     = 0.019086
train_tempcos     = 0.999572

val_weighted      = 0.115342
val_objective     = 0.255767
val_top           = 0.013323
val_top_std       = 0.421993
val_top_poolstd   = 0.012609
val_temploss      = 0.140425
val_tempmae       = 0.234556
val_tempcos       = 0.749503
```

另外直接看 motion prediction 误差：

```text
mae_forward_m     = 0.458817
mae_yaw_rad       = 0.011114
rmse_forward_m    = 0.532593
rmse_yaw_rad      = 0.012147
num_samples       = 232
```

阶段性结论：

```text
1. 这一步已经不只是“学 future feature”，
   而是开始直接学“下一时刻怎么动”。
2. 从研究目标上看，
   它比 next_top / delta_top / multi-horizon 更接近最终目标。
3. 训练集上 motion target 拟合很好，
   说明模型确实学到了可用的时序预测通路。
4. 但 cross-drive 验证集还有明显 generalization gap，
   说明它还没有稳定地学成“可迁移的运动规律”。
5. 因而现在的状态更准确地说是：
   我们已经把“目标定义”改对了，
   但还没有把“泛化能力”做强。
```

#### 18.77.4 frozen-feature ego-motion probe：task-aligned ego_motion

probe 输出：

```text
/home/lin/predify/kitti_ego_motion_probe_task_aligned_ego_motion_e10.p
```

验证集结果：

```text
mae_delta_forward_m  = 0.510476
mae_delta_yaw_rad    = 0.012843
rmse_delta_forward_m = 0.588023
rmse_delta_yaw_rad   = 0.017957
val_feature_std_global = 0.014376
```

与前面的 single-step delta_top 对比：

```text
single-step delta_top:
mae_forward = 0.511407
mae_yaw     = 0.011728
rmse_forward= 0.680875
rmse_yaw    = 0.017006
feature_std = 0.013559

task-aligned ego_motion:
mae_forward = 0.510476
mae_yaw     = 0.012843
rmse_forward= 0.588023
rmse_yaw    = 0.017957
feature_std = 0.014376
```

怎么解读：

```text
1. task-aligned ego_motion 并没有崩掉，
   它的冻结特征同样能被线性读出运动信息。
2. 前向位移误差基本打平，
   甚至 rmse_forward 更低。
3. 偏航角误差略差一些，
   说明它对不同运动分量的泛化还不均衡。
4. 这说明：
   我们已经不只是“让 temporal loss 好看”，
   而是真的把表征往动态任务方向推了一步。
5. 但它还没有形成一个非常明显、全面领先的结果，
   所以离“研究结论站稳”还差最后一轮针对泛化的优化。
```

### 18.78 KITTI dataset 改成固定 T_s 过滤

这一步是为了把数据入口和动力学假设对齐。

之前的状态是：

```text
1. 代码已经读取了真实时间戳。
2. 运动 target 也已经用到了真实 dt。
3. 但样本仍然是按相邻索引构造，
   不是按“固定采样时间”筛出来的。
```

这和我们的理论存在一个不完全一致的地方：

```text
模型更新里假设的是统一固定 T_s，
但 dataset 侧实际允许每个样本用自己的 dt。
```

所以这次代码改动把 KITTI temporal dataset 改成了：

```text
按相机 timestamps.txt 读取真实帧时间
只保留满足 |dt - T_s| <= tol 的样本
multi-horizon 样本要求每一步都满足这个条件
```

改动文件：

```text
/home/lin/predify2021_targetflow/predify2021/mce_scores/kitti_pairs.py
/home/lin/predify2021_targetflow/predify2021/mce_scores/train_kitti_targetflow_adjacent_pairs.py
/home/lin/predify2021_targetflow/predify2021/mce_scores/calculate_kitti_targetflow_pair_smoke.py
/home/lin/predify2021_targetflow/predify2021/mce_scores/calculate_kitti_ego_motion_linear_probe.py
```

新增配置：

```text
PREDIFY_FIXED_TS_S
PREDIFY_FIXED_TS_TOL_S
```

当前默认值：

```text
PREDIFY_FIXED_TS_S     = 0.1035
PREDIFY_FIXED_TS_TOL_S = 0.001
```

为什么不是 0.1000：

```text
因为 KITTI image_02 的实际相机相邻帧时间戳，
在当前两条 drive 上更接近 0.1035s，
而不是严格 0.1000s。
```

当前本地数据统计：

```text
drive_0005 image_02:
min dt  = 0.102944
max dt  = 0.103549
mean dt = 0.103266

drive_0011 image_02:
min dt  = 0.103338
max dt  = 0.103721
mean dt = 0.103558
```

因此先把固定 T_s 设成 0.1035，
是当前最贴近真实相机采样周期的选择。

最小 smoke 验证：

```text
PREDIFY_FIXED_TS_S=0.1035
PREDIFY_FIXED_TS_TOL_S=0.001
PREDIFY_TEMPORAL_HORIZONS=1,2
PREDIFY_MAX_PAIRS=4
```

输出：

```text
Time filter stats:
fixed_dt_s      = 0.1035
dt_tolerance_s  = 0.001
max_horizon     = 2
candidate_samples = 152
valid_samples     = 152
dropped_samples   = 0
subset_samples    = 4
```

阶段性结论：

```text
1. 现在 dataset 已经不再是“纯相邻索引配对”，
   而是“固定 T_s 约束下的时间样本”。
2. 这使得 targetflow / ego-motion 这条线，
   在数据层面更符合离散动力学假设。
3. 下一步所有正式实验，
   都应该默认带上这两个 fixed-T_s 参数。
```

### 18.79 固定 T_s 版本：正式 cross-drive 训练与验证

在固定时间样本版本上，正式重跑：

```text
train drive: 2011_09_26_drive_0005_sync
val drive:   2011_09_26_drive_0011_sync
fixed_ts_s:  0.1035
tol_s:       0.001
target:      ego_motion
top_target_source: ema_teacher
epochs: 10
```

输出文件：

```text
/home/lin/predify/kitti_targetflow_cross_drive_ego_motion_fixedts_ema_teacher_e10.p
/home/lin/predify/kitti_targetflow_cross_drive_ego_motion_fixedts_ema_teacher_e10_student.pt
/home/lin/predify/kitti_targetflow_cross_drive_ego_motion_fixedts_ema_teacher_e10_teacher.pt
```

时间过滤统计：

```text
train:
candidate_samples = 153
valid_samples     = 153
dropped_samples   = 0

val:
candidate_samples = 232
valid_samples     = 232
dropped_samples   = 0
```

说明：

```text
1. 这次 fixed-T_s 并没有删掉样本，
   因为当前两条 drive 的 image_02 时间戳本来就很稳定。
2. 但它依然是重要改动，
   因为现在 motion target 和时序假设已经严格绑定到相机时间上，
   不再混用原来较不稳定的时间差定义。
```

第 10 epoch 结果：

```text
train_weighted    = 0.139908
train_objective   = 0.139908
train_top         = 0.013075
train_top_std     = 0.639556
train_top_poolstd = 0.065180
train_temploss    = 0.091413
train_tempmae     = 0.222111
train_tempcos     = 0.805095

val_weighted      = 0.125108
val_objective     = 0.125108
val_top           = 0.016712
val_top_std       = 0.446090
val_top_poolstd   = 0.012338
val_temploss      = 0.223447
val_tempmae       = 0.242490
val_tempcos       = 0.721982
```

#### 18.79.1 temporal head 直接预测 motion

直接评估 temporal head 对验证集 motion target 的预测误差：

```text
mae_forward_m     = 0.471743
mae_yaw_rad       = 0.016335
rmse_forward_m    = 0.669352
rmse_yaw_rad      = 0.017905
num_samples       = 232
```

与之前非 fixed-T_s 版本对比：

```text
non-fixed-T_s head:
mae_forward = 0.458817
mae_yaw     = 0.011114
rmse_forward= 0.532593
rmse_yaw    = 0.012147

fixed-T_s head:
mae_forward = 0.471743
mae_yaw     = 0.016335
rmse_forward= 0.669352
rmse_yaw    = 0.017905
```

解读：

```text
1. fixed-T_s 之后，head 直接预测能力略有下降。
2. 这说明之前那版结果里，
   很可能混入了 variable-dt 带来的“额外可利用信息”。
3. fixed-T_s 版本更严格，也更诚实，
   但任务本身确实变难了。
```

#### 18.79.2 frozen-feature ego-motion probe

probe 输出：

```text
/home/lin/predify/kitti_ego_motion_probe_task_aligned_ego_motion_fixedts_e10.p
```

验证集结果：

```text
mae_delta_forward_m  = 0.552725
mae_delta_yaw_rad    = 0.047920
rmse_delta_forward_m = 0.784428
rmse_delta_yaw_rad   = 0.210020
val_feature_std_global = 0.013420
```

与之前 non-fixed-T_s probe 对比：

```text
non-fixed-T_s probe:
mae_forward = 0.510476
mae_yaw     = 0.012843
rmse_forward= 0.588023
rmse_yaw    = 0.017957
feature_std = 0.014376

fixed-T_s probe:
mae_forward = 0.552725
mae_yaw     = 0.047920
rmse_forward= 0.784428
rmse_yaw    = 0.210020
feature_std = 0.013420
```

怎么解读：

```text
1. fixed-T_s 版本下，冻结特征上的 linear probe 明显更难了，
   尤其 yaw 方向退化很大。
2. 但 temporal head 直接预测并没有同等幅度地崩掉，
   说明“头部任务能力”和“可线性读出的通用表征能力”并不是一回事。
3. 这反而给了我们一个更清晰的研究信号：
   现在模型学到的是较 task-specific 的 motion fitting，
   但还没有稳定地沉淀成更好的通用时序表征。
4. 因而下一步的关键，
   已经不是再争论 fixed-T_s 要不要做，
   而是如何让 fixed-T_s 条件下学到的动态信息，
   更稳定地进入顶层表征本身。
```

### 18.80 普通 VGG motion head baseline：fixed-T_s

为了判断当前 targetflow 机制到底有没有额外价值，
补了一组不带 targetflow 的标准监督 baseline：

```text
pretrained VGG16
+ frozen backbone
+ 一个普通 motion regression head
+ 直接监督 delta_forward_m / delta_yaw_rad
+ 同样使用 fixed-T_s dataset
```

脚本：

```text
/home/lin/predify2021_targetflow/predify2021/mce_scores/train_kitti_vgg_motion_baseline.py
```

正式设置：

```text
train drive: 2011_09_26_drive_0005_sync
val drive:   2011_09_26_drive_0011_sync
fixed_ts_s:  0.1035
tol_s:       0.001
pretrained:  True
freeze_backbone: True
epochs: 10
batchsize: 16
lr: 1e-4
```

输出文件：

```text
/home/lin/predify/kitti_vgg_motion_baseline_fixedts_frozen_e10.p
/home/lin/predify/kitti_vgg_motion_baseline_fixedts_frozen_e10_model.pt
```

第 10 epoch 验证结果：

```text
val_mae_forward_m  = 0.474006
val_mae_yaw_rad    = 0.004460
val_rmse_forward_m = 0.513257
val_rmse_yaw_rad   = 0.004832
```

与 fixed-T_s targetflow 对比：

```text
targetflow fixed-T_s head:
mae_forward = 0.471743
mae_yaw     = 0.016335
rmse_forward= 0.669352
rmse_yaw    = 0.017905

plain VGG motion head baseline:
mae_forward = 0.474006
mae_yaw     = 0.004460
rmse_forward= 0.513257
rmse_yaw    = 0.004832
```

阶段性结论：

```text
1. 在当前 fixed-T_s 设置下，
   普通 VGG motion head baseline 并不弱，
   甚至在 yaw 和 rmse 指标上明显优于当前 targetflow 版本。
2. 这说明：
   现在的 targetflow 线虽然“理论更强”，
   但还没有在直接 motion prediction 上打赢一个简单而合理的监督 baseline。
3. 这个结果很重要，
   因为它把问题定位清楚了：
   不是模型不能跑，
   也不是时序任务没定义出来，
   而是“当前机制的额外复杂度”还没有换来足够的性能收益。
4. 因而从研究推进角度看，
   下一步最关键的已经不是继续加功能，
   而是补最后一个反证对照：shuffled-pair control，
   然后再决定是继续优化 targetflow，
   还是收缩问题定义。
```

### 18.81 shuffled-pair control：fixed-T_s plain VGG baseline

为了检验当前任务到底有没有真正依赖时间配对，
补了一组反证对照：

```text
训练时保持 current frame 不变，
但把 future branch / motion target 用同一数据集中的其他样本打乱替换。
验证时仍然在真实配对上测试。
```

实现方式：

```text
ShuffledFuturePairDataset
/home/lin/predify2021_targetflow/predify2021/mce_scores/kitti_pairs.py

训练脚本：
/home/lin/predify2021_targetflow/predify2021/mce_scores/train_kitti_vgg_motion_baseline.py
```

正式设置：

```text
pretrained VGG16
frozen backbone
fixed_ts_s = 0.1035
tol_s      = 0.001
shuffle_train_pairs = True
shuffle_val_pairs   = False
shuffle_seed        = 0
```

输出文件：

```text
/home/lin/predify/kitti_vgg_motion_baseline_fixedts_frozen_shuffletrain_e10.p
/home/lin/predify/kitti_vgg_motion_baseline_fixedts_frozen_shuffletrain_e10_model.pt
```

第 10 epoch 验证结果：

```text
val_mae_forward_m  = 0.467648
val_mae_yaw_rad    = 0.001261
val_rmse_forward_m = 0.507469
val_rmse_yaw_rad   = 0.001510
```

和真实配对 baseline 对比：

```text
plain VGG baseline, real pairs:
mae_forward = 0.474006
mae_yaw     = 0.004460
rmse_forward= 0.513257
rmse_yaw    = 0.004832

plain VGG baseline, shuffled train pairs:
mae_forward = 0.467648
mae_yaw     = 0.001261
rmse_forward= 0.507469
rmse_yaw    = 0.001510
```

这是目前最重要的发现之一：

```text
1. 打乱时间配对之后，baseline 并没有明显崩掉。
2. 这说明当前这个 task 定义里，
   存在很强的“静态可猜偏置”或“均值可猜结构”。
3. 更直白地说：
   模型未必真的需要理解 current->future 的时序关系，
   也能在验证集上拿到看起来不差的 motion 指标。
4. 因而到这一步可以明确判断：
   现在的问题已经不再是模型能不能跑，
   而是当前 target / metric 设计还不足以强迫模型学到真正的时序因果结构。
```

这对研究方向的含义是：

```text
1. 继续在当前 target 上堆更复杂的 targetflow，
   很可能不会自动带来更可信的结论。
2. 下一步最合理的重点，
   应该转向“把任务本身变得更真正依赖时序”：
   - 更严格的 motion-aware target
   - 更长时间间隔或多步位姿变化
   - 去除可由单帧静态外观直接猜出的偏置
3. 也就是说，
   现在最该优化的已经不是 backbone，
   甚至不一定先是 targetflow 结构，
   而是 task definition 本身。
```

### 18.81 shuffled-pair control

为了验证模型是否真的在利用时间配对结构，
补了一组负对照：

```text
保持 current frame 不变
但把 future frame / motion target
替换成同一数据集里另一条样本的 future branch
```

也就是：

```text
输入还是 x_t
但监督不再是它真正的 t+1
而是别人的 t+1
```

如果模型真的依赖时序关系，
这种打乱应该让性能明显下降。

#### 18.81.1 plain VGG motion head：shuffled train pairs

设置：

```text
pretrained VGG16
frozen backbone
fixed-T_s
shuffle_train_pairs = True
shuffle_val_pairs   = False
```

输出文件：

```text
/home/lin/predify/kitti_vgg_motion_baseline_fixedts_frozen_shuffletrain_e10.p
/home/lin/predify/kitti_vgg_motion_baseline_fixedts_frozen_shuffletrain_e10_model.pt
```

第 10 epoch 验证结果：

```text
val_mae_forward_m  = 0.468602
val_mae_yaw_rad    = 0.001274
val_rmse_forward_m = 0.506285
val_rmse_yaw_rad   = 0.001638
```

与 non-shuffled baseline 对比：

```text
non-shuffled baseline:
mae_forward = 0.474006
mae_yaw     = 0.004460
rmse_forward= 0.513257
rmse_yaw    = 0.004832

shuffled-train baseline:
mae_forward = 0.468602
mae_yaw     = 0.001274
rmse_forward= 0.506285
rmse_yaw    = 0.001638
```

这个结果非常关键：

```text
1. 打乱时间配对后，baseline 几乎没有变差，
   甚至数值上还略好一些。
2. 这说明当前任务设置下，
   单帧外观本身就已经泄露了大量 motion 信息。
3. 换句话说：
   现在这个 cross-drive 任务并没有强到“必须依赖真实时序关系”。
4. 因而如果我们继续只盯着这个任务上的回归数值，
   很可能会高估模型的时序建模能力。
```

#### 18.81.2 targetflow shuffled smoke

由于正式 GPU 版 targetflow shuffled 任务在自动审批时被技术性中断，
先补了一组更安全的 smoke：

```text
train_pairs = 32
val_pairs   = 32
epochs      = 2
fixed-T_s
对比 shuffle_train_pairs = False vs True
```

结果：

```text
non-shuffled smoke epoch 2:
val_weighted = 0.523084
val_temploss = 0.701585
val_tempmae  = 0.618560

shuffled-train smoke epoch 2:
val_weighted = 0.502098
val_temploss = 0.679727
val_tempmae  = 0.600804
```

解读：

```text
1. 在这个很小的 smoke 上，
   shuffled 并没有出现预期中的明显崩塌。
2. 这和上面的 VGG baseline 结果是一致的：
   当前任务里，时序依赖并没有被充分“逼出来”。
3. 因而目前最重要的研究结论已经非常明确：
   我们不是“还没把模型调好”这么简单，
   而是“当前任务本身对真实时序关系不够敏感”。
```

阶段性总结：

```text
1. fixed-T_s 已经补齐。
2. plain VGG baseline 已经补齐。
3. shuffled-pair control 也已经给出了强信号：
   当前数据任务允许模型只靠单帧外观就做得不错。
4. 所以下一步最合理的，不再是继续堆 targetflow 细节，
   而是开始重审任务定义本身：
   怎样构造一个“必须依赖真实时序关系”的目标。
```

### 18.81 shuffled-pair control：fixed-T_s targetflow

为了检验模型到底是不是在利用真实时间关系，
补了一组 negative control：

```text
保持 current frame 不变
但把它对应的 future frame / motion target 固定乱配
训练集乱配，验证集保持真实时间配对
```

实现方式：

```text
/home/lin/predify2021_targetflow/predify2021/mce_scores/kitti_pairs.py
新增 ShuffledFuturePairDataset
```

训练脚本也新增了：

```text
PREDIFY_SHUFFLE_TRAIN_PAIRS=1
PREDIFY_SHUFFLE_VAL_PAIRS=0
PREDIFY_SHUFFLE_SEED=0
```

正式设置：

```text
fixed_ts_s: 0.1035
tol_s: 0.001
task_aligned_target: ego_motion
top_target_source: ema_teacher
train: shuffled pairs
val: proper pairs
epochs: 10
```

输出文件：

```text
/home/lin/predify/kitti_targetflow_cross_drive_ego_motion_fixedts_shuffletrain_e10.p
/home/lin/predify/kitti_targetflow_cross_drive_ego_motion_fixedts_shuffletrain_e10_student.pt
/home/lin/predify/kitti_targetflow_cross_drive_ego_motion_fixedts_shuffletrain_e10_teacher.pt
```

第 10 epoch 结果：

```text
train_weighted    = 0.049802
train_objective   = 0.049802
train_top         = 0.000306
train_top_std     = 0.105699
train_top_poolstd = 0.009928
train_temploss    = 0.117183
train_tempmae     = 0.243672
train_tempcos     = -0.573710

val_weighted      = 0.051978
val_objective     = 0.051978
val_top           = 0.000245
val_top_std       = 0.105037
val_top_poolstd   = 0.010045
val_temploss      = 0.252861
val_tempmae       = 0.251724
val_tempcos       = -0.645120
```

与正常 fixed-T_s targetflow 对比：

```text
normal fixed-T_s targetflow:
val_weighted = 0.125108
val_temploss = 0.223447
val_tempmae  = 0.242490
val_tempcos  = 0.721982

shuffle-train fixed-T_s targetflow:
val_weighted = 0.051978
val_temploss = 0.252861
val_tempmae  = 0.251724
val_tempcos  = -0.645120
```

怎么解读：

```text
1. 把时间关系打乱以后，
   top/local 这类 loss 甚至还能继续变小，
   说明这些量本身不等于“学到了正确时序关系”。
2. 但 temporal cosine 在验证集上从正的 0.721982
   直接掉到负的 -0.645120，
   这是一个非常强的反证信号。
3. 它说明：
   当前模型的 temporal 分支并不是随便都能学，
   而是真正在依赖“正确的前后时序配对”。
4. 所以到这一步为止，
   我们已经可以较有把握地说：
   这个模型确实在利用时序结构，
   而不是只靠静态场景统计或损失函数巧合。
5. 但另一面也很清楚：
   即便它确实在用时序，
   目前还没有在 fixed-T_s 条件下，
   打赢简单的 plain VGG motion baseline。
```

#### 18.81.3 targetflow shuffled：train+val 全乱配正式收口

为了把 shuffled-pair control 彻底收口，
又补了一组更“严格但也更偏负对照”的正式运行：

```text
train: shuffled pairs
val:   shuffled pairs
fixed_ts_s = 0.1035
tol_s      = 0.001
task_aligned_target = ego_motion
top_target_source   = ema_teacher
epochs = 10
```

输出文件：

```text
/home/lin/predify/kitti_targetflow_cross_drive_ego_motion_fixedts_shuffled_ema_teacher_e10.p
/home/lin/predify/kitti_targetflow_cross_drive_ego_motion_fixedts_shuffled_ema_teacher_e10_student.pt
/home/lin/predify/kitti_targetflow_cross_drive_ego_motion_fixedts_shuffled_ema_teacher_e10_teacher.pt
```

第 10 epoch 结果：

```text
train_weighted    = 0.052345
train_top         = 0.000332
train_top_std     = 0.108601
train_temploss    = 0.136890
train_tempmae     = 0.269384
train_tempcos     = -0.870460

val_weighted      = 0.054030
val_top           = 0.000566
val_top_std       = 0.108970
val_temploss      = 0.271401
val_tempmae       = 0.280120
val_tempcos       = -0.658474
```

这组结果的意义是：

```text
1. 当 train 和 val 都被乱配以后，
   top/local 这类损失仍然能非常小，
   说明“loss 变小”本身并不等于“学到了真实时间关系”。
2. 因而真正更可信的 negative control，
   仍然是上一节那个：
   train shuffled, val proper pairs。
3. 那组结果里 temporal cosine 从正值翻到负值，
   才是最关键的证据：
   模型的 temporal 分支确实依赖真实前后配对。
4. 而这一次 train+val 全乱配的正式结果，
   则补上了另一面：
   如果监督本身就是错的，
   模型依然可以把一部分表面 loss 做得很好。
```

### 18.82 当前阶段收口

到这里，这一阶段其实已经不是“永远做不完”，而是已经可以明确收口了。

目前已经完成的事：

```text
1. Predify / targetflow 入口已经能稳定跑。
2. KITTI 数据已经改成固定 T_s 过滤。
3. dynamic error / targetflow / ego-motion 训练链路都已打通。
4. plain VGG baseline 已经补齐。
5. shuffled-pair control 已经补齐，而且给出了明确结论。
```

当前最重要的阶段结论：

```text
1. 模型改动不是“没写进去”。
   temporal 分支已经真的在利用时序配对。
2. 但当前任务定义仍然太容易被静态外观或数据偏置投机。
3. 所以现在的瓶颈已经不是“模型能不能跑”，
   而是“任务是否足够强，能逼着模型学真正有用的时序信息”。
```

因此下一步不应该再无上限地继续堆对照，
而应该只保留一条主线：

```text
做一个更强的 task-aligned temporal target，
让模型必须依赖真实前后时序，才能把目标做好。
```

### 18.83 把显式动力学误差正式写进五层 pvgg_tf 主模型

这一步的目的不是再加一个外部实验脚本，
而是把下面这条误差动力学：

```text
epsilon_n^{k+1}
= (T_s / tau_n) * (e_n^k - d_n^k)
+ (1 - K_n T_s / tau_n) * epsilon_n^k
```

直接写进 `pvgg_tf` 的五层主架构，
让它成为统一的层状态更新规则。

#### 18.83.1 本次代码改动

改动文件：

```text
/home/lin/predify2021_targetflow/predify2021/model_factory/targetflow/core.py
/home/lin/predify2021_targetflow/predify2021/model_factory/targetflow/__init__.py
/home/lin/predify2021_targetflow/predify2021/model_factory/pvgg16_targetflow.py
/home/lin/predify2021_targetflow/predify2021/model_factory/get_model.py
/home/lin/predify2021_targetflow/predify2021/mce_scores/train_kitti_targetflow_adjacent_pairs.py
/home/lin/predify2021_targetflow/predify2021/mce_scores/calculate_kitti_targetflow_pair_smoke.py
```

核心结构变化：

```text
1. 新增 TargetFlowDynamicErrorConfig
   用来统一保存:
   - sample_time = T_s
   - time_constant = tau_n
   - error_gain = K_n

2. 扩充 TargetFlowLayerState
   每层现在不再只存瞬时误差，
   而是显式保存:
   - instant_error
   - previous_error
   - error (= epsilon_state)

3. 新增 build_targetflow_instant_error
   定义瞬时误差 e_n - d_n

4. 新增 build_dynamic_targetflow_error
   按离散动力学公式递推 epsilon_n

5. pvgg_tf 五层统一走同一套更新:
   - 先 backward target flow 生成 d_n
   - 再算 instant_error = e_n - d_n
   - 再用上一时刻 epsilon_n 递推出新的 epsilon_n
   - 再用 epsilon_n 生成 learn_signal 和 local_loss

6. 模型内部新增 error_state_memory
   step_frame 时不 reset，
   因此 epsilon 状态可以跨帧保留
```

现在 `T_s/tau/K` 不再只是外面的实验参数，
而是已经进入模型构造器：

```text
dynamic_error
error_sample_time
error_time_constant
error_gain
```

#### 18.83.2 最小 smoke：模型可实例化、可前向、可更新 epsilon

运行设置：

```text
KITTI drive: 2011_09_26_drive_0005_sync
task_aligned_target = ego_motion
fixed_ts_s = 0.1035
error_sample_time = 0.1035
error_time_constant = (0.1, 0.1, 0.1, 0.1, 0.1)
error_gain = (1.0, 1.0, 1.0, 1.0, 1.0)
dynamic_error = True
pretrained = False
pairs = 4
```

结果：

```text
Pairs: 4
Mean top local loss: 0.000317
Mean top error rms: 0.017805
Mean top instant error rms: 0.017203
Mean top grad rms: 0.000001
Mean total local loss: 0.095845
Mean temporal loss: 0.074612
Mean temporal mae: 0.203872
Mean temporal cosine: -0.970627
```

这里最关键的不是数值高低，
而是：

```text
top instant error rms != top error rms
```

这说明现在的 `error` 已经不是简单的瞬时差值，
而是经过动力学递推后的 `epsilon_state`。

#### 18.83.3 五层证据：不是只改顶层

单次前向时，5 层都同时产生了各自的瞬时误差与 epsilon：

```text
layer=1 instant_rms=0.173248 epsilon_rms=0.179312
layer=2 instant_rms=0.143767 epsilon_rms=0.148799
layer=3 instant_rms=0.110125 epsilon_rms=0.113979
layer=4 instant_rms=0.080924 epsilon_rms=0.083757
layer=5 instant_rms=0.012935 epsilon_rms=0.013388
```

这说明这次不是“只改一层”或“只改顶层 temporal head”，
而是 5 个 VGG stage 全部接上了新的误差动力学。

#### 18.83.4 跨帧证据：epsilon 会被模型记住

连续两次 `step_frame`，不 reset：

```text
step=1
layer=1 prev_rms=0.000000 epsilon_rms=0.198708
layer=2 prev_rms=0.000000 epsilon_rms=0.183138
layer=3 prev_rms=0.000000 epsilon_rms=0.159821
layer=4 prev_rms=0.000000 epsilon_rms=0.119865
layer=5 prev_rms=0.000000 epsilon_rms=0.019328

step=2
layer=1 prev_rms=0.198708 epsilon_rms=0.191762
layer=2 prev_rms=0.183138 epsilon_rms=0.178428
layer=3 prev_rms=0.159821 epsilon_rms=0.155187
layer=4 prev_rms=0.119865 epsilon_rms=0.116216
layer=5 prev_rms=0.019328 epsilon_rms=0.020470
```

这条证据很重要：

```text
第二步的 previous_error 已经等于第一步的 epsilon
```

所以现在可以明确说：

```text
动力学误差已经进入模型内部状态，
而不是每次前向都重新现算一个静态 error。
```

#### 18.83.5 这一步完成到什么程度

到这里为止，已经完成的是：

```text
1. 显式误差动力学已经写进 pvgg_tf 主模型
2. 五层都接上了这套更新
3. T_s 已经既用于数据筛选，也用于模型内部误差递推
4. step_frame 已经可以跨帧保留 epsilon 状态
```

还没有完成的是：

```text
1. 把这套新动力学重新跑完整 formal 训练
2. 把 W_b ≈ W_f^{-1} 的结构假设正式编码进去
3. 重新验证新主模型在 task-aligned target 上的表现
```

### 18.84 动态误差主模型：formal cross-drive 训练结果

在 18.83 把显式动力学写进五层 `pvgg_tf` 主模型之后，
正式重跑一版 cross-drive KITTI 训练：

```text
train drive: 2011_09_26_drive_0005_sync
val drive:   2011_09_26_drive_0011_sync
target:      ego_motion
fixed_ts_s:  0.1035
tol_s:       0.001
dynamic_error = True
error_sample_time = 0.1035
error_time_constant = (0.1, 0.1, 0.1, 0.1, 0.1)
error_gain = (1.0, 1.0, 1.0, 1.0, 1.0)
top_target_source = ema_teacher
epochs = 10
pretrained = True
```

输出文件：

```text
/home/lin/predify/kitti_targetflow_cross_drive_ego_motion_fixedts_dynerr_e10.p
/home/lin/predify/kitti_targetflow_cross_drive_ego_motion_fixedts_dynerr_e10_student.pt
/home/lin/predify/kitti_targetflow_cross_drive_ego_motion_fixedts_dynerr_e10_teacher.pt
```

第 10 epoch 结果：

```text
train_weighted    = 0.049329
train_top         = 0.006695
train_top_std     = 0.409476
train_temploss    = 0.087062
train_tempmae     = 0.211570
train_tempcos     = 0.915628

val_weighted      = 0.049902
val_top           = 0.005753
val_top_std       = 0.222324
val_temploss      = 0.220986
val_tempmae       = 0.244000
val_tempcos       = 0.717777
```

和之前 fixed-T_s、但还没有显式动态误差的版本对比：

```text
previous fixed-T_s targetflow:
val_weighted = 0.125108
val_temploss = 0.223447
val_tempmae  = 0.242490
val_tempcos  = 0.721982

dynamic-error fixed-T_s targetflow:
val_weighted = 0.049902
val_temploss = 0.220986
val_tempmae  = 0.244000
val_tempcos  = 0.717777
```

第一层解读：

```text
1. 加入显式动态误差以后，
   local / weighted loss 显著下降。
2. temporal cosine 仍保持在较高正值附近，
   说明模型没有失去时序对齐能力。
3. 这证明：
   这版新主模型不仅能跑通，
   还能够被正式训练。
```

#### 18.84.1 temporal head 直接预测 motion

直接评估 student checkpoint 的 temporal head：

```text
mae_forward_m     = 0.469517
mae_yaw_rad       = 0.019677
rmse_forward_m    = 0.665571
rmse_yaw_rad      = 0.020068
num_samples       = 232
```

与之前 fixed-T_s targetflow 对比：

```text
previous fixed-T_s targetflow head:
mae_forward = 0.471743
mae_yaw     = 0.016335
rmse_forward= 0.669352
rmse_yaw    = 0.017905

dynamic-error fixed-T_s targetflow head:
mae_forward = 0.469517
mae_yaw     = 0.019677
rmse_forward= 0.665571
rmse_yaw    = 0.020068
```

第二层解读：

```text
1. dynamic error 后，forward 方向略有改善。
2. 但 yaw 方向没有同步改善，甚至略有变差。
3. 这说明：
   显式动力学已经成功进入主模型，
   但它带来的第一版收益更偏向“优化与状态建模方式变化”，
   还没有转化成稳定、全面的 motion 指标提升。
```

#### 18.84.2 当前阶段结论

```text
1. 到这一步可以明确说：
   显式动力学误差已经正式写进五层主模型，
   并且已经完成 formal training。
2. 现在已经不再是“只有 smoke 证据”，
   而是有正式 cross-drive 训练结果。
3. 但从性能上看，
   这版 dynamic-error 主模型还没有形成决定性优势。
4. 因而后续如果继续推进，
   重点不该再是“有没有写进去”，
   而是“怎样把这套动力学变成真正有用的任务收益”。
```

#### 18.84.3 dynamic-error 主模型：shuffled-pair control 正式收口

为了验证这版 dynamic-error 主模型是否仍然真的在利用时序，
我们补了最关键的反证对照：

```text
train: shuffled pairs
val:   proper adjacent pairs
```

命令配置核心如下：

```text
train drive: 2011_09_26_drive_0005_sync
val drive:   2011_09_26_drive_0011_sync
target:      ego_motion
fixed_ts_s:  0.1035
tol_s:       0.001
dynamic_error = True
error_sample_time = 0.1035
error_time_constant = (0.1, 0.1, 0.1, 0.1, 0.1)
error_gain = (1.0, 1.0, 1.0, 1.0, 1.0)
top_target_source = ema_teacher
shuffle_train_pairs = True
shuffle_val_pairs   = False
shuffle_seed        = 0
epochs = 10
pretrained = True
```

输出文件：

```text
/home/lin/predify/kitti_targetflow_cross_drive_ego_motion_fixedts_dynerr_shuffletrain_e10.p
/home/lin/predify/kitti_targetflow_cross_drive_ego_motion_fixedts_dynerr_shuffletrain_e10_student.pt
/home/lin/predify/kitti_targetflow_cross_drive_ego_motion_fixedts_dynerr_shuffletrain_e10_teacher.pt
```

第 10 epoch 结果：

```text
train_weighted    = 0.038928
train_top         = 0.000403
train_top_std     = 0.090273
train_temploss    = 0.125128
train_tempmae     = 0.251858
train_tempcos     = -0.873287

val_weighted      = 0.040895
val_top           = 0.000304
val_top_std       = 0.087596
val_temploss      = 0.261809
val_tempmae       = 0.264013
val_tempcos       = -0.620335
```

与正常 dynamic-error 训练对比：

```text
dynamic-error proper-pair training:
val_weighted = 0.049902
val_temploss = 0.220986
val_tempmae  = 0.244000
val_tempcos  = 0.717777

dynamic-error shuffled-train control:
val_weighted = 0.040895
val_temploss = 0.261809
val_tempmae  = 0.264013
val_tempcos  = -0.620335
```

这组结果的正确解读：

```text
1. shuffled 之后，temporal cosine 从 +0.717777 掉到 -0.620335，
   说明模型失去了正确的时序方向对齐。
2. temporal loss 和 temporal MAE 都变差，
   说明它学不到真正的下一时刻运动目标。
3. 虽然 weighted/local loss 反而更低，
   但这不是更好，
   而是模型在乱配对数据上收缩到了一个更“保守”的表示，
   顶层方差也明显更小（0.222324 -> 0.087596）。
4. 因而这组 shuffled-pair control 给出了我们需要的核心证据：
   dynamic-error 新主模型下，
   “确实利用时序结构”这件事仍然成立。
```
