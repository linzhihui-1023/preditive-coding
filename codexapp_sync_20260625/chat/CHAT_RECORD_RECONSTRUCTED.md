# Chat Record Reconstructed

说明：

这不是系统原始逐字聊天导出，
而是根据本次长期协作过程重建的对话纪要。

它保留了：

1. 你提出的关键目标
2. 我们做过的主要修改
3. 每个阶段为什么转向
4. 最终形成的实验结论

---

## 阶段 1：复现 Predify 论文与 alpha / beta / lambda 路线

最开始，你的目标是：

1. 确认源码可用
2. 基于论文公式做 ablation
3. 对比 `with_alpha` 和 `without_alpha`
4. 在噪声环境下看准确率变化

这个阶段做过的事情包括：

1. 跑 PVGG16 的 ImageNet / ImageNet-C 路线
2. 下载并定位 feedback weights
3. 处理 CUDA、依赖、OOM、运行过慢等问题
4. 增加更快的 Gaussian-noise 验证入口
5. 做 `alpha = 0, 0.01, 0.05, 0.1` 等 sweep
6. 做去掉 `alpha`、去掉 forward term、去掉 feedback term 的对照

这一阶段的核心结论是：

1. 原论文里的提升不是“巨大到肉眼一看就非常夸张”的那种。
2. 在我们这套复现实验里，某些项的去掉并没有带来你一开始预期的巨大差异。
3. 你因此开始怀疑：真正值得继续的方向不是只做旧公式的局部 ablation，
   而是往你自己的动力学结构走。

---

## 阶段 2：从分类复现转向时序 / 动力学

接下来，你明确提出新的研究兴趣是：

1. 时序
2. 误差累计
3. 固定真实时间
4. 把动力学应用到模型上

这时候我们做了两件很重要的判断：

1. `predify_mnist` 被你抛弃了，不再作为主线。
2. 真正还值得继续的主线，是 `predify` 核心库 + 新的 `predify2021_targetflow` 外壳。

你也提出了后面一直保持不变的几个基本要求：

1. 时间步长要真实、明确、可控
2. 误差机制要从原来的即时误差改成带动力学累积的误差
3. 最终要证明模型确实利用了时序结构

---

## 阶段 3：引入 KITTI，转到视频时序场景

因为分类任务不适合你的研究目标，
我们开始转向 KITTI Raw。

这个阶段做过：

1. 下载 KITTI 小 drive
2. 写 stream 评测脚本
3. 在 KITTI 上跑逐帧预测与内部状态统计
4. 记录每帧推理时间、稳定性、PCoder error 等

你反复强调的关键点是：

1. “固定时间”不是 timesteps 数量，而是真实 wall-clock 对应的采样间隔
2. 每一步更新都应该对应明确的 `T_s`

这也是后面 fixed-`T_s` 路线的来源。

---

## 阶段 4：把 dynamic error 正式写进 Predify 路线

之后我们开始修改误差机制本身。

你给出了新的误差动力学思路，核心形式接近：

```text
epsilon^{k+1}
= (T_s / tau) * (e - d)
+ (1 - K T_s / tau) * epsilon^k
```

在这个阶段，我们做了两类工作：

1. 先在旧的 PVGG / KITTI stream 路线上引入 dynamic error 开关
2. 再把这套显式 dynamic error 正式写进新的五层 targetflow 主模型

你当时最关心的并不是“有没有立刻提升很多性能”，
而是：

1. 改动是不是真的进模型了
2. 每层是不是都改了
3. 这是不是一个真正的新主模型，而不是局部补丁

后面的验证基本都围绕这 3 个问题展开。

---

## 阶段 5：targetflow 主模型与新架构

你给出了一张更接近 target flow / learn flow 的图，
目标不再是只改某个项，而是逐渐把整个模型推向新的结构。

我们当时分析过：

1. 直接重写全部模型风险太大
2. 更合理的是保留现有外壳，逐步把 targetflow / learn flow / dynamic error 写进去

因此，最终形成了一条新的 worktree 主线：

`/home/lin/predify2021_targetflow`

其中最关键的代码文件包括：

1. `model_factory/targetflow/core.py`
2. `model_factory/pvgg16_targetflow.py`
3. `mce_scores/train_kitti_targetflow_adjacent_pairs.py`
4. `mce_scores/kitti_pairs.py`
5. `mce_scores/calculate_kitti_targetflow_pair_smoke.py`

---

## 阶段 6：固定真实时间间隔 fixed T_s

你的一个基本要求是：

1. 模型理论上依赖固定采样时间
2. 所以实验数据也应该尽量符合统一固定的 `T_s`

因此我们后来不再满足于“每个样本用自己的时间差”，
而是增加了：

1. `PREDIFY_FIXED_TS_S`
2. `PREDIFY_FIXED_TS_TOL_S`

并在 KITTI pair 构造中筛选更接近统一采样时间的相邻帧，
让训练假设和动力学公式更一致。

---

## 阶段 7：从 smoke 到 formal training

等新主模型能实例化并前向跑通以后，
我们没有停在 smoke test，
而是继续做了正式 cross-drive training。

这时候的路线已经比较清晰：

1. train drive 用 `2011_09_26_drive_0005_sync`
2. val drive 用 `2011_09_26_drive_0011_sync`
3. temporal target 逐步演化到更 task-aligned 的 `ego_motion`
4. top target 先以 `ema_teacher` 为主

结果说明：

1. 新主模型不是只能前向跑
2. 它是能训练的
3. temporal 指标是有可读性的

---

## 阶段 8：证明“模型真的利用了时序”

这是后期最重要的一步。

你的要求已经从“模型能跑”变成：

1. 模型要能学到时序信息
2. 模型要能根据前面信息之间的误差去预测下一时刻的信息
3. 要有证据，不只是直觉

于是我们补了最关键的 negative control：

### A. plain VGG baseline shuffled-pair

作为最简单对照，
说明普通冻结特征 + 线性头在时序被破坏时会退化。

### B. targetflow shuffled-pair

说明 targetflow 路线不是凭空成立。

### C. dynamic-error 主模型 shuffled-pair control

这是最后最关键的一组。

最终结果是：

1. 正常 proper-pair 训练时，
   `val_tempcos = 0.717777`
2. shuffled-train control 时，
   `val_tempcos = -0.620335`

这说明：

1. dynamic-error 主模型确实依赖时序关系
2. 一旦时序对齐被打乱，它就失去正确的时序目标学习能力

---

## 阶段 9：当前已经完成到什么程度

到目前为止，可以明确说：

1. 显式 `dynamic error` 已经正式写进五层主模型
2. KITTI fixed-`T_s` 时序训练已经跑通
3. cross-drive 训练已经完成
4. shuffled-pair control 已经证明模型确实利用时序

但也要诚实说明：

1. `W_b ≈ W_f^{-1}` 这个假设还没有正式落地到最终主模型
2. 图里所有模块虽然大体进入主线，但还不是最终定稿版研究系统
3. 目前最强的证据是“时序确实被利用了”
4. 还不是“已经拿到特别漂亮、无可争议的任务收益”

---

## 阶段 10：这次迁移包为什么要整理

你现在希望迁移到 `Codex app` 去继续总结和推进。

所以这个包的目的就是：

1. 把代码改动集中起来
2. 把关键实验结果集中起来
3. 把长期聊天里的关键决策转成文档
4. 避免下一次协作的人重新从零梳理

---

## 建议下一步在 Codex app 里先看什么

1. `docs/HANDOFF_SUMMARY.md`
2. `docs/README_DYNAMIC_ERROR_KITTI.md`
3. `results/kitti_targetflow_cross_drive_ego_motion_fixedts_dynerr_e10.p`
4. `results/kitti_targetflow_cross_drive_ego_motion_fixedts_dynerr_shuffletrain_e10.p`
5. `code/predify2021_targetflow/model_factory/pvgg16_targetflow.py`
6. `code/predify2021_targetflow/model_factory/targetflow/core.py`

这样最快能接上当前主线。
