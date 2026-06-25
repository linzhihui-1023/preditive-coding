# Handoff Summary

## 项目目标

当前项目的核心目标有三条：

1. 在原有 `Predify / PVGG16` 路线上，把新的动力学误差机制正式写进模型。
2. 让模型进入时序数据场景，重点围绕 KITTI 相邻帧学习时间结构。
3. 证明模型不仅“能跑”，还“确实利用了时序信息来预测下一时刻目标”。

## 已完成内容

### 1. `predify` 核心侧

已经在 `predify` 核心库中做过两类重要改动：

1. 为 `PCoder` 路径做过可控 ablation，支持把 `alpha / gradient` 路径关闭。
2. 支持 stream / frame-by-frame 的状态更新方式，便于后续时序实验。

### 2. `predify2021_targetflow` 新主模型侧

已经在新 worktree 中搭出并推进了新的 targetflow 主线：

1. 新建 `pvgg_tf / targetflow` 路线，不再只依赖旧的共享 `PCoder` 表达。
2. 把 target flow 正式写入五层模型。
3. 在 target flow 基础上，把显式 `dynamic error` 状态更新写进五层主模型。
4. 训练脚本已经支持：
   - KITTI 相邻帧配对
   - cross-drive train / val
   - fixed `T_s`
   - task-aligned temporal target
   - dynamic-error 开关
   - shuffle pair control

### 3. 主要实验结论

#### A. dynamic-error 已经真实进入模型

我们已经证明：

1. 它不是只停留在公式层面。
2. 它不是只改了单层。
3. 它已经进入五层主模型的实际状态更新与训练流程。

#### B. 模型已经能在 KITTI 时序任务上跑通

已经完成：

1. pair smoke
2. cross-drive targetflow training
3. dynamic-error formal training

#### C. 模型确实在利用时序

最关键证据是：

1. 正常相邻帧训练时，dynamic-error 主模型在验证集上的 `temporal cosine` 为显著正值。
2. 把训练 pair 打乱后，`temporal cosine` 明显掉到负值。

这说明模型不是单纯在记静态图像特征，而是真的依赖时序配对关系。

## 当前最关键结果

### dynamic-error proper training

- `val_weighted = 0.049902`
- `val_temploss = 0.220986`
- `val_tempmae = 0.244000`
- `val_tempcos = 0.717777`

### dynamic-error shuffled-train control

- `val_weighted = 0.040895`
- `val_temploss = 0.261809`
- `val_tempmae = 0.264013`
- `val_tempcos = -0.620335`

## 如何解释这组结果

1. `weighted/local loss` 更低，不代表模型更好。
2. 在 shuffled 数据上，模型更可能收缩到一个更保守、更低方差的表示。
3. 真正说明“时序学到了没有”的关键指标，是 temporal 方向上的对齐质量。
4. 从 `+0.717777` 到 `-0.620335` 的变化，已经足够说明时序结构被破坏后，模型失去正确预测能力。

## 还没有完成的事

1. `W_b ≈ W_f^{-1}` 这个假设还没有正式落进最终主模型。
2. 图里所有模块虽然已经大体落成主线，但还不是最终定稿版研究系统。
3. 还没有完成一套适合论文汇报的“强任务收益”验证。
4. 下一阶段更合理的方向是：
   - 继续完善 task-aligned temporal target
   - 补更强的 control
   - 把结果整理成更适合论文叙事的结构

## 迁移时优先看的文件

1. `docs/README_DYNAMIC_ERROR_KITTI.md`
2. `chat/CHAT_RECORD_RECONSTRUCTED.md`
3. `code/predify2021_targetflow/model_factory/pvgg16_targetflow.py`
4. `code/predify2021_targetflow/model_factory/targetflow/core.py`
5. `code/predify2021_targetflow/mce_scores/train_kitti_targetflow_adjacent_pairs.py`
6. `results/kitti_targetflow_cross_drive_ego_motion_fixedts_dynerr_e10.p`
7. `results/kitti_targetflow_cross_drive_ego_motion_fixedts_dynerr_shuffletrain_e10.p`
