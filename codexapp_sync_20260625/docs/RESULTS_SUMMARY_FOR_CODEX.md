# Results Summary For Codex

这份文件是给 `Codex App` 看的轻量结果摘要。

目的不是保留所有实验原始二进制，
而是让后续总结代码的人立刻知道：

1. 我们改了什么
2. 哪些实验最关键
3. 当前最应该围绕哪些结果来讲

---

## 1. 当前最重要的主线

当前真正的主线不是早期的 ImageNet-C alpha ablation，
而是：

```text
KITTI
+ fixed Ts
+ dynamic error
+ targetflow main model
+ cross-drive evaluation
+ shuffled-pair control
```

---

## 2. 最重要的正式结果

### dynamic-error proper training

```text
val_weighted = 0.049902
val_temploss = 0.220986
val_tempmae  = 0.244000
val_tempcos  = 0.717777
```

解释：

```text
1. 模型能训练。
2. temporal cosine 为显著正值。
3. 说明模型在正常相邻帧配对下，
   学到了合理的时序方向对齐。
```

### dynamic-error shuffled-train control

```text
val_weighted = 0.040895
val_temploss = 0.261809
val_tempmae  = 0.264013
val_tempcos  = -0.620335
```

解释：

```text
1. 把训练 pair 打乱后，
   temporal cosine 从正值掉到负值。
2. temporal loss 和 temporal MAE 也变差。
3. 这说明模型确实依赖真实时序关系，
   而不是仅仅在记静态图像特征。
```

---

## 3. 当前最强结论

```text
显式 dynamic error 已经正式写进五层主模型，
并且在 fixed-Ts 的 KITTI cross-drive 设置下，
已经证明模型确实利用时序结构。
```

---

## 4. 还没完成的内容

```text
1. W_b ≈ W_f^{-1} 这个假设还没正式落地。
2. 还没有把整套研究叙事打磨成“任务收益特别漂亮”的论文结果。
3. 现在最稳的说法是：
   模型改动已落地，
   时序利用已被证明，
   但收益强化仍需继续推进。
```

---

## 5. 推荐 Codex 总结提问

把这个目录交给 Codex App 之后，
最推荐直接问：

```text
请基于当前目录，总结：
1. 新主模型相对原始 Predify 改了哪些模块；
2. dynamic error 是如何进入五层模型的；
3. fixed Ts 在数据构造和训练中是怎么落地的；
4. shuffled-pair control 为什么能证明模型确实利用了时序；
5. 当前离“最终研究模型”还差哪些部分。
```
