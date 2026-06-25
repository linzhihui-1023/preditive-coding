# Codex App Sync Package

这是一个专门为 `Git 同步 + Windows/Codex App 总结` 准备的轻量包。

## 为什么不用大包

完整汇总包 `codexapp_bundle_20260625/` 大约 34G，
其中包含大量 `.pt` checkpoint 和实验结果副本。

这不适合直接推到 Git，原因是：

1. 太大，上传和拉取都慢。
2. 很多二进制结果对代码总结并不是必须的。
3. Windows 端的 Codex 更需要的是：
   - 代码
   - 设计文档
   - 实验结论摘要
   - 关键结果数字

## 这个轻量包包含什么

- `docs/`
  - 项目主 README
  - handoff summary
  - 关键实验结果摘要
- `chat/`
  - 重建版聊天纪要
- `code/`
  - `predify` 核心改动
  - `predify2021_targetflow` 主模型关键源码
- `tools/`
  - 用于检查公式/误差机制的小工具

## 最适合在 Codex App 里先看的文件

1. `docs/HANDOFF_SUMMARY.md`
2. `docs/RESULTS_SUMMARY_FOR_CODEX.md`
3. `docs/README_DYNAMIC_ERROR_KITTI.md`
4. `chat/CHAT_RECORD_RECONSTRUCTED.md`
5. `code/predify2021_targetflow/model_factory/pvgg16_targetflow.py`
6. `code/predify2021_targetflow/model_factory/targetflow/core.py`

## 不包含什么

这个轻量包不包含大 checkpoint 和全量实验产物。

如果以后确实需要二进制结果，
再从本地大包或原目录单独拷贝即可。
