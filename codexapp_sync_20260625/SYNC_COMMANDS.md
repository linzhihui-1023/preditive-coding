# Sync Commands

下面这组命令是把当前轻量同步包提交到 Git，并准备推送到你自己的私有仓库。

## 1. 在当前仓库创建 handoff 分支并提交

```bash
cd /home/lin/predify

git checkout -b codexapp-sync-20260625

git add .gitignore
git add predify/modules/base.py
git add predify/networks/base.py
git add README_DYNAMIC_ERROR_KITTI.md
git add tools/check_dynamic_error_formula.py
git add codexapp_sync_20260625

git commit -m "handoff: codexapp sync package and dynamic-error summary"
```

## 2. 检查提交内容

```bash
git status
git log --oneline -n 3
```

## 3. 推到你自己的私有仓库

不要直接推到当前 `origin`，
因为当前 `origin` 指向的是上游公开仓库：

```text
https://github.com/miladmozafari/predify.git
```

推荐你新建一个自己的 private repository，
然后执行：

```bash
git remote add myprivate <YOUR_PRIVATE_REPO_URL>
git push -u myprivate codexapp-sync-20260625
```

如果你已经有自己的远程仓库，
也可以直接改成：

```bash
git push -u <YOUR_REMOTE_NAME> codexapp-sync-20260625
```

## 4. Windows / Codex App 端建议

在 Windows 上：

```bash
git clone <YOUR_PRIVATE_REPO_URL>
cd predify
git checkout codexapp-sync-20260625
```

然后把 Codex App 指到这个目录，
优先看：

1. `codexapp_sync_20260625/docs/HANDOFF_SUMMARY.md`
2. `codexapp_sync_20260625/docs/RESULTS_SUMMARY_FOR_CODEX.md`
3. `codexapp_sync_20260625/chat/CHAT_RECORD_RECONSTRUCTED.md`
4. `codexapp_sync_20260625/code/...`

## 5. 推荐直接问 Codex App 的问题

```text
请基于当前分支，总结：
1. 新主模型相对原始 Predify 改了哪些模块；
2. dynamic error 是如何进入五层模型的；
3. fixed Ts 在数据构造和训练中是怎么落地的；
4. shuffled-pair control 为什么能证明模型确实利用了时序；
5. 当前离最终研究模型还差哪些部分。
```
