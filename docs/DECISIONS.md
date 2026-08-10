# Research Decisions

## 2026-08-10: Use a real video stream

Status: accepted

The active paradigm processes each real video frame once. The first frame
initializes five layer states. Later frames inherit the previous five layer
states, predictions, and dynamic errors. State is reset only at a true sequence
boundary.

Training and validation must preserve drive and frame order. Stream mode uses
batch size 1 and does not permit shuffled frame pairs.

## 2026-08-10: Cancel repeated timesteps on one image

Status: accepted

The old design that ran one static image through multiple model timesteps is no
longer an active research route. It must not be used as the implementation or
interpretation of continuous video state.

## 2026-08-10: Separate mechanism validation from generalization

Status: accepted

The two downloaded KITTI drives are used first for tightly controlled
comparisons. More drives are downloaded only after state inheritance and
dynamic error show a stable advantage over their controls. Two-drive results
are preliminary mechanism evidence, not a broad generalization claim.

## 2026-08-10: Keep research history durable

Status: accepted

Every meaningful code update should include a user-readable explanation and,
when relevant, updates to `PROJECT_STATE.md` or `EXPERIMENT_LOG.md`. Every
reported experiment should identify the Git commit, data split, configuration,
best epoch, metrics, and artifact location.

## 2026-08-10: Enforce causal temporal prediction

Status: accepted

The temporal predictor for transition `t -> t+1` may consume the current
forward feature `F_t` and memories completed at `t-1`. It must not consume an
error or target that requires `I_{t+1}`. Future-frame teacher features remain
valid detached supervision for local and temporal losses only after the causal
prediction has been formed.

The temporal prediction loss has a positive default weight of 1.0. Setting it
to zero is an explicit ablation and emits a warning because the temporal head
then receives no training signal.

## 2026-08-10: Make temporal controls reproducible

Status: accepted

Formal temporal comparisons use an explicit random seed and select the student
checkpoint with the lowest mean validation temporal loss. Final-epoch student
and teacher checkpoints remain available for diagnostics, but they are not
substituted for the selected checkpoint in result tables.

The core control matrix consists of inherited state with dynamic error,
reset-each-frame with dynamic error, and inherited state with instantaneous
error. Every condition processes each real frame exactly once and preserves
drive order. A reset-each-frame condition clears every recurrent memory,
including dynamic-error history, so it is interpreted as a control for the
complete cross-frame state mechanism.
