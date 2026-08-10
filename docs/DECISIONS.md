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

The first seeded A/B/C matrix is superseded because its filtered-error loss
changed current gradient scale and because reset-each-frame changed that loss
state. Its results remain in the experiment log but are not treated as clean
mechanism evidence.

## 2026-08-10: Separate error memory from local optimization

Status: accepted

The cross-frame error state and the error used for local optimization are
independent choices. Formal memory controls use the instantaneous error for all
five local losses, so they have the same current-frame loss, gradient
coefficient, learning rate, and target-flow architecture.

Error-state conditions are `instant`, recursive EMA, and lag-1 mixing. Lag-1
uses `alpha*e_t + (1-K*alpha)*e_(t-1)` and, for the formal `K=1` setting,
provides a one-step memory baseline with the same coefficients and
constant-signal scale as recursive EMA. Using the filtered state directly in
local loss is retained only as an explicitly named legacy ablation.

## 2026-08-10: Use a five-layer future target chain

Status: accepted

The active target-flow mode is `recursive`. The detached future-frame top
feature defines `T5`, then feedback modules propagate targets through
`T5 -> T4 -> T3 -> T2 -> T1`. The previous `quasi_steady` mode used current
higher-layer forward outputs below the top layer and therefore must not be
described as five-layer future Target Flow. It remains available only as an
explicit ablation.
