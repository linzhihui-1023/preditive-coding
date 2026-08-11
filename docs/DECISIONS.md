# Research Decisions

## 2026-08-11: Make next-frame feature prediction the primary task

Status: accepted

The 2-DoF longitudinal-yaw motion target is no longer a primary result. It is
retained only as a completed proxy-task diagnostic because the two-drive split
is dominated by motion-regime mismatch and does not establish a temporal-state
benefit.

The primary research objective is now to predict the next-frame feature,
improve feature consistency in a continuous video stream, resist image noise
and blur, and support online adaptation after an environmental change. The
first experiment under this objective keeps the existing Target Flow residual
and dynamic state unchanged, replaces the motion head target with the next-
frame feature, and retests whether inherited state helps. Error design changes
are considered only if state still provides no benefit on the feature task.

## 2026-08-11: Define the Target Flow residual without overclaiming

Status: accepted

At layer `l`, the implemented instantaneous residual is
`r_t^l = F_t^l - T_t^l`. At the top layer,
`T_t^L = stopgrad(F_(t+1)^L)`. In recursive mode, lower targets are produced by
the learned feedback chain, `T_t^l = D_l(T_t^(l+1))`. Thus `r_t^l` contains
information about the difference between the current representation and a
future-frame-guided target.

This residual is not the strict temporal feature-prediction error
`F_(t+1)^l - Fhat_(t+1|t)^l`. Reports must call it a Target Flow residual or a
future-guided inter-frame representation residual, not the error of the
next-feature predictor.

The current dynamic state remains
`epsilon_t^l = (Ts/tau) r_t^l + (1 - K Ts/tau) epsilon_(t-1)^l`. This is the
implemented recurrence; there is no independent `d_t` term. The existing
stability condition remains required.

## 2026-08-11: Preserve the causal feature-prediction order

Status: accepted

The prediction for `t -> t+1` is formed before `I_(t+1)` is used to construct
the current Target Flow target or residual. Its legal inputs are the current
feature `F_t` and state completed by the preceding transition, including
`epsilon_(t-1)`. It must not consume `r_t` or `epsilon_t`, because both require
the arrival of `I_(t+1)`.

The causal loop is:

```text
F_t + epsilon_(t-1)
    -> Fhat_(t+1|t)
    -> observe F_(t+1)
    -> r_t
    -> epsilon_t
    -> Fhat_(t+2|t+1)
```

Previous prediction memories may also be consumed only when they were
completed and detached before the current prediction. Future features are
valid supervision after prediction, never prediction context for the same
transition.

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

## 2026-08-10: Train the feedback decoder

Status: accepted

The feedback modules are learned components of recursive Target Flow and are
included in the same Adam optimizer as the temporal predictor. Forward stages
join that optimizer only in the explicit backbone-adaptation ablation. Feedback
modules must not retain gradients while being omitted from optimizer parameter
groups. The EMA teacher continues to track their learned parameters.

## 2026-08-10: Reserve temporal variance for backbone adaptation

Status: accepted

Batch variance is invalid when stream execution enforces batch size 1. The
optional backbone-adaptation regularizer therefore computes per-channel
variance over a sequence-local window of pooled top features. Stored previous-
frame features are detached, so the current frame receives a gradient without
retaining a graph across optimizer steps. The window resets whenever model
state resets and defaults to 16 frames.

The numerical guard requires `target_std > sqrt(eps)` whenever the variance
weight is positive. The defaults are `target_std=0.01` and `eps=1e-6`; the old
`eps=1e-4` made the hinge identically zero. Frozen-backbone mechanism controls
fix the weight at zero because top-feature variance has no gradient path to a
trainable module. A positive weight is allowed only with
`PREDIFY_TRAIN_BACKBONE=1`.

## 2026-08-10: Reset at every fixed-dt discontinuity

Status: accepted

Filtering invalid timestamp transitions creates separate contiguous segments,
not one sparse sequence. Stream loaders construct one sequence record per
segment. Student state, EMA-teacher state, and the temporal variance window
reset before the first frame of every segment, so a state computed at one side
of a rejected time step is never applied at the other side.

## 2026-08-10: Standardize the 2-DoF motion target

Status: accepted

The task target is `[forward displacement m, yaw change rad]`; it is not full
ego-motion because lateral translation is absent. New reports call it the
2-DoF longitudinal-yaw motion target. The internal `ego_motion` identifier is
retained only for configuration compatibility.

Mean and population standard deviation are estimated independently for each
component and prediction horizon using training segments only. Training and
best-checkpoint selection use standardized MSE. Predictions are transformed
back to physical units for separate forward-displacement MAE in metres and yaw
MAE in radians. Mixed-unit aggregate MSE and raw-vector cosine are not primary
physical performance claims.

## 2026-08-10: Separate duplicated current-top information from history

Status: accepted

An inherited previous top target contains the current image's top
representation. Therefore, the earlier inherited-versus-reset comparison does
not isolate temporal memory. With frozen VGG, teacher and student top features
are identical. The corrected reset-each-frame control duplicates the detached
current student top feature directly into the top previous-prediction slot;
dynamic-error and lower prediction history remain zero. The old 15.7%
difference is treated as confounded evidence and is not attributed to long-
term temporal memory.

## 2026-08-10: Use one-step online temporal credit assignment

Status: accepted

Cross-frame error and prediction memories are detached before reuse. The
method is stateful forward recurrence with one-step gradients, not BPTT. This
keeps the online predictive-coding interpretation and bounds graph memory.
Reports must not claim that gradients are propagated through a sequence or
that long-range dependencies are learned by cross-frame backpropagation.

## 2026-08-10: Freeze the pretrained VGG backbone by default

Status: accepted

The primary mechanism experiments freeze all VGG forward stages and train the
feedback decoders plus temporal predictor. This gives a clear frozen-backbone
predictive-coding method. Backbone fine-tuning is retained only as an explicit
adaptation ablation through `PREDIFY_TRAIN_BACKBONE=1`; it must be labelled as
a different training regime.

## 2026-08-10: Enforce stable dynamic-error parameters

Status: accepted

The implemented quantities are the instantaneous target error
`e_t = F_t - T_t` and the dynamic state
`epsilon_t = (Ts/tau)e_t + (1 - K Ts/tau)epsilon_(t-1)`. There is no separate
`d_t` term. Every layer must satisfy `abs(1 - K Ts/tau) < 1`; invalid or
non-finite parameters fail during model construction rather than allowing an
unstable recurrence to run.

## 2026-08-11: Fix the formal frozen-backbone control matrix

Status: accepted

The five formal conditions are inherited EMA (A), reset EMA (B), reset EMA
with a detached current-top duplicate (C), inherited instantaneous error (D),
and inherited lag-1 error (E). A versus C is the primary state-memory test; B
versus C checks the duplicated current feature; A versus D tests dynamic error;
and A versus E tests recursive memory against finite one-step memory.

All conditions fix recursive target flow, frozen VGG, instantaneous local
loss, zero variance weight, temporal prediction weight one, dynamic
parameters, initialization seed, drives, normalization statistics, and
checkpoint criterion. Formal execution requires explicit disjoint train and
validation drives through `PREDIFY_FORMAL_SPLIT=1`.

Because the frozen student and teacher forward stages are identical,
`TOP_TARGET_SOURCE=ema_teacher` is equivalent to a detached student top target
in these experiments. EMA teacher behavior is not claimed as a mechanism of
the frozen-backbone results.

## 2026-08-11: Isolate formal runs from the login shell

Status: accepted

The canonical five-group runner executes each process through `env -i` and
explicitly supplies all data, model, state, optimizer, timing, seed, checkpoint,
and compatibility variables. In particular, it sets the legacy
`PREDIFY_CURRENT_TEACHER_CONTEXT=0` in every group and explicitly sets the new
current-top duplicate switch per group. Every result stores the full Git
revision. A GitHub Actions workflow supplies an independent unit-test status
for the exact revision used by the matrix.

## 2026-08-11: Add a separate causal future-feature head

Status: accepted

The 2-DoF motion head remains intact for reproducibility. The primary task is
selected with `PREDIFY_TASK=future_feature` and uses a separate stage-5
residual predictor:

`Fhat_(t+1|t) = F_t + P_theta(F_t, H_t)`.

The learned current-only, instant, lag-1, and recursive conditions share the
same predictor architecture and parameter count. Only `H_t` changes. A
copy-current condition bypasses the predictor and reports `Fhat_(t+1|t)=F_t`.
The old current-top-duplicate control belongs only to the motion experiment and
is invalid in the future-feature matrix.

Future-frame features are resolved only after prediction. Instant, lag-1, and
recursive history inputs are snapshots produced by the previous completed
transition; the residual formed after observing the current pair target is
stored only for the next prediction. This is causal stateful recurrence with
one-step gradients, not BPTT.

Future-feature checkpoint selection uses validation feature MSE. Reports also
include feature cosine, normalized feature error, and copy-current metrics.
Future-feature MSE and residual-delta MSE are computed together and must remain
numerically equal.
