# Research Decisions

## 2026-08-13: Separate prediction stage from the Target Flow top

Status: accepted

Future-feature prediction may target VGG Stage 3, 4, or 5 through
`PREDIFY_FUTURE_FEATURE_STAGE`, but the Predify Target Flow top remains Stage
5. The future frame therefore supplies two logically separate targets:
`T_TF=F_(t+1)^5` for recursive Target Flow and
`T_future=F_(t+1)^s_pred` for the future-prediction loss. They are resolved
after the current prediction and can be extracted together in one frozen-VGG
pass.

The Future Predictor and optional fusion module derive their channels from the
selected prediction stage. Cross-frame spatial memory stores detached
`F_t^s_pred` and is named as prediction-stage memory, not top memory. Motion
radius remains configurable in prediction-stage feature cells and is not
silently rescaled when the stage changes.

All checkpoints and replays must record both stages. Absolute Stage-4 and
Stage-5 MSE values are not comparable because they belong to different feature
spaces. A method passes only by improving MSE, cosine, and normalized error
against Copy-current computed on the same stage and same frame stream.

## 2026-08-12: Do not use the persistence score as a 2-DoF degradation trigger

Status: accepted

Gate 3 at revision `80c4aee` froze both models and the detector definition.
Gate 2's score remained the unmodified eight-frame mean
`cos(e_t,e_(t-1))`, higher for more persistence. Its result hashes were locked,
and all 2,400 Persistent/Shuffled detector rows reproduced exactly. The task
model was the formal Group A seed-0 2-DoF epoch-6 checkpoint; signed forward
and yaw MAE degradation used same-frame independently reset clean trajectories.

On held-out drive 0011, forward score-versus-degradation Pearson was
`-0.001291`, Spearman was `0.057782`, and score AUROC for `D>0` was `0.500651`.
Persistent and Shuffled mean forward degradation was only `+0.000411 m` and
`+0.000335 m` relative to clean (`+0.126%` and `+0.103%`). Yaw MAE improved in
79 of 80 windows, so the one-positive-window yaw AUROC is not useful evidence.

Gate 2 therefore detects temporal organization in prediction error, but that
score is not a reliable task-degradation trigger for the tested 2-DoF model.
Do not proceed directly to a controller driven by this score. A future trigger
must be validated against a stronger downstream task with meaningful clean
performance before online adaptation is enabled.

The conclusion is task- and checkpoint-specific. The existing 2-DoF model has
known cross-drive weaknesses, especially for yaw, so this does not establish
that persistence is unrelated to every downstream task.

## 2026-08-12: Pass the matched-marginal persistence gate

Status: accepted

Revision `99b7e21` isolated temporal organization by giving persistent and
shuffled blur exactly the same kernel, four-value sigma multiset, per-value
counts, disturbed-frame occupancy, raw video frames, and frozen checkpoint.
Four counterbalanced replicates also made every absolute frame see every sigma
once per condition. No optimizer was created and no network parameter changed.

Drive 0005 selected higher `cos(e_t,e_(t-1))`. With metric and direction
frozen, drive 0011 reached primary nonoverlapping-eight-frame-window AUROC
`0.9725`. Excluding the onset window gave `0.972222`; all four held-out
replicates remained high (`0.95` to `1.00`). This passes the user-defined
`go_promising` threshold and permits selective-online-adaptation mechanism
design on `predify-selective-adaptation-v2`.

This is a mechanistic go/no-go, not a population-level performance claim.
Only two drives were available, and windows within a drive reuse the same
video content. Do not report the 160 windows as independent drives or attach a
naive confidence interval. Preserve the frozen `predify-temporal-v1` baseline
and keep all adaptation changes on the new branch.

## 2026-08-12: Pass the first error-separability gate, not the persistence gate

Status: accepted

The inference-only go/no-go at revision `a28fed5` used the existing strict
Temporal Error checkpoint without updating any network. Calibration drive 0005
selected low `||e_t||` as the score for persistent blur. With that direction
and statistic frozen, held-out drive 0011 reached AUROC 0.959609 against pooled
clean and i.i.d.-Gaussian-noise controls. This exceeds the user-defined 0.8
`go_promising` threshold.

The result justifies continuing prediction-error-based shift detection as a
small diagnostic direction. It does not yet justify a controller. Blur reduced
error norm, and raw instantaneous norm outperformed the temporal statistics on
the calibration drive. More importantly, the persistent positive uses blur
while the random negative uses Gaussian noise, so corruption type and temporal
persistence are confounded.

The next required gate is persistent blur versus an i.i.d. or temporally
permuted blur control with matched marginal severity. Keep the same clean
control, score directions, 40/80/30 schedule, frozen checkpoint, and no network
updates. A controller or online adaptation mechanism remains downstream of
that matched-persistence test.

## 2026-08-12: Keep warp transport, pause residual generalization claims

Status: accepted

The formal stage-5 matrix at revision `79de554` confirmed that causal
historical warp beats Copy-current on both tested drives, but very unevenly:
6.646% on training drive 0005 and only 0.219% on held-out drive 0011. This
retains causal spatial transport as a useful model component while keeping the
held-out evidence explicitly weak.

The learned post-warp residual improved train MSE by 10.747% relative to Copy,
but its best validation checkpoint was 2.408% worse than Copy and 2.633% worse
than warp-only. Training continued to improve while validation degraded. The
tested residual head therefore fails the current cross-drive generalization
gate. This is not a claim that motion-compensated residuals are intrinsically
unlearnable.

Do not add Temporal Error to the warp-residual predictor or tune either error
time constant yet. The next change must target held-out residual
generalization, preferably by increasing training-drive coverage or by a
predeclared capacity/regularization diagnostic. Keep the deterministic causal
warp as a separate baseline in every follow-up.

## 2026-08-12: Treat same-drive corruption as a mechanistic null

Status: accepted

The independent same-drive controlled-corruption experiment completed at
revision `ae90a9f`. On the clean 45-transition validation segment,
Current-only remained 3.62918% worse than Copy-current and Temporal Error was
0.25853% worse than Current-only. Temporal Error changed signed excess AUEC
relative to Current-only by +0.22189% for step bias, +0.22728% for ramp bias,
and -1.00328% for i.i.d. noise. This mixed pattern does not show selective
adaptation to persistent systematic disturbance.

State diagnostics show a normally scaled, actively used state rather than an
ignored input. `RMS(E)/RMS(F)` is approximately 0.104--0.143, the first-layer
history/feature weight RMS ratio is 1.01452, and history contributes roughly
0.166--0.227 of predicted-delta RMS. However, zeroing history in the same
Temporal Error checkpoint improves phase-mean feature MSE for every trajectory
phase. The current failure is therefore not explained by vanishing state
scale; on this trace, the learned use of `E` is harmful.

This one-seed, short same-drive result is retained as a mechanistic null. It
does not establish broad robustness or cross-drive generalization and does not
justify tuning `tau_e`. Work returns to predictor sufficiency and
generalization; Temporal Error seeds and time-constant sweeps remain paused
until Current-only beats Copy-current.

## 2026-08-12: Keep Temporal Error tuning behind the failed predictor gate

Status: accepted

The first three-group strict Temporal Prediction Error matrix completed at
revision `3ffbff0`. Validation feature MSE was `0.060080099` for Copy-current,
`0.061797074` for Current-only, and `0.061903913` for Temporal Error.
Current-only was 2.85781% worse than Copy-current, so the predictor gate failed.
Temporal Error was 0.17289% worse than Current-only, so the state-benefit gate
also failed.

This result does not justify tuning `tau_e`, running seeds 1 and 2, or claiming
that prediction-error memory is broadly ineffective. The Temporal Error
comparison is not decisive while Current-only fails its prerequisite. The
same-drive controlled-corruption experiment may proceed as a mechanistic test
of `e -> E -> L` transients, but it cannot replace the failed cross-drive gate
or support a broad robustness claim.

## 2026-08-12: Add top-layer Temporal Prediction Error state

Status: accepted

Future-feature history now gets a separate top-layer Temporal Prediction Error
state. It is not the Target Flow residual. The strict prediction error is
`e_t^5 = F_t^5 - Fhat_(t|t-1)^5`, computed only after the prediction made at
`t-1` and the frame-`t` top feature are both available. The carried state is
`E_t^5 = alpha_e e_t^5 + (1 - K_e alpha_e) E_(t-1)^5`.

The first implementation tests only the top layer because the active task is
`F_t^5 -> F_(t+1)^5`. It adds a distinct `temporal_error` history mode for the
future-feature predictor and leaves `recursive` with its historical meaning:
recursive Target Flow residual memory. Reports must not merge these two
mechanisms.

Temporal-error parameters are separate from Target Flow parameters:
`PREDIFY_TEMPORAL_ERROR_TS`, `PREDIFY_TEMPORAL_ERROR_TAU`, and
`PREDIFY_TEMPORAL_ERROR_GAIN`. The first version uses the same numerical values
as the formal Target Flow runs, `Ts=0.1035`, `tau=0.5`, and `K=1`, but tuning
one family must not silently change the other.

## 2026-08-12: Require predictor sufficiency before judging state

Status: accepted

Current-only must beat Copy-current on held-out ordered video before the
history matrix is used to judge inherited state. Diagnostics must first report
the true and predicted feature-change scale, norm ratio, and delta direction
on both train and validation data. A training-only gain with a validation
reversal is treated as cross-drive overfitting, not evidence that history is
useless.

The first 3x3 spatial-predictor diagnostic increases late-epoch training
capacity but still fails the held-out Copy-current gate. Its best validation
checkpoint predicts a smaller delta and has worse delta cosine than 1x1, so
its MSE being closer to Copy-current must not be described as improved spatial
motion prediction. Therefore `tau` tuning, history reruns, seeds 1 and 2, and
robustness experiments remain paused. This null result applies only to the
tested top residual history; it is not a conclusion about prediction-state
memory or lower-layer feedback-decoder state.

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

Error-state conditions are `instant`, recursive EMA, and two-tap mixing. Two-tap
uses `alpha*e_t + (1-K*alpha)*e_(t-1)` and, for the formal `K=1` setting,
provides a finite two-tap baseline with the same coefficients and
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

The implemented Target Flow quantity is the residual
`r_t = F_t - T_t` and its dynamic state
`epsilon_t = (Ts_r/tau_r)r_t + (1 - K_r Ts_r/tau_r)epsilon_(t-1)`. There is no separate
`d_t` term. Every layer must satisfy `abs(1 - K Ts/tau) < 1`; invalid or
non-finite parameters fail during model construction rather than allowing an
unstable recurrence to run.

## 2026-08-11: Fix the formal frozen-backbone control matrix

Status: accepted

The five formal conditions are inherited EMA (A), reset EMA (B), reset EMA
with a detached current-top duplicate (C), inherited instantaneous error (D),
and inherited two-tap error (E). A versus C is the primary state-memory test; B
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

The learned current-only, latest, two-tap, and recursive conditions share the
same predictor architecture and parameter count. Only `H_t` changes. A
copy-current condition bypasses the predictor and reports `Fhat_(t+1|t)=F_t`.
The old current-top-duplicate control belongs only to the motion experiment and
is invalid in the future-feature matrix.

Future-frame features are resolved only after prediction. Latest, two-tap, and
recursive history inputs are snapshots produced by the previous completed
transition; the residual formed after observing the current pair target is
stored only for the next prediction. This is causal stateful recurrence with
one-step gradients, not BPTT.

Future-feature checkpoint selection uses validation feature MSE. Reports also
include feature cosine, normalized feature error, and copy-current metrics.
Future-feature MSE and residual-delta MSE are computed together and must remain
numerically equal.

The strict predictor error is stored with the theory-consistent sign
`prediction_error_top = F_(t+1) - Fhat_(t+1|t)`. This differs from the Target
Flow residual sign and must not be silently interchanged with it.

At prediction time, `latest` is `e_(t-1)`. The finite condition is named
`two_tap`, because it is
`alpha e_(t-1) + (1-K alpha)e_(t-2)`, not a pure lag-1 residual. `instant` and
`lag1` remain accepted only as legacy configuration aliases for `latest` and
`two_tap` respectively.

The frozen-backbone feature matrix uses `TOP_TARGET_SOURCE=student_self` and
`TEMPORAL_TARGET_MODE=next_top`. A frozen EMA teacher would produce the same
top target and add no independent mechanism, so it is omitted from this
matrix.

## 2026-08-11: Pause feature seeds after the first matrix

Status: accepted

At revision `94059be`, copy-current validation feature MSE was `0.060080099`,
while current-only was 2.858% worse at `0.061797074`. Recursive history reached
`0.061796151`, only 0.00149% below current-only. Latest and two-tap were also
worse than copy-current. Thus the predictor failed the first gate and inherited
history provided no meaningful improvement once `F_t` was known.

Seeds 1 and 2, corruption robustness, and online-adaptation experiments remain
paused. The next work returns to mechanism diagnosis: inspect next-feature
delta scale and predictor behavior, then revise the residual/history design as
planned. Additional seeds are not used to rescue a failed primary comparison.

## 2026-08-12: Isolate same-drive controlled corruption from training

Status: accepted

Controlled step-bias, ramp-bias, and i.i.d.-noise trajectories live in a
separate evaluation module. They are independent experiments, not phases of
one composite trajectory, and model state resets before every clean and
corrupted stream. Training remains clean and unchanged except for an explicit
same-drive split option. The split is defined over raw frame positions: the
first 60% is training data, the next 20 raw frames form a gap, and the next
contiguous 20% is validation. Samples are included only if every raw frame they
read lies inside one region, and train/validation raw-frame sets must be
disjoint.

Corruption is applied after resize and center crop in unnormalized `[0,1]` RGB
space, before ImageNet normalization. Persistent systematic bias contains only
fixed RGB bias. Per-frame Gaussian noise is a separate negative control with no
bias. Its deterministic realization is keyed by experiment seed, drive,
camera, and absolute frame name, so re-reading an absolute frame as a future
image and then as the next current image returns the identical tensor.

Evaluation runs clean and corrupted counterfactual streams in frame order and
saves each frame's strict prediction error `e`, accumulated Temporal Error
state `E`, feature MSE `L`, and signed
`delta L=L_corrupted-L_clean`. Signed excess and its signed/absolute integrals
are primary; negative values are retained to expose improvement, overshoot, or
state overcompensation. Positive-part excess, raw Peak Error, and raw AUEC are
secondary. Recovery is censored when signed excess does not remain within its
configured absolute threshold for the required consecutive frames. The
evaluator also reports `RMS(F_t)`, `RMS(E_t)`, predictor input-weight scale, and
`P(F_t,E_t)-P(F_t,0)` to distinguish an ignored or very small state from a
normally scaled but uninformative state. These results test controlled
within-drive response only; they are not evidence of cross-drive
generalization.

## 2026-08-12: Limit the first Temporal Error matrix to three groups

Status: accepted

The first formal matrix contains only Copy-current, Current-only, and
top-layer Temporal Error. Its two gates are:

1. Current-only validation MSE must beat Copy-current.
2. Temporal Error validation MSE must beat Current-only.

Latest, two-tap, and Target-Flow recursive residual histories remain available
for historical checkpoint compatibility, but are not accepted by the formal
future-feature matrix runner and are not part of this first controlled-
corruption comparison.
