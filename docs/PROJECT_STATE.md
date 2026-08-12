# Project State

Last updated: 2026-08-12

## Active research direction

The primary task is next-frame feature prediction, not the completed 2-DoF
motion proxy. The intended outcome is a video representation that is more
consistent across adjacent frames, robust to noise and blur, and able to adapt
online when the environment changes.

The active model operates on a real video-frame stream:

1. The first frame initializes five layer states.
2. Each later frame executes the model exactly once.
3. Each frame inherits the previous five layer states, predictions, and dynamic
   errors.
4. The current feature and state completed at the previous transition predict
   the next-frame feature.
5. Training and validation preserve drive and frame order. Stream mode requires
   batch size 1 and rejects shuffled pair loaders.

The earlier route that repeatedly ran multiple timesteps on the same image is
cancelled and is not part of the active experiment design.

The active experiment order is:

1. Keep the existing Target Flow residual and dynamic recurrence.
2. Use next-frame feature prediction as the primary task.
3. Add a separate, strict top-layer Temporal Prediction Error state without
   changing the historical Target Flow residual recurrence.
4. Test Copy-current, Current-only, and Temporal Error under matched causal
   controls before tuning the new error state.
5. Use same-drive controlled corruption only as a mechanistic transient test;
   broader robustness and online-adaptation claims require a viable held-out
   predictor first.

For transition `t -> t+1`, prediction must use `F_t` and history completed at
`t-1`, such as `epsilon_(t-1)`. The current residual `r_t` and dynamic state
`epsilon_t` are formed only after `I_(t+1)` arrives and are available for the
next transition.

## Current implementation

- `predify2021/model_factory/pvgg16_targetflow.py`
  - Implements five target-flow stages.
  - Defaults to recursive target flow: the detached future top feature is
    propagated through `T5 -> T4 -> T3 -> T2 -> T1`.
  - Separates the cross-frame error state (`instant`, `ema`, or `two_tap`) from the
    local-loss error source.
  - Defaults local optimization to instantaneous error so memory controls have
    identical current-frame local losses and gradient coefficients.
  - Keeps the original 2-output motion `temporal_predictor` and adds an
    independent full-stage-5 `future_feature_predictor` selected through
    `PREDIFY_TASK`.
  - Predicts a residual feature map with
    `Fhat_(t+1|t) = F_t + P(F_t, H_t)`.
  - Supports an explicit first-layer spatial kernel of 1 or 3 for the future
    predictor. The default and formal history matrix remain 1x1.
  - Uses the same future predictor for `none`, `latest`, `two_tap`, and
    `recursive` history conditions; only `H_t` changes. `copy_current` bypasses
    the predictor as a non-learned baseline.
  - Keeps per-layer error and prediction memories across `step_frame` calls.
  - Detaches every stored error and prediction state. Execution is stateful
    forward recurrence with one-step gradients, not BPTT.
  - Clears memories at sequence boundaries through `reset`.
  - Resolves the next-frame target through a deferred provider only after the
    future prediction is complete, then updates Target Flow residuals and
    history for the following transition.
  - Maintains causal top-layer latest, two-tap, and recursive history snapshots
    independently of the configured local-loss error state.
  - Adds an independent top-layer Temporal Prediction Error state for the
    future-feature task:
    `e_t^5 = F_t^5 - Fhat_(t|t-1)^5` and
    `E_t^5 = alpha_e e_t^5 + (1-K_e alpha_e)E_(t-1)^5`.
    The predictor can select this state through
    `PREDIFY_FEATURE_HISTORY_MODE=temporal_error`; the older `recursive` mode
    remains Target Flow residual memory.
- `predify2021/model_factory/targetflow/core.py`
  - Defines target-flow state, dynamic-error integration, local losses, and
    gradient diagnostics.
- `predify2021/mce_scores/kitti_pairs.py`
  - Loads adjacent KITTI frames, timestamps, and 2-DoF longitudinal-yaw
    targets `[forward displacement m, yaw change rad]`.
  - Splits fixed-dt-valid sample indices into contiguous segments whenever a
    raw timestamp transition is rejected.
  - Provides an explicit same-drive raw-frame split: first 60% train, 20-frame
    gap, then a contiguous 20% validation range. It verifies that the complete
    train and validation raw-frame sets are disjoint.
- `predify2021/mce_scores/train_kitti_targetflow_adjacent_pairs.py`
  - Defaults to stream mode.
  - Resets once per drive or sequence, then calls `step_frame` once per frame.
  - Treats each contiguous fixed-dt segment as a separate sequence, resetting
    student, teacher, and variance history before the next segment.
  - Requires `PREDIFY_BATCHSIZE=1` and disables shuffled pairs in stream mode.
  - Supports deterministic runs through `PREDIFY_SEED`.
  - Supports the `PREDIFY_RESET_EACH_FRAME=1` control without adding repeated
    model executions.
  - Supports a reset/no-history current-top duplicate control that copies only
    detached `F_student(I_t)` into the top prediction-context slot. Under the
    frozen backbone this equals `F_teacher(I_t)`.
  - Freezes pretrained VGG forward stages by default. Feedback decoders and the
    task-selected predictor train; `PREDIFY_TRAIN_BACKBONE=1` is an explicit
    adaptation ablation.
  - Records separate Target Flow error parameters and Temporal Prediction Error
    parameters so future `tau_e` changes do not alter Target Flow dynamics.
  - Supports clean same-drive checkpoint training through
    `PREDIFY_SAME_DRIVE_SPLIT=1`; controlled corruption remains outside this
    training loop.
  - Selects motion checkpoints by validation temporal loss and future-feature
    checkpoints by validation feature MSE.
  - Reports feature MSE, cosine, normalized feature error, the equivalent delta
    MSE, and matched copy-current metrics.
  - Computes optional collapse prevention across a sequence-local temporal
    window of pooled top features instead of across the batch dimension.
  - Disables that window completely when its weight is zero and clears it on
    every reset-each-frame step when it is enabled.
  - Detaches stored history so only the current frame receives variance
    gradients; the default window is 16 frames.
  - Estimates per-horizon motion mean and standard deviation from training
    segments only, optimizes standardized MSE, and reports forward MAE in
    metres and yaw MAE in radians separately.
- `predify2021/mce_scores/calculate_kitti_targetflow_pair_smoke.py`
  - Provides a sequential stream smoke check and resets when frame indices
    cross a filtered fixed-dt discontinuity.
- `scripts/run_kitti_seed0_five_group_matrix.sh`
  - Is the canonical A-E seed-0 runner for the frozen-backbone matrix.
  - Starts from `env -i`, explicitly sets every mechanism and optimization
    variable, forces the legacy current-teacher variable to zero, and records
    the exact Git revision in each result.
- `scripts/run_kitti_seed0_future_feature_matrix.sh`
  - Runs the first formal three-group matrix: Copy-current, Current-only, and
    top-layer Temporal Error. It rejects Latest, Two-tap, and Target-Flow
    Recursive groups for this matrix.
  - Stores outputs under `/tmp/predify-storage` by default and retains only the
    best validation checkpoint per group.
- `predify2021/mce_scores/diagnose_kitti_future_feature_delta.py`
  - Re-evaluates a Current-only best checkpoint on ordered train and validation
    pairs and reports true/predicted delta scale, L2/RMS quantiles, norm ratio,
    delta cosine, projection, and matched Copy-current MSE.
- `scripts/run_kitti_seed0_future_feature_predictor_sufficiency.sh`
  - Runs a Current-only seed-0 predictor with an explicitly selected 1x1 or
    3x3 first convolution while keeping all history and dynamic-error controls
    fixed.
- `predify2021/mce_scores/kitti_controlled_corruption.py`
  - Applies deterministic absolute-frame corruption after resize/crop in RGB
    pixel space and before ImageNet normalization.
  - Keeps fixed RGB bias and per-frame i.i.d. Gaussian noise as separate
    corruption types.
  - Defines independent step-bias, ramp-bias, and noise-negative-control
    trajectories and computes signed, absolute, and positive-part excess
    integrals plus recovery metrics.
- `predify2021/mce_scores/evaluate_kitti_same_drive_controlled_corruption.py`
  - Resets before every paired clean/corrupted trajectory and saves per-frame
    `e`, `E`, `L`, and signed `delta L` traces as JSONL/CSV plus one recovery
    PNG per trajectory.
  - Reports `RMS(F_t)`, `RMS(E_t)`, predictor input-weight scale, and the
    counterfactual history contribution `P(F_t,E_t)-P(F_t,0)`.
  - Rejects cross-drive or mismatched checkpoints through saved split and
    architecture checks.
- `scripts/run_kitti_seed0_same_drive_controlled_corruption.sh`
  - Trains the same three clean groups on the same-drive split, then invokes
    the independent controlled-corruption evaluator.
- `.github/workflows/tests.yml`
  - Runs the unit tests on pushes to `targetflow-arch` and pull requests.
  - Pins the public base `predify` dependency by commit and uses CPU PyTorch.

## Future-feature implementation status

The causal future-feature path is implemented and unit tested. An initial
two-pair train/two-pair validation GPU smoke run completed on `cuda:0` with
pretrained feedback weights, backward optimization, feature metrics, and
best-checkpoint selection. The smoke run verified exact equality of future
feature MSE and residual-delta MSE; its numerical accuracy is not an experiment
result. The formal frozen-backbone matrix uses a detached student-self target,
because an EMA teacher produces the same top feature in this regime.

The seed-0 five-condition feature matrix completed at revision `94059be`.
Copy-current achieved validation feature MSE `0.060080099`. Current-only was
2.858% worse at `0.061797074`, so the predictor did not learn a useful future
change on validation. Recursive reached `0.061796151`, only 0.00149% better
than current-only and therefore a numerical tie, not evidence for useful
history. Latest and two-tap reached `0.061849746` and `0.061899316`.

All learned conditions selected epoch 2 and then overfit while training MSE
continued to improve. The follow-up best-checkpoint diagnostic showed that the
1x1 Current-only delta norm is only 15.8% of the true train delta and its delta
cosine is 0.195 on train and 0.068 on validation. It improves train MSE by
3.82% but is 2.86% worse than Copy-current on validation.

A 3x3-first Current-only sufficiency run completed at revision `1ae6b27`. Its
best epoch-1 validation MSE is 0.060632, still 0.917% worse than Copy-current,
while its online train MSE reaches 0.093191 by epoch 10 and validation degrades
to 0.095457. The best 3x3 checkpoint predicts a smaller correction than 1x1
and has worse validation delta cosine (0.027 versus 0.068); its MSE being closer
to Copy-current is therefore not evidence that it learned spatial motion more
accurately. The late-epoch train gain shows additional fitting capacity while
the validation trajectory shows severe cross-drive overfitting. The 3x3-first
model is not parameter matched: it has 9.96M predictor parameters versus 1.57M
for the 1x1 model.

The first strict top-layer Temporal Prediction Error matrix completed at
revision `3ffbff0`. Copy-current again reached `0.060080099`, Current-only
reached `0.061797074`, and Temporal Error reached `0.061903913`. Current-only
was 2.85781% worse than Copy-current, and Temporal Error was 0.17289% worse
than Current-only. Both formal gates therefore failed. The independent
Temporal Error state had a measurable numerical effect in this seed, but no
held-out benefit. Because the basic predictor still fails the Copy-current
gate, this does not establish that prediction-error memory is generally
unhelpful.

Seeds 1 and 2, `tau_e` tuning, and additional history reruns remain paused.
Current-only must first beat Copy-current on held-out video. The next
cross-drive decision is about data coverage and predictor
regularization/capacity, not either error time constant. The current data do
not separate spatial architecture effects from parameter count or
conservative near-zero prediction. Broad noise, blur, online adaptation, and
broader state claims remain downstream experiments.

The same-drive controlled-corruption run completed at revision `ae90a9f` on
drive 0011, with raw frames 0--138 for training, a 20-frame raw gap, and 45
ordered validation transitions from frames 159--204. Copy-current reached
`0.071884151`, Current-only `0.074492955`, and Temporal Error `0.074685545`.
Thus Current-only was 3.62918% worse than Copy-current and Temporal Error was
0.25853% worse than Current-only; both gates failed again.

Temporal Error versus Current-only signed excess AUEC changed by +0.22189% for
step bias, +0.22728% for ramp bias, and -1.00328% for the i.i.d.-noise negative
control. These small mixed differences do not support selective adaptation to
systematic bias. The state was not ignored: `RMS(E)/RMS(F)` was about
0.104--0.143, history and feature input-weight RMS were comparable, and the
history contribution was about 0.166--0.227 of predicted-delta RMS. Yet the
same-checkpoint zero-history ablation improved phase-mean MSE everywhere. This
is a one-seed, short same-drive mechanism diagnostic, not robustness or
generalization evidence.

The original evaluator `summary.json` and all nine 45-row per-frame CSV traces
are versioned under
`results/seed0_same_drive_controlled_corruption_ae90a9f/`, together with
provenance and SHA-256 hashes. The approximately 4 GB checkpoints remain only
on the experiment server.

## Available data

- Training drive: `2011_09_26_drive_0005_sync`, 153 accepted adjacent pairs.
- Validation drive: `2011_09_26_drive_0011_sync`, 232 accepted adjacent pairs.
- Camera: `image_02`.
- Fixed frame interval: 0.1035 seconds with 0.001-second tolerance.

These two drives are enough for controlled mechanism validation, but not for a
final claim about broad KITTI generalization.

Both downloaded drives currently form one uninterrupted fixed-dt segment, so
the newly fixed segment-boundary bug did not change their old sample order. The
fix is required before adding drives that contain rejected timestamp steps.

## Formal seed-0 control result

The corrected frozen-backbone A-E matrix completed at Git revision `6c446d9`.
All conditions used explicit isolated environments, disjoint drives,
training-only target normalization, recursive target flow, instantaneous local
loss, variance weight zero, and best-checkpoint selection by validation
standardized 2-DoF longitudinal-yaw MSE.

| Group | Condition | Best epoch | Standardized MSE | Forward MAE (m) | Yaw MAE (rad) |
| --- | --- | ---: | ---: | ---: | ---: |
| A | Inherit, recursive EMA | 6 | 11.045343 | 0.473039 | 0.008406 |
| B | Reset, no extra top | 5 | 11.443965 | 0.465796 | 0.005455 |
| C | Reset, current-top duplicate | 6 | **11.018217** | **0.458364** | 0.006219 |
| D | Inherit, instant error | 5 | 11.146288 | 0.477386 | **0.003696** |
| E | Inherit, two-tap error | 5 | 11.128227 | 0.477661 | 0.004047 |

A is 0.246% worse than C in the primary clean history comparison, so seed 0
does not show a benefit from inherited temporal history. C is 3.720% better
than B, confirming that the duplicated current top representation materially
affects the no-history baseline. A is 0.906% better than D and 0.745% better
than E in standardized MSE, but these small differences are not yet stable
evidence for recursive dynamic error and are not consistent across the two
physical component MAEs.

The full result, comparison definitions, CI identifiers, and server artifact
path are recorded in `docs/EXPERIMENT_LOG.md`.

## Cheap-diagnostic decision

The required constant, static, and best-checkpoint diagnostics completed at
revision `5b64cad`. The clean no-history C condition reaches standardized
joint MSE 11.018217 versus 11.266117 for the train-mean constant, only a 2.20%
reduction. A reaches 11.045345, a 1.96% reduction. The Frozen VGG plus current-
frame-only MLP reaches 11.376018 at epoch 1 and is 0.98% worse than the
constant; later epochs overfit strongly.

All learned conditions degrade yaw MSE relative to the constant. The constant
yaw MSE is `2.7766e-6 rad2`, compared with `8.3509e-5` for A and `5.4291e-5`
for C. Their small joint-MSE gains come from forward displacement and are not
consistent across mean, median, and P95 absolute error.

The two drives have mismatched motion regimes. Train drive 0005 has
forward/yaw standard deviations `0.107030 m / 0.016870 rad`; validation drive
0011 has `0.507264 m / 0.001657 rad`. Thus validation standardized MSE is
almost entirely controlled by forward displacement and cannot currently test
the yaw mechanism well.

Seeds 1 and 2 are paused. More training drives and a motion-regime-aware split
are required before repeating the matrix. Full component and quantile results
are in `docs/EXPERIMENT_LOG.md`.

This pause applies to the 2-DoF motion-proxy matrix. It does not block the new
next-frame feature experiment, which now precedes any additional motion seeds.

## Superseded seeded control result

The causal model and two controls were run for ten epochs at Git revision
`77f0ad0`, with seed 0 and best-checkpoint selection by validation temporal
loss. These results are retained for traceability but are not valid clean
mechanism comparisons.

| Run | State policy | Error policy | Best epoch | Legacy mixed-unit MSE | Legacy mixed-unit MAE |
| --- | --- | --- | ---: | ---: | ---: |
| A | Inherit | Dynamic, `tau=0.5` | 6 | **0.108731** | **0.222764** |
| B | Reset each frame | Dynamic, `tau=0.5` | 1 | 0.129041 | 0.240693 |
| C | Inherit | Instantaneous | 10 | 0.111454 | 0.223388 |

All three runs used `quasi_steady`, so only the top layer received a future
target. They also used the filtered error directly in the local MSE. At
`Ts=0.1035` and `tau=0.5`, the EMA current-error coefficient was 0.207, while
the instantaneous control coefficient was 1. This changed the local gradient
scale and optimization dynamics. Reset-each-frame also reset that filtered
loss state. Consequently, neither the reported A/B nor A/C difference can be
attributed cleanly to inherited memory.

The feedback decoders were also absent from the optimizer despite retaining
gradients. This has been corrected; all future recursive target-flow runs train
the feedback decoders. The earlier variance regularizer was inactive at stream
batch size 1 and is replaced by a temporal-window implementation.

The historical target named `ego_motion` contained only forward displacement
and yaw change, not lateral translation. Its direct MSE mixed metres and
radians. Corrected runs call it `longitudinal_yaw_2dof`, standardize each
component using training-only statistics, and report physical component MAEs.

The old inherited condition also contained `F_teacher(I_t)` through its saved
previous top target, whereas reset-each-frame did not. The reported 15.7% A/B
difference therefore cannot isolate long-term history. With frozen VGG,
`F_teacher(I_t)=F_student(I_t)`, so the corrected matrix adds a reset/current-
top-duplicate/no-history control.

The earlier unseeded corrected run reached MSE 0.094732 at epoch 8, but its
best weights were not saved. It remains exploratory evidence and is not used
in the seeded control comparison.

## Invalid predecessor stream result

Configuration:

- Stream state inheritance enabled.
- Dynamic error enabled at all five layers.
- `tau=0.5`, `gain=1.0`, `dt=0.1035` seconds.
- Ego-motion temporal target.
- EMA teacher with decay 0.99.
- Ten epochs, learning rate `1e-4`, batch size 1.

Key validation results:

- Best temporal cosine: 0.750144 at epoch 2.
- Epoch-10 temporal cosine: 0.651919.
- Epoch-10 weighted validation loss: 0.007630.

This run is not valid evidence of temporal prediction quality. Its temporal
context included errors formed with the future-frame teacher target, and the
temporal prediction loss weight defaulted to zero. The run only proves that the
continuous stateful execution path completed end to end. All prediction metrics
must be rerun after the causal-context and positive-loss-weight correction.

## Required next experiments

1. Preserve the completed same-drive corruption result as a mechanistic null:
   the state is used but does not improve systematic-bias response.
2. Improve cross-drive predictor generalization through broader training data
   or a controlled capacity/regularization study, then require Current-only to
   beat Copy-current before interpreting inherited-state value.
3. Keep seeds 1 and 2 and both Target Flow and Temporal Error `tau` sweeps
   paused while the primary gate fails.
4. Treat the existing 2-DoF matrix as a proxy-task diagnostic only. Additional
   motion seeds remain paused unless motion is later reintroduced as a
   secondary evaluation.

| Group | Cross-frame state | Error state | Extra current-top context | Purpose |
| --- | --- | --- | --- | --- |
| A | Inherit | Recursive EMA | From inherited state | Full method |
| B | Reset | EMA, no effective history | None | Stateless baseline |
| C | Reset | EMA, no effective history | Detached current-top duplicate | Extra-current-feature control |
| D | Inherit | Instant | From inherited state | Test recursive dynamic error |
| E | Inherit | Two-tap | From inherited state | Test EMA against finite two-tap memory |

All five groups use recursive target flow, frozen VGG, instantaneous local
loss, variance weight zero, temporal prediction weight one, identical dynamic
parameters, seed, drives, training-only normalization statistics, and best-
checkpoint selection by validation temporal loss. Formal runs set
`PREDIFY_FORMAL_SPLIT=1` and explicitly provide disjoint train and validation
drives.

The seed-0 runner fixes 10 epochs, learning rate `1e-4`, pretrained predictive-
VGG weights, train drive 0005, validation drive 0011, and the existing training-
only normalization procedure for all five conditions. It also explicitly sets
`PREDIFY_CURRENT_TEACHER_CONTEXT=0` so a stale login-shell value cannot alter a
run.

A positive variance weight is invalid while the backbone is frozen because
the top-feature variance has no gradient path to the trainable feedback or
temporal modules. Variance calibration is reserved for the separate
`PREDIFY_TRAIN_BACKBONE=1` adaptation experiment.

For the current complete training drive, horizon-1 normalization is based on
153 samples: forward mean/std `0.466140/0.107030 m`, yaw mean/std
`-0.001483/0.016870 rad`. Validation data is not used for these statistics.

Target Flow uses residual `r_t=F_t-T_t` and
`epsilon_t=(Ts_r/tau_r)r_t+(1-K_r*Ts_r/tau_r)epsilon_(t-1)`. Temporal
Prediction Error is separate: `e_t=F_t-Fhat_(t|t-1)` and
`E_t=(Ts_e/tau_e)e_t+(1-K_e*Ts_e/tau_e)E_(t-1)`. There is no independent
`d_t` term. Each recurrence must satisfy its own stability condition; the
current `Ts=0.1035, tau=0.5, K=1` gives memory coefficient 0.793 for both, but
their parameter families and physical meanings remain independent.

## Repository policy

Code, configuration, Markdown records, and lightweight metric summaries belong
in Git. KITTI data, pretrained weights, checkpoints, caches, and large pickle
outputs remain on the server and are excluded by `.gitignore`.

Completed code changes must be verified, committed, and pushed to
`myprivate/targetflow-arch` without waiting for a separate push request.
Completed formal training must likewise be audited, recorded in the tracked
research documents, committed, and pushed. Large training artifacts remain on
the server; their exact code revision and server path belong in the pushed
record. Use an ordinary fast-forward push and never rewrite remote history
unless the user explicitly requests it.
