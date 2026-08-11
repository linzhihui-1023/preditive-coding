# Project State

Last updated: 2026-08-11

## Active research direction

The active model operates on a real video-frame stream:

1. The first frame initializes five layer states.
2. Each later frame executes the model exactly once.
3. Each frame inherits the previous five layer states, predictions, and dynamic
   errors.
4. The current state predicts next-time features or a standardized 2-DoF
   longitudinal-yaw motion target.
5. Training and validation preserve drive and frame order. Stream mode requires
   batch size 1 and rejects shuffled pair loaders.

The earlier route that repeatedly ran multiple timesteps on the same image is
cancelled and is not part of the active experiment design.

## Current implementation

- `predify2021/model_factory/pvgg16_targetflow.py`
  - Implements five target-flow stages.
  - Defaults to recursive target flow: the detached future top feature is
    propagated through `T5 -> T4 -> T3 -> T2 -> T1`.
  - Separates the cross-frame error state (`instant`, `ema`, or `lag1`) from the
    local-loss error source.
  - Defaults local optimization to instantaneous error so memory controls have
    identical current-frame local losses and gradient coefficients.
  - Trains the four feedback decoder modules and temporal predictor. The VGG
    forward stages are frozen in the primary mechanism experiments.
  - Keeps per-layer error and prediction memories across `step_frame` calls.
  - Detaches every stored error and prediction state. Execution is stateful
    forward recurrence with one-step gradients, not BPTT.
  - Clears memories at sequence boundaries through `reset`.
  - Builds the causal temporal prediction context from the current top feature,
    five previous-frame errors, and five previous-frame predictions.
  - Forms the temporal prediction before resolving targets derived from future
    frames.
- `predify2021/model_factory/targetflow/core.py`
  - Defines target-flow state, dynamic-error integration, local losses, and
    gradient diagnostics.
- `predify2021/mce_scores/kitti_pairs.py`
  - Loads adjacent KITTI frames, timestamps, and 2-DoF longitudinal-yaw
    targets `[forward displacement m, yaw change rad]`.
  - Splits fixed-dt-valid sample indices into contiguous segments whenever a
    raw timestamp transition is rejected.
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
    temporal predictor train; `PREDIFY_TRAIN_BACKBONE=1` is an explicit
    adaptation ablation.
  - Saves the best student checkpoint by validation temporal loss as well as
    the final student and teacher checkpoints.
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
- `.github/workflows/tests.yml`
  - Runs the 21 unit tests on pushes to `targetflow-arch` and pull requests.
  - Pins the public base `predify` dependency by commit and uses CPU PyTorch.

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
| E | Inherit, lag-1 error | 5 | 11.128227 | 0.477661 | 0.004047 |

A is 0.246% worse than C in the primary clean history comparison, so seed 0
does not show a benefit from inherited temporal history. C is 3.720% better
than B, confirming that the duplicated current top representation materially
affects the no-history baseline. A is 0.906% better than D and 0.745% better
than E in standardized MSE, but these small differences are not yet stable
evidence for recursive dynamic error and are not consistent across the two
physical component MAEs.

The full result, comparison definitions, CI identifiers, and server artifact
path are recorded in `docs/EXPERIMENT_LOG.md`.

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

1. Repeat the corrected five-condition matrix with seeds 1 and 2, keeping the
   top variance weight fixed at zero in every frozen-backbone condition.
2. Treat A versus C as the primary state-memory comparison, B versus C as the
   current-top duplication check, A versus D as the dynamic-error comparison,
   and A versus E as recursive memory versus a lag-1 FIR baseline.
3. After seeds 1 and 2, report mean, standard deviation, and per-seed paired
   differences. Do not select a mechanism from seed 0 alone.
4. Only if a mechanism advantage is stable, run a tau sweep and then add more
   train and validation drives.
5. Reserve a separate test-drive set before reporting final generalization.

| Group | Cross-frame state | Error state | Extra current-top context | Purpose |
| --- | --- | --- | --- | --- |
| A | Inherit | Recursive EMA | From inherited state | Full method |
| B | Reset | EMA, no effective history | None | Stateless baseline |
| C | Reset | EMA, no effective history | Detached current-top duplicate | Extra-current-feature control |
| D | Inherit | Instant | From inherited state | Test recursive dynamic error |
| E | Inherit | Lag-1 | From inherited state | Test EMA against finite one-step memory |

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

Dynamic error uses `e_t=F_t-T_t` and
`epsilon_t=(Ts/tau)e_t+(1-K*Ts/tau)epsilon_(t-1)`. There is no independent
`d_t` term. Configurations must satisfy
`abs(1-K*Ts/tau)<1`; the current `Ts=0.1035, tau=0.5, K=1` gives 0.793.

## Repository policy

Code, configuration, Markdown records, and lightweight metric summaries belong
in Git. KITTI data, pretrained weights, checkpoints, caches, and large pickle
outputs remain on the server and are excluded by `.gitignore`.
