# Causal Future-Feature Experiment on KITTI

## Task

The future-feature task predicts a complete VGG feature map of the next video
frame. `PREDIFY_FUTURE_FEATURE_STAGE` selects Stage 3, 4, or 5 and defaults to
5. It does not replace the existing 2-DoF motion proxy head.

- `PREDIFY_TASK=motion`: original temporal motion/feature path.
- `PREDIFY_TASK=future_feature`: independent future-feature residual head.

For the active task:

`delta_hat = future_feature_predictor(F_t, H_t)`

`Fhat_(t+1|t) = F_t + delta_hat`

The backbone runs once for the current frame. The next-frame prediction-stage
feature is a detached supervision target and is computed only after
prediction. It is separate from the Predify Target Flow top target:

```text
T_TF = F_(t+1)^5
T_future = F_(t+1)^s_pred, s_pred in {3,4,5}
```

Changing `s_pred` never changes Target Flow's Stage-5 top. Checkpoint config
records both stages, the prediction-stage channels, and the two target
definitions.

## Causal Order

Each stream transition executes in this order:

1. Read state completed by the previous transition.
2. Extract the current feature `F_t`.
3. Predict `Fhat_(t+1|t)` from `F_t` and old history `H_t`.
4. Extract detached `T_TF` and `T_future` in one future-frame pass.
5. Compute prediction and Target Flow losses.
6. Update latest, two-tap, and recursive state for the next transition.

The current pair's residual never enters its own prediction. Stored state is
detached, so this is stateful forward recurrence with one-step gradients, not
BPTT.

## Seed-0 Matrix

The first formal Temporal Error matrix contains exactly three groups:

| Group | `PREDIFY_FEATURE_HISTORY_MODE` | History input |
| --- | --- | --- |
| Copy-current | `copy_current` | Predictor bypassed; `Fhat=F_t` |
| Current-only | `none` | Zeros |
| Temporal error | `temporal_error` | Previous completed top-layer Temporal Prediction Error state |

Current-only and Temporal Error use the same predictor architecture and
parameter count; only the history tensor changes. Latest, two-tap, and
Target-Flow recursive modes remain implemented for historical reproducibility
but are excluded from this first formal matrix and from the runner's accepted
groups.

Temporal error is a separate top-layer state. It uses the strict future-feature
prediction error `e_t^5 = F_t^5 - Fhat_(t|t-1)^5`, then stores
`E_t^5 = alpha_e e_t^5 + (1-K_e alpha_e)E_(t-1)^5` for the next transition. In
the first version `alpha_e=0.207`, controlled by
`PREDIFY_TEMPORAL_ERROR_TS=0.1035`, `PREDIFY_TEMPORAL_ERROR_TAU=0.5`, and
`PREDIFY_TEMPORAL_ERROR_GAIN=1.0`. This is not the Target Flow residual and
does not change the meaning of `recursive`.

Run the three formal conditions:

```bash
conda activate predifyproject
cd /home/lin/predify2021_targetflow
scripts/run_kitti_seed0_future_feature_matrix.sh
```

The runner also accepts a selected subset of those three conditions:

```bash
scripts/run_kitti_seed0_future_feature_matrix.sh copy_current current_only temporal_error
```

Run the Current-only spatial-predictor sufficiency diagnostic:

```bash
scripts/run_kitti_seed0_future_feature_predictor_sufficiency.sh 3
```

The value selects the first predictor convolution's kernel size. The second
convolution remains 1x1. The default is 3 for this diagnostic runner; the
formal history matrix explicitly fixes the value to 1.

Recompute delta scale and direction statistics from a best checkpoint:

```bash
PREDIFY_DIAGNOSTIC_CHECKPOINT=/path/to/best_student.pt \
PREDIFY_TRAIN_DRIVES=2011_09_26/2011_09_26_drive_0005_sync \
PREDIFY_VAL_DRIVES=2011_09_26/2011_09_26_drive_0011_sync \
python -m predify2021.mce_scores.diagnose_kitti_future_feature_delta
```

The diagnostic accepts only checkpoints whose saved config identifies
`prediction_task=future_feature` and `future_feature_history_mode=none`. It
verifies the recorded kernel against the first predictor weight. A legacy
checkpoint without a kernel field is accepted only when its weight is verified
to be 1x1, preventing silent evaluation of a Recursive model with history
forcibly removed.

Outputs default to
`/tmp/predify-storage/experiments/seed0_future_feature_matrix_<git-sha>`.
Set `PREDIFY_MATRIX_OUTPUT_ROOT` to override that location. The runner starts
each process with `env -i`, explicitly fixes all mechanism variables, and keeps
only the best validation checkpoint per condition.

The frozen-backbone matrix explicitly uses
`PREDIFY_TOP_TARGET_SOURCE=student_self` and
`PREDIFY_TEMPORAL_TARGET_MODE=next_top`. The detached student target is equal
to a frozen EMA-teacher top feature without keeping a redundant teacher model.

## Losses And Metrics

The optimized objective is:

`L = lambda_rec * L_local + lambda_feature * L_feature`

The frozen-backbone matrix fixes variance weight to zero. `L_feature` is both
computed as future-feature MSE and residual-delta MSE; these expressions must
be numerically equal.

Validation reports:

- Feature MSE.
- Feature cosine similarity.
- Normalized feature error
  `||Fhat-F_next||_2 / (||F_next||_2 + eps)`.
- The same three metrics for copy-current.
- Maximum future/delta MSE equivalence error.
- Signed and relative improvement against Copy-current from the same stage.

Absolute metrics from different VGG stages are not directly comparable. A
Stage-4 experiment must be judged against Stage-4 Copy-current on the same
frame stream, not against the historical Stage-5 MSE. Spatial motion radius is
measured in prediction-stage feature cells; it is configurable and is not
automatically rescaled across stages.

The signed diagnostic is the Temporal Prediction Error
`e_t = F_t - Fhat_(t|t-1)`. In the implementation of transition `t -> t+1`,
this is stored canonically as
`prediction_error_feature = F_next - Fhat_(t+1|t)` after the prediction is
made. `prediction_error_top` remains only as a compatibility alias for older
Stage-5 evaluators. MSE is sign invariant, but this stored tensor must retain
the documented sign.

Best checkpoints are selected by validation feature MSE.

## Same-Drive Controlled Corruption

Controlled corruption is a separate experiment and is not injected into the
training loop. The canonical runner first trains Copy-current, Current-only,
and Temporal Error checkpoints on clean frames from one drive, then invokes
the independent evaluation module:

```bash
scripts/run_kitti_seed0_same_drive_controlled_corruption.sh
```

The split is defined in raw-frame space:

```text
train: first 60% of raw frames
gap:   next 20 raw frames
val:   next contiguous 20% of raw frames
```

Every pair is admitted only when all raw frames it reads lie inside its split.
The implementation collects every raw frame index used by train and validation
and fails unless the sets are disjoint. The current runner uses the longer
drive 0011: raw train frames 0--138, gap frames 139--158, and raw validation
frames 159--204, giving 45 evaluated transitions. The remaining tail is
unused. This is longer than the old 29-transition drive-0005 diagnostic but is
still explicitly classified as a short mechanistic trace, not a paper-scale
adaptation curve.

Corruption is applied after the existing resize and center crop in the
unnormalized `[0,1]` RGB pixel domain, then ImageNet normalization is applied.
Its deterministic key is `(seed, drive, camera, absolute frame name)`. Thus an
absolute frame used first as `future` and then as the next sample's `current`
is exactly the same corrupted tensor. The script never calls
`torch.randn_like(image)` without an absolute-frame seed.

Step, ramp, and random noise are not concatenated. The evaluator runs three
independent trajectories, resetting model state before every clean and
corrupted stream:

```text
step_bias: baseline -> abrupt fixed RGB bias -> hold -> recovery
ramp_bias: baseline -> gradual fixed RGB bias -> hold -> recovery
iid_noise: baseline -> per-frame i.i.d. Gaussian noise -> recovery
```

Because phases are assigned by the absolute future frame, the 45 transitions
contain 9 baseline transitions. Step-bias then has 1 step, 17 hold, and 18
recovery transitions; ramp-bias has 8 ramp, 10 hold, and 18 recovery
transitions; i.i.d. noise has 18 disturbed and 18 recovery transitions.

The bias trajectories contain no Gaussian noise. The i.i.d. noise trajectory
contains no RGB bias and acts as a negative control for an unpredictable
disturbance. Evaluation saves one JSONL row per frame and one recovery CSV per
checkpoint and trajectory, plus a PNG per trajectory with:

- strict Temporal Prediction Error magnitude `e_t`;
- accumulated Temporal Error state magnitude `E_t`;
- prediction loss `L_t` as stage-5 feature MSE;
- signed `delta L_t = L_t(corrupted)-L_t(clean)` as the primary curve;
- signed, absolute, and positive-part excess AUEC;
- secondary raw Peak Error and raw AUEC;
- `RMS(F_t)`, `RMS(E_t)`, predictor input-weight scale, and the activation
  contribution `P(F_t,E_t)-P(F_t,0)`.

Signed excess is never clipped, so negative values expose temporary
improvement, overshoot, or overcompensation. The positive-part integral is
retained only as an auxiliary metric. Recovery Time is measured after
corruption returns to zero: signed excess must remain within 10% of the peak
absolute excursion from its paired-clean baseline for three consecutive
frames. Unrecovered streams are marked censored rather than assigned a fake
finite recovery time.

The evaluator accepts only checkpoints that record the matching same-drive
split. Existing cross-drive checkpoints are intentionally rejected: this
experiment tests controlled transient response within a drive, not cross-drive
generalization.

## Interpretation Order

1. `Current-only < Copy-current`: the predictor learned useful future change.
2. `Temporal Error < Current-only`: strict prediction-error state adds useful
   information beyond `F_t`.

Do not interpret the Temporal Error comparison until Current-only passes step
1 on the held-out drive. A predictor that only fits the training drive cannot
establish whether history contains useful generalizable information.

The current two-drive split is suitable for this initial mechanism check, not
for a broad KITTI generalization claim. More drives are required after the
mechanism clears these thresholds and before noise, blur, and online-adaptation
claims are treated as formal results.
