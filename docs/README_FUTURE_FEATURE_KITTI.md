# Causal Future-Feature Experiment on KITTI

## Task

The primary experiment predicts the complete VGG stage-5 feature map of the
next video frame. It does not replace the existing 2-DoF motion proxy head.

- `PREDIFY_TASK=motion`: original temporal motion/feature path.
- `PREDIFY_TASK=future_feature`: independent future-feature residual head.

For the active task:

`delta_hat = future_feature_predictor(F_t, H_t)`

`Fhat_(t+1|t) = F_t + delta_hat`

The backbone runs once for the current frame. The next-frame backbone feature
is a detached supervision target and is computed only after prediction.

## Causal Order

Each stream transition executes in this order:

1. Read state completed by the previous transition.
2. Extract the current feature `F_t`.
3. Predict `Fhat_(t+1|t)` from `F_t` and old history `H_t`.
4. Extract the detached target `F_(t+1)`.
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
prediction error
`e_(t+1)^5 = F_(t+1)^5 - Fhat_(t+1|t)^5`, then stores
`E_(t+1)^5 = alpha_e e_(t+1)^5 + (1-K_e alpha_e)E_t^5` for the next
transition. In the first version `alpha_e=0.207`, controlled by
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

The signed diagnostic is
`prediction_error_top = F_next - Fhat_(t+1|t)`, matching the project theory.
MSE is sign invariant, but this stored tensor must retain the documented sign.

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
and fails unless the sets are disjoint. For drive 0005 this produces raw train
frames 0--91, gap frames 92--111, and raw validation frames 112--141; the
remaining tail is unused.

Corruption is applied after the existing resize and center crop in the
unnormalized `[0,1]` RGB pixel domain, then ImageNet normalization is applied.
Its deterministic key is `(seed, drive, camera, absolute frame name)`. Thus an
absolute frame used first as `future` and then as the next sample's `current`
is exactly the same corrupted tensor. The script never calls
`torch.randn_like(image)` without an absolute-frame seed.

The validation trajectory contains baseline, step change, ramp change,
persistent bias, and recovery phases. Evaluation runs paired clean and
corrupted counterfactual streams and saves one JSONL row per frame, a recovery
curve CSV, and a PNG with:

- strict Temporal Prediction Error magnitude `e_(t+1)`;
- accumulated Temporal Error state magnitude `E_(t+1)`;
- prediction loss `L_t` as stage-5 feature MSE;
- Peak Error, Recovery Time, AUEC, and clean-adjusted excess AUEC.

Recovery Time is measured after corruption returns to zero: excess MSE must
remain below baseline plus 10% of the peak excursion for three consecutive
frames. Unrecovered streams are marked censored rather than assigned a fake
finite recovery time. AUEC integrates the full disturbance-and-recovery curve
using the fixed sample interval.

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
