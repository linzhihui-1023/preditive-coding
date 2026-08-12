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

All learned conditions use one shared predictor architecture. Only the history
input changes.

| Group | `PREDIFY_FEATURE_HISTORY_MODE` | History input |
| --- | --- | --- |
| Copy-current | `copy_current` | Predictor bypassed; `Fhat=F_t` |
| Current-only | `none` | Zeros |
| Latest | `latest` | `e_(t-1)` from the previous transition |
| Two-tap | `two_tap` | `alpha e_(t-1) + (1-K alpha)e_(t-2)` |
| Recursive | `recursive` | Previous completed recursive state |

With the matrix parameters, `alpha=0.1035/0.5=0.207`. Thus two-tap history is
`0.207e_(t-1)+0.793e_(t-2)`, while recursive history is
`0.207e_(t-1)+0.793epsilon_(t-2)`. The legacy names `instant` and `lag1` are
accepted as aliases, but new result tables use Latest and Two-tap.

Run all five conditions:

```bash
conda activate predifyproject
cd /home/lin/predify2021_targetflow
scripts/run_kitti_seed0_future_feature_matrix.sh
```

Run selected conditions:

```bash
scripts/run_kitti_seed0_future_feature_matrix.sh current_only recursive
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

## Interpretation Order

1. `Current-only < Copy-current`: the predictor learned useful future change.
2. `Recursive < Current-only`: history adds information beyond `F_t`.
3. Compare Recursive with Latest and Two-tap to test whether recursive memory is
   better than simpler causal history.

Do not interpret history comparisons until Current-only passes step 1 on the
held-out drive. A predictor that only fits the training drive cannot establish
whether history contains useful generalizable information.

The current two-drive split is suitable for this initial mechanism check, not
for a broad KITTI generalization claim. More drives are required after the
mechanism clears these thresholds and before noise, blur, and online-adaptation
claims are treated as formal results.
