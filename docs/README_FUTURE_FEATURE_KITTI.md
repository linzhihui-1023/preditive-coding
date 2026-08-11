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
6. Update instant, lag-1, and recursive state for the next transition.

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
| Instant | `instant` | Previous completed instant residual |
| Lag-1 | `lag1` | Previous completed finite-memory state |
| Recursive | `recursive` | Previous completed recursive state |

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

Outputs default to
`/tmp/predify-storage/experiments/seed0_future_feature_matrix_<git-sha>`.
Set `PREDIFY_MATRIX_OUTPUT_ROOT` to override that location. The runner starts
each process with `env -i`, explicitly fixes all mechanism variables, and keeps
only the best validation checkpoint per condition.

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

Best checkpoints are selected by validation feature MSE.

## Interpretation Order

1. `Current-only < Copy-current`: the predictor learned useful future change.
2. `Recursive < Current-only`: history adds information beyond `F_t`.
3. Compare Recursive with Instant and Lag-1 to test whether recursive memory is
   better than simpler causal history.

The current two-drive split is suitable for this initial mechanism check, not
for a broad KITTI generalization claim. More drives are required after the
mechanism clears these thresholds and before noise, blur, and online-adaptation
claims are treated as formal results.
