# Project State

Last updated: 2026-08-10

## Active research direction

The active model operates on a real video-frame stream:

1. The first frame initializes five layer states.
2. Each later frame executes the model exactly once.
3. Each frame inherits the previous five layer states, predictions, and dynamic
   errors.
4. The current state predicts next-time features or an ego-motion target.
5. Training and validation preserve drive and frame order. Stream mode requires
   batch size 1 and rejects shuffled pair loaders.

The earlier route that repeatedly ran multiple timesteps on the same image is
cancelled and is not part of the active experiment design.

## Current implementation

- `predify2021/model_factory/pvgg16_targetflow.py`
  - Implements five target-flow stages.
  - Keeps per-layer error and prediction memories across `step_frame` calls.
  - Clears memories at sequence boundaries through `reset`.
  - Builds the causal temporal prediction context from the current top feature,
    five previous-frame errors, and five previous-frame predictions.
  - Forms the temporal prediction before resolving targets derived from future
    frames.
- `predify2021/model_factory/targetflow/core.py`
  - Defines target-flow state, dynamic-error integration, local losses, and
    gradient diagnostics.
- `predify2021/mce_scores/kitti_pairs.py`
  - Loads adjacent KITTI frames, timestamps, and ego-motion targets.
- `predify2021/mce_scores/train_kitti_targetflow_adjacent_pairs.py`
  - Defaults to stream mode.
  - Resets once per drive or sequence, then calls `step_frame` once per frame.
  - Requires `PREDIFY_BATCHSIZE=1` and disables shuffled pairs in stream mode.
- `predify2021/mce_scores/calculate_kitti_targetflow_pair_smoke.py`
  - Provides a sequential stream smoke check.

## Available data

- Training drive: `2011_09_26_drive_0005_sync`, 153 accepted adjacent pairs.
- Validation drive: `2011_09_26_drive_0011_sync`, 232 accepted adjacent pairs.
- Camera: `image_02`.
- Fixed frame interval: 0.1035 seconds with 0.001-second tolerance.

These two drives are enough for controlled mechanism validation, but not for a
final claim about broad KITTI generalization.

## Latest corrected stream result

The corrected causal model was run for ten epochs at Git revision `4793bf3`
with temporal prediction weight 1.0. The data and remaining configuration match
the two-drive setup below.

Key validation results:

- Best temporal MSE: 0.094732 at epoch 8.
- Best temporal MAE: 0.212064 at epoch 8.
- Epoch-8 temporal cosine: 0.750689.
- Final epoch temporal MSE: 0.127408.
- Final epoch temporal MAE: 0.248920.

A constant predictor using the training-drive mean ego motion obtains validation
MSE 0.129003, MAE 0.235466, and cosine 0.751176. The epoch-8 model improves MSE
by 26.6% and MAE by 9.9%, but does not improve cosine. This suggests useful
magnitude prediction while showing that cosine is dominated by the common
forward-motion direction.

The training script saved only the final epoch checkpoint, not epoch 8. The
next formal run must save the best validation-temporal-loss checkpoint and set
an explicit random seed.

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

1. Add deterministic seeding and save the best validation temporal-loss
   checkpoint.
2. Add and run a reset-each-frame control while keeping one execution per real
   video frame.
3. Run `PREDIFY_DYNAMIC_ERROR=0` with the same ordered stream.
4. Repeat the core comparisons with multiple seeds.
5. If the mechanism advantage is stable, run a tau sweep and then add more
   train and validation drives.

## Repository policy

Code, configuration, Markdown records, and lightweight metric summaries belong
in Git. KITTI data, pretrained weights, checkpoints, caches, and large pickle
outputs remain on the server and are excluded by `.gitignore`.
