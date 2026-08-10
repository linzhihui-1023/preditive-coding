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
  - Supports deterministic runs through `PREDIFY_SEED`.
  - Supports the `PREDIFY_RESET_EACH_FRAME=1` control without adding repeated
    model executions.
  - Saves the best student checkpoint by validation temporal loss as well as
    the final student and teacher checkpoints.
- `predify2021/mce_scores/calculate_kitti_targetflow_pair_smoke.py`
  - Provides a sequential stream smoke check.

## Available data

- Training drive: `2011_09_26_drive_0005_sync`, 153 accepted adjacent pairs.
- Validation drive: `2011_09_26_drive_0011_sync`, 232 accepted adjacent pairs.
- Camera: `image_02`.
- Fixed frame interval: 0.1035 seconds with 0.001-second tolerance.

These two drives are enough for controlled mechanism validation, but not for a
final claim about broad KITTI generalization.

## Latest seeded control result

The corrected causal model and two controls were run for ten epochs at Git
revision `77f0ad0`, with seed 0 and best-checkpoint selection by validation
temporal loss.

| Run | State policy | Error policy | Best epoch | Temporal MSE | Temporal MAE |
| --- | --- | --- | ---: | ---: | ---: |
| A | Inherit | Dynamic, `tau=0.5` | 6 | **0.108731** | **0.222764** |
| B | Reset each frame | Dynamic, `tau=0.5` | 1 | 0.129041 | 0.240693 |
| C | Inherit | Instantaneous | 10 | 0.111454 | 0.223388 |

Run A improves MSE by 15.7% over B and 2.4% over C. This seed supports the
value of inherited cross-frame state. The dynamic-error advantage over
instantaneous error is small and needs repeated seeds. Cosine remains close to
0.75 for every condition and is not a useful discriminator on this split.

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

1. Repeat the A/B/C control matrix with at least seeds 1 and 2.
2. Report mean, standard deviation, and per-seed paired differences.
3. If the mechanism advantage is stable, run a tau sweep and then add more
   train and validation drives.
4. Reserve a separate test-drive set before reporting final generalization.

## Repository policy

Code, configuration, Markdown records, and lightweight metric summaries belong
in Git. KITTI data, pretrained weights, checkpoints, caches, and large pickle
outputs remain on the server and are excluded by `.gitignore`.
