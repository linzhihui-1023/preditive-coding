# Stage-4 Aligned Temporal-Difference Predictor

This directory contains the lightweight auditable outputs from the formal
seed-0 run trained at Git revision
`fdc47437931e03f1284a852b997952b29b963910`.

## Fixed Protocol

- Train drive: KITTI `2011_09_26_drive_0005_sync`, 153 transitions.
- Held-out drive: KITTI `2011_09_26_drive_0011_sync`, 232 transitions.
- Future prediction target: frozen pretrained VGG Stage 4.
- Predify Target Flow top target: frozen pretrained VGG Stage 5.
- Temporal input: `D_t=F_t^4-align(F_(t-1)^4,F_t^4)` using radius 1 in
  Stage-4 feature cells and patch size 3.
- Prediction: `Fhat_(t+1)^4=F_t^4+P([F_t^4,D_t])`. No Temporal Fusion module
  is created; previous prediction-stage features are detached.
- Training: 10 epochs, seed 0, learning rate `1e-4`; only the Future Predictor
  is optimized.
- Checkpoint selection: minimum held-out Stage-4 feature MSE, selecting epoch
  1.
- Gate: Copy-current computed in Stage 4 on the exact same held-out frames.
  Stage-5 absolute metrics are not used as references.

## Held-Out Gate

| Metric | Aligned difference | Stage-4 Copy-current | Required | Pass |
| --- | ---: | ---: | :---: | :---: |
| Feature MSE | 0.74586091 | 0.74053836 | `<` | No |
| Cosine | 0.85806100 | 0.86544334 | `>` | No |
| Normalized error | 0.49728997 | 0.48848719 | `<` | No |

The MSE change relative to same-stage Copy-current is `-0.718741%`. The
all-three held-out gate failed. This is a result for one seed, one training
drive, radius 1, and this residual predictor; it is not a cross-stage ranking
or a claim that Stage-4 motion information is absent.

The fixed-checkpoint replay improved train-drive MSE from `2.16380702` to
`2.01633201` (`+6.815535%`) while worsening held-out MSE. During training,
train MSE fell every epoch and held-out MSE rose every epoch after the selected
epoch 1. Alignment was applied to `383/385` rows, Temporal Fusion was never
applied, and the maximum difference between prediction base and `F_t^4` was
exactly `0.0`.

## Audit

The audit verified 385 unique condition/split/frame keys, exact 153/232 split
counts, adjacent raw-frame indices, finite metrics, Stage-4 prediction and
Stage-5 Target Flow metadata, `512x28x28` prediction features, per-row
same-stage gain arithmetic, all CSV-to-summary means, ten epochs, the epoch-1
checkpoint contract, and predictor weight shapes `1024x1024x1x1` then
`512x1024x1x1`. Logs contain no traceback, OOM, CUDA error, NaN, or Inf.

Tracked artifacts:

- `summary.json`: aggregate distributions, checkpoint contract, and gate.
- `per_frame.csv`: 385 frame-level rows and alignment/same-stage diagnostics.
- `training_history.json`: all ten epoch records and formal configuration.
- `manifest.txt`: revision, drive roles, stage split, and metric scope.

The checkpoint and logs remain on the server at
`/tmp/predify-storage/experiments/seed0_stage4_aligned_temporal_difference_fdc4743`.
The 1.41 GB checkpoint SHA-256 is
`b17ed5c8988c6b11761c8cb7a68d74e655a122b82303162f8224193ca226c4ee`.

Tracked artifact SHA-256 values:

- `summary.json`: `b45b7d8346f5ade5fbdb9220ed833ad8ad7ae5f26c9c8b70b0f7e9e30fb8dad4`
- `per_frame.csv`: `86cbe6352042ce154aefe6c85985bd11d6888402876f45f81075bb78a14727a5`
- `training_history.json`: `5f7e0eca2334f225989b8450ad00c73f171c5ead456528eae1610db9f99580da`
- `manifest.txt`: `4baa59e7244e7bfa93e4755489a6d00fff8edcf0f1bb7714af9b674777f3c86e`
