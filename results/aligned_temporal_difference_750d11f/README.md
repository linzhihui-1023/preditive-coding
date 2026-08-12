# Stage-5 Aligned Temporal-Difference Predictor

This directory contains the lightweight auditable outputs from the formal
seed-0 run trained at Git revision
`750d11f5dab703cd089518eede05037e740d9c2e`.

## Fixed Protocol

- Train drive: KITTI `2011_09_26_drive_0005_sync`, 153 transitions.
- Held-out drive: KITTI `2011_09_26_drive_0011_sync`, 232 transitions.
- Feature target: frozen pretrained VGG stage 5 at horizon 1 (`0.1035 s`).
- Alignment: local radius-1, patch-size-3 matching from `F_(t-1)` to `F_t`;
  the matched source is used in current coordinates. Historical future
  splatting is not used.
- Temporal input: `D_t = F_t - align(F_(t-1), F_t)`. The predictor receives
  `[F_t, D_t]`, with `D_t = 0` immediately after a drive reset.
- Prediction: `Fhat_(t+1) = F_t + P([F_t, D_t])`. No Temporal Fusion module is
  created, and `F_t` remains the prediction base.
- Training: 10 epochs, seed 0, learning rate `1e-4`. The VGG backbone and all
  Target Flow feedback decoders are frozen; only the Future Predictor is
  optimized. Previous features and the aligned temporal difference are
  detached.
- Checkpoint selection: minimum held-out feature MSE, selecting epoch 2.
- Gate: the previously fixed Copy-current metrics from the same held-out
  drive. No additional experimental condition was run.

## Held-Out Gate

| Metric | Aligned difference | Copy-current gate | Required | Pass |
| --- | ---: | ---: | :---: | :---: |
| Feature MSE | 0.06180378 | 0.06008010 | `<` | No |
| Cosine | 0.91604762 | 0.91990469 | `>` | No |
| Normalized error | 0.38434362 | 0.37576611 | `<` | No |

The predeclared all-three gate failed. Alignment was applied to `152/153`
training samples and `231/232` held-out samples. Across all 385 replay rows,
Temporal Fusion was never applied and the maximum absolute difference between
the prediction base and `F_t` was exactly `0.0`.

## Artifacts

Tracked here:

- `summary.json`: aggregate distributions, checkpoint contract, and gate.
- `per_frame.csv`: 385 frame-level rows and difference/alignment diagnostics.
- `training_history.json`: all 10 epoch records and the formal configuration.
- `manifest.txt`: revision, output path, drive roles, and fixed gate values.

The checkpoint and logs remain on the server at
`/tmp/predify-storage/experiments/seed0_aligned_temporal_difference_750d11f`.
The 1.41 GB best checkpoint SHA-256 is
`7deb9945ea751cc6d71e5819b5d2333e5bf277ed095337a3dd0e2a79890e613f`.

Tracked artifact SHA-256 values:

- `summary.json`: `b542de4aa23f8a6f5666a2637de8b65efc0ee9cc34edee086dca67ba6478f7cc`
- `per_frame.csv`: `b6808f85523e7dbe086517debd2410d22b4479b85bcaa759ac909249b7506fab`
- `training_history.json`: `339c198d88d4ffc8761f4865a63134ac7f8ce789bcd69a7ad1134a8bfce40444`
- `manifest.txt`: `9a591fa79d8d76d8d99564f4f2a1da214a8aea794e15a957fd58c82ccf726bba`
