# Stage-5 Aligned-History Temporal Fusion

This directory contains the lightweight auditable outputs from the formal
seed-0 run trained at Git revision
`95aa61d911dead2a89e0f749a74a0cef3329e42f`.

## Fixed Protocol

- Train drive: KITTI `2011_09_26_drive_0005_sync`, 153 transitions.
- Held-out drive: KITTI `2011_09_26_drive_0011_sync`, 232 transitions.
- Feature target: frozen pretrained VGG stage 5 at horizon 1 (`0.1035 s`).
- Alignment: local radius-1, patch-size-3 matching from `F_(t-1)` to `F_t`;
  the matched source is used in current coordinates. The historical future
  splat is not used.
- Model: `Z_t = F_t + T([F_(t-1)_aligned, F_t])`, followed by the existing
  residual future predictor.
- Training: 10 epochs, seed 0, learning rate `1e-4`. The VGG backbone and all
  Target Flow feedback decoders are frozen; only Temporal Fusion and the
  Future Predictor are optimized. Previous features are detached.
- Checkpoint selection: minimum held-out feature MSE, selecting epoch 2.
- Gate: the previously fixed Copy-current metrics from the same held-out
  drive. No Copy-current retraining or additional condition was run.

## Held-Out Gate

| Metric | Aligned Temporal Fusion | Copy-current gate | Required | Pass |
| --- | ---: | ---: | :---: | :---: |
| Feature MSE | 0.06191402 | 0.06008010 | `<` | No |
| Cosine | 0.91520128 | 0.91990469 | `>` | No |
| Normalized error | 0.38581802 | 0.37576611 | `<` | No |

The predeclared all-three gate failed. Alignment was applied to `152/153`
training samples and `231/232` held-out samples; sample 0 after each drive
reset had no previous feature. On alignment-applied held-out samples, the mean
previous-to-current MSE was `0.06032671` before matching and `0.05621678`
after matching.

## Artifacts

Tracked here:

- `summary.json`: aggregate distributions, checkpoint contract, and gate.
- `per_frame.csv`: 385 frame-level rows and alignment/fusion diagnostics.
- `training_history.json`: all 10 epoch records and the formal configuration.
- `manifest.txt`: revision, output path, drive roles, and fixed gate values.

The checkpoint and logs remain on the server at
`/tmp/predify-storage/experiments/seed0_aligned_temporal_fusion_95aa61d`.
The 1.41 GB best checkpoint SHA-256 is
`8e85c39f5380aa7aa4a4a778617ba3fc3522894144d146cf961ff7a0ad102dc6`.

Tracked artifact SHA-256 values:

- `summary.json`: `ad4d4f2e2aa24b0cdb8a5f0bfe4c5ce89e7064c161df01a3b8d230b77f5573f8`
- `per_frame.csv`: `fb4d07c9a553009f8230370ed3effd014a039ed6197454914f6fbfc4ad2ce534`
- `training_history.json`: `1ceaa39d58dcb0c6bd7fdb48fcf78718db2f7ee1dc8126a81d8c5a39d0ad0771`
- `manifest.txt`: `d43a3dd5c4c356cb93e3e16592471337c0e35a45d76eedb33d5d19ffdedf8f68`
