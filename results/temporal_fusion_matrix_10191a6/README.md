# Stage-5 Two-Frame Temporal Fusion Matrix

This directory contains the lightweight auditable outputs from the formal
seed-0 matrix trained at Git revision
`10191a6d2d04df325ffbc7959f9e633f0af43d3b`.

## Fixed Protocol

- Train drive: KITTI `2011_09_26_drive_0005_sync`, 153 transitions.
- Held-out drive: KITTI `2011_09_26_drive_0011_sync`, 232 transitions.
- Feature target: frozen pretrained VGG stage 5 at horizon 1 (`0.1035 s`).
- Copy-current: no optimized parameters, one exact evaluation epoch.
- Current-only: train the future predictor for 10 epochs.
- Temporal Fusion: train the residual 1x1 Conv-ReLU-Conv fusion module and
  future predictor for 10 epochs.
- The VGG backbone and all Target Flow feedback decoders are frozen. State is
  reset at each drive boundary, and previous top features are detached.
- Checkpoint selection uses held-out feature MSE. The selected epochs were 1,
  2, and 1 for Copy-current, Current-only, and Temporal Fusion respectively.

## Held-Out Results

| Condition | Feature MSE | Cosine | Normalized error |
| --- | ---: | ---: | ---: |
| Copy-current | 0.06008010 | 0.91990469 | 0.37576611 |
| Current-only | 0.06179707 | 0.91581467 | 0.38497547 |
| Temporal Fusion | 0.06192617 | 0.91600015 | 0.38513246 |

Temporal Fusion increased held-out MSE by `3.0727%` relative to Copy-current
and by `0.2089%` relative to Current-only. Its cosine was `0.0001855` higher
than Current-only, while its normalized error was `0.0408%` higher. Thus this
specific minimal fusion did not improve the held-out next-frame prediction
matrix.

This is not a claim that temporal information is generally useless. It is a
single-seed result for stage 5, two frames, 1x1 fusion, training on drive 0005,
and testing on drive 0011. Both learned conditions reduced training-drive MSE
but generalized below Copy-current.

## Causal Audit

Temporal fusion was applied on `152/153` train transitions and `231/232`
held-out transitions. The only unfused transition was sample 0 after each
drive reset. The held-out mean fusion-residual RMS was `0.0464521`, so the
trained fusion path was active rather than numerically ignored. The 1155 CSV
rows have 1155 unique `(condition, split, raw frame)` keys.

## Artifacts

Tracked here:

- `summary.json`: aggregate distributions, checkpoint contracts, and all
  pairwise comparisons.
- `per_frame.csv`: all 1155 frame-level predictions and fusion diagnostics.
- `training_history.json`: complete epoch curves and serialized formal
  configuration for all three conditions.
- `manifest.txt`: revision, output path, groups, and drive roles.

Large checkpoints and logs remain on the server at
`/tmp/predify-storage/experiments/seed0_temporal_fusion_matrix_10191a6`
(`4.0G` total). Best-checkpoint SHA-256 values:

- Copy-current: `410604e43674b8d372ef95cf697cb6b70d7af5c00aa64d922a9d6c32b926843e`
- Current-only: `f366deff9779608fd2b5043ce5d7dc0c202fcb057103c2cd3673c1635923baf0`
- Temporal Fusion: `df054256ad1ad07e01091bfb93c1905a33320fa335465e8cb1d04d085ce57239`

Tracked artifact SHA-256 values are:

- `summary.json`: `b6b90534783561300b757fa52fccb5f361a012562e32b2110d70a53ef092a9fd`
- `per_frame.csv`: `81c81919f1d4dda5119dccc3bb191c53228c17fd4c63e3692570dc825e7da315`
- `training_history.json`: `fa27e4779b4d4831c451e0640b030bde9a647b7d40cf535752366bb3dcc5716b`
- `manifest.txt`: `eef751812c0bc68b71f7ff506b53ea234422f2d9f5dac5646cce701bf72be0b9`

The server-generated CSV uses standard library CRLF output and has SHA-256
`b7a2ac66d7674c7d09c79bbf3bbec8d90c489640eb00e2e67cace0fc0abcae62`.
The tracked copy differs only by LF normalization; all 1155 parsed records and
numeric values are identical.
