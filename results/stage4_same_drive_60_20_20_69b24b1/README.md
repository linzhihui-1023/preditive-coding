# Stage-4 Same-Drive 60/20/20 Diagnostic

This directory contains the lightweight auditable outputs from the one-time
seed-0 diagnostic trained at revision
`69b24b1e0d6e23f82011ed6533565485c771b783`.

## Protocol

- Drive: KITTI 0005 only.
- Raw-frame Train: `0--91`; 91 transitions.
- Raw-frame Val: `92--122`; 30 transitions.
- Raw-frame Test: `123--153`; 30 transitions.
- Boundary-crossing transition starts 91 and 122 were excluded. The three raw
  frame sets are disjoint.
- All samples remained in true time order; no loader shuffled.
- Model: Stage-4 aligned temporal difference, radius 1 Stage-4 cell, patch
  size 3, residual Future Predictor, fixed Stage-5 Target Flow top.
- Training: ten epochs on Train. Val alone selected epoch 10. Test was replayed
  once after checkpoint selection.

## Result

| Split | Metric | Predictor | Same-segment Copy | Direction |
| --- | --- | ---: | ---: | :---: |
| Val | MSE | 1.98922046 | 2.41822275 | Better |
| Val | Cosine | 0.56127751 | 0.55808507 | Better |
| Val | Normalized error | 0.85218414 | 0.93812825 | Better |
| Test | MSE | 1.51209933 | 1.67944006 | Better |
| Test | Cosine | 0.69367177 | 0.70890513 | Worse |
| Test | Normalized error | 0.72217820 | 0.76012611 | Better |

Test MSE improved by `9.964079%`, and normalized error improved in the same
direction. Cosine decreased by `0.01523336`, so the strict all-three gate did
not pass. The result supports same-drive forward generalization for Euclidean
feature error in this fixed run, but not metric-uniform superiority. It does
not remove the earlier cross-drive failure and does not motivate more
same-drive variants.

## Audit

The audit verified 151 unique split/frame rows, exact 91/30/30 counts, exact
raw-frame ranges, two excluded boundaries, no shared raw frames, no shuffle,
adjacent frame indices, Stage-4/Stage-5 target separation, `512x28x28`
prediction features, CSV-to-summary means, epoch-10 Val selection, Test access
policy, predictor-only optimized parameters, and finite metrics. Each split
reset history at its start. Logs contain no traceback, OOM, CUDA error, NaN,
or Inf.

Tracked artifacts:

- `summary.json`
- `per_frame.csv`
- `training_history.json`
- `manifest.txt`

The checkpoint and logs remain at
`/tmp/predify-storage/experiments/seed0_stage4_same_drive_60_20_20_69b24b1/`.
Checkpoint SHA-256:
`566a7da87e4c51063d43433ff8638e9f4499cda32509e6d1dae018e5da22106e`.

Tracked artifact SHA-256 values:

- `summary.json`: `22beafb2489d2bf5b0207d0090e1e32d38af5cf0f19d27216adb021afab01f09`
- `per_frame.csv`: `9d186cf8e9634a8a74cbf27f251d48625f2bd64222f338fd103301872e519f8f`
- `training_history.json`: `ce57a5b322dcbbec6a4ee9fbf56ab2ff2887bbea2aedcb7335ee6eebe95a2909`
- `manifest.txt`: `9b7752150241247d121460e97d57d231ad8bab5b880f066a80e2798f9d2a9198`
