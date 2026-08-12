# Stage-5 causal warp-residual matrix

This directory contains the lightweight audit outputs from the formal seed-0
matrix trained and evaluated at Git revision:

```text
79de55494d24f4e01083851f8331e6c96b2d87f1
```

The three conditions are Copy-current, deterministic causal historical warp,
and historical warp plus a learned residual predictor. Motion uses only
`F_(t-1)` and `F_t`, with radius 1 and 3x3 feature descriptors. The learned
condition predicts the residual relative to the warped base.

`summary.json` records checkpoint provenance, aggregate distributions, and
the train/validation comparisons. `per_frame.csv` contains all 1,155 replayed
predictions with raw-frame provenance, Copy/base/final errors, residual scale,
and warp coverage. `training_epochs.csv` contains all 60 condition/epoch/split
records. Large checkpoints and logs remain at:

```text
/tmp/predify-storage/experiments/seed0_warp_residual_matrix_79de554/
```
