# VGG local matching and causal historical warp

This directory contains the complete small outputs from the ordered diagnostic
at Git revision:

```text
06ec8e79832b981873854da14939bcebb33f2974
```

Execution order was stage-5 future-selected local matching, stage-4 local
matching, then causal historical motion and forward warp. The original files
were copied byte-for-byte from:

```text
/tmp/predify-storage/experiments/vgg_local_motion_06ec8e7/
```

`summary.json` contains all 48 aggregate entries, distributions, method
definitions, execution order, and the predeclared causal Copy gate.
`per_frame.csv` contains 9,096 rows with frame provenance, Copy and result MSE,
gain, matching cost, motion scale, forward-warp coverage, and collision rate.

Future-selected local matching is noncausal. Causal historical warp uses only
`F_(t-1)` and `F_t` to estimate motion, then forward-splats `F_t`; collisions
are averaged and holes fall back to Copy-current.
