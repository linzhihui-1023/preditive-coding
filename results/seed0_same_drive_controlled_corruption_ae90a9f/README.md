# Same-drive controlled corruption, seed 0

This directory contains the small, reviewable outputs from the formal
same-drive controlled-corruption run trained and evaluated at Git revision:

```text
ae90a9f74f1560f5b3d6d72905cfe1b16a5b34c4
```

The original evaluator output was copied byte-for-byte from:

```text
/tmp/predify-storage/experiments/seed0_same_drive_controlled_corruption_ae90a9f/evaluation/
```

Contents:

- `evaluation/summary.json` contains the split, schedules, checkpoint
  provenance, selected-epoch metrics, metric definitions, trajectory metrics,
  phase means, state-scale diagnostics, and state-utilization diagnostics.
- Each `evaluation/*_recovery_curve.csv` contains all 45 ordered validation
  transitions for one checkpoint and one independent trajectory. The columns
  include clean and corrupted `e`, completed and input `E`, feature MSE `L`,
  signed/absolute/positive excess, `RMS(F)`, and the zero-history ablation.
- `SHA256SUMS` records the exact hashes of the versioned evaluator outputs.

The approximately 4 GB checkpoints remain server-only. PNG plots, JSONL files,
training logs, and pickle histories are also omitted because the CSV files and
summary retain the numerical data needed to audit AUEC, phase curves, causal
state traces, recovery metrics, and the reported state-utilization conclusion.

This is a one-seed, 45-transition same-drive mechanistic diagnostic. It is not
evidence of broad robustness or cross-drive generalization.
