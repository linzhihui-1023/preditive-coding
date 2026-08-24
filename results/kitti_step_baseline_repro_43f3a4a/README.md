# RAFT Baseline Reproduction

This run reproduces the Noisy Static and unaligned Closed-loop baselines from
the RAFT paired evaluation with `seed=0` and Gaussian noise `sigma=0.10`.
It uses the KITTI-STEP validation split and does not run RAFT or Alignment.

Results:

- Noisy Static mIoU: `0.294076490502174`
- Unaligned Closed-loop mIoU: `0.301130569379060`
- Absolute differences from the RAFT references: `2.17e-12` and `2.09e-11`

Decision: `BASELINE REPRODUCTION PASS`.
