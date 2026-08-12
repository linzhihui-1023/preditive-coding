# Prediction-error persistent-shift separability

This directory contains the lightweight outputs from the inference-only
go/no-go experiment at Git revision:

```text
a28fed59116f12fc641384d786b86600ebcf04e4
```

It reuses the strict Temporal Error best checkpoint from revision
`3ffbff0156dc9435fe3b060f4c9999703729ea8a`. No optimizer was created and no
network parameter changed. Each of two drives ran three independently reset
150-transition trajectories: clean, persistent Gaussian blur, and absolute-
frame deterministic i.i.d. Gaussian noise. The schedule was 40 clean, 80
disturbed, and 30 recovery transitions.

`summary.json` records the predeclared statistics, calibration-drive score
directions, all AUROCs, checkpoint hash, phase distributions, and the user-
defined decision threshold. `per_frame.csv` contains all 900 causal prediction
errors and statistics with raw-frame provenance.

The selected score is low `||e_t||`, not high error. It reaches held-out AUROC
0.959609 against the pooled clean and i.i.d.-noise controls. This establishes
separability for the tested corruption types, not separability of persistence
itself; blur and Gaussian noise do not have matched marginal corruption.
