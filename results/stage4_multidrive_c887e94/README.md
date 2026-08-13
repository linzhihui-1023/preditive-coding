# Stage-4 Multi-Drive Aligned-Difference Experiment

This directory contains the lightweight auditable outputs from the formal
seed-0 experiment trained at revision
`c887e94edda86885fd0bedfaaf25e56c957719bb`.

## Protocol

- Train: drives 0005, 0013, 0014, and 0036; 1411 ordered transitions.
- Val: drives 0011 and 0039; 626 ordered transitions.
- Frozen Test: drives 0051 and 0056; 728 ordered transitions.
- Model: Stage-4 aligned temporal difference, radius 1 Stage-4 cell, patch
  size 3, residual Future Predictor, and fixed Stage-5 Target Flow top.
- Training: seed 0, ten epochs, no shuffled transitions, frozen VGG and
  feedback decoders, Future Predictor only.
- Val MSE alone selected epoch 7. The training process did not receive Test
  drive names.
- The selected checkpoint atomically claimed one Test access before reading
  0051/0056. Test metrics were used only for final reporting.
- Drive 0051 was evaluated as two independently reset fixed-time segments of
  56 and 379 transitions. Drive 0056 had one 293-transition segment.

## Result

| Split | Predictor MSE | Copy MSE | MSE gain | Predictor cosine | Copy cosine | Predictor NFE | Copy NFE | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| Train | 0.93394784 | 1.15164056 | +18.902836% | 0.79362428 | 0.76854944 | 0.59781858 | 0.66215935 | Pass |
| Val | 1.01279691 | 1.14439157 | +11.499093% | 0.82207205 | 0.81639943 | 0.55275378 | 0.58144107 | Pass |
| Frozen Test | 0.77194043 | 0.89291654 | +13.548423% | 0.83322664 | 0.82239107 | 0.54544640 | 0.58067843 | Pass |

Both frozen Test drives pass all three same-stage checks independently:

| Drive | Segments | Predictor MSE | Copy MSE | MSE gain | Cosine delta | NFE delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0051 | 56 + 379 | 0.70959795 | 0.80359568 | +11.697142% | +0.00721870 | -0.02687930 |
| 0056 | 293 | 0.86449668 | 1.02552602 | +15.702122% | +0.01620533 | -0.04763284 |

For this fixed seed and protocol, multi-drive training generalizes the
Stage-4 aligned-difference predictor beyond its training drives and clears the
predeclared frozen-Test Copy-current gate. This is evidence for the value of
training-drive diversity in this configuration. It is not a multi-seed claim
or a comparison of absolute Stage-4 and Stage-5 MSE values.

## Audit

The audit verified 2765 unique split/drive/sample rows; exact 1411/626/728
split counts; exact per-drive counts; 0051's two reset segments; Stage-4
prediction and Stage-5 Target Flow separation; `512x28x28` prediction
features; no Temporal Fusion; unchanged `F_t` prediction base; CSV-to-summary
means; ten training epochs; epoch-7 Val-only selection; predictor-only
optimized parameters; absence of Test drives from training config; completed
single-access receipt; finite metrics; and clean logs. The receipt and summary
agree with the 1.41 GB checkpoint SHA-256.

Tracked artifacts:

- `summary.json`
- `per_frame.csv`
- `training_history.json`
- `manifest.txt`
- `frozen_test_receipt.json`

The checkpoint and logs remain at
`/tmp/predify-storage/experiments/seed0_stage4_multidrive_c887e94/`.
Checkpoint SHA-256:
`8a29c40fc71f0d54a2d21306a9444407d25865a01262713d05f522b495ab7754`.

Tracked artifact SHA-256 values:

- `summary.json`: `91a75c1860b14e98ce8372d2cc22a5f5aba359b35c3aeb61da4db6224d484bb6`
- `per_frame.csv`: `475f4886167d9cf6a4411f507c3859659c2c57da886f208c3d55425ef03898ca`
- `training_history.json`: `94bfb07a2cb5be56612aab1ebc1b142c4eca099805aec68c7c190856a7666a28`
- `manifest.txt`: `eb83a8b44ad01fb15b724608e3bb3dc78f0def381b3081c6d6ab09e797745557`
- `frozen_test_receipt.json`: `60883ca0f607fbaaff8c699cd917ac07941bb0f64b965f3f83b075bad142eec1`
