# Z4-only Stage P

Stage P trains only the existing Z4 predictor with strict causal order
`Zhat_t -> Z_t -> e_t -> Zhat_(t+1)`. The run started from the original
FAST-B epoch-3 checkpoint and used TBPTT=16 on all 12 training sequences.

Zero-step equivalence passed (`max_abs_z4_difference=0.0`). After 15 epochs,
the best full9 validation result was epoch 15:

- frame-weighted prediction MSE: `7.70453731205473e-05`
- frame-weighted persistence MSE: `8.15197619535933e-05`
- `R_pred`: `0.9451128324492271`
- sequences better than persistence: `8/9`
- effective validation frames: `2972`

The Stage P gate (`R_pred < 1` and at least 6/9 sequences better) passed.
Stage C/W/J were not started automatically.
