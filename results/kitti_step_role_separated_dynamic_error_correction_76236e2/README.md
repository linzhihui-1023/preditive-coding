# Role-Separated Dynamic Error Correction

This experiment keeps the Role-Separated Predictor frozen and tests whether its Dynamics prediction error, accumulated with `0.207 / 0.793`, improves the Semantic Temporal Prior on the final host segmentation task. The Predictor history contains noisy observations only; correction posterior states are not fed back.

The Role Separation Quick Gate was rerun with the formal Z4 ratio limit `<= 1.10`. The Quick Gate and the existing full validation now both report `DYNAMICS_SEMANTIC_ROLE_SEPARATION: GO`.

Correction training updated only the two Z1/Z4 `ErrorGainCorrection` modules (`328192` parameters), for 3 epochs. Each epoch visited 5003 train and 2963 validation frames. Best epoch: 3, validation state MSE `0.0173455021`.

| Path | mIoU |
| --- | ---: |
| Clean Static Host | 0.6552125562 |
| Noisy Static Host | 0.2940764905 |
| No Correction (Semantic Prior) | 0.0137861915 |
| Dynamic Error Correction | 0.0143510873 |

This historical result is invalid as a mechanism conclusion: the Semantic Diagnostic State used as the no-correction prior collapsed from the noisy host mIoU `0.2940765` to `0.0137862` after writeback. Its former `PREDICTION_ERROR_DRIVEN_CORRECTION: GO` label is retained as `historical_decision` only and superseded by `INVALID_BASELINE_COLLAPSE: NO-GO`. The checkpoint and raw metrics remain preserved for reproducibility.
