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

The primary paired difference is `+0.0005648957`, so the result is `PREDICTION_ERROR_DRIVEN_CORRECTION: GO`. This is a positive but small contribution; correction remains far below the noisy static reference under the fixed host-conditioned writeback path.
