# Gradient-Isolated Semantic Recurrent Predictor

The fixed `lambda_state=1593.4333` objective was trained from random initialization for 3 epochs; epoch 3 was selected as best by validation combined loss (`1.4784920`). Semantic gradient was stopped only on the `h4 -> Z1` path and on the semantic-loss decode of predicted Z4; the Z4 state-loss path remained differentiable.

The one-clip gradient Gate passed: Z4 semantic gradient `0`, Z4 state gradient `0.0248288`, and Z1 semantic gradient `18.1128`.

Quick Gate on Val sequences `0002, 0006` (499 effective frames):

| Method | Mean MSE | Z1 MSE | Z4 MSE | NMSE | mIoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| Persistence | 0.0001615951 | 0.0002658827 | 0.0000573074 | 1.0000 | 0.1137686 |
| Balanced New | 0.0003159114 | 0.0002855130 | 0.0003463098 | 1.9550 | 0.2696656 |
| Gradient Isolated | 0.0003257101 | 0.0005905364 | 0.0000608837 | 2.0156 | 0.1768476 |
| Old CNN | 0.0001322474 | 0.0002116448 | 0.0000528501 | 0.8184 | 0.1114309 |

Decision: `GRADIENT_ISOLATION: NO-GO`. The semantic gradient was isolated as intended, but overall dynamics remained worse than Persistence and semantic retention fell below `0.2497`; the full 9-sequence evaluation was not run.
