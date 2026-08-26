# Z4 Context-Only Predictor Ablation

Two-sequence Val quick Gate, 499 effective frames, used the frozen `b0dcb11` checkpoint. `persist_z4=True` keeps the Z4 ConvGRU hidden state and its high-to-low Z1 path, while setting predicted Z4 to the previous observation state.

| Method | Mean MSE | Z1 MSE | Z4 MSE | NMSE | mIoU | wIoU | mVC8 | mVC16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Persistence | 0.0001615951 | 0.0002658827 | 0.0000573074 | 1.0000 | 0.1137686 | 0.3356716 | 0.3698967 | 0.3330185 |
| Old CNN | 0.0001322474 | 0.0002116448 | 0.0000528501 | 0.8184 | 0.1114309 | 0.3280606 | 0.3633676 | 0.3274189 |
| Balanced New | 0.0003159114 | 0.0002855130 | 0.0003463098 | 1.9550 | 0.2696656 | 0.7478527 | 0.8883986 | 0.9053662 |
| Z4 Context-Only | 0.0001874425 | 0.0003175775 | 0.0000573074 | 1.1600 | 0.1301691 | 0.3822648 | 0.4017662 | 0.3633525 |

Direct Z4 prediction is harmful for Z4 MSE: context-only returns to Persistence-level Z4 MSE. Overall NMSE remains above 1 and mIoU remains below the required 0.2497 semantic-retention threshold.

Decision: `Z4_NOT_SOLE_DYNAMICS_BOTTLENECK`.
