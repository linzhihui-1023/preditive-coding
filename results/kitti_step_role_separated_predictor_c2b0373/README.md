# Role-Separated Recurrent Predictor

The recurrent predictor was split into independent Dynamics and Semantic branches for Z1 and Z4. Dynamics predictions alone define future prediction errors. Semantic diagnostic states use detached Dynamics predictions and cannot enter prediction-error recurrence.

The Gradient Responsibility Gate passed: semantic-to-dynamics gradient norm was `0`, semantic-to-semantic was `46.0485`, dynamics-to-semantic was `0`, and dynamics-to-dynamics was `0.0522`.

The best checkpoint is epoch 2, selected by the lowest 64-clip validation combined loss (`0.5366522`). Full validation used all 9 KITTI-STEP sequences and 2,963 effective frames.

| Method | Mean MSE | MSE ratio vs Persistence | Predicted-state mIoU | wIoU | mVC8 | mVC16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Persistence | 0.0002596572 | 1.0000 | 0.1191175 | 0.4125660 | 0.5150084 | 0.4749011 |
| Old CNN | 0.0001874364 | 0.7219 | 0.1159907 | 0.4038862 | 0.5075491 | 0.4681940 |
| Balanced New | 0.0003709039 | 1.4284 | 0.2946898 | 0.7375642 | 0.8850869 | 0.8956489 |
| Gradient Isolated | 0.0003456005 | 1.3310 | 0.1801848 | 0.5645712 | 0.6753043 | 0.6453434 |
| Role Separated | 0.0002032951 | 0.7829 | 0.3162580 | 0.7271152 | 0.8762185 | 0.8861476 |

Role Separated improves the mean state MSE over Persistence and maintains a predicted-state mIoU above `0.24`. Z1 MSE is lower than Persistence; Z4 remains near Persistence (`1.0255x`) rather than reproducing the Balanced predictor's large Z4 error. Decision: `DYNAMICS_SEMANTIC_ROLE_SEPARATION: GO`.

The next stage remains Prediction Error -> Dynamic Error -> Correction. Alignment is not indicated by this experiment.
