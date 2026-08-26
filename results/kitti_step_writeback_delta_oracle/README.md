# Delta writeback oracle diagnostic

The output adapters for `z1` and `z4` were trained for one epoch on 5,027 KITTI-STEP Train frames. Only 294,912 writeback parameters were updated; labels, Predictor, Correction, and closed-loop state were not used by the training objective.

## Training

- Mean writeback loss: `0.0086301687`
- `z1 -> c1` MSE: `0.0090728314`
- `z4 -> c4` MSE: `0.0081875060`
- Checkpoint: `/home/lin/predify/experiments/kitti_step_writeback_delta/writeback_delta_epoch1.pt`

## Validation

The paired Oracle evaluation used all 9 KITTI-STEP Val sequences, 2,981 total frames, 2,963 post-warm-up frames, seed 0, and Gaussian noise sigma 0.10.

| Path | Before | Trained writeback | Delta |
| --- | ---: | ---: | ---: |
| Learned context | 0.3157625378 | 0.3209319916 | +0.0051694538 |
| Clean-state injection | 0.4053099845 | 0.4582736605 | +0.0529636760 |
| Oracle clipped gain | 0.3569645412 | 0.3882007270 | +0.0312361858 |

Both predefined thresholds passed, and all evaluation gates passed. Decision: `WRITEBACK GO`.
