# Host-conditioned writeback epoch-3 evaluation

The host-conditioned writeback modules were initialized from the prior delta-writeback checkpoint and trained for 3 epochs on the 12 KITTI-STEP Train sequences. Each epoch visited 5,027 frames, for 15,081 frame visits total. The run used AdamW, learning rate `1e-4`, weight decay `0.01`, batch size 1, seed 0, and Gaussian noise sigma `0.10`.

Checkpoint: `/home/lin/predify/experiments/kitti_step_host_conditioned_writeback/host_conditioned_writeback_epoch3.pt`

The paired validation used 9 sequences, 2,981 total frames, and 2,963 post-warm-up frames.

| Path | mIoU |
| --- | ---: |
| Clean static | 0.6552125562 |
| Noisy static | 0.2940764905 |
| Learned context | 0.3376793292 |
| Clean-state injection | 0.5305473217 |
| Oracle clipped gain | 0.4222064482 |

Relative to the required references, learned correction improved by `+0.0219167914`, and clean-state injection improved by `+0.1252373372`. Both thresholds and all existing evaluator gates passed. Decision: `WRITEBACK GO`.
