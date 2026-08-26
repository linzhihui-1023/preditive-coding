# Direct State Correction

Direct State Correction was trained from a new initialization for 3 epochs on the 12 KITTI-STEP Train sequences, using the fixed host-conditioned writeback checkpoint, fixed Predictor, and Gaussian noise sigma `0.10`. Only the two `z1/z4` correction modules were trained; no labels or segmentation loss were used. Each epoch processed 5,003 post-initialization frames in sequence order.

Training losses were `0.0005802987`, `0.0005165225`, and `0.0004973021`. The correction checkpoint is:
`/home/lin/predify/experiments/kitti_step_direct_state_correction/direct_state_correction_epoch3.pt`

The paired Val evaluation used 9 sequences, 2,981 total frames, and 2,963 post-warm-up frames.

| Path | mIoU |
| --- | ---: |
| Clean Static | 0.6552125562 |
| Noisy Static | 0.2940764905 |
| Direct State Correction | 0.2870732213 |

Direct State Correction was below both Noisy Static and the Oracle Gain reference `0.4222064482` by `0.1351332269`. Static reference checks, finite checks, and the evaluation protocol passed. Decision: `NO-GO`.
