# Host-conditioned writeback oracle diagnostic

Host-conditioned residual writeback was trained for one epoch on 5,027 frames from the 12 KITTI-STEP Train sequences. The run used batch size 1, AdamW, learning rate `1e-4`, weight decay `0.01`, seed 0, and Gaussian noise sigma `0.10`. Only 884,736 parameters in the `c1` and `c4` writeback modules were trainable; the static host, adapters, Predictor, and Correction remained frozen.

Training loss was `0.0056816564` (`c1`: `0.0047347058`, `c4`: `0.0066286069`). Checkpoint:
`/home/lin/predify/experiments/kitti_step_host_conditioned_writeback/host_conditioned_writeback_epoch1.pt`

The paired Oracle validation used 9 Val sequences, 2,981 total frames, 2,963 post-warm-up frames, seed 0, and sigma `0.10`.

| Path | mIoU |
| --- | ---: |
| Clean static | 0.6552125562 |
| Noisy static | 0.2940764905 |
| Learned context with writeback | 0.3418815657 |
| Clean-state injection with writeback | 0.4978327134 |
| Oracle clipped-gain with writeback | 0.4137196502 |

Relative to the recorded pre-writeback references, learned context improved by `+0.0261190279`, and clean-state injection improved by `+0.0925227289`. Both requested thresholds passed: `WRITEBACK GO`. All evaluator gates passed, including finite values, checkpoint loading, protocol, and no-training parameter invariance during validation.
