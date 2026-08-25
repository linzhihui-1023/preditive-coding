# Oracle upper-bound diagnostic

The diagnostic code is in commit 314d9d986b1ff44b5ec23875b3086ac05b1f900d.

The complete KITTI-STEP validation run used 9 sequences, 2981 total frames, 2963 post-warm-up evaluation frames, seed 0, and Gaussian noise sigma 0.10. No optimizer or backward pass was used, and all loaded parameters remained unchanged.

## Results

| Path | mIoU |
| --- | ---: |
| Clean static | 0.6552125562 |
| Noisy static | 0.2940764905 |
| Learned GAIN_DYNAMIC_INSTANT | 0.3157625378 |
| Clean-state injection | 0.4053099845 |
| Oracle clipped-gain closed loop | 0.3569645412 |

The oracle clipped-gain headroom over the learned context correction is 0.0412020034 mIoU. Clean-state injection recovers 30.8009929% of the clean-to-noisy loss. All protocol, finite-value, gain-range, exact-injection, MSE-dominance, reproduction, and no-training gates passed.

Diagnosis: `ADAPTER_WRITEBACK_BOTTLENECK`.
