# Oracle upper-bound diagnostic

The diagnostic code is in commit 314d9d986b1ff44b5ec23875b3086ac05b1f900d.

Execution is blocked in the current agent environment: the KITTI-STEP dataset and model checkpoints are not mounted, and CUDA is unavailable. No mIoU, recovery, headroom, gain statistics, or gate result is recorded here. Historical values are not substituted.

The code uses the formal GAIN_DYNAMIC_INSTANT input mask, evaluates all five paths on the same 2963 post-warm-up frames, and checks all five logits for finite values.

Run on the research host:

```bash
python -m predify2021.mce_scores.evaluate_kitti_step_oracle_upper_bound
```