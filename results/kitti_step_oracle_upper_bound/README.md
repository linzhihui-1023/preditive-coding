# Oracle upper-bound diagnostic

The diagnostic code is in commit 217f199d582e59c014d4ff65af8f34b5be671ae4.

Execution is blocked in the current agent environment: the KITTI-STEP dataset and model checkpoints are not mounted, and CUDA is unavailable. No mIoU, recovery, headroom, gain statistics, or gate result is recorded here. Historical values are not substituted.

Run on the research host:

```bash
python -m predify2021.mce_scores.evaluate_kitti_step_oracle_upper_bound
```

The script writes this directory's `summary.json` after checking the paired 9-sequence, 2981-frame, 2963-evaluation-frame protocol.