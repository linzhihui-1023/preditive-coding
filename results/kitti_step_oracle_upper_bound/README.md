# Oracle upper-bound diagnostic

The diagnostic code was added in commit b9151188b621d8c36cde993aabea730b054ef761.

Execution is blocked in the current agent environment: the KITTI-STEP dataset and model checkpoints are not mounted, and CUDA is unavailable. Therefore no mIoU, recovery, headroom, gain statistics, or gate result is recorded here. Historical values are not substituted.

The script is ready to run on the research host:

```bash
python -m predify2021.mce_scores.evaluate_kitti_step_oracle_upper_bound
```

It writes this directory's `summary.json` after checking the paired 9-sequence, 2981-frame, 2963-evaluation-frame protocol.