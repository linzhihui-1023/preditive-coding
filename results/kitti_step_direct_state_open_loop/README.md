# Direct State Correction open-loop diagnostic

This evaluation used the fixed Direct State Correction epoch3 checkpoint and the fixed host-conditioned writeback epoch3 checkpoint. No parameters were updated. The KITTI-STEP Val protocol was 9 sequences, 2,981 total frames, 2,963 effective frames, seed 0, and Gaussian noise sigma `0.10`.

The current-frame Direct posterior was decoded through the writeback path, while the next-frame Predictor history used only the noisy observation. Clean state was not placed in history. All parameters remained unchanged, values were finite, and the protocol gate passed.

| Quantity | Value |
| --- | ---: |
| Noisy -> Clean state MSE | 0.0004620672 |
| Direct Open-loop -> Clean state MSE | 0.0003377442 |
| Direct Open-loop mIoU | 0.3347162547 |
| Historical Noisy Static mIoU | 0.2940764905 |
| Historical Direct Closed-loop mIoU | 0.2870732213 |

Direct Open-loop improves the state MSE over the noisy observation and improves mIoU over both historical references. The resulting diagnosis is `CLOSED-LOOP ACCUMULATION BOTTLENECK`: the single-frame correction is useful, but feeding the corrected posterior back into the closed-loop recurrence reduced performance in the prior closed-loop result.
