# Residual Writeback Experiment B Rerun

This rerun verifies that semantic correction training uses the same residual writeback path as evaluation.

The model was trained from fresh correction initialization for three epochs and evaluated on all nine KITTI-STEP validation sequences. Clean and noisy reference mIoU were reproduced. Corrected B reached `0.3503165758`; this is `-0.0002320540` versus the previous `0.3505486298` reference and remains above the `0.350` GO threshold.
