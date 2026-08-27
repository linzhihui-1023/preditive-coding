# Error Decomposition + Reliability Correction

This experiment separates observation corruption from predictor mismatch, accumulates only the estimated corruption with the fixed positive dynamic recursion, and uses a reliability map to gate a free direct state correction.

The model was trained for three epochs and evaluated on all nine KITTI-STEP validation sequences. Both corruption NMSE gates passed, but the final mIoU was `0.3410877488`, below the Corrected B reference `0.3505486298`; the result is `ERROR_DECOMPOSITION_RELIABILITY: NO-GO`. The conditional dynamic-error ablation was not run.
