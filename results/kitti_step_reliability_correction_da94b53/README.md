# Reliability-aware Correction

This experiment tests whether learned prediction and observation uncertainty
can improve the existing closed-loop correction. Only the new Z1/Z4
ReliabilityAwareCorrection modules were trained; Host, Adapter, Predictor,
legacy Correction, and the fixed dynamic-error coefficients remained frozen.

On KITTI-STEP Val with seed `0` and Gaussian noise `sigma=0.10`, Legacy
Closed-loop mIoU was `0.301130569379060` and Reliability-aware Closed-loop
mIoU was `0.293920443992310`, a difference of `-0.007210125386750`.

The reliability direction diagnostic was correct for both Z1 and Z4:
mean gain was higher where the observation residual was smaller than the
prediction residual. This did not translate into segmentation improvement.

Decision: `RELIABILITY CORRECTION NO-GO`.
