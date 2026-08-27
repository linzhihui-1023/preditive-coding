# Semantic Temporal Error Correction

This rerun fixes the local attention Softmax axis and evaluates the semantic-temporal correction on the persistent Gaussian blur protocol (kernel 11, sigma levels 0.75/1.5/2.25/3.0). The model was retrained for 3 epochs with frozen host, adapters, predictor, and writeback.

The best checkpoint is epoch 3. Full validation used 9 KITTI-STEP sequences and 2,963 effective frames. All ten runtime gates passed. The new mIoU is `0.5081481503`, versus `0.4963800845` for the current error-decomposition reference, so the fixed-threshold result is `WEAK` (below `0.5165`).

The 9-position attention remained near uniform (`max weight` about `0.1115/0.1118`, entropy about `ln(9)`), so the local-correlation sanity status is `NEAR_UNIFORM_CORRELATION`; no further tuning was performed.
