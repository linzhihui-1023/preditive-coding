# Semantic Temporal Error Correction Rerun

This run removes Q/K L2 normalization while keeping the 3x3 window, 32-dimensional projections, and scaled dot-product score `Q^T K / sqrt(32)`. The alignment ratio `R_align = ||Vobs - Valign_pred||_1 / ||Vobs - Vpred||_1` is recorded in the validation summary.

The model was retrained from initialization for 3 epochs on the persistent Gaussian blur protocol. All ten runtime gates passed. The new mIoU is `0.5083952797`, an improvement of `+0.0120151952` over the current best reference `0.4963800845`, so the fixed mIoU classification is `WEAK`.

The alignment ratio is `2.6460` for Z1 and `2.2698` for Z4, both above 1, so projected local alignment did not make prediction features closer to the observation. No additional tuning was performed.
