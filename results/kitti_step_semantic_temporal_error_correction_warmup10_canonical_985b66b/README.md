# Semantic Temporal Error Correction

This canonical result uses the persistent-blur protocol with `floor(0.10*T)` clean warm-up frames per sequence. Warm-up frames initialize temporal state only and are excluded from reported metrics; the remaining 2686 of 2981 validation frames are evaluated.

The correction module was trained for 7 epochs from the fixed protocol. Best epoch: 7. Validation total loss: `0.4935865566`.

Results on the post-warm-up validation frames:

- Clean host mIoU: `0.6657964071`
- Corrupted host mIoU: `0.3936403210`
- Corrected B mIoU: `0.4354863514`
- New semantic temporal correction mIoU: `0.4634747094`
- New minus corrupted host: `+0.0698343885`
- Recovery ratio: `0.2565968282`

All recorded finite, causality, no-clean-leakage, writeback, and correction-path gates passed. Historical results using the former one-third warm-up protocol are retained separately and are not treated as directly comparable.
