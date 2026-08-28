# Semantic Temporal Error Correction: 10% Warm-up / 7 Epochs

This run changes only the persistent-blur protocol warm-up from one third to `floor(0.10T)` and trains the existing correction for 7 epochs. Blur remains kernel `11`, sigma levels `0.75, 1.5, 2.25, 3.0`; warm-up frames only initialize causal state and are excluded from the main metrics.

The validation set contains 9 sequences and 2,981 frames. The post-warm-up evaluation contains 2,686 frames. Clean Host mIoU is `0.6657964071`, post-warm-up Corrupted Host mIoU is `0.3936403210`, and Corrected mIoU is `0.4634747094`, giving `Corrected - Corrupted = +0.0698343885` and recovery ratio `0.2565968282`. The historical `0.5083952797` result used a different one-third warm-up protocol.

All protocol, training, evaluation, causality, leakage, identity, and finite gates passed. Best epoch is `7` with validation total loss `0.4935865566`.
