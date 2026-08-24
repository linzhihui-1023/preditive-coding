# RAFT Dynamic Error Memory Alignment

This paired validation experiment tests whether a frozen, pretrained
TorchVision RAFT-Large can provide a useful spatial correspondence for the
previous dynamic error memory. Clean frames are used only for the RAFT input;
the model path uses the same `sigma=0.10` noisy frames for Prediction Only,
Unaligned Closed-loop, and RAFT Flow-Aligned Closed-loop.

Only Z1 and Z4 use the backward-flow warp. Z2 and Z3 retain the unaligned
dynamic-error update. Host, Adapter, Predictor, Correction, and RAFT are all
frozen; no optimizer or new learnable parameter is used.

On the full Val split (2,981 frames, 9 sequences), Flow-Aligned mIoU is
`0.302095623` versus Unaligned Closed-loop `0.301130569`, a gain of
`+0.000965053`. The valid warp ratios are `0.992543` for Z1 and `0.988708`
for Z4. All states are finite and the phase stability check passes.

Decision: `ALIGNMENT INCONCLUSIVE` because the gain is positive but below
`+0.005`.
