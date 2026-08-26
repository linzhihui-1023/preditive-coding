# Semantic Direct State Correction

Semantic-supervised Direct State Correction versus matched observation-only control. Fixed host, adapter, predictor, writeback, noise protocol, and open-loop history.

- Full error mIoU: `0.3437739181`
- Observation-only mIoU: `0.3442450865`
- Direct open-loop baseline: `0.3347162547`
- Full error minus observation-only: `-0.0004711684`

Both modes completed 3 epochs with 5003 frames per epoch. Validation used 9 sequences, 2981 frames, and 2963 evaluated frames. Observation-only is slightly higher; prediction/dynamic error inputs show no independent semantic correction contribution: **NO-GO**.
