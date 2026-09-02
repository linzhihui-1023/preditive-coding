# Semantic V3-TemporalStats Task-Aware Fast Validation

Protocol: fixed existing fast validation (`3 Epoch`, `TBPTT=8`, sequences
`0002/0010/0018`, Blur-Mid `sigma=2.25`, Blur-Max `sigma=3.0`). Only semantic
V3-TemporalStats parameters were trained. The objective was
`1.0 * L_Z4 + 1e-4 * L_seg`; `L_seg` was evaluated once at each TBPTT window
end (632 windows/epoch). Best checkpoint selection remained validation mIoU;
best epoch is **2**.

## Result 1 — mIoU (best epoch 2)

| condition | Corrupted Host | Task-Aware Restored | Delta mIoU |
|---|---:|---:|---:|
| Blur-Mid | 0.437277 | 0.443069 | +0.005792 |
| Blur-Max | 0.349050 | 0.362950 | +0.013900 |

Mean Delta mIoU: **+0.009846**.

## Result 2 — training loss

| epoch | feature_loss (`L_Z4`) | segmentation_loss (`L_seg`) | total_loss |
|---:|---:|---:|---:|
| 1 | 5.19870e-5 | 0.189593 | 7.09013e-5 |
| 2 | 6.31822e-5 | 0.205035 | 8.36211e-5 |
| 3 | 5.00448e-5 | 0.182279 | 6.82041e-5 |

## Result 3 — internal diagnostics (best epoch 2)

| condition | Feature Recovery | Temporal Gain | State Recovery | Hidden Growth |
|---|---:|---:|---:|---:|
| Blur-Mid | 7.782% | -0.006419 | -2.141911 | 0.99432 |
| Blur-Max | 9.291% | +0.015750 | -1.389240 | 0.99517 |

Task-aware training improves mIoU in both conditions despite mixed internal
diagnostics. It resolves the observed “Z4 MSE improves but mIoU does not”
failure for this fast protocol.
