# Semantic V3-TemporalStats E3

Protocol: existing V3 fast validation protocol; 3 training epochs, fixed fast
validation sequences `0002/0010/0018`, post-warmup 781 frames, Blur-Mid
(`sigma=2.25`) and Blur-Max (`sigma=3.0`). Best checkpoint is epoch 1 by the
existing validation-mIoU rule.

## Result A — temporal/error diagnostics

| condition | Temporal Gain | Error Contribution |
|---|---:|---:|
| Blur-Mid | +0.022053 | +1.20e-7 |
| Blur-Max | +0.032731 | +9.50e-8 |

Temporal Gain passes the sign test, but Error Contribution does not reach the
required `1e-4` scale.

## Result B — semantic state

| condition | State Recovery | Hidden Growth |
|---|---:|---:|
| Blur-Mid | -1.331463 | 1.01256 |
| Blur-Max | -0.917674 | 1.01482 |

State Recovery is negative and worse than the paired V3-Structure reference;
hidden growth remains stable.

## Result C — feature restoration

| condition | Feature Recovery | Restoration Direction | Amplitude Ratio |
|---|---:|---:|---:|
| Blur-Mid | 14.456% | 0.47834 | 0.19131 |
| Blur-Max | 14.524% | 0.49969 | 0.17746 |

Feature Recovery exceeds the 8.5% floor and restoration direction is positive.

## Result D — fast mIoU (best epoch 1)

| condition | corrupted mIoU | restored mIoU | delta |
|---|---:|---:|---:|
| Blur-Mid | 0.437277 | 0.436440 | -0.000837 |
| Blur-Max | 0.349050 | 0.350575 | +0.001525 |

## GO / NO-GO

**NO-GO.** The Error Contribution and State Recovery hard gates fail, so the
9-sequence/15-epoch Step 3 experiment is not run.

Trainable parameters: 1,540,736 total; 49,152 newly added history-projection
parameters; dynamics remain frozen. Additional inference state is O(1) per
batch element (`mu`, `A`, `Q`, `R`, `e_prev`).
