# Temporal–semantic interface quick screen

Diagnostic Blur-Mid/Blur-Max conditions (sigma 2.25/3.0), not formal corruption severities.

## mIoU

| Variant | Clean | Blur-Mid | Blur-Max |
|---|---:|---:|---:|
| F0 | 0.550480 | 0.475816 | 0.431939 |
| F1 | 0.549431 | 0.469952 | 0.403819 |
| F2 | 0.549988 | 0.470485 | 0.402877 |
| F3 | 0.549795 | 0.470639 | 0.404007 |

## mVC16

| Variant | Clean | Blur-Mid | Blur-Max |
|---|---:|---:|---:|
| F0 | 0.955605 | 0.913139 | 0.856835 |
| F1 | 0.956532 | 0.875847 | 0.763258 |
| F2 | 0.956758 | 0.875698 | 0.762204 |
| F3 | 0.956637 | 0.876341 | 0.763537 |

## Deltas and judgement

{
  "F1 - F0": {
    "mean_blur_mIoU_delta": -0.016992218574165774,
    "mean_blur_mVC16_delta": -0.0654343179973027
  },
  "F2 - F0": {
    "mean_blur_mIoU_delta": -0.017197079778027757,
    "mean_blur_mVC16_delta": -0.06603603739251351
  },
  "F3 - F0": {
    "mean_blur_mIoU_delta": -0.016554656311756094,
    "mean_blur_mVC16_delta": -0.06504816122245938
  },
  "F3 - best(F1,F2)": {
    "Blur-Mid mIoU": 0.00015473742030569504,
    "Blur-Max mIoU": 0.00018805281989031641
  }
}

{
  "CHANNEL-GATE BOTTLENECK": "NOT SUPPORTED",
  "TEMPORAL-CONDITION BOTTLENECK": "NOT SUPPORTED",
  "COMBINED": "GO",
  "IDEAL F3 VS OLD": false
}

## Initialization and gradient checks

```json
{
  "posterior_max_abs_diff": {
    "full-scalar": 0.0,
    "full-channel-gate": 0.0,
    "full-temporal-semantic": 0.0,
    "full-channel-temporal": 0.0
  },
  "logits_max_abs_diff": {
    "full-scalar": 0.0,
    "full-channel-gate": 0.0,
    "full-temporal-semantic": 0.0,
    "full-channel-temporal": 0.0
  },
  "gradients": {
    "full-scalar": {
      "seg_gate": 0.01703026401810348,
      "seg_semantic": 0.5941349300555885,
      "seg_temporal": 0.0,
      "temporal_branch": 5.814429308055878,
      "temporal_gate": 0.0,
      "temporal_semantic": 0.0
    },
    "full-channel-gate": {
      "seg_gate": 0.047012092429213226,
      "seg_semantic": 0.5941349300555885,
      "seg_temporal": 0.0,
      "temporal_branch": 5.814429308055878,
      "temporal_gate": 0.0,
      "temporal_semantic": 0.0
    },
    "full-temporal-semantic": {
      "seg_gate": 0.01703026401810348,
      "seg_semantic": 0.6017457949928939,
      "seg_temporal": 0.0,
      "temporal_branch": 5.814429308055878,
      "temporal_gate": 0.0,
      "temporal_semantic": 0.0
    },
    "full-channel-temporal": {
      "seg_gate": 0.047012092429213226,
      "seg_semantic": 0.6017457949928939,
      "seg_temporal": 0.0,
      "temporal_branch": 5.814429308055878,
      "temporal_gate": 0.0,
      "temporal_semantic": 0.0
    }
  },
  "indexing": {
    "601": {
      "warmup": 60,
      "first_temporal_pair": [
        60,
        61
      ],
      "temporal_pairs": 540,
      "bptt": 4,
      "sequence_boundary_reset": true
    },
    "503": {
      "warmup": 50,
      "first_temporal_pair": [
        50,
        51
      ],
      "temporal_pairs": 452,
      "bptt": 4,
      "sequence_boundary_reset": true
    }
  }
}
```
