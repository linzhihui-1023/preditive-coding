# Temporal–semantic interface same-epoch quick screen

Diagnostic Blur-Mid/Blur-Max conditions (sigma 2.25/3.0), not formal corruption severities. Every row is evaluated from its own epoch checkpoint.

## mIoU

| Model | Epoch | Clean | Blur-Mid | Blur-Max | Blur Mean |
|---|---:|---:|---:|---:|---:|
| F0 | 1 | 0.549595 | 0.469867 | 0.402917 | 0.436392 |
| F1 | 1 | 0.549428 | 0.469951 | 0.403819 | 0.436885 |
| F2 | 1 | 0.549987 | 0.470485 | 0.402876 | 0.436681 |
| F3 | 1 | 0.549796 | 0.470640 | 0.404006 | 0.437323 |
| F0 | 2 | 0.545581 | 0.470276 | 0.425084 | 0.447680 |
| F1 | 2 | 0.542369 | 0.465108 | 0.421202 | 0.443155 |
| F2 | 2 | 0.543809 | 0.467724 | 0.421410 | 0.444567 |
| F3 | 2 | 0.542279 | 0.464594 | 0.419797 | 0.442196 |
| F0 | 3 | 0.550470 | 0.475807 | 0.431926 | 0.453867 |
| F1 | 3 | 0.547126 | 0.473143 | 0.432655 | 0.452899 |
| F2 | 3 | 0.548213 | 0.473695 | 0.429494 | 0.451595 |
| F3 | 3 | 0.546458 | 0.471792 | 0.430343 | 0.451068 |

## mVC16

| Model | Epoch | Clean | Blur-Mid | Blur-Max | Blur Mean |
|---|---:|---:|---:|---:|---:|
| F0 | 1 | 0.956660 | 0.875091 | 0.761817 | 0.818454 |
| F1 | 1 | 0.956531 | 0.875846 | 0.763259 | 0.819553 |
| F2 | 1 | 0.956759 | 0.875699 | 0.762205 | 0.818952 |
| F3 | 1 | 0.956636 | 0.876339 | 0.763537 | 0.819938 |
| F0 | 2 | 0.955721 | 0.910401 | 0.854048 | 0.882225 |
| F1 | 2 | 0.955085 | 0.903903 | 0.839681 | 0.871792 |
| F2 | 2 | 0.955788 | 0.911076 | 0.855927 | 0.883501 |
| F3 | 2 | 0.955788 | 0.905873 | 0.844471 | 0.875172 |
| F0 | 3 | 0.955604 | 0.913141 | 0.856840 | 0.884991 |
| F1 | 3 | 0.954437 | 0.909231 | 0.849697 | 0.879464 |
| F2 | 3 | 0.955677 | 0.912523 | 0.856318 | 0.884420 |
| F3 | 3 | 0.955205 | 0.909667 | 0.851269 | 0.880468 |

## Deltas and judgement

{
  "F1 - F0": {
    "epoch1_blur_mIoU_delta": 0.0004930829642654788,
    "epoch1_blur_mVC16_delta": 0.0010983700800698548,
    "epoch2_blur_mIoU_delta": -0.004524534141571279,
    "epoch2_blur_mVC16_delta": -0.010432633485857545,
    "epoch3_blur_mIoU_delta": -0.000967609690191884,
    "epoch3_blur_mVC16_delta": -0.005527022711785401
  },
  "F2 - F0": {
    "epoch1_blur_mIoU_delta": 0.00028866293233936746,
    "epoch1_blur_mVC16_delta": 0.0004978549224324835,
    "epoch2_blur_mIoU_delta": -0.00311286456147003,
    "epoch2_blur_mVC16_delta": 0.0012766383084592547,
    "epoch3_blur_mIoU_delta": -0.0022717328057049735,
    "epoch3_blur_mVC16_delta": -0.0005704446908406657
  },
  "F3 - F0": {
    "epoch1_blur_mIoU_delta": 0.0009309373848779834,
    "epoch1_blur_mVC16_delta": 0.0014839726900751637,
    "epoch2_blur_mIoU_delta": -0.0054838930636120775,
    "epoch2_blur_mVC16_delta": -0.007052576649877351,
    "epoch3_blur_mIoU_delta": -0.002798830769001226,
    "epoch3_blur_mVC16_delta": -0.00452264292418425
  },
  "same_epoch_final": {
    "F0_blur_mean_mIoU": 0.4538665114229227,
    "F1_blur_mean_mIoU": 0.45289890173273084,
    "F2_blur_mean_mIoU": 0.45159477861721775,
    "F3_blur_mean_mIoU": 0.4510676806539215,
    "F1_minus_F0": -0.000967609690191884,
    "F3_minus_F0": -0.002798830769001226
  }
}

{
  "CHANNEL-GATE": "RETEST / SUPPORTED",
  "COMBINED": "NO-GO; F3 Blur Mean mIoU is below F0 at the same epoch",
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
