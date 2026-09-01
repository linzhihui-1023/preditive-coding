# Residual semantic correction quick screen

Diagnostic Blur-Mid/Blur-Max conditions use sigma 2.25/3.0 and are not formal corruption severities.

## mIoU

| Model | Clean | Blur-Mid | Blur-Max | Blur Mean |
|---|---:|---:|---:|---:|
| old | 0.545940 | 0.482805 | 0.434205 | 0.458505 |
| full-explicit | 0.549270 | 0.476828 | 0.432094 | 0.454461 |
| full-residual | 0.549021 | 0.476482 | 0.431921 | 0.454201 |

## mVC16

| Model | Clean | Blur-Mid | Blur-Max | Blur Mean |
|---|---:|---:|---:|---:|
| old | 0.953850 | 0.890799 | 0.806483 | 0.848641 |
| full-explicit | 0.955541 | 0.913320 | 0.857097 | 0.885208 |
| full-residual | 0.955597 | 0.912846 | 0.855983 | 0.884414 |

## Best epoch

{
  "old": {
    "mIoU": 0.5459401806927648,
    "epoch": 1
  },
  "full-explicit": {
    "mIoU": 0.5492699024003885,
    "epoch": 3
  },
  "full-residual": {
    "mIoU": 0.5490209417413188,
    "epoch": 3
  }
}

## Parameters

{
  "old": 2598080,
  "full-explicit": 2557378,
  "full-residual": 2770882,
  "residual_semantic_extra_vs_explicit": 213504
}

## Comparisons

{
  "Residual - Explicit": {
    "Clean mIoU delta": -0.0002489606590697635,
    "Mean Blur mIoU delta": -0.0002593343604255782,
    "Mean Blur mVC16 delta": -0.0007936024568537259
  },
  "Residual - Old": {
    "Clean mIoU delta": 0.003080761048553926,
    "Mean Blur mIoU delta": -0.004303325357534887,
    "Mean Blur mVC16 delta": 0.03577351419050001
  },
  "SEMANTIC REFINEMENT": "NOT SUPPORTED"
}

## Initialization / gradient sanity

{
  "posterior_max_abs_diff": 0.0,
  "semantic_residual_max_abs_diff": 0.0,
  "seg_semantic_grad": 0.5959126754896715,
  "seg_temporal_grad": 0.0,
  "local_refine_expand_grad": 0.04207054851576686,
  "context_refine_expand_grad": 0.041795605910010636,
  "temporal_branch_grad": 3.9142026312217695,
  "temporal_semantic_grad": 0.0,
  "temporal_gate_grad": 0.0
}
