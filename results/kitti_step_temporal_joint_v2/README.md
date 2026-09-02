# Temporal Joint V2

Controlled KITTI-STEP Clean temporal training, initialized from FAST-B epoch 3.
The zero-step evaluator reproduced the stored FAST-B dev3 reference within the
configured floating-point tolerance. Stage 1 was run for one epoch; the V2
automatic gate stopped before any Dynamics-unfrozen Stage 2 or full9 run.

| epoch | mIoU | mVC8 | mVC16 | mTC |
|---:|---:|---:|---:|---:|
| 0 (zero step) | 57.8829% | 94.5077% | 94.2116% | 70.5006% |
| 1 (Stage 1) | 55.5757% | 94.5659% | 94.3007% | 69.2104% |

Stage 1 status: `SEMANTIC_TEMPORAL_OBJECTIVE_CONFLICT` (mIoU < 57.0%).
No Stage 2 or full9 evaluation is authorized by the V2 protocol after this
gate failure.
