# Temporal Joint V2 — Stage 1 rerun from original FAST-B epoch 3

This run was initialized directly from the original FAST-B `best.pt`; the
failed V2 Epoch 1 checkpoint was not loaded. The zero-step dev3 evaluator
matched the stored FAST-B reference within floating-point tolerance.

| epoch | mIoU | mVC8 | mVC16 | mTC |
|---:|---:|---:|---:|---:|
| 0 (zero step) | 57.8829% | 94.5077% | 94.2116% | 70.5006% |
| 1 (Stage 1) | 55.4345% | 94.5779% | 94.3224% | 69.1049% |

`Lpreserve = 0.0051639935 > 0`.

The Stage 1 semantic gate failed (`mIoU < 57.0%`), so Stage 2 and full9 were
not run.
