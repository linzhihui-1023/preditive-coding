# Local Error-Memory Alignment

The experiment removes the extra `sqrt(32)` scaling from normalized cosine
attention scores and trains only the Layer 1 and Layer 4 query/key
projections. The fixed Host, Adapter, Predictor, Correction, dynamic-error
coefficients, noise protocol, and 7x7 local window are unchanged.

The training stability abort now stops only on non-finite states. The
post-training matching gate requires at least one layer to have mean maximum
attention weight above `0.03` and normalized entropy below `0.98`, while the
other layer must not remain uniform.

Result: `MATCHING NOT LEARNED`. Layer 1 reached max weight `0.0327463` and
entropy `0.9906956`; Layer 4 reached max weight `0.0232061` and entropy
`0.9997768`. The state traces were finite and passed the stability check, but
the matching gate failed, so paired mIoU evaluation was not interpreted.
