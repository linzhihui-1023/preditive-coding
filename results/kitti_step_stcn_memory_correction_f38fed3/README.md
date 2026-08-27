# STCN Memory Correction

This experiment adds a frozen-base, trainable STCN-style space-time memory readout for Z4. The memory stores only corrupted observation states from the previous four frames; Z1 keeps the existing semantic-temporal correction path. Training used persistent Gaussian blur, AdamW, three epochs, and 24,705 trainable parameters.

The full validation used 9 KITTI-STEP sequences, 2,981 frames, and 2,963 effective frames. Clean and corrupted baselines reproduced the existing values (`0.6552125562` and `0.4701953800`). Current d7 correction reached `0.5083952797`; STCN memory reached `0.3483662123`.

The memory reference ratio was `4.8122813e17`, direction cosine `0.0662731`, and normalized affinity entropy `0.9900464`. All finite, identity, causality, observation-only-memory, and frozen-base gates passed, but the memory reference and final semantic result failed the usefulness criterion: `MEMORY REFERENCE NO-GO`.
