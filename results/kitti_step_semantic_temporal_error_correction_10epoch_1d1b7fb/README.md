# Semantic Temporal Error Correction: 10-Epoch Evaluation

The model and persistent-blur training protocol are unchanged. Training uses 10 epochs, 10% clean warm-up, truncated BPTT 4, and selects the minimum validation total loss. The best checkpoint is epoch 8 (`0.4853148598`); epoch 10 is `0.4857070738`, so the 7-to-10 extension did not improve on epoch 8 and ended with a slight rebound.

The main post-warm-up evaluation contains 2686 frames from 9 sequences. VSPW-style video consistency is computed on the same post-warm-up stream: 2623 valid VC8 windows and 2551 valid VC16 windows.

| Method | mIoU | wIoU | mVC8 | mVC16 |
| --- | ---: | ---: | ---: | ---: |
| Clean Host | 0.665796 | 0.862479 | 0.943587 | 0.936405 |
| Corrupted Host | 0.393640 | 0.729683 | 0.830361 | 0.776704 |
| Corrected B | 0.435486 | 0.771308 | 0.909981 | 0.894626 |
| Error Decomposition + Reliability | 0.436583 | 0.766160 | 0.908073 | 0.893401 |
| Current Model | 0.466109 | 0.795249 | 0.910080 | 0.893240 |

Relative to Corrupted Host, the current model gains `+0.072469` mIoU, `+0.065566` wIoU, `+0.079719` mVC8, and `+0.116536` mVC16. Recovery ratio is `0.266278`.

The additional ImageNet-C Gaussian Blur benchmark uses the standard sigma levels `[1, 2, 3, 4, 6]`; the benchmark reference model is Corrupted Host. Mean blur mIoU is `0.377515` for Corrupted Host and `0.430100` for Current Model. Current Model CD is `0.915524` and rCD is `0.890260` versus the Corrupted Host reference value `1.0`.

Measured on post-warm-up model inference with CUDA synchronization, Host-only FPS is `48.4962` and Full Current Model FPS is `7.6413`, a `84.2434%` drop.
