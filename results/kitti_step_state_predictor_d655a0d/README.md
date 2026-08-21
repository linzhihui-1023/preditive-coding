# KITTI-STEP Multilayer State Predictor

The static host was restored from the original static fine-tuning code at
`16d0229` for one epoch. Its validation mIoU was `0.6541174065`; the stable
checkpoint SHA256 is
`19057b288434956d0a6dd22e8a719fc03c44417025aad5512e74516fdc15dea0`.

The Adapter + Predictor experiment used implementation `d655a0d`, seed 0,
three epochs, batch size 1, AdamW at `1e-4`, weight decay `0.01`, and
`lambda_recon=0.1`. The official KITTI-STEP split produced 5,003 train and
2,963 validation triplets.

Epoch 3 was selected by validation mean Predictor MSE:

| Layer | Predictor MSE | Copy MSE | Improvement |
| --- | ---: | ---: | ---: |
| 1 | 0.0003058777 | 0.0004377310 | 30.12% |
| 2 | 0.0000804503 | 0.0001115653 | 27.89% |
| 3 | 0.0000339056 | 0.0000452137 | 25.01% |
| 4 | 0.0000807818 | 0.0000815763 | 0.97% |
| Mean | 0.0001252539 | 0.0001690216 | 25.89% |

Decision: **GO**. Mean Predictor MSE beat Copy MSE and all four layers beat
their corresponding Copy baseline. No prediction-error, dynamic-error,
correction, or host-write-back path was used.

Large checkpoints and logs remain outside Git:

- Static host: `/home/lin/predify/checkpoints/kitti_step_static_deeplabv3plus_epoch1/best_kitti_step_static_deeplabv3plus.pt`
- Best Adapter + Predictor: `/home/lin/predify/experiments/kitti_step_state_predictor_d655a0d/best_state_predictor.pt`
