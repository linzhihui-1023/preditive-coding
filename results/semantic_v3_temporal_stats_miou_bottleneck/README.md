# Fast mIoU Bottleneck Diagnosis

Checkpoint: `V3-TemporalStats` best Epoch 1. No training or parameter updates
were performed. The fixed fast protocol used sequences `0002/0010/0018`,
Blur-Mid (`sigma=2.25`) and Blur-Max (`sigma=3.0`), with 781 post-warmup frames
per condition. Clean/blurred images shared one backbone forward per frame;
the four paths diverged after feature extraction.

| condition | corrupted Host | learned restored Z4 | oracle Z4 → Writeback | clean C4 |
|---|---:|---:|---:|---:|
| Blur-Mid | 0.437285 | 0.436451 | 0.486699 | 0.584753 |
| Blur-Max | 0.349069 | 0.350594 | 0.426530 | 0.584753 |

| condition | corrupted Z4 MSE | learned Z4 MSE | Writeback Transfer Ratio | Learned Transfer Efficiency |
|---|---:|---:|---:|---:|
| Blur-Mid | 2.6093e-4 | 2.2360e-4 | 0.3351 | -0.0169 |
| Blur-Max | 3.5381e-4 | 3.0303e-4 | 0.3287 | 0.0197 |

The learned Z4 reduces feature MSE, but produces essentially no segmentation
gain (and a small Mid loss), while the oracle Z4 improves mIoU substantially.
This satisfies the task-relevance failure pattern. Writeback Transfer Ratio is
between 0.20 and 0.50 in both conditions, so Writeback also limits transfer;
the combined diagnosis is **Both**, with learned task-relevant Z4 direction the
dominant issue.
