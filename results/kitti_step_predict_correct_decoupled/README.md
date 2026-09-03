# Predict / Correct Decoupled Training — S1

Initialized from original FAST-B epoch3. S1 trained only the Semantic Error
Encoder and Semantic State Cell for one epoch.

| Stage | mIoU | mVC8 | mVC16 | mTC |
|---|---:|---:|---:|---:|
| zero step | 57.8829% | 94.5077% | 94.2116% | 70.5006% |
| S1 epoch 1 | 57.7525% | 94.5105% | 94.2191% | 70.4888% |

S1 mIoU remained above 57.70%, but mTC was slightly below 70.50%; D1 was not
run.
