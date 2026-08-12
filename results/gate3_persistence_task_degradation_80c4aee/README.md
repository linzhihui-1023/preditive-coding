# Gate 3: persistence score versus task degradation

This directory contains the lightweight outputs from the frozen, inference-
only Gate 3 experiment at revision
`80c4aee7f1493479764c1f21c638dbac40ca4e32`.

Gate 2's detector was locked to the mean
`cos(e_t,e_(t-1))` over nonoverlapping eight-frame windows. The Gate 2 result
files were hash-locked, and all 2,400 Persistent/Shuffled detector rows were
reproduced with maximum absolute delta `0.0`. No metric or direction was
reselected.

Physical task errors came from the frozen formal Group A 2-DoF motion
checkpoint at revision `6c446d9`, epoch 6. For every corrupted window, signed
degradation was computed against an independently reset clean trajectory on
the same raw frames and OXTS targets.

On held-out drive 0011, forward `S-D` Pearson was `-0.001291`, Spearman was
`0.057782`, and `S` AUROC for positive degradation was `0.500651`. Yaw mean
degradation was negative and only one of 80 windows degraded, so its nominal
positive-degradation AUROC is not stable evidence. Gate 3 therefore does not
support using the Gate 2 score as a task-degradation trigger for this frozen
2-DoF model.

`per_frame.csv` contains 2,700 records, `windows.csv` contains all 180 Clean/
Persistent/Shuffled windows, and `degradations.csv` contains the 160 corrupted
windows paired to clean. These correlated windows are diagnostic units, not
independent drives.
