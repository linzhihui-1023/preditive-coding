# Experiment Log

## Continuous stream, dynamic error, two-drive validation

Date: 2026-08-10

Git revision: the first private-repository commit containing this document

Data:

- Train: `2011_09_26_drive_0005_sync`, 153 frame pairs.
- Validation: `2011_09_26_drive_0011_sync`, 232 frame pairs.
- Camera: `image_02`.
- Fixed interval: 0.1035 seconds, tolerance 0.001 seconds.

Configuration:

```text
PREDIFY_STREAM_MODE=1
PREDIFY_BATCHSIZE=1
PREDIFY_EPOCHS=10
PREDIFY_PRETRAINED=1
PREDIFY_LR=1e-4
PREDIFY_WEIGHT_DECAY=0
PREDIFY_EMA_DECAY=0.99
PREDIFY_TOP_TARGET_SOURCE=ema_teacher
PREDIFY_TEMPORAL_TARGET_MODE=ego_motion
PREDIFY_TASK_ALIGNED_TARGET=ego_motion
PREDIFY_DYNAMIC_ERROR=1
PREDIFY_ERROR_TS=0.1035
PREDIFY_ERROR_TAU=0.5,0.5,0.5,0.5,0.5
PREDIFY_ERROR_GAIN=1.0,1.0,1.0,1.0,1.0
```

Validation history:

| Epoch | Weighted loss | Temporal loss | Temporal MAE | Temporal cosine |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.050456 | 0.238344 | 0.244059 | 0.741795 |
| 2 | 0.027820 | 0.236265 | 0.242751 | **0.750144** |
| 3 | 0.019656 | 0.240420 | 0.246158 | 0.688198 |
| 4 | 0.014163 | 0.240545 | 0.244538 | 0.720874 |
| 5 | 0.010404 | 0.240569 | 0.242745 | 0.746884 |
| 6 | 0.009099 | 0.240348 | 0.242796 | 0.747342 |
| 7 | 0.007869 | 0.241156 | 0.244133 | 0.729381 |
| 8 | 0.007367 | 0.240475 | 0.244872 | 0.705784 |
| 9 | 0.006928 | 0.241004 | 0.245344 | 0.696293 |
| 10 | 0.007630 | 0.240110 | 0.247483 | 0.651919 |

Conclusion:

Retrospective validity: invalid as temporal-prediction evidence. The temporal
context consumed current errors that had already been formed from the
future-frame teacher feature, which leaked `I_{t+1}` into the `t -> t+1`
prediction. In addition, the temporal prediction loss weight used its old
default value of zero, so the temporal predictor received no temporal-loss
gradient. The run is retained only as an execution smoke record and its metrics
must not be compared with corrected experiments.

Server artifacts, not tracked by Git:

```text
/home/lin/predify/kitti_targetflow_stream_ego_motion_dynerr_tau0p5_e10.p
/home/lin/predify/kitti_targetflow_stream_ego_motion_dynerr_tau0p5_e10_student.pt
/home/lin/predify/kitti_targetflow_stream_ego_motion_dynerr_tau0p5_e10_teacher.pt
```

## Earlier adjacent-pair result

This older experiment reset model state each batch and therefore tested
pair-level temporal supervision, not continuous video-state inheritance.

| Training condition | Val weighted | Val temporal loss | Val MAE | Val cosine |
| --- | ---: | ---: | ---: | ---: |
| Ordered adjacent pairs | 0.049902 | 0.220986 | 0.244000 | 0.717777 |
| Shuffled-future control | 0.040895 | 0.261809 | 0.264013 | -0.620335 |
