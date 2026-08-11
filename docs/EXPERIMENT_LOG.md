# Experiment Log

## Superseded seeded temporal control matrix

Date: 2026-08-10

Git revision: `77f0ad042a031fb151e12cb1174c628f20e22a44`

Retrospective validity: the runs are causal, but they are not clean mechanism
controls and do not evaluate the active five-layer future target chain. All
three used `target_flow_mode=quasi_steady` and optimized
`mean(state_error ** 2)`. Their feedback decoders produced gradients but were
omitted from the optimizer, so those gradients accumulated without updating
the decoder parameters.

Shared configuration:

- Train: `2011_09_26_drive_0005_sync`, 153 frame pairs.
- Validation: `2011_09_26_drive_0011_sync`, 232 frame pairs.
- Camera: `image_02`; fixed interval 0.1035 seconds.
- Seed 0, ten epochs, batch size 1, ordered stream, no shuffle.
- Ego-motion target, temporal prediction weight 1.0, EMA decay 0.99.
- Best student checkpoint selected by lowest validation temporal loss.

Results:

| Run | State policy | Error policy | Best epoch | Legacy mixed-unit MSE | Legacy mixed-unit MAE | Raw cosine |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| A | Inherit | Dynamic, `tau=0.5` | 6 | **0.108731** | **0.222764** | **0.752622** |
| B | Reset each frame | Dynamic, `tau=0.5` | 1 | 0.129041 | 0.240693 | 0.750757 |
| C | Inherit | Instantaneous | 10 | 0.111454 | 0.223388 | 0.751711 |

With `alpha=Ts/tau=0.207`, the dynamic local loss has gradient
`dL/de_t=2*alpha*epsilon_t`, whereas the instantaneous condition has
`dL/de_t=2*e_t`. A versus C therefore changes memory, smoothing, gradient
scale, and effective optimization dynamics together. Resetting B every frame
also resets the error integrator, so A versus B changes its local-loss gradient
trajectory as well as temporal state inheritance.

The numerical differences are retained only as a record that the causal stream
executed. They cannot support claims for dynamic-error memory or state
inheritance. In addition, `quasi_steady` supplied the future target only at the
top layer; lower targets came from current-frame higher-layer forward outputs.
The corrected controls must use `recursive` target flow and an identical
instantaneous local loss in every memory condition, with trainable feedback
decoders.

The old runs set the variance weight to zero, so the inactive variance term did
not alter their reported objectives. However, the old batch-based implementation
would also have been identically zero in stream mode, and its default
`target=0.01, eps=1e-4` made the hinge zero for any batch size.

The target called `ego_motion` was only `[forward displacement in metres,
yaw change in radians]`. Its unstandardized MSE mixed incompatible units and
was dominated by forward displacement. These historical MSE/MAE/cosine values
must not be compared with corrected standardized 2-DoF longitudinal-yaw runs.

The inherited condition also received the previous top target, which at time
`t` contains `F_teacher(I_t)`, while reset-each-frame did not. Its A/B
difference therefore mixed temporal history with an extra current-frame top
representation. The old 15.7% difference is not evidence for long-term
memory. With the backbone frozen, `F_teacher(I_t)=F_student(I_t)`, so the
corrected reset/no-history control duplicates the detached current student top
feature in the top context slot.

Server artifacts, not tracked by Git:

```text
/home/lin/predify/kitti_targetflow_seed0_A_inherit_dynerr_tau0p5_tw1_e10.p
/home/lin/predify/kitti_targetflow_seed0_A_inherit_dynerr_tau0p5_tw1_e10_best_student.pt
/home/lin/predify/kitti_targetflow_seed0_B_resetframe_dynerr_tau0p5_tw1_e10.p
/home/lin/predify/kitti_targetflow_seed0_B_resetframe_dynerr_tau0p5_tw1_e10_best_student.pt
/home/lin/predify/kitti_targetflow_seed0_C_inherit_instanterr_tw1_e10.p
/home/lin/predify/kitti_targetflow_seed0_C_inherit_instanterr_tw1_e10_best_student.pt
```

## Corrected causal stream, dynamic error, two-drive validation

Date: 2026-08-10

Git revision: `4793bf387ca70fa3ae941b4ee64c0e77c1bda60d`

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
PREDIFY_TEMPORAL_PREDICTION_WEIGHT=1.0
PREDIFY_DYNAMIC_ERROR=1
PREDIFY_ERROR_TS=0.1035
PREDIFY_ERROR_TAU=0.5,0.5,0.5,0.5,0.5
PREDIFY_ERROR_GAIN=1.0,1.0,1.0,1.0,1.0
```

Validation history:

| Epoch | Weighted loss | Temporal loss | Temporal MAE | Temporal cosine | Objective |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.053068 | 0.134883 | 0.249643 | 0.750227 | 0.187952 |
| 2 | 0.030311 | 0.143171 | 0.254734 | 0.750417 | 0.173483 |
| 3 | 0.020068 | 0.140171 | 0.254326 | 0.750190 | 0.160238 |
| 4 | 0.016547 | 0.135199 | 0.247631 | 0.750714 | 0.151746 |
| 5 | 0.020322 | 0.126555 | 0.241959 | 0.751024 | 0.146878 |
| 6 | 0.011090 | 0.119205 | 0.234498 | 0.750196 | 0.130294 |
| 7 | 0.011140 | 0.117941 | 0.232466 | **0.751353** | 0.129080 |
| 8 | 0.024938 | **0.094732** | **0.212064** | 0.750689 | **0.119669** |
| 9 | 0.027394 | 0.132564 | 0.251984 | 0.746482 | 0.159958 |
| 10 | 0.008007 | 0.127408 | 0.248920 | 0.749358 | 0.135415 |

Reference baselines on the same validation targets:

| Predictor | Legacy mixed-unit MSE | Legacy mixed-unit MAE | Raw-vector cosine |
| --- | ---: | ---: | ---: |
| All zeros | 0.249872 | 0.246986 | 0.000000 |
| Training-drive mean motion | 0.129003 | 0.235466 | 0.751176 |
| Corrected stream, epoch 8 | **0.094732** | **0.212064** | 0.750689 |

Conclusion:

This historical run improved the mixed-unit metrics, but those values are not
physically balanced: forward displacement dominated yaw. It is retained only
for traceability and cannot be compared with standardized longitudinal-yaw
training. Raw-vector cosine was likewise dominated by forward motion.

The script saved only epoch 10, so the best epoch-8 weights are not available.
Best-checkpoint saving and deterministic seeding are required before the formal
control matrix.

Server artifacts, not tracked by Git:

```text
/home/lin/predify/kitti_targetflow_stream_causal_ego_motion_dynerr_tau0p5_tw1_e10.p
/home/lin/predify/kitti_targetflow_stream_causal_ego_motion_dynerr_tau0p5_tw1_e10_student.pt
/home/lin/predify/kitti_targetflow_stream_causal_ego_motion_dynerr_tau0p5_tw1_e10_teacher.pt
```

## Invalid predecessor stream experiment

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
