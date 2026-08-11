# Experiment Log

## Seed-0 cheap diagnostics before additional seeds

Date: 2026-08-11

Diagnostic code revision: `5b64cad42ba53ffd4dc256949f53e7c5f2b91401`

Matrix checkpoint revision: `6c446d901a581323208a335489a7c065f1946adc`

The diagnostic used the same 153 ordered training pairs from drive 0005, 232
ordered validation pairs from drive 0011, fixed 0.1035-second interval, and
training-only longitudinal-yaw normalization as the formal seed-0 matrix. All
A-E values below were recomputed frame by frame from each best checkpoint.
Recomputed joint MSE values match the saved histories to floating-point
precision.

The static baseline takes only the current frame, extracts the frozen ImageNet
VGG16 feature immediately before the final max-pool, applies global average
pooling, and predicts the standardized target through a `512 -> 256 -> 2` MLP.
It has no future-frame input, recurrent state, feedback decoder, or temporal
context. It used seed 0, batch size 1, ordered samples, ten epochs, Adam at
`1e-4`, and the same training-only target statistics. Its best checkpoint was
selected by validation standardized joint MSE.

| Model | Best epoch | Std. joint MSE | Fwd MSE (m2) | Fwd median AE (m) | Fwd P95 AE (m) | Yaw MSE (rad2) | Yaw median AE (rad) | Yaw P95 AE (rad) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train-mean constant | n/a | 11.266117 | 0.258004 | 0.466013 | 0.764155 | **2.7766e-6** | **0.001415** | **0.003343** |
| A: inherit, EMA | 6 | 11.045345 | **0.249697** | 0.537760 | 0.641254 | 8.3509e-5 | 0.008745 | 0.013345 |
| B: reset | 5 | 11.443966 | 0.260506 | **0.414215** | 0.779019 | 4.1857e-5 | 0.004754 | 0.012248 |
| C: reset, top duplicate | 6 | **11.018217** | 0.250251 | 0.418766 | 0.758862 | 5.4291e-5 | 0.005448 | 0.013330 |
| D: inherit, instant | 5 | 11.146297 | 0.254559 | 0.549633 | 0.648188 | 2.0179e-5 | 0.003322 | 0.007552 |
| E: inherit, lag-1 | 5 | 11.128227 | 0.253999 | 0.559413 | **0.625530** | 2.3806e-5 | 0.003529 | 0.008240 |
| Frozen VGG + static MLP | 1 | 11.376018 | 0.260237 | 0.497094 | 0.739814 | 9.8636e-6 | 0.003209 | 0.004537 |

Key comparisons:

- C lowers joint MSE by only 2.20% relative to the train-mean constant; A
  lowers it by 1.96%. B and the static MLP are worse than the constant by
  1.58% and 0.98%, respectively.
- A and C lower forward MSE by 3.22% and 3.00% relative to the constant. A's
  forward median absolute error is nevertheless worse, while its P95 is 16.1%
  better. Its MSE improvement is therefore not a uniform per-frame gain.
- Every learned model is worse than the constant on yaw MSE. A and C have
  30.1x and 19.6x the constant yaw MSE; even the static MLP has 3.55x.
- The static MLP's best checkpoint occurs at epoch 1. Its training MSE falls
  from 0.864610 at epoch 1 to 0.027601 at epoch 10 while validation MSE rises
  from 11.376018 to 11.526384, showing severe overfitting.

Target-distribution diagnosis:

| Drive | Fwd mean/std (m) | Fwd min/max (m) | Yaw mean/std (rad) | Yaw min/max (rad) |
| --- | --- | --- | --- | --- |
| Train 0005 | 0.466140 / 0.107030 | 0.317148 / 0.667017 | -0.001483 / 0.016870 | -0.027689 / 0.020870 |
| Val 0011 | 0.492365 / 0.507264 | -0.003405 / 1.235956 | -0.001305 / 0.001657 | -0.006213 / 0.000399 |

The drives have sharply different motion regimes: validation forward standard
deviation is 4.74x the training value, while validation yaw standard deviation
is only 0.098x the training value. Consequently, the validation
joint standardized MSE is almost entirely determined by forward displacement:
the constant baseline has forward/yaw standardized MSE 22.522478/0.009756.
The current two-drive split therefore provides weak evidence about temporal
memory and poor evidence about yaw prediction.

Decision:

Do not run seeds 1 and 2 yet. Repeating seeds would quantify initialization
noise around a split where the strongest learned condition improves on a
constant by only 2.20%, the static baseline does not beat the constant, and
all learned conditions degrade yaw. Add more training drives and construct a
motion-regime-aware train/validation split first; then rerun constant, static,
A, and C before spending on the full five-condition multi-seed matrix.

Server artifacts, not tracked by Git:

```text
/home/lin/predify/experiments/seed0_cheap_diagnostics_5b64cad/
```

## Formal frozen-backbone five-group matrix, seed 0

Date: 2026-08-11

Git revision: `6c446d901a581323208a335489a7c065f1946adc`

CI status for the tested revision: passed 21 unit tests in GitHub Actions run
`31473188368` (job `93720888577`).

Shared configuration:

- Train: `2011_09_26_drive_0005_sync`, 153 ordered frame pairs.
- Validation: `2011_09_26_drive_0011_sync`, 232 ordered frame pairs.
- Camera `image_02`; fixed interval 0.1035 seconds with tolerance 0.001
  seconds; both drives contain one accepted contiguous segment.
- Seed 0, ten epochs, batch size 1, learning rate `1e-4`, no shuffle.
- Recursive five-layer target flow, frozen VGG backbone, trainable feedback
  decoders and temporal predictor.
- Instantaneous local-loss error, variance weight zero, standardized 2-DoF
  longitudinal-yaw temporal loss weight 1.0.
- Dynamic parameters recorded identically in all groups: `Ts=0.1035`,
  `tau=0.5`, and `K=1.0`.
- Best student checkpoint selected by minimum validation standardized
  longitudinal-yaw MSE.

The runner started each condition with `env -i`, explicitly set the current
and legacy top-context variables, and recorded the exact Git revision. A
post-run config audit found only the intended differences: B changes
`reset_each_frame`; C additionally changes `current_top_duplicate`; D changes
the error mode to `instant` (with the derived legacy `dynamic_error` flag
false); E changes the error mode to `lag1`. Training-only normalization was
identical in all five histories: forward mean/std `0.466140/0.107030 m` and
yaw mean/std `-0.001483/0.016870 rad`.

Best-checkpoint validation results:

| Group | State/error condition | Best epoch | Standardized MSE | Standardized MAE | Forward MAE (m) | Yaw MAE (rad) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A | Inherit, recursive EMA | 6 | 11.045343 | 2.458972 | 0.473039 | 0.008406 |
| B | Reset, no extra top context | 5 | 11.443965 | 2.337672 | 0.465796 | 0.005455 |
| C | Reset, current-top duplicate | 6 | **11.018217** | **2.325611** | **0.458364** | 0.006219 |
| D | Inherit, instant error | 5 | 11.146288 | 2.339699 | 0.477386 | **0.003696** |
| E | Inherit, lag-1 error | 5 | 11.128227 | 2.351370 | 0.477661 | 0.004047 |

Predefined comparisons, using the second condition as the percentage
denominator when A is compared with another group:

- A versus C, the primary history comparison: A is 0.027126 MSE higher, or
  0.246% worse. Seed 0 therefore shows no state-inheritance benefit after
  controlling for the duplicated current top feature.
- B versus C, the extra-current-feature check: C is 0.425748 lower than B, a
  3.720% reduction relative to B. A substantial part of the old A/B gap can
  therefore be explained by access to the extra current top representation.
- A versus D: A is 0.100945 lower, a 0.906% reduction relative to D.
- A versus E: A is 0.082884 lower, a 0.745% reduction relative to E.
- A versus B is only an auxiliary comparison: A is 3.483% lower than B, but
  this contrast remains confounded by the extra top representation.

Conclusion:

The corrected seed-0 matrix does not support a claim that inherited history
improves the motion target: the clean no-history control C is marginally
better than A. Recursive EMA is slightly better than instant and lag-1 error
in standardized MSE, but both differences are below one percent and the
component MAEs do not improve consistently. These are single-seed,
two-drive mechanism-validation results, not generalization evidence. Seeds 1
and 2 are required before judging whether any A/D or A/E difference is stable.

All five runs completed without errors. Server artifacts, not tracked by Git:

```text
/home/lin/predify/experiments/seed0_frozen_matrix_6c446d9/
```

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
