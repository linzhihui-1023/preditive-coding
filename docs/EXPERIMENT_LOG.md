# Experiment Log

## Real-frame observation plus accumulated error-memory recurrence

Date: 2026-08-14

Code revision: `268e92698019344b9d091f50177819154ba9e87f`

Evaluation revision: `1178f7bc042c4c36d76cd902669723ddaa05a76c`

This run reorganized the learned real-frame PC recurrence into the constrained
form:

```text
e_t = F_t - Fhat_t
epsilon_t = 0.207 e_t + 0.793 epsilon_(t-1)
h_t = T(h_(t-1), F_t, E(error_input), feedback)
```

The observation feature, previous recurrent state, and top-down feedback are
kept in all formal conditions. VGG, original Predify modules, and feedback
decoders are frozen. Only the ConvGRU recurrent transition and lightweight
signed-error encoder train. Training used clean ordered streams from
0005/0013/0014/0036, selected by Val next-frame prediction MSE on 0011/0039,
and did not read Frozen Test drives 0051/0056.

The three matched-capacity conditions differ only in the error input:
`temporal_only` zeros it, `instant_error` encodes instantaneous `e_t`, and
`error_memory` encodes accumulated `epsilon_t`. Validation used only Val drives
0011/0039 with the paired 40 clean / 80 corruption / 40 recovery protocol and
two fixed corruptions: Gaussian blur and brightness overexposure.

| Corruption | Condition | Next-frame MSE | Disturbance normalized L2 | Recovery first 10 | Recovery last 10 |
| --- | --- | ---: | ---: | ---: | ---: |
| Gaussian blur | temporal_only | 2.334490107 | 0.517564676 | 0.351396176 | 0.049391711 |
| Gaussian blur | instant_error | 1.907473086 | 0.618940690 | 0.455200213 | 0.128084182 |
| Gaussian blur | error_memory | 2.025569703 | 0.579650802 | 0.411636176 | 0.093996459 |
| Brightness overexposure | temporal_only | 3.242816785 | 0.184214546 | 0.127207593 | 0.021436488 |
| Brightness overexposure | instant_error | 2.767941671 | 0.218418550 | 0.159805887 | 0.048202725 |
| Brightness overexposure | error_memory | 2.956473686 | 0.199469868 | 0.143912379 | 0.048761494 |

`error_memory` improved over `instant_error` on the main representation
deviation metric by `6.347925%` for blur and `8.675399%` for brightness. It
was nevertheless worse than `temporal_only` by `11.995820%` and `8.281280%`,
respectively, with the same direction on both Val drives.

Decision: `benefit_mainly_from_temporal_recurrence`. The accumulated
prediction-error memory helps relative to instantaneous error but does not beat
the no-error temporal recurrent baseline, so it does not support an independent
anti-corruption value claim under this protocol. Auditable small artifacts are
under `results/real_frame_error_memory_1178f7b/`; checkpoints remain under
`/tmp/predify-storage/experiments/real_frame_error_memory_train_268e926/`.

## Stage-4 Dynamic Prediction Error state: phase 1

Date: 2026-08-13

Runtime revision: `eaf29a04fc69b6d4e2e71da7dc044cca027a7278`

Corrected analysis revision: `307139167ba22f9e1197ae915a6dad6d76859417`

The frozen epoch-7 `c887e94` Stage-4 aligned temporal-difference predictor ran
48 independently reset trajectories on Val drives 0011/0039. Gaussian blur,
i.i.d. Gaussian noise, and RGB bias each used four counterbalanced replicates
of persistent severity plateaus versus the same shuffled severity multiset.
Each trajectory contained 40 baseline, 80 disturbance, and 40 recovery
transitions. The network was not updated, the dynamic state was not a
predictor input, and frozen Test drives were not read.

| Score | Higher-is-persistent AUROC | Separability AUROC |
| --- | ---: | ---: |
| Instantaneous `RMS(e_t)` | 0.35102431 | 0.64897569 |
| Scalar `EMA(RMS(e_t))` | 0.34848958 | 0.65151042 |
| Dynamic `RMS(epsilon_t)` | 0.50489583 | 0.50489583 |

The original runtime summary incorrectly compared directional AUROCs directly
and was retained as `summary_before_separability_fix.json`. Revision `3071391`
recomputed only the analysis from the saved frame/window CSVs, retained the
predeclared score direction, and ranked separability using
`max(AUROC, 1-AUROC)`. No network replay occurred.

Dynamic-state AUROC was near chance for every corruption and was `0.14407986`
below instantaneous error and `0.14661458` below scalar EMA in aggregate. The
formula audit and matched tensor-EMA difference were both exactly zero.
Decision: no-go; do not begin selective online adaptation from this state.

Audited artifacts are under
`results/stage4_dynamic_error_state_phase1_eaf29a0/`; the frozen checkpoint and
execution log remain under
`/tmp/predify-storage/experiments/stage4_dynamic_error_state_phase1_eaf29a0*`.

## Stage-4 multi-drive aligned-difference predictor

Date: 2026-08-13

Git revision: `c887e94edda86885fd0bedfaaf25e56c957719bb`

The formal seed-0 run trained the unchanged Stage-4 aligned temporal-
difference predictor on drives 0005/0013/0014/0036, selected on 0011/0039,
and reserved 0051/0056 as frozen Test. All transitions remained ordered.
Training received 1411 Train and 626 Val transitions and no Test drive names.
Val MSE selected epoch 7 before an atomic receipt claimed one Test read.

| Split | MSE | Copy MSE | MSE gain | Cosine vs Copy | NFE vs Copy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 0.93394784 | 1.15164056 | +18.902836% | 0.79362428 vs 0.76854944 | 0.59781858 vs 0.66215935 |
| Val | 1.01279691 | 1.14439157 | +11.499093% | 0.82207205 vs 0.81639943 | 0.55275378 vs 0.58144107 |
| Frozen Test | 0.77194043 | 0.89291654 | +13.548423% | 0.83322664 vs 0.82239107 | 0.54544640 vs 0.58067843 |

Drive 0051 used two independently reset segments of 56 and 379 transitions;
its MSE gain was `11.697142%`. Drive 0056 used one 293-transition segment and
improved by `15.702122%`. Both passed MSE, cosine, and normalized-error checks
against their own same-stage Copy-current replay.

The audit verified all 2765 CSV rows, split and drive counts, reset boundaries,
summary aggregation, stage separation, best-Val selection, absence of Test
drives from training config, completed single-access receipt, artifact hashes,
and finite logs. Lightweight artifacts are under
`results/stage4_multidrive_c887e94/`; the 1.41 GB checkpoint and logs remain
under `/tmp/predify-storage/experiments/seed0_stage4_multidrive_c887e94/`.

## Stage-4 same-drive 60/20/20 diagnostic

Date: 2026-08-13

Git revision: `69b24b1e0d6e23f82011ed6533565485c771b783`

The sole same-drive diagnostic split drive 0005 raw frames into chronological
Train `0--91`, Val `92--122`, and Test `123--153`. Boundary-crossing starts 91
and 122 were excluded, leaving 91/30/30 transitions with disjoint raw-frame
sets. No loader shuffled. Stage 4, aligned temporal difference, radius 1,
patch size 3, seed 0, and ten epochs remained fixed. Val selected epoch 10;
Test was evaluated once afterward.

| Split | MSE | Copy MSE | MSE gain | Cosine vs Copy | NFE vs Copy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Val | 1.98922046 | 2.41822275 | +17.740396% | 0.56127751 vs 0.55808507 | 0.85218414 vs 0.93812825 |
| Test | 1.51209933 | 1.67944006 | +9.964079% | 0.69367177 vs 0.70890513 | 0.72217820 vs 0.76012611 |

Test MSE and normalized error improved, while cosine worsened. This supports
same-drive forward generalization for the Euclidean feature objective but not
all-metric superiority. It remains compatible with the previous finding that
the same architecture fails across drives. No same-drive variants follow.

Audited artifacts are under
`results/stage4_same_drive_60_20_20_69b24b1/`; the checkpoint and logs remain
under `/tmp/predify-storage/experiments/seed0_stage4_same_drive_60_20_20_69b24b1/`.

## Stage-4 aligned temporal-difference predictor

Date: 2026-08-13

Git revision: `fdc47437931e03f1284a852b997952b29b963910`

This single-condition run changed the future prediction space from VGG Stage 5
to Stage 4 while keeping the Predify Target Flow top target at Stage 5. It
trained only the Future Predictor on 153 ordered drive-0005 transitions and
selected by MSE over 232 held-out drive-0011 transitions. The temporal input
was `D_t=F_t^4-align(F_(t-1)^4,F_t^4)` with radius 1 in Stage-4 cells and a 3x3
descriptor. All other aligned-difference controls remained fixed.

Best-checkpoint replay selected epoch 1:

| Split | Method MSE | Stage-4 Copy MSE | Aggregate gain |
| --- | ---: | ---: | ---: |
| Drive 0005 train | 2.01633201 | 2.16380702 | +6.815535% |
| Drive 0011 held-out | 0.74586091 | 0.74053836 | -0.718741% |

Held-out cosine was `0.85806100` versus Copy `0.86544334`; normalized error
was `0.49728997` versus Copy `0.48848719`. All three same-stage checks failed.
Training MSE decreased over all ten epochs while held-out MSE increased after
epoch 1. This supports only a configuration-specific held-out generalization
failure; Stage-4 and Stage-5 absolute MSE are not compared.

The audit verified all 385 replay rows, summary aggregation, stage separation,
feature shape, target/memory configuration, and checkpoint parameter shapes.
Lightweight artifacts are under
`results/stage4_aligned_temporal_difference_fdc4743/`; the 1.41 GB checkpoint
and logs remain at
`/tmp/predify-storage/experiments/seed0_stage4_aligned_temporal_difference_fdc4743/`.

## Gate 3: frozen persistence score versus 2-DoF degradation

Date: 2026-08-12

Evaluation revision: `80c4aee7f1493479764c1f21c638dbac40ca4e32`

Detector checkpoint revision: `3ffbff0156dc9435fe3b060f4c9999703729ea8a`

Motion checkpoint revision: `6c446d901a581323208a335489a7c065f1946adc`

Gate 3 asked only whether Gate 2's high persistence score corresponds to worse
existing 2-DoF task performance. Both networks were frozen; no optimizer or
online update was present. The detector was not reselected: throughout Gate 3,

```text
S = mean_8[cos(e_t,e_(t-1))]
```

with higher `S` meaning more persistent. Gate 2's tracked `summary.json` and
`per_frame.csv` were SHA256 locked. Gate 3 reproduced all 2,400 corrupted
detector rows, including sigma and all four error statistics, with maximum
absolute delta exactly `0.0`.

The three conditions were Clean, Persistent blur, and Shuffled blur. The two
blur conditions were exactly the Gate 2 trajectories. Clean was an independent
reset on the identical raw-frame/OXTS sequence. Task performance used the
formal full-method Group A seed-0 motion checkpoint selected at epoch 6. Its
standardized predictions were converted back to metres and radians. For each
nonoverlapping eight-frame disturbed window:

```text
D_forward = MAE_forward(corrupted) - MAE_forward(clean)
D_yaw     = MAE_yaw(corrupted) - MAE_yaw(clean)
```

Negative degradation was retained. No threshold, score direction, task
metric, or window size was selected in Gate 3.

Held-out drive 0011 results over all 80 corrupted windows:

| Task component | Mean signed D | D > 0 | Pearson(S,D) | Spearman(S,D) | AUROC S for D > 0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Forward | +0.000373 m | 40.0% | -0.001291 | 0.057782 | 0.500651 |
| Yaw | -0.004657 rad | 1.25% | 0.027053 | 0.076020 | 0.974684* |

`*` The yaw AUROC has only one positive-degradation window and is not stable
evidence. Shuffled blur had zero positive yaw-degradation windows.

Condition means on held-out drive 0011:

| Condition | Mean S | Forward D | Forward D / clean | Yaw D | Yaw D / clean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Persistent | -0.122692 | +0.000411 m | +0.126% | -0.004450 rad | -58.49% |
| Shuffled | -0.322340 | +0.000335 m | +0.103% | -0.004865 rad | -63.94% |

Persistent raised `S` by `0.199648` relative to Shuffled, as Gate 2 requires,
but changed forward degradation by only `+0.000076 m`. Both corruptions
improved yaw relative to clean; Persistent merely improved less. On drive 0005,
both blur organizations improved both physical MAEs, and pooled score-versus-
degradation AUROCs were near 0.5.

The audit verified 2,700 unique trajectory rows, exact 40/80/30 phase counts,
identical OXTS targets across all nine trajectories per drive, all 180 window
means, all 160 signed clean differences, and independent Pearson/Spearman
recomputation. Clean-prefix detector traces were bitwise identical. Motion
replay variation across identical clean prefixes was at most `5.96e-8`, below
the `1e-7` audit tolerance and negligible relative to reported degradation.

Gate 3 answers no for this task/checkpoint: the matched persistence score is
not a useful proxy for 2-DoF performance degradation. This blocks a controller
that would treat high `S` alone as evidence that adaptation is needed. The
existing motion proxy's known weak cross-drive/yaw behavior remains an
important scope limitation.

Versioned artifacts:

```text
results/gate3_persistence_task_degradation_80c4aee/
```

## Matched-marginal blur persistence gate

Date: 2026-08-12

Evaluation revision: `99b7e210f7922c747a08d57b9340162f359527ad`

Checkpoint revision: `3ffbff0156dc9435fe3b060f4c9999703729ea8a`

This inference-only experiment tested whether the existing causal error trace
distinguishes persistence when blur marginals are exactly matched. Every
trajectory used 40 clean, 80 disturbed, and 30 recovery transitions. Both
conditions used an 11x11 Gaussian kernel and exactly 20 future frames at each
sigma in `{0.75, 1.5, 2.25, 3.0}`. Persistent blur arranged them as four
20-frame dwell blocks; shuffled blur was a deterministic permutation with 61
runs and maximum run length 3. Blur occupancy was 100% in both conditions.

There were four counterbalanced replicates. Within every replicate the two
conditions had identical sigma multisets. Across replicates, every absolute
video frame saw every sigma exactly once in each condition, controlling the
interaction between corruption strength and natural frame difficulty. The
same raw frames and frozen checkpoint were used throughout. Absolute-frame
scheduling kept a frame bit-identical when read as one sample's future and
the next sample's current.

The primary units were nonoverlapping eight-frame disturbance windows. Drive
0005 selected metric and direction; these were frozen on drive 0011.

| Statistic | Positive direction | 0005 window AUROC | 0011 window AUROC |
| --- | --- | ---: | ---: |
| `||e_t||` | Lower | 0.900000 | 0.976250 |
| `EMA(||e_t||)` | Lower | 0.878750 | 0.990000 |
| `cos(e_t,e_(t-1))` | Higher | **0.995000** | **0.972500** |
| `Var(e_(t-7:t))` | Lower | 0.875625 | 0.978125 |

Calibration selected cosine. Its held-out per-replicate window AUROCs were
`0.98`, `0.98`, `1.00`, and `0.95`; leave-one-replicate-out AUROC ranged from
`0.964444` to `0.981111`. Excluding the first eight-frame onset window gave
`0.972222`. The secondary held-out per-frame cosine AUROC was `0.704531`, so
temporal aggregation is doing meaningful work rather than merely multiplying
the same frame-level number.

Mean held-out disturbance cosine was `-0.122692` for persistent blur and
`-0.322340` for shuffled blur. The corresponding mean absolute frame-to-frame
sigma changes were `0.065625` and `0.965625`; this difference is the intended
temporal-organization manipulation, not a marginal mismatch.

The audit reproduced every summary AUROC from CSV, verified 2,400 unique
records and 160 window rows, exact 40/80/30 phase counts, finite statistics,
identical clean prefixes, exact pairwise sigma multisets, and exact per-frame
counterbalancing. No optimizer, parameter update, traceback, OOM, or NaN
occurred. Under the user-defined thresholds this is `go_promising`.

The evidence supports proceeding to selective-adaptation mechanism design,
but only as a two-drive mechanistic result. Windows within each drive share
video content and are not independent drive samples.

Versioned artifacts:

```text
results/matched_blur_persistence_99b7e21/
```

## Prediction-error persistent-shift go/no-go

Date: 2026-08-12

Evaluation revision: `a28fed59116f12fc641384d786b86600ebcf04e4`

Checkpoint revision: `3ffbff0156dc9435fe3b060f4c9999703729ea8a`

This inference-only diagnostic reused the strict Temporal Error best
checkpoint from the original three-group matrix. It created no optimizer and
the evaluator verified that every parameter version remained unchanged. Each
drive ran three independently reset trajectories over the same 150 ordered
transitions:

```text
40 clean -> 80 disturbance -> 30 clean recovery
```

The conditions were normal clean video, persistent Gaussian blur (11x11,
sigma 3.0), and per-absolute-frame i.i.d. Gaussian RGB noise (std 0.08).
Corruption was applied in `[0,1]` RGB space after resize/crop and before
ImageNet normalization. An absolute raw frame therefore remained identical
when read first as a future and then as the next current frame.

For strict Temporal Prediction Error
`e_t=F_t-Fhat_(t|t-1)`, the predeclared causal statistics were `||e_t||`, an
EMA of that norm with alpha 0.207, `cos(e_t,e_(t-1))`, and mean elementwise
temporal variance over the latest 8 errors. Only the 80 frame-matched
disturbance transitions entered classification: persistent blur was positive;
clean and i.i.d. noise were negative.

Drive 0005 was used only to choose the direction of each score and the best
single statistic. Those choices were frozen before reporting drive 0011.

| Statistic | Positive direction | 0005 pooled AUROC | 0011 pooled AUROC |
| --- | --- | ---: | ---: |
| `||e_t||` | Lower | **0.951094** | **0.959609** |
| `EMA(||e_t||)` | Lower | 0.921250 | 0.966875 |
| `cos(e_t,e_(t-1))` | Higher | 0.681484 | 0.772188 |
| `Var(e_(t-7:t))` | Lower | 0.902813 | 0.942734 |

The calibration rule selected raw `||e_t||`; the numerically higher held-out
EMA AUROC is reported but was not selected post hoc. For selected `||e_t||`,
drive-0011 pairwise AUROC was 0.937031 against clean and 0.982188 against i.i.d.
noise. Mean disturbance-phase norms were:

| Drive | Persistent blur | Clean | i.i.d. noise |
| --- | ---: | ---: | ---: |
| 0005 | 81.4584 | 117.5187 | 121.5470 |
| 0011 | 59.4953 | 86.7701 | 107.2486 |

The effect was not an onset-only shortcut. With the first eight disturbed
transitions excluded, selected-score AUROC was 0.958912 on 0005 and 0.971065
on 0011. AUROC over nonoverlapping eight-frame score means was 0.960 and 0.985.
On each drive, persistent blur had the correctly oriented score against each
matched control on 79 of 80 individual disturbance frames.

Under the user-specified engineering thresholds, held-out AUROC 0.959609 is a
`go_promising` result. The interpretation is narrower than “prediction error
detects persistence”: blur lowers feature prediction-error norm, and the
selected statistic is instantaneous. Because persistent blur and i.i.d.
Gaussian noise differ in corruption type and marginal effect, this experiment
shows that the tested conditions are separable from `e`; it does not isolate
temporal persistence. Before designing a controller, the next gate should
compare persistent blur with a time-randomized blur control matched in marginal
severity.

The audit verified 900 unique drive/condition/stream rows, exact 40/80/30 phase
counts, finite statistics, identical clean prefixes across all conditions,
calibration-only score orientation, and exact CSV-to-summary AUROCs. No
traceback, OOM, NaN, or network update occurred.

Versioned artifacts:

```text
results/prediction_error_separability_a28fed5/
```

## Stage-5 causal warp-residual matrix

Date: 2026-08-12

Git revision: `79de55494d24f4e01083851f8331e6c96b2d87f1`

This formal seed-0 matrix followed the local-motion gate without changing its
order. It used drive 0005 for 153 ordered training transitions and drive 0011
for 232 ordered validation transitions. All groups used the frozen pretrained
VGG, recursive Target Flow, a student-self future target, batch size 1, ten
epochs, and best-checkpoint selection by validation feature MSE.

The three conditions were:

| Condition | Prediction |
| --- | --- |
| Copy-current | `Fhat_(t+1)=F_t` |
| Historical warp | `Fhat_(t+1)=W(F_t,M(F_(t-1),F_t))` |
| Warp residual | `Fhat_(t+1)=W(F_t,M(F_(t-1),F_t))+Rhat_(t+1)` |

Motion used radius 1 and 3x3 feature descriptors. The local argmin consumed
only `F_(t-1)` and `F_t`; the future target was resolved after prediction.
Discrete forward splatting averaged collisions and filled holes with the
unwarped current feature. Stored previous features were detached.

Best-checkpoint replay on the complete ordered drives:

| Condition | Best epoch | Train MSE | Versus Copy | Val MSE | Versus Copy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Copy-current | 1 | 0.138954983 | 0.000% | 0.060080099 | 0.000% |
| Historical warp | 1 | 0.129720510 | +6.646% | 0.059948462 | +0.219% |
| Warp residual | 3 | 0.124021935 | +10.747% | 0.061526919 | -2.408% |

Warp residual improved over warp-only by 4.393% on train and degraded it by
2.633% on validation. Its final epoch reached train MSE `0.112814438` while
validation rose to `0.067354957`, so the selected checkpoint and full learning
curve show a cross-drive generalization failure in this configuration.

The deterministic warp reproduces the earlier causal gate: it helps strongly
on drive 0005 and only marginally on held-out drive 0011. This is causal
evidence that historical feature-space motion has predictive value on both
tested drives under the literal MSE gate, but the 0.219% held-out margin is too
small for a broad transport claim. The learned post-warp residual did not pass
the held-out Copy or warp-only gates. This does not show that residuals are
unlearnable; the experiment uses one seed, one training drive, and one small
predictor.

The audit checked the exact revision, all 1,155 unique condition/split/frame
keys, finite metrics, checkpoint forms and selected epochs, and exact
CSV-to-summary means. Warp coverage excluding the one bootstrap frame averaged
0.9252 on train and 0.9711 on validation. No traceback, OOM, NaN, or timestamp
drop was found.

Versioned artifacts:

```text
results/seed0_warp_residual_matrix_79de554/
```

Server-only checkpoints and logs:

```text
/tmp/predify-storage/experiments/seed0_warp_residual_matrix_79de554/
```

## Local matching and causal historical warp

Date: 2026-08-12

Git revision: `06ec8e79832b981873854da14939bcebb33f2974`

The fixed execution order was future-selected local matching for stage-5 at
`h=1,3`, the same diagnostic for stage-4, and finally causal historical motion
estimation and forward warp. Both radii `r=1,2` and 1x1/3x3 descriptors were
run. All methods used the same 150 drive-0005 and 229 drive-0011 origins valid
from `t-1` through `t+3`.

Future-selected local matching gain, defined as
`(MSE_copy-MSE_local)/MSE_copy` using aggregate means:

| Stage | h | Radius | Patch | Drive 0005 | Drive 0011 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5 | 1 | 1 | 1x1 | 18.374% | 12.072% |
| 5 | 1 | 2 | 1x1 | 19.391% | 13.231% |
| 5 | 3 | 1 | 1x1 | 35.624% | 28.985% |
| 5 | 3 | 2 | 1x1 | 44.777% | 33.115% |
| 5 | 1 | 1 | 3x3 | 12.386% | 6.824% |
| 5 | 1 | 2 | 3x3 | 13.282% | 7.944% |
| 5 | 3 | 1 | 3x3 | 28.859% | 22.411% |
| 5 | 3 | 2 | 3x3 | 38.666% | 26.200% |
| 4 | 1 | 1 | 1x1 | 46.850% | 29.529% |
| 4 | 1 | 2 | 1x1 | 50.683% | 31.868% |
| 4 | 3 | 1 | 1x1 | 31.910% | 28.577% |
| 4 | 3 | 2 | 1x1 | 46.544% | 37.373% |
| 4 | 1 | 1 | 3x3 | 43.049% | 25.456% |
| 4 | 1 | 2 | 3x3 | 47.688% | 28.626% |
| 4 | 3 | 1 | 3x3 | 23.928% | 21.398% |
| 4 | 3 | 2 | 3x3 | 40.528% | 31.985% |

These are noncausal values: each target location or patch uses the future
feature to choose a nearby source. The stable pointwise reductions support a
local spatial-correspondence hypothesis, but 1x1 matching can also select a
nearby feature with a convenient value and is not by itself motion prediction.
The 3x3 rows retain substantial reductions while imposing more local
structure.

Causal historical warp at `h=1`:

| Stage | Radius | Patch | 0005 Copy | 0005 warp | Gain | 0011 Copy | 0011 warp | Gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 1 | 1x1 | 0.139440 | 0.133229 | 4.454% | 0.060554 | 0.061838 | -2.121% |
| 5 | 2 | 1x1 | 0.139440 | 0.134018 | 3.888% | 0.060554 | 0.062033 | -2.442% |
| 5 | 1 | 3x3 | 0.139440 | 0.129907 | 6.837% | 0.060554 | 0.060420 | 0.221% |
| 5 | 2 | 3x3 | 0.139440 | 0.130164 | 6.652% | 0.060554 | 0.060516 | 0.062% |
| 4 | 1 | 1x1 | 2.177706 | 1.333488 | 38.766% | 0.744313 | 0.583636 | 21.587% |
| 4 | 2 | 1x1 | 2.177706 | 1.322291 | 39.281% | 0.744313 | 0.584560 | 21.463% |
| 4 | 1 | 3x3 | 2.177706 | 1.280373 | 41.205% | 0.744313 | 0.569829 | 23.442% |
| 4 | 2 | 3x3 | 2.177706 | 1.256662 | 42.294% | 0.744313 | 0.569650 | 23.466% |

The predeclared gate required one fixed radius/patch configuration to beat
stage-5 Copy-current on both drives. The 3x3 configurations passed and thereby
permitted the subsequent historical-warp-base residual experiment recorded
above. The held-out stage-5 gain is only 0.221% for `r=1` and 0.062% for
`r=2`; gate passage is therefore literal but weak. Stage-4 causal transport is
much stronger and must not be substituted for the formal stage-5 result.

The audit verified exact revision, 9,096 unique rows, frame horizons, finite
metrics, per-row arithmetic, all 48 CSV-to-summary aggregates, and the causal
gate. No network was trained in this diagnostic.

Versioned artifacts:

```text
results/vgg_local_motion_06ec8e7/
```

## VGG feature-task learnability matrix

Date: 2026-08-12

Git revision: `1605f29c28b214b25f2f4c2df6c06adf45c8f554`

This forward-only diagnostic evaluated ImageNet VGG16 stage 3, 4, and 5 at
`h=1,2,3,5`, corresponding to 0.1035, 0.2070, 0.3105, and 0.5175 seconds.
Drive 0005 supplied 148 forecast origins and drive 0011 supplied 227. All
origins have a valid history frame and valid futures through `h=5`, so every
horizon within one drive uses the same sample set.

The three comparisons are:

```text
Copy:       Fhat_(t+h) = F_t
Velocity:   Fhat_(t+h) = F_t + h(F_t-F_(t-1))
Translation oracle: future-selected integer translation of F_t
```

The oracle searched `dy,dx in {-1,0,1}` feature cells, used zero fill, and
computed full-map MSE. Stage 3/4/5 have strides 4/8/16 input pixels, so the
maximum searched displacement differs by stage. Because the future target
selects the shift, oracle values are diagnostic lower bounds within this small
translation family, not causal prediction results.

Drive 0005:

| Stage | h | Copy MSE | Velocity MSE | Velocity gain | Oracle MSE | Oracle gain | Adjacent delta cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 1 | 15.557438 | 43.886687 | -182.095% | 11.790630 | 24.212% | -0.3961 |
| 3 | 2 | 18.350077 | 110.058762 | -499.773% | 17.628484 | 3.932% | -0.3961 |
| 3 | 3 | 19.185980 | 204.529077 | -966.034% | 18.907528 | 1.451% | -0.3961 |
| 3 | 5 | 19.992369 | 485.845376 | -2330.154% | 19.871251 | 0.606% | -0.3961 |
| 4 | 1 | 2.185479 | 5.303839 | -142.685% | 1.542329 | 29.428% | -0.1969 |
| 4 | 2 | 3.436122 | 15.472533 | -350.291% | 2.723608 | 20.736% | -0.1969 |
| 4 | 3 | 3.973900 | 29.411911 | -640.127% | 3.562695 | 10.348% | -0.1969 |
| 4 | 5 | 4.384387 | 69.518976 | -1485.603% | 4.224353 | 3.650% | -0.1969 |
| 5 | 1 | 0.139504 | 0.321587 | -130.522% | 0.136297 | 2.298% | -0.1549 |
| 5 | 2 | 0.236044 | 0.905048 | -283.424% | 0.197146 | 16.479% | -0.1549 |
| 5 | 3 | 0.319733 | 1.790191 | -459.902% | 0.260107 | 18.649% | -0.1549 |
| 5 | 5 | 0.436755 | 4.424232 | -912.979% | 0.376618 | 13.769% | -0.1549 |

Drive 0011:

| Stage | h | Copy MSE | Velocity MSE | Velocity gain | Oracle MSE | Oracle gain | Adjacent delta cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 1 | 4.971944 | 12.730875 | -156.054% | 4.971709 | 0.005% | -0.2971 |
| 3 | 2 | 7.188927 | 34.655441 | -382.067% | 7.084794 | 1.449% | -0.2971 |
| 3 | 3 | 8.445830 | 66.180344 | -683.586% | 8.288973 | 1.857% | -0.2971 |
| 3 | 5 | 9.701977 | 157.552512 | -1523.922% | 9.549371 | 1.573% | -0.2971 |
| 4 | 1 | 0.750586 | 1.769467 | -135.745% | 0.750586 | 0.000% | -0.1977 |
| 4 | 2 | 1.236032 | 5.169621 | -318.243% | 1.232554 | 0.281% | -0.1977 |
| 4 | 3 | 1.527861 | 10.001076 | -554.580% | 1.526526 | 0.087% | -0.1977 |
| 4 | 5 | 1.850616 | 23.803775 | -1186.262% | 1.846047 | 0.247% | -0.1977 |
| 5 | 1 | 0.061063 | 0.140873 | -130.703% | 0.061063 | 0.000% | -0.1874 |
| 5 | 2 | 0.103505 | 0.394610 | -281.247% | 0.101744 | 1.702% | -0.1874 |
| 5 | 3 | 0.141344 | 0.781827 | -453.136% | 0.131837 | 6.726% | -0.1874 |
| 5 | 5 | 0.197528 | 1.922527 | -873.295% | 0.184736 | 6.476% | -0.1874 |

`Velocity gain` and `Oracle gain` are ratios of aggregate means:
`(MSE_copy-MSE_method)/MSE_copy`. Adjacent delta cosine is repeated across
horizons because all horizons use the same origins and it always compares the
one-step deltas immediately before and after `t`.

Direct observations are limited to this diagnostic. Raw constant-velocity
extrapolation was worse than Copy-current for every one of the 4,500 matrix
rows, consistent with the negative adjacent-delta cosine means. On drive 0011,
the small translation oracle reduced aggregate Copy-current MSE by 0--6.726%,
so a single global integer feature shift does not account for most of that
drive's error under this search. Drive 0005 has a different oracle pattern,
including 24.212% and 29.428% reductions for stage-3 and stage-4 at `h=1`.
This matrix does not test a learned predictor and does not by itself establish
that future-feature prediction is unlearnable.

The output audit verified revision, 4,500 unique matrix rows, exact raw-frame
horizons, finite metrics, oracle MSE no greater than Copy-current, and exact
agreement between CSV recomputation and all 24 summary entries.

Versioned artifacts:

```text
results/vgg_feature_learnability_1605f29/
```

## Top-layer Temporal Prediction Error matrix, seed 0

Date: 2026-08-12

Git revision: `3ffbff0156dc9435fe3b060f4c9999703729ea8a`

The first strict Temporal Prediction Error matrix completed on the existing
cross-drive split: 153 ordered training pairs from drive 0005 and 232 ordered
validation pairs from drive 0011. All conditions used the frozen ImageNet
VGG16 backbone, recursive Target Flow, the 1x1 future-feature predictor, seed
0, and best-checkpoint selection by validation feature MSE. The Temporal Error
condition used only the previous completed top-layer state:

```text
e_t^5 = F_t^5 - Fhat_(t|t-1)^5
E_t^5 = 0.207 e_t^5 + 0.793 E_(t-1)^5
Fhat_(t+1|t)^5 = F_t^5 + P(F_t^5, E_t^5)
```

The future target was observed only after prediction. Target Flow and Temporal
Error parameters were configured independently, although both used
`Ts=0.1035`, `tau=0.5`, and `K=1` in this run.

| Condition | History input | Best epoch | Feature MSE | Feature cosine | Normalized feature error |
| --- | --- | ---: | ---: | ---: | ---: |
| Copy-current | Predictor bypassed | 1 | **0.060080099** | **0.919904691** | **0.375766109** |
| Current-only | Zeros | 2 | 0.061797074 | 0.915814666 | 0.384975467 |
| Temporal Error | Previous completed `E_t^5` | 2 | 0.061903913 | 0.915660890 | 0.385387186 |

Gate results:

- Gate 1 failed. Current-only was 2.85781% worse than Copy-current.
- Gate 2 failed. Temporal Error was 0.17289% worse than Current-only.
- Temporal Error was 3.03564% worse than Copy-current.
- Copy-current and Current-only exactly reproduced their prior 1x1 matrix
  values, providing a direct configuration consistency check.

Interpretation:

- The strict Temporal Error path produces a numerically distinct result, but
  this seed provides no evidence that it improves held-out next-feature
  prediction.
- Current-only still fails the prerequisite Copy-current gate. Therefore the
  Temporal Error comparison is secondary and must not be generalized into a
  claim that prediction-error memory is broadly useless.
- Both learned conditions selected epoch 2 and then degraded on validation as
  training error continued to fall. Cross-drive predictor generalization
  remains the primary unresolved limitation.
- This run is not the same-drive controlled-corruption experiment. It contains
  no step, ramp, persistent-bias, recovery, Peak Error, Recovery Time, or AUEC
  result.
- Do not start a `tau_e` sweep or seeds 1 and 2 to rescue these failed primary
  comparisons. A future same-drive corruption run may test transient response
  mechanistically, but cannot replace the cross-drive predictor gate.

All three best checkpoints were reloaded and audited. Their selected epochs,
history modes, validation MSE values, checkpoint kind, and Git revision match
the saved histories and manifest. No final checkpoints were retained, and no
training log contains a traceback, runtime error, CUDA out-of-memory error, or
NaN.

Server artifacts, not tracked by Git:

```text
/tmp/predify-storage/experiments/seed0_future_feature_matrix_3ffbff0/
size: about 4.0 GB
```

## Same-drive controlled corruption, seed 0

Date: 2026-08-12

Code and experiment revision:
`ae90a9f74f1560f5b3d6d72905cfe1b16a5b34c4`

The formal runner trained clean Copy-current, Current-only, and Temporal Error
checkpoints on drive 0011. Raw frames 0--138 supplied 138 training pairs,
frames 139--158 formed a 20-frame gap, and frames 159--204 supplied 45 ordered
validation transitions. The train and validation samples share no raw frame.
All best checkpoints and the evaluator report the exact revision above.

Corruption is applied only by the independent evaluator after resize/crop and
before normalization. Step-bias, ramp-bias, and i.i.d.-noise negative-control
trajectories are run independently with a full reset before every clean and
corrupted stream. Fixed RGB bias contains no noise, and i.i.d. noise contains
no bias. Absolute-frame deterministic corruption preserves the identity of a
frame when it is read first as a future image and then as the next current
image.

Clean best-checkpoint results:

| Condition | Best epoch | Feature MSE | Feature cosine | Normalized feature error |
| --- | ---: | ---: | ---: | ---: |
| Copy-current | 1 | **0.071884151** | **0.941850869** | **0.335404187** |
| Current-only | 3 | 0.074492955 | 0.938545369 | 0.341990503 |
| Temporal Error | 3 | 0.074685545 | 0.938351866 | 0.342506164 |

Current-only was 3.62918% worse than Copy-current. Temporal Error was 0.25853%
worse than Current-only and 3.89709% worse than Copy-current. The same-drive
training therefore also failed both prerequisite gates.

Primary paired-corruption result, signed excess
`delta L_t=L_t(corrupted)-L_t(clean)` integrated with `dt=0.1035 s`:

| Trajectory | Copy-current signed AUEC | Current-only signed AUEC | Temporal Error signed AUEC | Temporal versus Current-only |
| --- | ---: | ---: | ---: | ---: |
| Step bias | 0.053056728 | 0.053893020 | 0.054012604 | +0.22189% |
| Ramp bias | 0.042441423 | 0.043303211 | 0.043401632 | +0.22728% |
| I.i.d. noise | 0.040481845 | 0.034820756 | 0.034471407 | -1.00328% |

All nine recovery measurements reached the signed-excess threshold after one
recovery frame and were not censored. This coarse result is identical across
conditions and should not be interpreted as a Temporal Error benefit. Temporal
Error has mixed, very small AUEC differences: it is slightly worse for both
systematic-bias trajectories and slightly better for the unpredictable-noise
negative control. It does not show the predicted selective adaptation to
persistent systematic bias.

State-scale and utilization diagnostics reject the explanation that the state
was simply too small or ignored. For the Temporal Error checkpoint, the first
predictor layer's history-input weight RMS is `0.0182370` versus `0.0179760`
for feature input, a ratio of `1.01452`. Across phases, `RMS(E_t)/RMS(F_t)` is
approximately 0.104--0.143, and
`RMS(P(F_t,E_t)-P(F_t,0))/RMS(delta_hat)` is approximately 0.166--0.227.
However, the same-checkpoint history ablation
`MSE(P(F_t,E_t))-MSE(P(F_t,0))` is positive in every phase mean, from about
`7.7e-5` to `3.5e-4`. The predictor materially uses `E_t`, but that use hurts
prediction on this validation trace.

The abrupt step also confirms the intended causal ordering in the saved
per-frame trace. At future raw frame 169, prediction-error RMS rises to
`0.466116` and the completed state rises from the input `0.111943` to
`0.123362`; only frame 170 receives that newly completed state as its history
input. Feature MSE at the step is `0.217264`, and the later state use does not
produce a consistent loss reduction.

Interpretation is deliberately narrow. This is one seed on one 45-transition
same-drive validation segment. Natural scene changes can still dominate raw
Peak Error, so signed paired excess is primary; signed negatives are retained.
The run is a mechanistic transient diagnostic, not cross-drive generalization,
broad robustness, or online-adaptation evidence. It does not justify a
Temporal Error time-constant sweep while Current-only still loses to
Copy-current.

All three checkpoints were selected by best validation feature MSE. Histories,
checkpoint metadata, manifest, and evaluator summary agree on revision,
history mode, split, and selected epoch. The runner exited successfully; its
logs contain no traceback, runtime error, CUDA OOM, or NaN. The full 53-test
suite had passed on the experiment revision before this run.

Server artifacts, not tracked by Git:

```text
/tmp/predify-storage/experiments/seed0_same_drive_controlled_corruption_ae90a9f/
size: about 4.0 GB
```

Versioned audit artifacts:

```text
results/seed0_same_drive_controlled_corruption_ae90a9f/
```

The tracked copy contains the original `summary.json`, all nine 45-row
per-checkpoint/per-trajectory CSV traces, provenance notes, and SHA-256 hashes.
It is sufficient to re-audit AUEC, phase curves, recovery metrics, causal
`e -> E` traces, and history-utilization measurements without a checkpoint.

## Current-only delta and spatial-predictor diagnostic, seed 0

Date: 2026-08-12

Diagnostic and 3x3 experiment revision:
`1ae6b2776c793e3d7d53300c9321dece8ba6a344`

The diagnostic evaluated best validation checkpoints on all 153 ordered train
pairs from drive 0005 and all 232 ordered validation pairs from drive 0011.
For each frame it measured the true `delta F=F_(t+1)-F_t`, the Current-only
prediction `delta_hat=P(F_t,0)`, their flattened L2 norms, norm ratio, cosine,
and MSE against the matched Copy-current baseline. The backbone remained
frozen. No history input or `tau` setting changed.

True feature-change scale:

| Split | Element mean | Element std | L2 P50 | L2 P90 | L2 P95 | RMS P50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train 0005 | 0.000254 | 0.372769 | 114.175 | 148.687 | 151.860 | 0.360418 |
| Validation 0011 | 0.000091 | 0.245113 | 80.726 | 98.295 | 101.554 | 0.254829 |

Best-checkpoint predictor diagnostics:

| Predictor | Best epoch | Split | MSE | Copy MSE | vs Copy | Norm ratio | Delta cosine |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1x1 | 2 | Train | 0.133647 | 0.138957 | -3.821% | 0.1582 | 0.1949 |
| 1x1 | 2 | Validation | 0.061798 | 0.060081 | +2.858% | 0.3562 | 0.0683 |
| 3x3-first | 1 | Train | 0.136159 | 0.138957 | -2.014% | 0.0952 | 0.1469 |
| 3x3-first | 1 | Validation | 0.060632 | 0.060081 | +0.917% | 0.1789 | 0.0269 |

The table uses a post-training pass through each saved best checkpoint. It is
more precise than the online training average printed while weights change
within an epoch. The old 1x1 checkpoint therefore improves train MSE by 3.82%,
although its epoch-2 online average showed a 1.88% improvement.

The 3x3 sufficiency run changed only the first future-predictor convolution
from 1x1 to 3x3 with padding 1; the output convolution stayed 1x1. Its train
MSE fell monotonically to 0.093191 by epoch 10, 32.93% below the fixed train
Copy-current value 0.138955, while validation MSE rose to 0.095457. The best
validation checkpoint remained epoch 1. This confirms only that this spatial,
much larger predictor can fit substantially more of the training-drive target
by late epochs; it does not show that spatial transport was learned more
accurately on the held-out drive.

Interpretation:

- The 1x1 predictor is not collapsing exactly to Copy-current, but its
  predicted feature change is much too small and weakly aligned with the true
  change. Validation direction is especially poor.
- The 3x3 best validation MSE is numerically closer to Copy-current than the
  1x1 result, but this is not evidence of better spatial-motion prediction. Its
  validation predicted-delta RMS is smaller (0.0307 versus 0.0612), its norm
  ratio is smaller (0.1789 versus 0.3562), and its delta cosine is worse
  (0.0269 versus 0.0683). The most direct interpretation is that its best
  checkpoint stays closer to zero correction and therefore closer to the
  Copy-current predictor.
- The stronger 3x3 predictor rapidly fits drive 0005 and rapidly overfits drive
  0011 at later epochs. This demonstrates additional train fitting capacity,
  not improved held-out spatial prediction. The present result therefore
  cannot evaluate whether inherited state is useful; Current-only has not
  passed the cross-drive Copy-current gate.
- The 3x3-first predictor has 9,963,008 parameters versus 1,574,400 for 1x1.
  This is a predictor-sufficiency diagnostic, not a parameter-matched
  architecture ablation.
- The current history matrix tests only the top Target Flow residual state. It
  does not test prediction-state memory or lower-layer feedback-decoder state,
  so its null result must not be generalized to the complete Predify state.

Decision: do not tune `tau`, rerun history groups, or start seeds 1 and 2 yet.
The next experiment should address cross-drive generalization with more
training sequences or a deliberately train-only predictor-capacity study,
then require Current-only to beat Copy-current before testing history again.

The checkpoint diagnostic rejects non-future-feature and non-Current-only
checkpoints. It also verifies the configured kernel against the first
predictor weight. Legacy checkpoints without a kernel field are accepted only
when that weight is explicitly verified as 1x1.

Server artifacts, not tracked by Git:

```text
/tmp/predify-storage/experiments/seed0_future_feature_matrix_94059be/current_only_seed0_delta_diagnostics.json
/tmp/predify-storage/experiments/seed0_future_feature_predictor_sufficiency_1ae6b27/
```

## Causal future-feature matrix, seed 0

Date: 2026-08-11

Git revision: `94059bed62f4d6ea5aa29d01af58319d37498da4`

The first primary-task matrix used 153 ordered training pairs from drive 0005
and 232 ordered validation pairs from drive 0011. All groups used the same
frozen ImageNet VGG16 backbone, recursive Target Flow, trainable feedback
decoders, full stage-5 residual predictor, instantaneous local loss, zero
variance weight, seed 0, and best-checkpoint selection by validation feature
MSE. Future labels were detached student-self top features. No EMA teacher was
constructed.

| Condition | History input at `t -> t+1` | Best epoch | Feature MSE | Feature cosine | Normalized feature error |
| --- | --- | ---: | ---: | ---: | ---: |
| Copy-current | Predictor bypassed | 1 | **0.060080099** | **0.919904691** | **0.375766109** |
| Current-only | Zeros | 2 | 0.061797074 | 0.915814666 | 0.384975467 |
| Latest | `e_(t-1)` | 2 | 0.061849746 | 0.916069347 | 0.384410881 |
| Two-tap | `0.207e_(t-1)+0.793e_(t-2)` | 2 | 0.061899316 | 0.915900138 | 0.384815322 |
| Recursive | `0.207e_(t-1)+0.793epsilon_(t-2)` | 2 | 0.061796151 | 0.915806531 | 0.384969822 |

Key comparisons:

- Current-only is 2.858% worse than copy-current, so the predictor did not pass
  the minimum requirement of learning a useful future change on validation.
- Recursive is only 0.00149% better than current-only. This numerical tie is
  not evidence that inherited history adds information beyond `F_t`.
- Latest and two-tap are 0.085% and 0.165% worse than current-only. Recursive
  is 0.087% better than latest and 0.167% better than two-tap, but those tiny
  differences are secondary because the learned predictor does not beat the
  non-learned copy baseline.
- Learned training feature MSE continues to fall while validation MSE reaches
  its minimum at epoch 2 and then rises. The matrix therefore shows rapid
  cross-drive overfitting.
- Future-feature MSE and residual-delta MSE agree within `2.235e-8` in every
  group, validating the residual-loss implementation.

Decision:

Do not run seeds 1 and 2 yet. The first feature-task gate failed, and recursive
history is indistinguishable from current-only at seed 0. Follow the planned
order: diagnose feature-delta scale and predictor behavior, then revise the
residual/history formulation before spending on robustness or additional
seeds. The two-drive split remains useful for this mechanism rejection but is
not sufficient for a broad generalization claim.

Server artifacts, not tracked by Git:

```text
/tmp/predify-storage/experiments/seed0_future_feature_matrix_94059be/
```

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

## Real-frame Predictive Coding Phase 1

Date: 2026-08-13

Evaluation revision: `1d2587743f36ce29850d15557d4e7fa795d888ea`

Protocol: frozen PVGG16 and PCoder decoders, Val drives 0011/0039, raw frames
0--159 on each drive, and a fixed 40-clean/80-persistent-Gaussian-blur/40-clean
recovery trajectory. Blur used kernel 11 and sigma 3.0. Feedforward reset on
every frame; PC-no-error retained representation and feedback state with
`alpha=0`; PC-dynamic-error used the complete `0.207/0.793` recurrence and
original `K/C_sqrt` correction. No training, optimizer, online update, future
predictor, or Frozen Test read occurred.

| Condition | Disturbance normalized L2 | Recovery normalized L2 | Recovery last 10 |
| --- | ---: | ---: | ---: |
| Feedforward | 0.890284632 | 0.000000000 | 0.000000000 |
| PC-no-error | 0.784450073 | 0.126158581 | 0.001216764 |
| PC-dynamic-error | 0.784296992 | 0.126198921 | 0.001205088 |

The dynamic condition improved disturbance distance over PC-no-error by only
`0.019514%`, below the predeclared 5% practical threshold, although both drives
had the same favorable direction. Dynamic recovery declined from `0.427156845`
in the first ten recovery frames to `0.001205088` in the final ten. Decision:
`NO-GO`. This rejects a meaningful incremental dynamic-error benefit only for
the fixed Phase-1 mechanism and blur protocol.

Tracked audit artifacts:

```text
results/real_frame_pc_phase1_1d25877/
```

Server-only log:

```text
/tmp/predify-storage/experiments/real_frame_pc_phase1_1d25877.log
```

### Four-condition mechanism decomposition

Evaluation revision: `9a3d9bd8b64bd13fd846add5b04d6e0610a83834`

The protocol was unchanged and added only Representation-memory-only, which
retained `R_(t-1)` while setting feedback and error correction to zero. The
three original condition summaries were exactly reproduced.

| Condition | Disturbance normalized L2 | Relative improvement from previous |
| --- | ---: | ---: |
| A Feedforward | 0.890284632 | - |
| B Representation-memory only | 0.803649702 | 9.731150% |
| C Representation + Feedback | 0.784450073 | 2.389054% |
| D Representation + Feedback + Dynamic error | 0.784296992 | 0.019514% |

A-to-C improved by `11.887722%`. Representation memory supplied `0.086634929`
of absolute reduction (`81.8588%`), while feedback supplied `0.019199629`
(`18.1412%`). Thus the gain came mainly from representation memory. Dynamic
error remained `NO-GO` under the unchanged Phase-1 criterion.

Tracked artifacts and server-only log:

```text
results/real_frame_pc_phase1_9a3d9bd/
/tmp/predify-storage/experiments/real_frame_pc_phase1_9a3d9bd.log
```

## Learned Recurrent-error Transition

Date: 2026-08-13

Training/evaluation revision: `c9fec52`

The learned path retained the original per-layer feedforward and top-down
feedback base update, disabled the fixed dynamic-error gradient projection,
and trained only 7,328,064 parameters in five 1x1 ConvGRU transition cells.
Ordered clean streams from 0005/0013/0014/0036 supplied 1,411 transition
frames per epoch. Clean prediction MSE on Val 0011/0039 selected epoch 1 at
`0.893984978`; the run used one fixed five-epoch, `lr=1e-4` configuration.

Formal validation reused the Phase-1 40-clean/80-persistent-Gaussian-blur/
40-clean protocol on 0011/0039. Learned and zeroed loaded the identical
checkpoint; zeroed changed only the recurrent transition's error input.

| Condition | Disturbance normalized L2 | Recovery first 10 | Recovery last 10 |
| --- | ---: | ---: | ---: |
| Current stateful | 0.784296992 | 0.427156845 | 0.001205088 |
| Learned recurrent-error | 0.501548966 | 0.280066796 | 0.004235435 |
| Learned recurrent-error zeroed | 0.521438669 | 0.301188880 | 0.045429500 |

Learned improved by `36.051143%` over current-stateful and by `3.814390%`
over zeroed. The learned-versus-zeroed improvement was `2.291069%` on drive
0011 and `5.414597%` on drive 0039. It was positive at every layer and largest
at Stage 5 (`18.557417%`). Learned versus current was worse at Stage 1 but
better at Stages 2--5. This supports both a learned recurrent-state benefit and
a smaller, separately identified dynamic-error contribution. Frozen Test
drives 0051/0056 were not read.

Tracked artifacts and server-only checkpoint:

```text
results/real_frame_recurrent_error_c9fec52/
/tmp/predify-storage/experiments/real_frame_recurrent_error_train_c9fec52/
checkpoint sha256: 26c5333da95a6b754af6036f9d16a5dd394673400e6f9456c0ebc0b27e714cf4
```

### Frozen Test

Evaluator revision: `8915f75`

The unchanged `c9fec52` epoch-1 checkpoint received one Frozen Test access on
0051/0056. The evaluator claimed an atomic checkpoint-bound receipt before
reading Test frames. It reused the Val conditions, preprocessing, seed,
40-clean/80-blur/40-recovery trajectory, and metrics without training or
tuning. Drive 0051 used the first qualifying continuous 160-frame segment
after its timestamp break (raw frames 58--217); 0056 used raw frames 0--159.

| Condition | Disturbance normalized L2 | Recovery first 10 | Recovery last 10 |
| --- | ---: | ---: | ---: |
| Current stateful | 0.819635281 | 0.446208276 | 0.001090620 |
| Learned recurrent-error | 0.536746322 | 0.305349457 | 0.001978208 |
| Learned recurrent-error zeroed | 0.548573083 | 0.310442773 | 0.040968571 |

| Drive | Learned vs current | Learned vs zeroed |
| --- | ---: | ---: |
| 0051 | 37.165928% | 0.568743% |
| 0056 | 31.765203% | 3.624299% |

Both drives passed both predeclared comparisons. Aggregate improvements were
`34.514005%` over current and `2.155914%` over zeroed, yielding
`PASS_MAIN_MECHANISM`. The layer-level direction was not uniform: Stage 1 was
worse than current and Stage 4 was worse than zeroed. The conclusion is the
predeclared drive-level main-metric result, not layer-uniform dominance.

Tracked artifacts and completed server receipt:

```text
results/real_frame_recurrent_error_frozen_test_c9fec52/
/tmp/predify-storage/experiments/real_frame_recurrent_error_train_c9fec52/best_recurrent_transition_real_frame_frozen_test_receipt.json
```

### Cross-corruption Val

Evaluator revision: `54e893e`

The frozen `c9fec52` epoch-1 checkpoint was evaluated only on Val drives
0011/0039 with the unchanged three conditions and 40/80/40 protocol. Gaussian
noise used deterministic absolute-frame realizations with pixel-space
`std=0.08`; brightness shift added uniform pixel-space RGB `+0.15`. Both were
applied before ImageNet normalization. No training, tuning, checkpoint
selection, severity sweep, or Frozen Test read occurred.

| Corruption | Condition | Disturbance L2 | Recovery first 10 | Recovery last 10 |
| --- | --- | ---: | ---: | ---: |
| Gaussian noise | Current | 0.650661257 | 0.392622591 | 0.001167595 |
| Gaussian noise | Learned | 0.490607851 | 0.316237350 | 0.007056333 |
| Gaussian noise | Zeroed | 0.487297715 | 0.307258255 | 0.036697759 |
| Brightness shift | Current | 0.272876398 | 0.156972760 | 0.000737066 |
| Brightness shift | Learned | 0.184116804 | 0.102595143 | 0.000840261 |
| Brightness shift | Zeroed | 0.197603509 | 0.114548865 | 0.020621419 |

Gaussian noise: learned improved `24.598576%` over current but was `0.679284%`
worse than zeroed. The learned-versus-zeroed direction was negative on 0011
and positive on 0039. Brightness shift: learned improved `32.527399%` over
current and `6.825134%` over zeroed, favorable on both drives. The macro mean
over the two fixed corruptions was `26.941295%` versus current and `1.485845%`
versus zeroed, but mechanism interpretation remains corruption-specific.

Tracked artifacts and server-only log:

```text
results/real_frame_recurrent_error_cross_corruption_54e893e/
/tmp/predify-storage/experiments/real_frame_recurrent_error_cross_corruption_54e893e.log
```

### Unified Real-frame Robustness Val Benchmark

Evaluator revision: `42a7029`

The unchanged epoch-1 `c9fec52` checkpoint was evaluated on Val drives
0011/0039 only. The benchmark used paired per-model clean references and the
same 40-clean/80-corruption/40-recovery protocol for Gaussian blur, Gaussian
noise, brightness, motion blur, contrast, fog, and JPEG compression, each at
three fixed severities. No training, tuning, checkpoint selection, or Frozen
Test access occurred. Original Predify used the legacy `pvgg` path and its
original independent per-frame `t=0..10` internal inference.

| Model | Blur | Noise | Brightness | Motion blur | Contrast | Fog | JPEG | Overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen VGG16 | 0.757655 | 0.653359 | 0.194903 | 0.737504 | 0.365978 | 0.226278 | 0.557366 | 0.499006 |
| Original Predify | 0.716148 | 0.535335 | 0.168492 | 0.667430 | 0.334296 | 0.201913 | 0.464285 | 0.441128 |
| Current stateful | 0.665489 | 0.525299 | 0.191541 | 0.600955 | 0.354962 | 0.214037 | 0.370445 | 0.417533 |
| Learned zeroed | 0.437786 | 0.386891 | 0.139895 | 0.410381 | 0.273876 | 0.158904 | 0.238413 | 0.292307 |
| Learned recurrent error | 0.420092 | 0.387542 | 0.130466 | 0.387374 | 0.259959 | 0.148264 | 0.220550 | 0.279178 |

Learned improved over current-stateful by `33.1362%` overall and by
`24.5981%` to `40.9063%` at every corruption/severity combination. It improved
over zeroed by `4.4913%` overall and for blur, brightness, motion blur,
contrast, fog, and JPEG. Gaussian noise was the exception: its three-severity
mean was `0.1683%` worse than zeroed, severity 2/3 were `0.3367%`/`0.6793%`
worse, and drive 0011 was `1.5002%` worse. The learned recurrent transition
therefore did not show a corruption-level failure against current-stateful,
but the dynamic-error input contribution did not generalize to i.i.d. noise.

Tracked artifacts and server-only log:

```text
results/real_frame_robustness_benchmark_42a7029/
/tmp/predify-storage/experiments/real_frame_robustness_benchmark_42a7029.log
```

### Strict Prediction-error-driven Recurrent Transition

Training revision: `a721fcb`

Evaluation revision: `d39ffa3`

The learned ConvGRU transition no longer receives the current feedforward
feature. The causal update is `Fhat_t -> e_t=F_t-Fhat_t -> epsilon_t -> h_t`,
with historical state and top-down feedback retained. Only transition
parameters trained, and the loss uses the next real-frame feature target after
the current recurrence is complete. Epoch 5 minimized Val next-frame MSE at
`3.362763003`. The fixed validation used only drives 0011/0039 and Gaussian
blur sigma 3 under the 40/80/40 protocol.

| Condition | Disturbance normalized L2 | Recovery first 10 | Recovery last 10 |
| --- | ---: | ---: | ---: |
| Current stateful | 0.784296992 | 0.427156845 | 0.001205088 |
| Error-zeroed | 0.000000000 | 0.000000000 | 0.000000000 |
| Full error-driven | 0.522036018 | 0.391024056 | 0.060610952 |

Full improved `33.438988%` over current-stateful on disturbance deviation.
Full versus zeroed has no defined relative percentage because the denominator
is exactly zero; the absolute change is `-0.522036018`. This zero is a control
degeneracy, not successful adaptation: without current feedforward or error,
zeroed is input-blind after initialization, making its clean and corrupted
trajectories identical. The predeclared conclusion is therefore that
prediction error was not established as the main state-update driver. Frozen
Test drives 0051/0056 were not read.

Tracked artifacts and server-only checkpoint/logs:

```text
results/real_frame_error_driven_d39ffa3/
/tmp/predify-storage/experiments/real_frame_recurrent_error_train_a721fcb/best_recurrent_transition.pt
/tmp/predify-storage/experiments/real_frame_recurrent_error_train_a721fcb.log
/tmp/predify-storage/experiments/real_frame_error_driven_eval_d39ffa3.log
checkpoint sha256: 7b743640cae5901874090b3eed438d3ea697eafd8415ebbfcf3fefddcb6015d3
```

### Matched Observation-driven Recurrent Control

Code/training revision: `1765d15`

The invalid input-blind zeroed control was replaced as the core comparison by
a matched-capacity observation-driven ConvGRU transition. Two recurrent
transitions were trained from the same seed and with identical train drives
0005/0013/0014/0036, epochs, Adam learning rate `1e-4`, and next-frame
prediction objective. The only difference was the second transition input:
`epsilon_t` for error-driven versus current observation feature `F_t` for
observation-driven. Validation used only drives 0011/0039 with Gaussian blur
sigma 3 and the unchanged 40-clean/80-blur/40-recovery protocol. Frozen Test
drives 0051/0056 were not read.

| Condition | Next-frame MSE | Disturbance normalized L2 | Recovery first 10 | Recovery last 10 |
| --- | ---: | ---: | ---: | ---: |
| Current stateful | 2.671453703 | 0.784296992 | 0.427156845 | 0.001205088 |
| Observation-driven recurrent | 2.344646957 | 0.510357280 | 0.341280196 | 0.042804741 |
| Error-driven recurrent | 2.341551072 | 0.522036018 | 0.391024056 | 0.060610952 |

Error-driven improved over current-stateful by `33.438988%`, but was
`2.288345%` worse than observation-driven on the primary disturbance
representation-deviation metric. Next-frame MSE was nearly tied
(`2.341551072` error-driven versus `2.344646957` observation-driven), while
observation-driven had better disturbance and recovery representation
deviation. The matched-control conclusion is therefore:
`benefit_mainly_from_recurrent_temporal_modeling`. Prediction error did not
show independent value over a same-capacity observation-driven recurrent
input. The error-zeroed condition remains only a sanity check and again had
zero paired representation deviation because it is input-blind after
initialization.

Tracked artifacts and server-only checkpoints/logs:

```text
results/real_frame_matched_recurrent_1765d15/
/tmp/predify-storage/experiments/real_frame_matched_recurrent_train_1765d15/best_error_driven_recurrent.pt
/tmp/predify-storage/experiments/real_frame_matched_recurrent_train_1765d15/best_observation_driven_recurrent.pt
/tmp/predify-storage/experiments/real_frame_recurrent_error_train_1765d15.log
/tmp/predify-storage/experiments/real_frame_recurrent_error_eval_1765d15.log
error checkpoint sha256: 05166f66273b5545fce2cd160c4a0f730882df8e153ad7c5cf30e670b4cf8fca
observation checkpoint sha256: 531dbef98d71475c58a978abb3e9a3b1ce893c4fd7882276172a85efaf7717f3
```

### Error-driven Recurrent V2 with Dedicated Error Encoder

Code/training/evaluation revision: `94ad47e`

This final error-driven structure replaced the frozen-stage error input with a
dedicated signed-error encoder:
`z_t=P([relu(F_t-Fhat_t), relu(Fhat_t-F_t)])`, followed by
`h_t=T(h_(t-1), z_t, feedback)`. The current observation feature does not
enter the error-driven transition. The recurrent transition, signed-error
encoder, and temporal prediction decoders were trained jointly with a
short-window truncated-BPTT window of 4. VGG and the non-recurrent Predify body
remained frozen. Observation-driven recurrent used the same train drives,
epochs, optimizer, learning rate, next-frame objective, and recurrent
transition capacity, with `F_t` as its update input. Frozen Test 0051/0056 was
not read.

Training selected epoch 5 for both learned conditions. Clean Val next-frame MSE
was `2.611095539` for error-driven v2 and `3.301208898` for
observation-driven.

| Condition | Next-frame MSE | Disturbance normalized L2 | Recovery first 10 | Recovery last 10 |
| --- | ---: | ---: | ---: | ---: |
| Current stateful | 2.671453703 | 0.784296992 | 0.427156845 | 0.001205088 |
| Observation-driven recurrent | 2.355917884 | 0.660923713 | 0.514935964 | 0.189150613 |
| Error-driven recurrent v2 | 1.805955697 | 0.660516654 | 0.524020957 | 0.175822271 |

Error-driven v2 improved the primary disturbance representation deviation by
`15.782330%` versus current-stateful, but only by `0.061589%` versus the
matched observation-driven control. The next-frame MSE advantage is real, but
the robustness/adaptation metric is effectively tied with observation-driven.
Final decision: **NO-GO for independent prediction-error state-update value**;
the observed robustness benefit is still best described as recurrent temporal
modeling rather than a clearly superior error-driven mechanism.

Tracked artifacts and server-only checkpoints/logs:

```text
results/real_frame_error_driven_v2_94ad47e/
/tmp/predify-storage/experiments/real_frame_error_driven_v2_train_94ad47e/best_error_driven_recurrent_v2.pt
/tmp/predify-storage/experiments/real_frame_error_driven_v2_train_94ad47e/best_observation_driven_recurrent.pt
/tmp/predify-storage/experiments/real_frame_recurrent_error_train_94ad47e.log
/tmp/predify-storage/experiments/real_frame_recurrent_error_eval_94ad47e.log
error checkpoint sha256: ba0927e64aa6c6cf6acdc994fb4f01f6aef35fae8e587570b271757afae4b35a
observation checkpoint sha256: b84e910e1f80775436a2691a32cba7ff606be89720b7f23f0ea1af517e1fc555
```

## Earlier adjacent-pair result

This older experiment reset model state each batch and therefore tested
pair-level temporal supervision, not continuous video-state inheritance.

| Training condition | Val weighted | Val temporal loss | Val MAE | Val cosine |
| --- | ---: | ---: | ---: | ---: |
| Ordered adjacent pairs | 0.049902 | 0.220986 | 0.244000 | 0.717777 |
| Shuffled-future control | 0.040895 | 0.261809 | 0.264013 | -0.620335 |
