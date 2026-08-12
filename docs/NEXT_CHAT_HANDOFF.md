# Next Chat Handoff

Last updated: 2026-08-12

## Start Here

Repository:

```text
/home/lin/predify2021_targetflow
```

Branch and private remote:

```text
branch: targetflow-arch
remote: myprivate -> git@github.com:linzhihui-1023/preditive-coding.git
latest implementation and experiment revision: ae90a9f
```

At handoff time the Git worktree was clean. Use this environment:

```bash
conda activate predifyproject
cd /home/lin/predify2021_targetflow
```

Read this file first, then consult:

1. `docs/PROJECT_STATE.md`
2. `docs/DECISIONS.md`
3. `docs/EXPERIMENT_LOG.md`
4. `docs/README_FUTURE_FEATURE_KITTI.md`

## Settled Research Paradigm

The active system processes a real ordered video stream. The old route that
ran several timesteps on one static image is cancelled.

1. The first frame initializes five layer states.
2. Each later frame executes the model once.
3. State is inherited across frames and reset only at a real sequence or
   fixed-time-segment boundary.
4. Training and validation preserve frame order. Stream mode uses batch size
   1 and rejects shuffled pairs.
5. Stored recurrent state is detached. This is stateful forward recurrence
   with one-step gradients, not BPTT.

The primary research objective is no longer the 2-DoF motion proxy. It is:

- Predict the next-frame feature.
- Improve feature consistency in continuous video.
- Resist noise and blur.
- Support online adaptation after an environmental shift.

The 2-DoF `[forward displacement, yaw change]` task remains only as historical
diagnostic work and must not be presented as the primary result.

## Current Causal Feature Task

The existing motion head is preserved. The active independent head is:

```text
delta_hat = future_feature_predictor(F_t^5, H_t^5)
Fhat_(t+1|t)^5 = F_t^5 + delta_hat
```

Strict causal order for transition `t -> t+1`:

```text
read state completed at t-1
extract F_t
predict Fhat_(t+1|t)
observe detached F_(t+1)
compute feature loss and Target Flow residual
update state for the next transition
```

The predictor must not consume `I_(t+1)`, current-pair residual `r_t`, or
updated state `epsilon_t` before making the current prediction.

The two residual/error families must remain distinct:

```text
Target Flow residual: r_t^l = F_t^l - T_t^l
Target Flow state: epsilon_t^l <- r_t^l
Temporal Prediction Error: e_t^5 = F_t^5 - Fhat_(t|t-1)^5
Temporal Error state: E_t^5 <- e_t^5
```

The top Target Flow residual uses the next-frame target only after prediction.
Recursive target flow propagates the detached top target through all five
feedback levels. There is no independent `d_t` term in the implemented error
formula.

The strict top-layer Temporal Prediction Error recurrence is:

```text
e_t^5 = F_t^5 - Fhat_(t|t-1)^5
E_t^5 = alpha_e e_t^5 + (1-K_e alpha_e)E_(t-1)^5
```

Prediction for `t -> t+1` can use only the previously completed `E_t^5`.
The current `e_(t+1)^5` is computed after observing the future target, updates
`E_(t+1)^5`, and becomes available only to the next transition. Stored `e` and
`E` are detached. The first implementation is top-layer only.

## Formal Mechanism Configuration

Frozen-backbone feature experiments use:

```text
PREDIFY_TASK=future_feature
PREDIFY_FORMAL_SPLIT=1
PREDIFY_TARGET_FLOW_MODE=recursive
PREDIFY_TRAIN_BACKBONE=0
PREDIFY_TOP_TARGET_SOURCE=student_self
PREDIFY_TEMPORAL_TARGET_MODE=next_top
PREDIFY_LOCAL_LOSS_ERROR_SOURCE=instant
PREDIFY_TOP_VARIANCE_WEIGHT=0
PREDIFY_FEATURE_PREDICTION_WEIGHT=1.0
PREDIFY_ERROR_TS=0.1035
PREDIFY_ERROR_TAU=0.5
PREDIFY_ERROR_GAIN=1.0
PREDIFY_TEMPORAL_ERROR_TS=0.1035
PREDIFY_TEMPORAL_ERROR_TAU=0.5
PREDIFY_TEMPORAL_ERROR_GAIN=1.0
PREDIFY_SEED=0
```

The frozen VGG backbone does not need an EMA teacher for the top target.
Feedback decoders and the selected predictor train. Variance weight stays zero
because the frozen top feature has no trainable variance-loss path.

The first formal Temporal Error matrix contains only:

| Name | Predictor history input |
| --- | --- |
| Copy-current | Predictor bypassed; `Fhat=F_t` |
| Current-only | Zeros |
| Temporal Error | Previous completed top-layer `E_t^5` |

The legacy Latest, Two-tap, and Recursive modes remain available only for old
experiments and checkpoint compatibility. They are Target Flow residual
histories, not Temporal Prediction Error. The formal three-group runner rejects
them. Do not silently change the meaning of `recursive`.

## Available Data

```text
KITTI root: /home/lin/predify/kitti_raw
train: 2011_09_26/2011_09_26_drive_0005_sync, 153 accepted pairs
validation: 2011_09_26/2011_09_26_drive_0011_sync, 232 accepted pairs
camera: image_02
fixed interval: 0.1035 s
tolerance: 0.001 s
```

Both current drives form one uninterrupted accepted segment. Dataset code also
handles future timestamp gaps by splitting indices into contiguous segments
and resetting state between them.

The two drives are enough for preliminary rejection/diagnostic checks, not for
broad KITTI generalization claims. Their feature-change scales differ:

| Split | Element std of true delta | L2 P50 | L2 P90 | L2 P95 |
| --- | ---: | ---: | ---: | ---: |
| Train 0005 | 0.372769 | 114.175 | 148.687 | 151.860 |
| Validation 0011 | 0.245113 | 80.726 | 98.295 | 101.554 |

Disk at handoff:

```text
/home: about 169 GB free
/tmp filesystem: about 255 GB free
```

## Strict Temporal Error Matrix Result

Code and experiment revision: `3ffbff0`

The three formal cross-drive groups completed successfully with the same 153
ordered train pairs and 232 ordered validation pairs:

| Condition | Best epoch | Validation feature MSE | Feature cosine | Normalized feature error |
| --- | ---: | ---: | ---: | ---: |
| Copy-current | 1 | **0.060080099** | **0.919904691** | **0.375766109** |
| Current-only | 2 | 0.061797074 | 0.915814666 | 0.384975467 |
| Temporal Error | 2 | 0.061903913 | 0.915660890 | 0.385387186 |

Both gates failed:

```text
Current-only versus Copy-current: +2.85781% MSE
Temporal Error versus Current-only: +0.17289% MSE
```

The strict Temporal Error state changed the numerical result but did not
improve held-out prediction. This is not a broad rejection of Temporal Error,
because Current-only still fails the prerequisite Copy-current gate. Do not
tune `tau_e` or run seeds 1 and 2 to rescue this matrix.

All checkpoint metadata was audited against the histories and manifest. The
three checkpoints are best-validation checkpoints from the exact `3ffbff0`
revision; only epochs 1, 2, and 2 respectively were retained. Logs contain no
training error.

Artifacts:

```text
/tmp/predify-storage/experiments/seed0_future_feature_matrix_3ffbff0/
size: about 4.0 GB
```

## Historical Five-Condition Feature Matrix Result

Code revision: `94059be`

| Condition | Best epoch | Validation feature MSE |
| --- | ---: | ---: |
| Copy-current | 1 | 0.060080099 |
| Current-only 1x1 | 2 | 0.061797074 |
| Latest | 2 | 0.061849746 |
| Two-tap | 2 | 0.061899316 |
| Recursive | 2 | 0.061796151 |

Interpretation:

- Current-only was 2.858% worse than Copy-current on validation.
- Recursive and Current-only were a numerical tie.
- History cannot be judged yet because the basic Current-only predictor did
  not pass the Copy-current validation gate.
- These history inputs cover only the top residual state. They do not test
  `prediction_state_memory` or lower-layer feedback-decoder state. Do not
  generalize this result to the complete Predify state.

Artifacts:

```text
/tmp/predify-storage/experiments/seed0_future_feature_matrix_94059be/
size: about 6.6 GB
```

## Current-Only Delta Diagnostic

The diagnostic reloads the best checkpoint and recomputes every ordered pair.
It reports true and predicted delta moments, L2/RMS quantiles, norm ratio,
cosine, projection, Current-only MSE, and matched Copy-current MSE.

For the 1x1 best checkpoint:

| Split | Current-only MSE | Copy MSE | Difference | Norm ratio | Delta cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 0.133647 | 0.138957 | -3.821% | 0.1582 | 0.1949 |
| Validation | 0.061798 | 0.060081 | +2.858% | 0.3562 | 0.0683 |

The 1x1 predictor learned a small amount on the training drive, but predicted
delta magnitude was too small and direction alignment was weak, especially on
validation. This is a train-to-validation generalization reversal, not exact
collapse to Copy-current.

## 3x3 Sufficiency Diagnostic

Code revision: `1ae6b27`

Architecture:

```text
concat input 1024 channels
Conv 1024 -> 1024, kernel 3x3, padding 1
ReLU
Conv 1024 -> 512, kernel 1x1
```

The spatial size is unchanged. The 3x3-first predictor has about 9.96M
parameters versus 1.57M for 1x1, so this is not a parameter-matched ablation.

Results:

```text
best epoch: 1
best validation MSE: 0.060632
Copy-current validation MSE: 0.060081
3x3 remains 0.917% worse than Copy-current
epoch-10 train MSE: 0.093191
epoch-10 validation MSE: 0.095457
```

Critical scientific interpretation:

- Do not say that 3x3 learned better spatial motion because its MSE is closer
  to Copy-current than 1x1.
- At the best validation checkpoint, 3x3 predicts a smaller delta than 1x1:
  validation delta RMS `0.0307` versus `0.0612`.
- Its validation norm ratio is smaller: `0.1789` versus `0.3562`.
- Its validation delta cosine is worse: `0.0269` versus `0.0683`.
- The direct interpretation is that the best 3x3 checkpoint stays closer to
  zero correction and therefore closer to Copy-current.
- Late epochs show that the larger spatial predictor can fit the train drive
  much more strongly, but validation degrades sharply. This is evidence of
  additional training capacity and overfitting, not held-out spatial-motion
  prediction improvement.

Artifacts:

```text
/tmp/predify-storage/experiments/seed0_future_feature_predictor_sufficiency_1ae6b27/
size: about 1.4 GB
```

## Diagnostic Safety Guards

Commit `53f7d0d` hardened
`predify2021/mce_scores/diagnose_kitti_future_feature_delta.py`.

It now rejects a checkpoint unless:

```text
config.prediction_task == future_feature
config.future_feature_history_mode == none
configured predictor kernel matches the first predictor weight shape
```

Legacy checkpoints without a kernel config field pass only when the first
weight is verified as 1x1. A Recursive checkpoint is rejected before model
construction, preventing silent evaluation with history forcibly removed.

The Temporal Error and controlled-corruption implementation at `ae90a9f`
passed the full local suite: 53 tests. This includes formula, no-current-future-
error leakage, next-frame state use, detach, deterministic absolute-frame
corruption, and disjoint raw-frame split tests. The exact `1ae6b27` GitHub
Actions unit test run was previously reported successful; CI status for
`ae90a9f` was not checked during this handoff.

## Current Decision And Next Step

Paused:

- Do not tune Target Flow `tau` or Temporal Error `tau_e`.
- Do not rerun Latest/Two-tap/Recursive yet.
- Do not run seeds 1 and 2 yet.
- Do not make broad noise, blur, robustness, or online-adaptation claims yet.

Failed gates at seed 0:

```text
Current-only validation MSE < Copy-current validation MSE       FAILED
Temporal Error validation MSE < Current-only validation MSE     FAILED
```

The completed same-drive controlled-corruption run used drive 0011 with raw
train frames 0--138, gap 139--158, and validation 159--204. It runs independent
step-bias, ramp-bias, and i.i.d.-noise trajectories, resetting before every
clean and corrupted stream. Bias and noise are never mixed. Primary outputs
are per-frame signed `delta L=L_corrupted-L_clean`, signed/absolute excess
integrals, Recovery Time, `RMS(F_t)`, `RMS(E_t)`, predictor input-weight scale,
and `P(F_t,E_t)-P(F_t,0)`. Raw Peak Error and raw AUEC are secondary. This is a
short mechanistic trace, not a substitute for the failed cross-drive gate or
broad robustness evidence.

Clean best-checkpoint MSE was `0.071884151` for Copy-current, `0.074492955`
for Current-only, and `0.074685545` for Temporal Error. Current-only was
3.62918% worse than Copy-current and Temporal Error was 0.25853% worse than
Current-only. Temporal Error versus Current-only signed excess AUEC changed by
+0.22189% for step bias, +0.22728% for ramp bias, and -1.00328% for i.i.d.
noise. The state was materially present and used, but zeroing its history
improved every phase-mean MSE. This does not motivate `tau_e` tuning.

Artifacts:

```text
/tmp/predify-storage/experiments/seed0_same_drive_controlled_corruption_ae90a9f/
size: about 4.0 GB
```

For the research path, the next discussion should choose a clean
way to address predictor generalization. Leading options are:

1. Download more training drives while keeping genuinely unseen drives for
   validation.
2. Design a capacity/regularization diagnostic that separates spatial context
   from the 6.3x parameter increase and from near-zero prediction.
3. Only after Current-only clears Copy-current, rerun the matched history
   matrix and then consider broader Predify state inputs.

Do not change or tune either error recurrence based on this matrix. The strict
Temporal Error implementation is now present and causally tested; the limiting
factor remains a next-feature predictor that does not generalize past
Copy-current on the held-out drive.

## Reproduction Commands

Run the three-condition 1x1 Temporal Error matrix:

```bash
scripts/run_kitti_seed0_future_feature_matrix.sh
```

Run the clean same-drive training plus independent controlled-corruption
evaluation:

```bash
scripts/run_kitti_seed0_same_drive_controlled_corruption.sh
```

Run only the 3x3-first Current-only diagnostic:

```bash
scripts/run_kitti_seed0_future_feature_predictor_sufficiency.sh 3
```

Run checkpoint delta diagnostics by following the environment example in:

```text
docs/README_FUTURE_FEATURE_KITTI.md
```

All formal runners use `env -i` and explicitly record critical variables. Do
not rely on inherited shell variables, especially legacy compatibility flags.

## Automatic GitHub Publication

After a code change is complete and verified, commit it and push
`targetflow-arch` to `myprivate` automatically. After a formal training run is
complete, audit the histories, checkpoints, logs, and manifest; update the
tracked result documents; commit those records; and push again automatically.
Do not wait for a separate push request. Checkpoint files and other large
artifacts stay on the server, but the pushed documentation must record their
server path and exact training revision. Use only ordinary fast-forward pushes;
never force-push unless the user explicitly requests history rewriting.
