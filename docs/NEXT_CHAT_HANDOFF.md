# Next Chat Handoff

Last updated: 2026-08-14

## Start Here

Repository:

```text
/home/lin/predify2021_selective_adaptation
```

The frozen local baseline is a separate worktree:

```text
/home/lin/predify2021_targetflow -> targetflow-arch @ 4cc7a21
```

Branch and private remote:

```text
active branch: predify-selective-adaptation-v2
frozen baseline tag: predify-temporal-v1
frozen baseline commit: 4cc7a21280881813dbb972415a74dc400843fcc6
remote: myprivate -> git@github.com:linzhihui-1023/preditive-coding.git
latest Gate 3 evaluation revision: 80c4aee
latest Stage-4 prediction training revision: c887e94
latest unified robustness evaluator revision: 42a7029
latest strict error-driven training/evaluation revisions: a721fcb / d39ffa3
latest matched recurrent-control revision: 1765d15
```

All selective-online-adaptation work must stay on
`predify-selective-adaptation-v2`. The annotated `predify-temporal-v1` tag is
the immutable temporal-prediction baseline and must not be moved. The research
details below document that frozen baseline and remain available if the
prediction direction is resumed later.

At handoff time the Git worktree was clean. Use this environment:

```bash
conda activate predifyproject
cd /home/lin/predify2021_selective_adaptation
```

Read this file first, then consult:

1. `docs/PROJECT_STATE.md`
2. `docs/DECISIONS.md`
3. `docs/EXPERIMENT_LOG.md`
4. `docs/README_FUTURE_FEATURE_KITTI.md`

## Latest Matched Recurrent-control Result

Revision `1765d15` added a fair observation-driven recurrent control for the
strict error-driven transition. It trained two same-capacity ConvGRU
transitions with identical train drives 0005/0013/0014/0036, seed, epoch
count, Adam optimizer, learning rate `1e-4`, and next-frame prediction
objective. The only difference was the recurrent input:

```text
Error-driven:      h_t=T(h_(t-1),epsilon_t,feedback)
Observation-driven h_t=T(h_(t-1),F_t,feedback)
```

Val used only drives 0011/0039, Gaussian blur sigma 3, and the unchanged
40-clean/80-blur/40-recovery protocol. Frozen Test 0051/0056 was not read.

| Condition | Next-frame MSE | Disturbance normalized L2 | Recovery first 10 | Recovery last 10 |
| --- | ---: | ---: | ---: | ---: |
| Current stateful | 2.671453703 | 0.784296992 | 0.427156845 | 0.001205088 |
| Observation-driven recurrent | 2.344646957 | 0.510357280 | 0.341280196 | 0.042804741 |
| Error-driven recurrent | 2.341551072 | 0.522036018 | 0.391024056 | 0.060610952 |

Error-driven improved over current by `33.438988%`, but was `2.288345%` worse
than observation-driven on the primary disturbance representation-deviation
metric. Next-frame MSE was effectively tied. Conclusion:
`benefit_mainly_from_recurrent_temporal_modeling`; prediction error did not
show independent value over the matched observation-driven recurrent input.
The zeroed condition remains only a sanity check because it is input-blind
after initialization. Results are in
`results/real_frame_matched_recurrent_1765d15/`. Checkpoints remain server-only:

```text
/tmp/predify-storage/experiments/real_frame_matched_recurrent_train_1765d15/best_error_driven_recurrent.pt
/tmp/predify-storage/experiments/real_frame_matched_recurrent_train_1765d15/best_observation_driven_recurrent.pt
```

## Latest Unified Robustness Result

Evaluator revision `42a7029` completed the Val-only 0011/0039 benchmark with
five models, seven corruptions, and three severities under the paired
40-clean/80-corruption/40-recovery protocol. Original Predify is the legacy
`pvgg` path with independent per-frame `t=0..10` inference. Learned and zeroed
use the same frozen epoch-1 `c9fec52` checkpoint. No training, tuning, or
Frozen Test read occurred.

Overall `mean_normalized_representation_deviation` was `0.499006257` (Frozen
VGG16), `0.441128454` (Original Predify), `0.417532706` (current-stateful),
`0.292306507` (learned zeroed), and `0.279178111` (learned recurrent error).
Learned beat current-stateful for every corruption/severity. It beat zeroed
for six corruptions, but was `0.1683%` worse on the Gaussian-noise mean; do not
claim a universal dynamic-error-input benefit. Results are in
`results/real_frame_robustness_benchmark_42a7029/`.

## Strict Error-driven Result

Revision `a721fcb` removed the current feedforward feature from the learned
transition and moved training supervision to `F_(t+1)`. The epoch-5 checkpoint
was evaluated at `d39ffa3` on Val 0011/0039 with blur sigma 3 only. Disturbance
normalized L2 was `0.784296992` (current), `0.522036018` (full), and `0`
(error-zeroed). Full improved `33.438988%` over current, but Full versus zeroed
is undefined because zeroed is input-blind and produces identical paired clean
and corrupted trajectories. The mechanism conclusion is `NOT ESTABLISHED`;
do not expand this branch or touch Frozen Test. Results are in
`results/real_frame_error_driven_d39ffa3/`.

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

Future-feature prediction now has an explicit stage boundary
`s_pred in {3,4,5}`. It is independent of the Predify Target Flow top:

```text
Target Flow top target: T_TF = F_(t+1)^5
Future prediction target: T_future = F_(t+1)^s_pred
Prediction memory: M_(t-1) = detach(F_(t-1)^s_pred)
```

`PREDIFY_FUTURE_FEATURE_STAGE` selects `s_pred`. Changing it must not change
the Stage-5 Target Flow target or feedback recursion. The deferred future
providers extract both requested stages in one future-frame VGG pass, after
the current prediction has been made.

The existing motion head is preserved. The original independent head remains:

```text
delta_hat = future_feature_predictor(F_t^5, H_t^5)
Fhat_(t+1|t)^5 = F_t^5 + delta_hat
```

After the ordered local-motion gate, the active tested form is:

```text
M_t = local_match(F_(t-1)^5, F_t^5)
B_(t+1)^5 = W(F_t^5, M_t)
Rhat_(t+1)^5 = future_feature_predictor(B_(t+1)^5, H_t^5)
Fhat_(t+1|t)^5 = B_(t+1)^5 + Rhat_(t+1)^5
```

`M_t` uses only detached historical/current features. The deterministic
`historical_warp` form returns `B` directly; `historical_warp_residual` learns
only the post-warp residual. `copy_current` still bypasses both paths.

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
Target Flow state: epsilon_t^(TF,l) <- r_t^l
Stage-4 Prediction Error: e_t^4 = Fhat_(t|t-1)^4 - F_t^4
Dynamic Prediction Error state: epsilon_t^(pred,4) <- e_t^4
```

The top Target Flow residual uses the next-frame target only after prediction.
Recursive target flow propagates the detached top target through all five
feedback levels. There is no independent `d_t` term in the implemented error
formula.

The active Stage-4 Dynamic Prediction Error recurrence is:

```text
e_t^4 = Fhat_(t|t-1)^4 - F_t^4
epsilon_t^4 = alpha_e e_t^4 + (1-K_e alpha_e)epsilon_(t-1)^4
```

For phase 1, prediction never consumes this state. The current `e_(t+1)^4` is
computed only after observing `F_(t+1)^4`, updates `epsilon_(t+1)^4` exactly
once, and is detached before the next real-video transition. Drive and
fixed-time-segment boundaries reset it. There are no same-frame iterations,
online updates, new fusion modules, or backbone changes.

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

The feature-learnability diagnostic at `1605f29` passed the expanded full
local suite: 57 tests. Its exact GitHub Actions status was not checked here.

The causal warp-residual implementation at `79de554` passed the full local
suite: 69 tests plus 3 subtests. A real `cuda:0` smoke verified historical
motion on the second stream step, exact equality of future/delta/residual MSE,
and nonzero residual-predictor gradients. CI status for this revision was not
checked locally.

## Current Decision And Next Step

Gate 3 completed at revision `80c4aee` with both networks frozen. It locked
Gate 2's detector to the eight-frame mean error cosine and reproduced all
2,400 corrupted detector rows exactly. Physical forward/yaw MAE came from the
formal Group A seed-0 epoch-6 2-DoF checkpoint and was compared with an
independently reset clean trajectory on the same frames and OXTS targets.

On held-out drive 0011, forward `S-D` Pearson was `-0.001291`, Spearman was
`0.057782`, and score AUROC for positive degradation was `0.500651`. Yaw MAE
improved in 79 of 80 windows; its nominal AUROC has only one positive and must
not be interpreted as a successful trigger. Persistent had much higher `S`
than Shuffled but nearly identical forward degradation.

Decision: do not build a controller that treats Gate 2's score alone as task
degradation. The next research step must either validate a trigger against a
stronger downstream task or redesign the utility signal. Keep both old models
frozen and continue only in `/home/lin/predify2021_selective_adaptation`.

Versioned audit artifacts:

```text
results/gate3_persistence_task_degradation_80c4aee/
```

The strict matched-marginal persistence gate completed at revision `99b7e21`
on `predify-selective-adaptation-v2`. Persistent and shuffled conditions used
the same 11x11 Gaussian blur, exactly 20 frames at each sigma in
`{0.75, 1.5, 2.25, 3.0}`, identical 100% disturbance occupancy, the same raw
frames, and the same frozen checkpoint. Four replicates counterbalanced every
absolute frame across sigma values. Only temporal ordering differed.

Drive 0005 selected higher `cos(e_t,e_(t-1))`; frozen drive-0011 evaluation
reached `0.9725` AUROC over nonoverlapping eight-frame windows. Excluding the
onset window gave `0.972222`, and individual held-out replicate AUROCs ranged
from `0.95` to `1.00`. This passes the user-defined persistence gate, so the
next work may design a small selective-online-adaptation mechanism. Do not
modify or move `predify-temporal-v1`.

The evidence is limited to two drives. Windows reuse video content and are
diagnostic units, not independent drive samples. Full lightweight audit data:

```text
results/matched_blur_persistence_99b7e21/
```

The latest inference-only go/no-go reused the strict Temporal Error checkpoint
from revision `3ffbff0` and evaluated at revision `a28fed5`. No optimizer was
created and parameter versions remained unchanged. Both drives ran clean,
persistent-blur, and i.i.d.-Gaussian-noise trajectories with exactly 40 clean,
80 disturbed, and 30 recovery prediction transitions.

Drive 0005 selected lower raw `||e_t||` as the positive direction and best
single statistic. Frozen evaluation on drive 0011 reached pooled AUROC
`0.959609` (`0.937031` versus clean, `0.982188` versus i.i.d. noise). Excluding
the first eight disturbance frames produced AUROC `0.971065`; eight-frame block
means produced `0.985`. Under the user-defined threshold this is
`go_promising`.

Do not interpret this as evidence that `e` detects temporal persistence. Blur
reduced error norm, the winning statistic was instantaneous, and the i.i.d.
negative used a different corruption type. The next gate is a matched-marginal
persistent-blur versus temporally randomized-blur experiment, with the same
checkpoint, frozen score direction, clean control, and no network updates.
Do not build a controller before that gate.

Versioned audit artifacts:

```text
results/prediction_error_separability_a28fed5/
```

The ordered local-motion diagnostic at `06ec8e7` completed without training.
Future-selected local matching reduced stage-5 Copy MSE on both drives by
12--19% at `h=1` and 29--45% at `h=3` for 1x1 matching; 3x3 matching retained
smaller but positive reductions. Stage-4 reductions were larger.

The causal historical 3x3 warp technically passed the stage-5 Copy gate on
both drives. At `r=1`, diagnostic gain was 6.837% on 0005 and only 0.221% on
0011; at `r=2`, it was 6.652% and 0.062%. Stage-4 causal gains were 21--42%.

Versioned audit artifacts:

```text
results/vgg_local_motion_06ec8e7/
```

The forward-only VGG feature-task learnability matrix completed at revision
`1605f29` on 148 drive-0005 and 227 drive-0011 forecast origins. It evaluated
stage 3/4/5 at `h=1,2,3,5` with Copy-current, causal raw feature velocity, and
a future-selected one-cell translation oracle. The full 24-entry matrix is in
`docs/EXPERIMENT_LOG.md`.

Raw constant velocity lost to Copy-current in every one of 4,500 per-frame
rows, and adjacent one-step feature-delta cosine was negative at every stage
on both drives. The oracle reduced aggregate Copy-current MSE by only
0--6.726% on drive 0011 under this small global-shift search; drive 0005 had a
different pattern and larger short-horizon stage-3/4 reductions. Do not turn
this into a general unlearnability claim: no predictor was trained, and the
oracle tests only one global integer translation.

Versioned audit artifacts:

```text
results/vgg_feature_learnability_1605f29/
```

They contain the exact `summary.json`, all 4,500 per-frame CSV rows,
provenance, and SHA-256 hashes.

The permitted next step was implemented and formally run at revision
`79de554` as Copy-current, deterministic causal warp, and learned post-warp
residual. Best-checkpoint replay gave:

| Condition | Train MSE | Val MSE | Val versus Copy |
| --- | ---: | ---: | ---: |
| Copy-current | 0.138954983 | 0.060080099 | 0.000% |
| Historical warp | 0.129720510 | 0.059948462 | +0.219% |
| Warp residual, epoch 3 | 0.124021935 | 0.061526919 | -2.408% |

Warp residual improved over warp-only by 4.393% on train but degraded it by
2.633% on validation. At epoch 10, train MSE reached `0.112814438` while val
rose to `0.067354957`. Keep deterministic warp as a causal baseline, but do
not claim that this residual predictor generalizes across drives. The complete
audit copy is:

```text
results/seed0_warp_residual_matrix_79de554/
```

Paused:

- Do not tune Target Flow `tau` or Temporal Error `tau_e`.
- Do not rerun Latest/Two-tap/Recursive yet.
- Do not run seeds 1 and 2 yet.
- Do not make broad noise, blur, robustness, or online-adaptation claims yet.

Failed gates at seed 0:

```text
Current-only validation MSE < Copy-current validation MSE       FAILED
Temporal Error validation MSE < Current-only validation MSE     FAILED
Warp-residual validation MSE < historical-warp validation MSE   FAILED
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
versioned audit copy: results/seed0_same_drive_controlled_corruption_ae90a9f/
```

The versioned audit copy includes `summary.json`, all nine per-frame CSV
traces, provenance, and hashes. Use it to re-audit reported AUEC, phase curves,
recovery, `e -> E` timing, and state utilization. Checkpoints remain
server-only.

For the research path, the next discussion should choose a clean way to
address post-warp residual generalization. Leading options are:

1. Download more training drives while keeping genuinely unseen drives for
   validation.
2. Predeclare a capacity/regularization diagnostic for the residual head while
   retaining warp-only as a separate checkpoint-free baseline.
3. Only after warp-residual clears warp-only on held-out video, add Temporal
   Error or broader Predify state inputs to that predictor.

Do not change or tune either error recurrence based on this matrix. The strict
Temporal Error implementation is now present and causally tested; the limiting
factor remains a learned correction that does not generalize past the causal
warp or Copy-current on the held-out drive.

## Stage-5 Two-Frame Temporal Fusion Result

The strict three-condition matrix was implemented and trained at revision
`10191a6d2d04df325ffbc7959f9e633f0af43d3b`. It compared only Copy-current,
Current-only, and two-frame residual Temporal Fusion, with the pretrained VGG
and Target Flow feedback decoders frozen. Drive 0005 supplied 153 training
transitions and held-out drive 0011 supplied 232 transitions.

Best-checkpoint held-out metrics were:

| Condition | Feature MSE | Cosine | Normalized error |
| --- | ---: | ---: | ---: |
| Copy-current | 0.06008010 | 0.91990469 | 0.37576611 |
| Current-only | 0.06179707 | 0.91581467 | 0.38497547 |
| Temporal Fusion | 0.06192617 | 0.91600015 | 0.38513246 |

Temporal Fusion therefore did not pass the Copy-current gate in this matrix.
It was active: its held-out fusion-residual RMS was `0.0464521`, and fusion was
applied to all held-out samples except sample 0 immediately after the drive
reset. Both learned methods improved on drive 0005 while worsening over epochs
on drive 0011, so the observed limitation is held-out generalization rather
than failure to optimize the training objective. Do not generalize this one
seed, one stage, one architecture result to all forms of temporal modeling.

Auditable outputs are tracked under
`results/temporal_fusion_matrix_10191a6/`. The 4.0 GB server artifact remains at
`/tmp/predify-storage/experiments/seed0_temporal_fusion_matrix_10191a6`.

## Aligned-History Temporal Fusion Result

The single aligned-history follow-up was implemented and trained at revision
`95aa61d911dead2a89e0f749a74a0cef3329e42f`. It uses local radius-1,
patch-size-3 matching to place `F_(t-1)` directly in `F_t` coordinates before
the residual Temporal Fusion module. It does not use the historical-warp
future splat. The protocol remained seed 0, train drive 0005, held-out drive
0011, 10 epochs, frozen VGG and feedback decoders, and the previous
Copy-current metrics as fixed gates.

The best checkpoint was epoch 2. Held-out MSE was `0.06191402` against the
required `< 0.06008010`; cosine was `0.91520128` against the required
`> 0.91990469`; normalized error was `0.38581802` against the required
`< 0.37576611`. All three checks failed. Alignment was active on `231/232`
held-out samples, with only the first post-reset sample lacking history.

Small auditable artifacts are tracked under
`results/aligned_temporal_fusion_95aa61d/`. The 1.41 GB best checkpoint and
logs remain at
`/tmp/predify-storage/experiments/seed0_aligned_temporal_fusion_95aa61d`.

## Aligned Temporal-Difference Result

The single aligned temporal-difference experiment was implemented and trained
at revision `750d11f5dab703cd089518eede05037e740d9c2e`. It computes
`D_t = F_t - align(F_(t-1), F_t)` with current-coordinate local matching and
predicts `Fhat_(t+1) = F_t + P([F_t, D_t])`. It does not create or train a
Temporal Fusion module, and it never uses the historical future splat. The
formal protocol remained seed 0, train drive 0005, held-out drive 0011, 10
epochs, radius 1, patch size 3, and frozen VGG and feedback decoders.

The best checkpoint was epoch 2. Held-out MSE was `0.06180378` against the
required `< 0.06008010`; cosine was `0.91604762` against the required
`> 0.91990469`; normalized error was `0.38434362` against the required
`< 0.37576611`. All three checks failed. The independent replay recorded zero
Temporal Fusion applications and an exact maximum base-to-`F_t` difference of
`0.0` across all 385 rows.

Small auditable artifacts are tracked under
`results/aligned_temporal_difference_750d11f/`. The 1.41 GB best checkpoint
and logs remain at
`/tmp/predify-storage/experiments/seed0_aligned_temporal_difference_750d11f`.

## Reproduction Commands

Run Gate 3 with both detector and 2-DoF task model frozen:

```bash
scripts/run_kitti_gate3_persistence_task_degradation.sh
```

Run the strict matched-marginal persistence diagnostic on the active branch:

```bash
scripts/run_kitti_matched_blur_persistence.sh
```

Run the inference-only prediction-error separability diagnostic:

```bash
scripts/run_kitti_prediction_error_separability.sh
```

Run the formal causal warp-residual matrix and per-frame replay:

```bash
scripts/run_kitti_seed0_warp_residual_matrix.sh
```

Run the strict three-condition stage-5 temporal-fusion matrix:

```bash
scripts/run_kitti_seed0_temporal_fusion_matrix.sh
```

Run only the aligned-history Temporal Fusion follow-up:

```bash
scripts/run_kitti_seed0_aligned_temporal_fusion.sh
```

Run only the aligned temporal-difference predictor follow-up:

```bash
scripts/run_kitti_seed0_aligned_temporal_difference.sh
```

Run the same single aligned temporal-difference condition at Stage 4:

```bash
scripts/run_kitti_seed0_stage4_aligned_temporal_difference.sh
```

The Stage-4 replay computes Copy-current on the same Stage-4 frame stream.
Judge MSE, cosine, and normalized error against that same-stage reference;
never compare Stage-4 and Stage-5 absolute MSE as if they shared a feature
space. Radius 1 means one prediction-stage feature cell and is deliberately
not auto-rescaled across stages.

The formal Stage-4 run completed at revision `fdc4743`. Best-checkpoint replay
selected epoch 1. On held-out drive 0011, aligned temporal difference had MSE
`0.74586091` versus Stage-4 Copy-current `0.74053836`, cosine `0.85806100`
versus `0.86544334`, and normalized error `0.49728997` versus `0.48848719`.
All three same-stage checks failed. Train replay improved MSE by `6.815535%`,
while the held-out change was `-0.718741%`. Treat this narrowly as a
cross-drive generalization failure for the tested predictor, not as a
cross-stage MSE comparison or evidence that Stage-4 lacks motion signal.

Versioned audit artifacts:

```text
results/stage4_aligned_temporal_difference_fdc4743/
```

Server-only checkpoint and logs:

```text
/tmp/predify-storage/experiments/seed0_stage4_aligned_temporal_difference_fdc4743/
```

The next diagnostic is deliberately one-time and same-drive. Drive 0005 raw
frames are fixed as Train `0--91`, Val `92--122`, and Test `123--153`.
Transitions crossing boundaries at starts 91 and 122 are excluded, leaving
`91/30/30` ordered transitions and no shared raw frames. Stage 4, aligned
temporal difference, radius 1, patch size 3, seed 0, and ten epochs remain
fixed. Val selects the checkpoint; Test is replayed once afterward. Do not
create further same-drive variants based on its result.

```bash
scripts/run_kitti_seed0_stage4_same_drive_diagnostic.sh
```

This one-time diagnostic completed at revision `69b24b1` and selected epoch
10 using only the middle 20% Val segment. On the once-read last 20% Test
segment, Stage-4 MSE was `1.51209933` versus same-segment Copy `1.67944006`, a
`9.964079%` improvement. Normalized error improved from `0.76012611` to
`0.72217820`; cosine worsened from `0.70890513` to `0.69367177`. Therefore the
model shows same-drive forward generalization for the trained Euclidean error,
but does not pass all three metrics. Do not run further same-drive variants.

Artifacts:

```text
results/stage4_same_drive_60_20_20_69b24b1/
/tmp/predify-storage/experiments/seed0_stage4_same_drive_60_20_20_69b24b1/
```

The main Stage-4 experiment completed at revision `c887e94` using the
predeclared protocol below without changing the aligned temporal-difference
architecture:

```text
Train: 0005 + 0013 + 0014 + 0036  (1411 transitions)
Val:   0011 + 0039                (626 transitions)
Test:  0051 + 0056                (728 transitions, frozen)
```

Training receives no Test-drive environment variable and checkpoint selection
uses Val only. The evaluator first replays Train/Val, atomically creates a
checkpoint-specific frozen-Test receipt, and then reads 0051/0056 exactly once.
Drive 0051 has a timestamp discontinuity and is replayed as two independently
reset segments of 56 and 379 transitions. Never rerun the evaluator for a
checkpoint whose receipt already exists.

```bash
scripts/run_kitti_seed0_stage4_multidrive.sh
```

Val selected epoch 7. The once-read frozen Test aggregate passed all three
same-stage Copy-current checks: MSE `0.77194043` versus `0.89291654`
(`+13.548423%`), cosine `0.83322664` versus `0.82239107`, and normalized error
`0.54544640` versus `0.58067843`. Drive 0051 improved MSE by `11.697142%` and
drive 0056 by `15.702122%`; both passed all three checks independently. Treat
this as one-seed evidence that training-drive diversity resolves the earlier
single-train-drive generalization failure for this fixed architecture. It is
not a multi-seed or cross-stage result.

Artifacts:

```text
results/stage4_multidrive_c887e94/
/tmp/predify-storage/experiments/seed0_stage4_multidrive_c887e94/
```

The frozen Test receipt is completed and bound to checkpoint SHA-256
`8a29c40fc71f0d54a2d21306a9444407d25865a01262713d05f522b495ab7754`.
Do not evaluate 0051/0056 again or use their result for model selection.

The active next experiment is the inference-only Stage-4 Dynamic Prediction
Error state diagnostic. It reuses the frozen epoch-7 `c887e94` checkpoint,
keeps `future_feature_history_mode=aligned_difference`, and therefore does not
feed `epsilon_t^4` into the predictor. Val drives 0011/0039 receive matched
persistent and shuffled severity sequences for blur, i.i.d. Gaussian noise,
and an RGB-bias domain-shift proxy. The same severity multiset, occupancy, raw
frames, and checkpoint are used; only temporal ordering differs.

```bash
scripts/run_kitti_stage4_dynamic_error_state_phase1.sh
```

Primary fixed scores are `RMS(e_t)`, scalar `EMA(RMS(e_t))`, and
`RMS(epsilon_t)`. A matched tensor EMA is also recorded only to audit the fact
that the formal `K=1` dynamics are mathematically identical to a tensor EMA
with `alpha=0.207`. No score or direction is selected from the data, no
network is updated, and frozen Test drives are not read.

This phase-1 diagnostic completed at evaluation revision `eaf29a0`. The
corrected offline analysis is revision `3071391`; it used only the saved CSVs
and did not replay the network. Direction-independent aggregate AUROC was
`0.64897569` for instantaneous RMS, `0.65151042` for scalar EMA, and
`0.50489583` for dynamic-state RMS. Dynamic AUROC was `0.51312500` for blur,
`0.50687500` for i.i.d. noise, and `0.50343750` for RGB bias. Formula error and
dynamic-versus-matched-tensor-EMA difference were both exactly zero.

Decision: `no_go_dynamic_state_not_better_than_controls`. Do not start phase 2
online adaptation from this state. The result is a mechanistic null for the
fixed signed tensor recurrence, not a rejection of all possible persistence
features.

Artifacts:

```text
results/stage4_dynamic_error_state_phase1_eaf29a0/
/tmp/predify-storage/experiments/stage4_dynamic_error_state_phase1_eaf29a0*
```

## Real-frame predictive-coding recurrence (causal review accepted)

The Stage-4 future-prediction route is now stopped. Do not add another future
predictor, fusion module, online update, or Stage-4 training variant. The new
default `PVGG16TargetFlow` task is `real_frame_pc`, which uses the original
five PVGG16 PCoder decoders and performs one update per observed video frame.

For layer `i`, the complete frame-`t-1` snapshot is taken before any layer is
advanced:

```text
R_(t-1)^i              previous representation
p_(t-1)^i              previous same-layer prediction
p_(t-1)^(i+1)          previous higher-layer prediction / feedback
epsilon_(t-1)^i        previous dynamic Target Flow error
```

The current update is the original Predify location and ordering:

```text
g_(t-1)^i = grad_R MSE(
    P_i(R_(t-1)^i),
    stopgrad(p_(t-1)^i + epsilon_(t-1)^i)
)

s_i = K_i / C_sqrt_i

R_t^i = beta_i ff_t^i
      + lambda_i p_(t-1)^(i+1)
      + (1 - beta_i - lambda_i) R_(t-1)^i
      - alpha_i s_i g_(t-1)^i
```

The highest layer has no feedback term. On a segment's first frame,
`R_1^i = ff_1^i`. Current lower-layer representations remain the targets of
their current higher-layer predictions, exactly as in the original hook
schedule. `K_i` is the number of elements in the layer prediction and
`C_sqrt_i` is calibrated once with the original `PCoderN.compute_C_sqrt`
ten-perturbation procedure; it is a non-parameter buffer and is not cleared by
segment reset. Once the single update is complete:

```text
p_t^i       = P_i(R_t^i)
r_t^i       = target_t^i - p_t^i
epsilon_t^i = 0.207 r_t^i + 0.793 epsilon_(t-1)^i
```

The current `R_t`, `p_t`, `r_t`, and `epsilon_t` are detached before storage.
The entire causal chain is therefore:

```text
Frame 1 -> r_1 -> epsilon_1
                    |
                    v
Frame 2 state update -> r_2 -> epsilon_2
```

`real_frame_pc` rejects `next_x`, `future_x`, all future-target providers, and
all target overrides before executing a layer. It creates neither
`temporal_predictor` nor `future_feature_predictor`; all parameters have
`requires_grad=False`, no local training loss is exposed, and reset clears the
segment state and frame counter. Unit tests audit the exact second-frame update
equation, historical feedback identity, dynamic formula, one update per layer,
detachment, reset, future-input rejection, and parameter immutability.

The causal chain has been accepted. The subsequent scaling-only revision
aligned the second-frame error correction numerically with the original
`PCoderN` `K/C_sqrt` update; it did not run a formal KITTI experiment.

### Phase-1 mechanism result

Real-frame Predictive Coding Phase 1 ran from clean evaluation revision
`1d25877` on Val drives 0011 and 0039 only. Each drive used raw frames 0--159:
40 clean, 80 persistent Gaussian-blur (`kernel=11`, `sigma=3.0`), then 40
clean recovery frames. Every condition had a synchronized condition-matched
clean counterfactual. No parameter trained, no optimizer or future predictor
existed, and Frozen Test drives 0051/0056 were not read.

Mean corrupted-to-clean representation normalized L2 across five layers and
both drives was:

| Condition | Disturbance | Recovery | Recovery last 10 |
| --- | ---: | ---: | ---: |
| Feedforward | 0.890284632 | 0.000000000 | 0.000000000 |
| PC-no-error | 0.784450073 | 0.126158581 | 0.001216764 |
| PC-dynamic-error | 0.784296992 | 0.126198921 | 0.001205088 |

Dynamic error improved the primary disturbance metric over PC-no-error by only
`0.019514%`, despite being directionally lower on both drives. This is below
the predeclared `5%` practical threshold. Recovery did not show persistent
drift: dynamic distance fell from `0.427156845` over its first 10 recovery
frames to `0.001205088` over its final 10. Decision: **NO-GO** for a meaningful
incremental benefit from the tested dynamic error correction. This is limited
to the fixed recurrence, coefficients, and single blur trajectory.

Versioned artifacts:

```text
results/real_frame_pc_phase1_1d25877/
```

Server-only log:

```text
/tmp/predify-storage/experiments/real_frame_pc_phase1_1d25877.log
```

The strict four-condition decomposition then reran the identical protocol at
evaluation revision `9a3d9bd`, adding only Representation-memory-only
(`lambda=0`, `alpha=0`). The original Feedforward, PC-no-error, and
PC-dynamic-error summaries were exactly unchanged.

| Condition | Disturbance normalized L2 |
| --- | ---: |
| A Feedforward | 0.890284632 |
| B Representation-memory only | 0.803649702 |
| C Representation + Feedback | 0.784450073 |
| D Representation + Feedback + Dynamic error | 0.784296992 |

Adjacent relative improvements were A->B `9.731150%`, B->C `2.389054%`, and
C->D `0.019514%`. The full A->C gain was `11.887722%`. By absolute reduction,
representation memory contributed `81.8588%` and feedback `18.1412%` of that
gain. Therefore the earlier approximately 11.9% result came mainly from
representation memory, with a smaller feedback contribution; dynamic error
remained negligible. The dynamic-error decision remains **NO-GO**.

Four-condition artifacts:

```text
results/real_frame_pc_phase1_9a3d9bd/
```

Server-only log:

```text
/tmp/predify-storage/experiments/real_frame_pc_phase1_9a3d9bd.log
```

### Learned recurrent-error result

The current learned mechanism revision is `c9fec52`. It preserves the original
Predify feedforward and previous-frame top-down feedback base update, disables
the fixed `dynamic error -> gradient projection -> pc_error_multiplier`
correction only in learned mode, and adds one 1x1 ConvGRU transition per layer.
The dynamic recurrence remains
`epsilon_t=0.207e_t+0.793epsilon_(t-1)`. Cross-frame states detach and reset at
drive/segment boundaries. Only 7,328,064 transition parameters train.

Training used ordered clean streams from 0005/0013/0014/0036 and one fixed
five-epoch `lr=1e-4` run. Val clean prediction MSE on 0011/0039 selected epoch
1 (`0.893984978`). Formal controlled-blur validation then compared only:

| Condition | Disturbance normalized L2 | Recovery first 10 | Recovery last 10 |
| --- | ---: | ---: | ---: |
| Current stateful | 0.784296992 | 0.427156845 | 0.001205088 |
| Learned recurrent-error | 0.501548966 | 0.280066796 | 0.004235435 |
| Same checkpoint, error input zeroed | 0.521438669 | 0.301188880 | 0.045429500 |

Learned improved by `36.051143%` over current and by `3.814390%` over zeroed.
Both Val drives improved in both comparisons. Interpret the first gap as the
value of the learned recurrent transition and the second as the isolated
incremental dynamic-error contribution. Do not assign the full 36.1% to
dynamic error. Frozen Test drives 0051/0056 were not read.

Artifacts:

```text
results/real_frame_recurrent_error_c9fec52/
/tmp/predify-storage/experiments/real_frame_recurrent_error_train_c9fec52/
checkpoint sha256: 26c5333da95a6b754af6036f9d16a5dd394673400e6f9456c0ebc0b27e714cf4
```

The single Frozen Test access for this checkpoint is now complete. Evaluator
revision `8915f75` reused the identical three conditions and
40-clean/80-blur/40-recovery protocol on 0051/0056, with no training, tuning,
or checkpoint update.

| Condition | Disturbance normalized L2 | Recovery first 10 | Recovery last 10 |
| --- | ---: | ---: | ---: |
| Current stateful | 0.819635281 | 0.446208276 | 0.001090620 |
| Learned recurrent-error | 0.536746322 | 0.305349457 | 0.001978208 |
| Same checkpoint, error input zeroed | 0.548573083 | 0.310442773 | 0.040968571 |

Learned improved over current/zeroed by `37.165928%/0.568743%` on 0051 and
`31.765203%/3.624299%` on 0056. Aggregate improvements were
`34.514005%/2.155914%`. Both drives passed both comparisons, so the
predeclared decision is `PASS_MAIN_MECHANISM`. Per-layer directions are not
uniform; retain the conclusion at the predeclared drive-level aggregate.

Frozen Test artifacts and receipt:

```text
results/real_frame_recurrent_error_frozen_test_c9fec52/
/tmp/predify-storage/experiments/real_frame_recurrent_error_train_c9fec52/best_recurrent_transition_real_frame_frozen_test_receipt.json
```

Do not rerun 0051/0056 for the `c9fec52` checkpoint. The receipt is completed
and bound to checkpoint SHA-256
`26c5333da95a6b754af6036f9d16a5dd394673400e6f9456c0ebc0b27e714cf4`.

Cross-corruption validation then ran at evaluator revision `54e893e` on Val
0011/0039 only. The c9fec52 epoch-1 checkpoint, three conditions, and 40/80/40
protocol stayed fixed. Gaussian noise used pre-normalization `std=0.08` and
brightness shift used uniform pre-normalization RGB `+0.15`; each had one
fixed severity.

| Corruption | Current | Learned | Zeroed | Learned vs current | Learned vs zeroed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gaussian noise | 0.650661257 | 0.490607851 | 0.487297715 | 24.598576% | -0.679284% |
| Brightness shift | 0.272876398 | 0.184116804 | 0.197603509 | 32.527399% | 6.825134% |

The learned recurrent transition improved over current for both corruptions.
Dynamic-error input improved over zeroed for brightness on both drives, but
not for Gaussian noise, where the two drives had opposite directions. Frozen
Test was not read. Artifacts:

```text
results/real_frame_recurrent_error_cross_corruption_54e893e/
/tmp/predify-storage/experiments/real_frame_recurrent_error_cross_corruption_54e893e.log
```

## Frozen historical runners

The commands below belong to completed predictor research. They remain for
reproducibility and are not the next experiment on this branch.

Run the three-condition 1x1 Temporal Error matrix:

```bash
scripts/run_kitti_seed0_future_feature_matrix.sh
```

Run the clean same-drive training plus independent controlled-corruption
evaluation:

```bash
scripts/run_kitti_seed0_same_drive_controlled_corruption.sh
```

Run the forward-only VGG feature-task learnability matrix:

```bash
scripts/run_kitti_vgg_feature_learnability_matrix.sh
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
`predify-selective-adaptation-v2` to `myprivate` automatically. After a formal training run is
complete, audit the histories, checkpoints, logs, and manifest; update the
tracked result documents; commit those records; and push again automatically.
Do not wait for a separate push request. Checkpoint files and other large
artifacts stay on the server, but the pushed documentation must record their
server path and exact training revision. Use only ordinary fast-forward pushes;
never force-push unless the user explicitly requests history rewriting.
