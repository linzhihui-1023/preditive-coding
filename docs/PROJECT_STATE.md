# Project State

Last updated: 2026-08-13

## Active research direction

Stage-4 next-frame feature prediction is closed to further extension. Its
completed results remain versioned below, but it is no longer the active model
or the next experiment.

The active model is the inference-only `PVGG16TargetFlow` real-frame
predictive-coding recurrence:

1. A segment starts with `reset`; its first frame initializes five PCoder
   representations from current feedforward drives.
2. Each observed frame advances every PCoder exactly once. There are no
   same-frame iterations.
3. Frame `t` inherits only detached representations, predictions, and dynamic
   Target Flow errors completed on frame `t-1`.
4. Previous higher-layer predictions enter the original feedback terms.
   Previous same-layer prediction errors are projected through frozen PCoder
   decoders into the original error-correction terms.
5. Current hierarchical residuals and dynamic errors are formed only after the
   current representations and predictions exist, then detached for frame
   `t+1`.
6. The recurrence accepts no future frame or future target, creates no future
   predictor, exposes no training loss, and performs no parameter or online
   update.

The accepted causal state chain is:

```text
Frame 1 -> r_1 -> epsilon_1 -> Frame 2 update -> r_2 -> epsilon_2
```

The earlier same-image repeated-timestep route and all Stage-4 predictor
variants are inactive. The `K/C_sqrt` alignment revision itself is code and
unit-test work only; no formal KITTI experiment was run from it.

For the formal dynamic Target Flow state, `Ts=0.1035`, `tau=0.5`, and `K=1`,
so `epsilon_t = 0.207 r_t + 0.793 epsilon_(t-1)`.

Phase 1 has now completed on Val drives 0011/0039 with a fixed
clean-40/blur-80/recovery-40 trajectory. Dynamic error changed the disturbance
normalized representation distance from `0.784450073` (PC-no-error) to
`0.784296992`, an improvement of only `0.019514%`; therefore the result is
`NO-GO` under the predeclared 5% practical threshold. The dynamic stream did
recover without lasting drift, reaching `0.001205088` over the last ten clean
recovery frames. Frozen Test drives were not read. Auditable outputs are in
`results/real_frame_pc_phase1_1d25877/`.

## Current implementation

- `predify2021/model_factory/pvgg16_targetflow.py`
  - Defaults to `task=real_frame_pc` and implements five real-frame PCoder
    stages using the original PVGG16 feedforward boundaries and decoders.
  - Snapshots every layer's previous state before processing the current frame,
    preventing current-frame higher-layer predictions from leaking into the
    historical feedback slots.
  - Uses the original update coefficients
    `beta=(0.2,0.4,0.4,0.5,0.6)`,
    `lambda=(0.05,0.1,0.1,0.1,0)`, and
    `alpha=(0.01,0.01,0.01,0.01,0.01)`.
  - Applies the original `PCoderN` error normalization `K/C_sqrt` to the
    pseudo-target gradient. Each layer's `C_sqrt` is calibrated with the
    original ten-perturbation procedure, persists across segment resets, and
    remains a non-trainable buffer.
  - Creates the original Stage-1-to-image decoder in addition to the existing
    Stage-2-through-Stage-5 decoders. It does not create the temporal motion
    head, future-feature predictor, or temporal fusion module in real-frame
    mode.
  - Freezes every parameter and stores detached representation, prediction,
    instantaneous residual, and dynamic-error memories per layer.
  - Rejects every future-frame/target argument before any layer update and
    resets all state at segment boundaries.
  - Retains the older motion/future-feature tasks only to reproduce completed
    versioned experiments; they are not the active direction.
  - Historical implementation notes follow.
  - Defaults to recursive target flow: the detached future top feature is
    propagated through `T5 -> T4 -> T3 -> T2 -> T1`.
  - Separates the cross-frame error state (`instant`, `ema`, or `two_tap`) from the
    local-loss error source.
  - Defaults local optimization to instantaneous error so memory controls have
    identical current-frame local losses and gradient coefficients.
  - Keeps the original 2-output motion `temporal_predictor` and adds an
    independent full-map `future_feature_predictor` selected through
    `PREDIFY_TASK`. `PREDIFY_FUTURE_FEATURE_STAGE` selects Stage 3, 4, or 5.
  - Keeps the Predify Target Flow top fixed at future Stage 5 while resolving
    a separate configured-stage target for future prediction. Both targets are
    deferred until after the current prediction.
  - Predicts a residual feature map with
    `Fhat_(t+1|t) = F_t + P(F_t, H_t)`.
  - Also supports causal historical warp
    `W(F_t,M(F_(t-1),F_t))` and a warp-residual form that predicts only the
    remaining feature residual. Motion estimation uses detached prior/current
    features and prediction still precedes future-target resolution.
  - Supports an explicit first-layer spatial kernel of 1 or 3 for the future
    predictor. The default and formal history matrix remain 1x1.
  - Uses the same future predictor for `none`, `latest`, `two_tap`, and
    `recursive` history conditions; only `H_t` changes. `copy_current` bypasses
    the predictor as a non-learned baseline.
  - Keeps per-layer error and prediction memories across `step_frame` calls.
    Spatial history uses a detached previous prediction-stage feature rather
    than a misleadingly named top-feature memory.
  - Detaches every stored error and prediction state. Execution is stateful
    forward recurrence with one-step gradients, not BPTT.
  - Clears memories at sequence boundaries through `reset`.
  - Resolves the next-frame target through a deferred provider only after the
    future prediction is complete, then updates Target Flow residuals and
    history for the following transition.
  - Maintains causal top-layer latest, two-tap, and recursive history snapshots
    independently of the configured local-loss error state.
  - Adds an independent prediction-stage Temporal Prediction Error state for the
    future-feature task:
    the active selective-adaptation branch now uses
    `e_t^4 = Fhat_(t|t-1)^4 - F_t^4` and
    `epsilon_t^4 = alpha_e e_t^4 + (1-K_e alpha_e)epsilon_(t-1)^4`.
    It updates once after observation, detaches, and resets at real sequence
    boundaries.
    The predictor can select this state through
    `PREDIFY_FEATURE_HISTORY_MODE=temporal_error`; the older `recursive` mode
    remains Target Flow residual memory.
- `predify2021/model_factory/targetflow/core.py`
  - Defines target-flow state, dynamic-error integration, local losses, and
    gradient diagnostics.
- `predify2021/model_factory/targetflow/spatial_motion.py`
  - Implements shared local patch matching, integer feature translation, and
    discrete forward splatting with collision averaging and Copy-filled holes.
- `predify2021/mce_scores/kitti_pairs.py`
  - Loads adjacent KITTI frames, timestamps, and 2-DoF longitudinal-yaw
    targets `[forward displacement m, yaw change rad]`.
  - Splits fixed-dt-valid sample indices into contiguous segments whenever a
    raw timestamp transition is rejected.
  - Provides an explicit same-drive raw-frame split: first 60% train, 20-frame
    gap, then a contiguous 20% validation range. It verifies that the complete
    train and validation raw-frame sets are disjoint.
- `predify2021/mce_scores/train_kitti_targetflow_adjacent_pairs.py`
  - Defaults to stream mode.
  - Resets once per drive or sequence, then calls `step_frame` once per frame.
  - Treats each contiguous fixed-dt segment as a separate sequence, resetting
    student, teacher, and variance history before the next segment.
  - Requires `PREDIFY_BATCHSIZE=1` and disables shuffled pairs in stream mode.
  - Supports deterministic runs through `PREDIFY_SEED`.
  - Supports the `PREDIFY_RESET_EACH_FRAME=1` control without adding repeated
    model executions.
  - Supports a reset/no-history current-top duplicate control that copies only
    detached `F_student(I_t)` into the top prediction-context slot. Under the
    frozen backbone this equals `F_teacher(I_t)`.
  - Freezes pretrained VGG forward stages by default. Feedback decoders and the
    task-selected predictor train; `PREDIFY_TRAIN_BACKBONE=1` is an explicit
    adaptation ablation.
  - Records separate Target Flow error parameters and Temporal Prediction Error
    parameters so future `tau_e` changes do not alter Target Flow dynamics.
  - Supports clean same-drive checkpoint training through
    `PREDIFY_SAME_DRIVE_SPLIT=1`; controlled corruption remains outside this
    training loop.
  - Selects motion checkpoints by validation temporal loss and future-feature
    checkpoints by validation feature MSE.
  - Reports feature MSE, cosine, normalized feature error, the equivalent delta
    and residual MSE, prediction-base MSE, matched same-stage Copy-current
    metrics, and within-stage absolute/relative improvements.
  - Records the prediction stage, Target Flow top stage, channels, separate
    target equations, and feature-cell radius units in checkpoint config.
  - Computes optional collapse prevention across a sequence-local temporal
    window of pooled top features instead of across the batch dimension.
  - Disables that window completely when its weight is zero and clears it on
    every reset-each-frame step when it is enabled.
  - Detaches stored history so only the current frame receives variance
    gradients; the default window is 16 frames.
  - Estimates per-horizon motion mean and standard deviation from training
    segments only, optimizes standardized MSE, and reports forward MAE in
    metres and yaw MAE in radians separately.
- `predify2021/mce_scores/calculate_kitti_targetflow_pair_smoke.py`
  - Provides a sequential stream smoke check and resets when frame indices
    cross a filtered fixed-dt discontinuity.
- `scripts/run_kitti_seed0_five_group_matrix.sh`
  - Is the canonical A-E seed-0 runner for the frozen-backbone matrix.
  - Starts from `env -i`, explicitly sets every mechanism and optimization
    variable, forces the legacy current-teacher variable to zero, and records
    the exact Git revision in each result.
- `scripts/run_kitti_seed0_future_feature_matrix.sh`
  - Runs the first formal three-group matrix: Copy-current, Current-only, and
    top-layer Temporal Error. It rejects Latest, Two-tap, and Target-Flow
    Recursive groups for this matrix.
  - Stores outputs under `/tmp/predify-storage` by default and retains only the
    best validation checkpoint per group.
- `scripts/run_kitti_seed0_warp_residual_matrix.sh`
  - Runs Copy-current, causal historical warp, and learned post-warp residual
    from one clean revision, then replays every best checkpoint into a common
    per-frame evaluator.
- `predify2021/mce_scores/evaluate_kitti_warp_residual_matrix.py`
  - Rejects mismatched checkpoint forms or revisions and records Copy/base/final
    MSE, residual scale, frame provenance, warp coverage, and collisions.
- `scripts/run_kitti_seed0_stage4_aligned_temporal_difference.sh`
  - Runs the one-condition Stage-4 aligned temporal-difference follow-up from
    a clean revision while retaining the Stage-5 Target Flow top.
  - Fixes radius 1 in Stage-4 feature cells and evaluates only against
    same-stage Copy-current over the same frames.
- `scripts/run_kitti_seed0_stage4_same_drive_diagnostic.sh`
  - Runs the sole 0005 chronological 60/20/20 Stage-4 diagnostic without
    shuffled transitions or shared raw frames.
  - Selects only on the middle Val segment and evaluates the final Test
    segment once after best-checkpoint selection.
- `scripts/run_kitti_seed0_stage4_multidrive.sh`
  - Trains the unchanged Stage-4 aligned temporal-difference model on drives
    0005/0013/0014/0036 and selects checkpoints on 0011/0039 only.
  - Keeps frozen Test drives 0051/0056 out of the training process and invokes
    their one-time evaluator only after best-Val selection.
- `predify2021/mce_scores/evaluate_kitti_stage4_multidrive.py`
  - Enforces the exact Train/Val checkpoint contract and refuses checkpoints
    that contain Test drives.
  - Atomically claims one checkpoint-specific frozen-Test read, resets at
    every fixed-time segment, and records aggregate and per-drive same-stage
    Copy-current gates.
- `predify2021/mce_scores/evaluate_kitti_aligned_temporal_difference.py`
  - Enforces prediction-stage/Target-Flow-stage checkpoint separation and
    replays both train and held-out drives into one per-frame schema.
  - Records Stage, feature shape, MSE/cosine/normalized error, and signed and
    relative improvements against same-stage Copy-current.
- `predify2021/mce_scores/evaluate_kitti_stage4_dynamic_error_state.py`
  - Keeps the `c887e94` Stage-4 predictor frozen and validates that the dynamic
    error state is not a predictor input.
  - Compares fixed instantaneous RMS, scalar EMA-envelope, and dynamic-state
    RMS scores under matched persistent/shuffled blur, noise, and RGB bias on
    Val drives only; it also audits exact tensor-EMA equivalence at `K=1`.
- `scripts/run_kitti_stage4_dynamic_error_state_phase1.sh`
  - Runs the inference-only phase-1 diagnostic without reading frozen Test
    drives or creating an optimizer.
- `predify2021/mce_scores/diagnose_kitti_future_feature_delta.py`
  - Re-evaluates a Current-only best checkpoint on ordered train and validation
    pairs and reports true/predicted delta scale, L2/RMS quantiles, norm ratio,
    delta cosine, projection, and matched Copy-current MSE.
- `scripts/run_kitti_seed0_future_feature_predictor_sufficiency.sh`
  - Runs a Current-only seed-0 predictor with an explicitly selected 1x1 or
    3x3 first convolution while keeping all history and dynamic-error controls
    fixed.
- `predify2021/mce_scores/kitti_controlled_corruption.py`
  - Applies deterministic absolute-frame corruption after resize/crop in RGB
    pixel space and before ImageNet normalization.
  - Keeps fixed RGB bias and per-frame i.i.d. Gaussian noise as separate
    corruption types.
  - Defines independent step-bias, ramp-bias, and noise-negative-control
    trajectories and computes signed, absolute, and positive-part excess
    integrals plus recovery metrics.
- `predify2021/mce_scores/evaluate_kitti_same_drive_controlled_corruption.py`
  - Resets before every paired clean/corrupted trajectory and saves per-frame
    `e`, `E`, `L`, and signed `delta L` traces as JSONL/CSV plus one recovery
    PNG per trajectory.
  - Reports `RMS(F_t)`, `RMS(E_t)`, predictor input-weight scale, and the
    counterfactual history contribution `P(F_t,E_t)-P(F_t,0)`.
  - Rejects cross-drive or mismatched checkpoints through saved split and
    architecture checks.
- `scripts/run_kitti_seed0_same_drive_controlled_corruption.sh`
  - Trains the same three clean groups on the same-drive split, then invokes
    the independent controlled-corruption evaluator.
- `.github/workflows/tests.yml`
  - Runs the unit tests on pushes to `targetflow-arch` and pull requests.
  - Pins the public base `predify` dependency by commit and uses CPU PyTorch.

## Future-feature implementation status

The one-time 0005 same-drive 60/20/20 diagnostic completed at revision
`69b24b1`. Its Val-selected epoch-10 checkpoint improved untouched Test-segment
Stage-4 MSE by `9.964079%` and normalized error, while cosine worsened. This is
same-drive forward generalization for the Euclidean objective, not an
all-metric pass. No further same-drive variants are allowed; the next result
must use the fixed multi-drive Train/Val/Test protocol.

That multi-drive protocol is fixed before training: Train is
0005/0013/0014/0036, Val is 0011/0039, and frozen Test is 0051/0056. The
training process cannot receive Test drive names, all loaders remain ordered,
and only Val selects the epoch. A checkpoint-specific atomic receipt prevents
0051/0056 from being evaluated more than once. Their metrics are final
reporting only and cannot feed back into model or parameter selection.

The formal revision `c887e94` run selected epoch 7 on Val. Frozen Test MSE was
`0.77194043` versus Stage-4 Copy `0.89291654`, a `13.548423%` improvement;
cosine improved by `0.01083557` and normalized error improved by `0.03523203`.
Both 0051 and 0056 passed all three checks independently. Lightweight audit
artifacts and the completed one-time Test receipt are under
`results/stage4_multidrive_c887e94/`. The frozen Test is now consumed for this
checkpoint and must not be replayed or used for follow-up selection.

The independent Stage-4 Dynamic Prediction Error phase-1 diagnostic completed
at runtime revision `eaf29a0` with corrected CSV-only analysis at `3071391`.
It compared persistent versus shuffled severity organization across blur,
i.i.d. noise, and RGB bias on Val drives 0011/0039. Aggregate direction-
independent AUROC was `0.64897569` for instantaneous RMS, `0.65151042` for
scalar EMA, and `0.50489583` for dynamic-state RMS. The dynamic state was near
chance for each corruption and exactly matched tensor EMA at `K=1`.

This fails the phase-1 state-utility gate. No predictor feedback, optimizer,
online adaptation, or frozen-Test read occurred, and selective online
adaptation must not proceed from this state. Artifacts are under
`results/stage4_dynamic_error_state_phase1_eaf29a0/`.

The Stage-4 aligned temporal-difference run completed at revision `fdc4743`.
Its epoch-1 checkpoint improved fixed replay MSE by `6.815535%` on train drive
0005 but changed held-out drive-0011 MSE by `-0.718741%` relative to Stage-4
Copy-current. Cosine and normalized error also worsened, so all three
same-stage checks failed. Audited artifacts are under
`results/stage4_aligned_temporal_difference_fdc4743/`. This is a narrow
cross-drive generalization result and must not be turned into an absolute
Stage-4 versus Stage-5 MSE ranking.

The causal future-feature path is implemented and unit tested. An initial
two-pair train/two-pair validation GPU smoke run completed on `cuda:0` with
pretrained feedback weights, backward optimization, feature metrics, and
best-checkpoint selection. The smoke run verified exact equality of future
feature MSE and residual-delta MSE; its numerical accuracy is not an experiment
result. The formal frozen-backbone matrix uses a detached student-self target,
because an EMA teacher produces the same top feature in this regime.

The first inference-only prediction-error separability gate completed at
revision `a28fed5` using the existing strict Temporal Error checkpoint. With
drive 0005 selecting score direction and statistic, low raw `||e_t||` separated
persistent blur from frame-matched clean and i.i.d.-noise controls on drive
0011 at AUROC `0.959609`. The first-eight-frame-excluded AUROC was `0.971065`,
so the result is not carried by the shift onset. All 900 per-frame records are
versioned under `results/prediction_error_separability_a28fed5/`.

This passes the user-defined first go threshold but not a persistence-specific
mechanism gate. Blur and Gaussian noise have different marginals, and the
selected raw norm does not require temporal history. The next experiment must
use a time-randomized blur negative matched to the persistent blur in marginal
severity before any controller is added.

That matched-marginal gate completed on the new
`predify-selective-adaptation-v2` branch at revision `99b7e21`. Persistent and
shuffled blur had exactly the same kernel, sigma multiset, counts, occupancy,
raw frames, and checkpoint; four replicates counterbalanced sigma exposure at
every absolute frame. Drive 0005 selected higher error cosine, and frozen
drive-0011 evaluation reached `0.9725` AUROC over nonoverlapping eight-frame
windows (`0.972222` after excluding the onset window). Results are versioned
under `results/matched_blur_persistence_99b7e21/`.

This passes the user-defined persistence go/no-go and allows selective online
adaptation work to begin on the new branch. It remains a two-drive mechanism
diagnostic: correlated windows must not be represented as independent drives
or used for a naive population confidence interval.

Gate 3 froze Gate 2's eight-frame cosine detector and compared it with signed
clean-relative physical 2-DoF MAE degradation at revision `80c4aee`. All Gate
2 corrupted detector traces reproduced exactly. On held-out drive 0011,
forward Pearson/Spearman were `-0.001291/0.057782` and score AUROC for positive
degradation was `0.500651`. Yaw improved in 79 of 80 windows. The high
persistence score therefore does not indicate task degradation for the tested
motion checkpoint, and it must not directly drive an adaptation controller.

This is not a universal downstream-task rejection. The existing 2-DoF proxy
has weak cross-drive performance and especially poor yaw behavior. Results
are versioned under
`results/gate3_persistence_task_degradation_80c4aee/`.

A forward-only task-learnability matrix completed at revision `1605f29` for
VGG stage 3/4/5 and horizons 1/2/3/5 on both existing drives. Raw causal
constant-velocity feature extrapolation was worse than Copy-current for all
4,500 per-frame matrix rows; mean adjacent one-step delta cosine was negative
at every stage on both drives. The future-selected one-cell translation oracle
reduced aggregate Copy-current MSE by 0--6.726% on drive 0011, while drive 0005
showed a different pattern with larger short-horizon stage-3/4 reductions.
These are simple-baseline and spatial-alignment diagnostics, not a learned-
predictor result or proof that the task is unlearnable. The summary and full
per-frame CSV are versioned under `results/vgg_feature_learnability_1605f29/`.

The ordered local-motion diagnostic completed at revision `06ec8e7`. Local
future-selected matching reduced stage-5 Copy MSE on both drives by 12--19% at
`h=1` and 29--45% at `h=3` for pointwise matching; structured 3x3 matching also
reduced it, though less at `h=1`. Causal historical 3x3 warp passed the formal
stage-5 Copy gate on both drives, but the held-out gain was only 0.221% for
`r=1` and 0.062% for `r=2`. Stage-4 causal gains were much larger at 21--42%.
This permits a warp-plus-residual predictor implementation while requiring the
weak stage-5 margin to remain explicit. Exact summary and per-frame outputs
are versioned under `results/vgg_local_motion_06ec8e7/`.

The resulting stage-5 formal matrix completed at revision `79de554`.
Deterministic causal warp improved MSE relative to Copy-current by 6.646% on
drive 0005 and 0.219% on held-out drive 0011. The learned post-warp residual
improved train MSE by 10.747% relative to Copy, but its best epoch-3 validation
MSE was `0.061526919`: 2.408% worse than Copy and 2.633% worse than warp-only.
Its final train/validation MSE diverged to `0.112814438/0.067354957`. Thus
historical transport remains a causal but weak held-out baseline, while this
residual head fails the cross-drive generalization gate. Exact summary, all
1,155 per-frame rows, and the complete 60-row training curve are versioned
under `results/seed0_warp_residual_matrix_79de554/`.

The seed-0 five-condition feature matrix completed at revision `94059be`.
Copy-current achieved validation feature MSE `0.060080099`. Current-only was
2.858% worse at `0.061797074`, so the predictor did not learn a useful future
change on validation. Recursive reached `0.061796151`, only 0.00149% better
than current-only and therefore a numerical tie, not evidence for useful
history. Latest and two-tap reached `0.061849746` and `0.061899316`.

All learned conditions selected epoch 2 and then overfit while training MSE
continued to improve. The follow-up best-checkpoint diagnostic showed that the
1x1 Current-only delta norm is only 15.8% of the true train delta and its delta
cosine is 0.195 on train and 0.068 on validation. It improves train MSE by
3.82% but is 2.86% worse than Copy-current on validation.

A 3x3-first Current-only sufficiency run completed at revision `1ae6b27`. Its
best epoch-1 validation MSE is 0.060632, still 0.917% worse than Copy-current,
while its online train MSE reaches 0.093191 by epoch 10 and validation degrades
to 0.095457. The best 3x3 checkpoint predicts a smaller correction than 1x1
and has worse validation delta cosine (0.027 versus 0.068); its MSE being closer
to Copy-current is therefore not evidence that it learned spatial motion more
accurately. The late-epoch train gain shows additional fitting capacity while
the validation trajectory shows severe cross-drive overfitting. The 3x3-first
model is not parameter matched: it has 9.96M predictor parameters versus 1.57M
for the 1x1 model.

The first strict top-layer Temporal Prediction Error matrix completed at
revision `3ffbff0`. Copy-current again reached `0.060080099`, Current-only
reached `0.061797074`, and Temporal Error reached `0.061903913`. Current-only
was 2.85781% worse than Copy-current, and Temporal Error was 0.17289% worse
than Current-only. Both formal gates therefore failed. The independent
Temporal Error state had a measurable numerical effect in this seed, but no
held-out benefit. Because the basic predictor still fails the Copy-current
gate, this does not establish that prediction-error memory is generally
unhelpful.

Seeds 1 and 2, `tau_e` tuning, and additional history reruns remain paused.
Current-only must first beat Copy-current on held-out video. The next
cross-drive decision is about data coverage and predictor
regularization/capacity, not either error time constant. The current data do
not separate spatial architecture effects from parameter count or
conservative near-zero prediction. Broad noise, blur, online adaptation, and
broader state claims remain downstream experiments.

The same-drive controlled-corruption run completed at revision `ae90a9f` on
drive 0011, with raw frames 0--138 for training, a 20-frame raw gap, and 45
ordered validation transitions from frames 159--204. Copy-current reached
`0.071884151`, Current-only `0.074492955`, and Temporal Error `0.074685545`.
Thus Current-only was 3.62918% worse than Copy-current and Temporal Error was
0.25853% worse than Current-only; both gates failed again.

Temporal Error versus Current-only signed excess AUEC changed by +0.22189% for
step bias, +0.22728% for ramp bias, and -1.00328% for the i.i.d.-noise negative
control. These small mixed differences do not support selective adaptation to
systematic bias. The state was not ignored: `RMS(E)/RMS(F)` was about
0.104--0.143, history and feature input-weight RMS were comparable, and the
history contribution was about 0.166--0.227 of predicted-delta RMS. Yet the
same-checkpoint zero-history ablation improved phase-mean MSE everywhere. This
is a one-seed, short same-drive mechanism diagnostic, not robustness or
generalization evidence.

The original evaluator `summary.json` and all nine 45-row per-frame CSV traces
are versioned under
`results/seed0_same_drive_controlled_corruption_ae90a9f/`, together with
provenance and SHA-256 hashes. The approximately 4 GB checkpoints remain only
on the experiment server.

## Available data

- Training drive: `2011_09_26_drive_0005_sync`, 153 accepted adjacent pairs.
- Validation drive: `2011_09_26_drive_0011_sync`, 232 accepted adjacent pairs.
- Camera: `image_02`.
- Fixed frame interval: 0.1035 seconds with 0.001-second tolerance.

These two drives are enough for controlled mechanism validation, but not for a
final claim about broad KITTI generalization.

Both downloaded drives currently form one uninterrupted fixed-dt segment, so
the newly fixed segment-boundary bug did not change their old sample order. The
fix is required before adding drives that contain rejected timestamp steps.

## Formal seed-0 control result

The corrected frozen-backbone A-E matrix completed at Git revision `6c446d9`.
All conditions used explicit isolated environments, disjoint drives,
training-only target normalization, recursive target flow, instantaneous local
loss, variance weight zero, and best-checkpoint selection by validation
standardized 2-DoF longitudinal-yaw MSE.

| Group | Condition | Best epoch | Standardized MSE | Forward MAE (m) | Yaw MAE (rad) |
| --- | --- | ---: | ---: | ---: | ---: |
| A | Inherit, recursive EMA | 6 | 11.045343 | 0.473039 | 0.008406 |
| B | Reset, no extra top | 5 | 11.443965 | 0.465796 | 0.005455 |
| C | Reset, current-top duplicate | 6 | **11.018217** | **0.458364** | 0.006219 |
| D | Inherit, instant error | 5 | 11.146288 | 0.477386 | **0.003696** |
| E | Inherit, two-tap error | 5 | 11.128227 | 0.477661 | 0.004047 |

A is 0.246% worse than C in the primary clean history comparison, so seed 0
does not show a benefit from inherited temporal history. C is 3.720% better
than B, confirming that the duplicated current top representation materially
affects the no-history baseline. A is 0.906% better than D and 0.745% better
than E in standardized MSE, but these small differences are not yet stable
evidence for recursive dynamic error and are not consistent across the two
physical component MAEs.

The full result, comparison definitions, CI identifiers, and server artifact
path are recorded in `docs/EXPERIMENT_LOG.md`.

## Cheap-diagnostic decision

The required constant, static, and best-checkpoint diagnostics completed at
revision `5b64cad`. The clean no-history C condition reaches standardized
joint MSE 11.018217 versus 11.266117 for the train-mean constant, only a 2.20%
reduction. A reaches 11.045345, a 1.96% reduction. The Frozen VGG plus current-
frame-only MLP reaches 11.376018 at epoch 1 and is 0.98% worse than the
constant; later epochs overfit strongly.

All learned conditions degrade yaw MSE relative to the constant. The constant
yaw MSE is `2.7766e-6 rad2`, compared with `8.3509e-5` for A and `5.4291e-5`
for C. Their small joint-MSE gains come from forward displacement and are not
consistent across mean, median, and P95 absolute error.

The two drives have mismatched motion regimes. Train drive 0005 has
forward/yaw standard deviations `0.107030 m / 0.016870 rad`; validation drive
0011 has `0.507264 m / 0.001657 rad`. Thus validation standardized MSE is
almost entirely controlled by forward displacement and cannot currently test
the yaw mechanism well.

Seeds 1 and 2 are paused. More training drives and a motion-regime-aware split
are required before repeating the matrix. Full component and quantile results
are in `docs/EXPERIMENT_LOG.md`.

This pause applies to the 2-DoF motion-proxy matrix. It does not block the new
next-frame feature experiment, which now precedes any additional motion seeds.

## Superseded seeded control result

The causal model and two controls were run for ten epochs at Git revision
`77f0ad0`, with seed 0 and best-checkpoint selection by validation temporal
loss. These results are retained for traceability but are not valid clean
mechanism comparisons.

| Run | State policy | Error policy | Best epoch | Legacy mixed-unit MSE | Legacy mixed-unit MAE |
| --- | --- | --- | ---: | ---: | ---: |
| A | Inherit | Dynamic, `tau=0.5` | 6 | **0.108731** | **0.222764** |
| B | Reset each frame | Dynamic, `tau=0.5` | 1 | 0.129041 | 0.240693 |
| C | Inherit | Instantaneous | 10 | 0.111454 | 0.223388 |

All three runs used `quasi_steady`, so only the top layer received a future
target. They also used the filtered error directly in the local MSE. At
`Ts=0.1035` and `tau=0.5`, the EMA current-error coefficient was 0.207, while
the instantaneous control coefficient was 1. This changed the local gradient
scale and optimization dynamics. Reset-each-frame also reset that filtered
loss state. Consequently, neither the reported A/B nor A/C difference can be
attributed cleanly to inherited memory.

The feedback decoders were also absent from the optimizer despite retaining
gradients. This has been corrected; all future recursive target-flow runs train
the feedback decoders. The earlier variance regularizer was inactive at stream
batch size 1 and is replaced by a temporal-window implementation.

The historical target named `ego_motion` contained only forward displacement
and yaw change, not lateral translation. Its direct MSE mixed metres and
radians. Corrected runs call it `longitudinal_yaw_2dof`, standardize each
component using training-only statistics, and report physical component MAEs.

The old inherited condition also contained `F_teacher(I_t)` through its saved
previous top target, whereas reset-each-frame did not. The reported 15.7% A/B
difference therefore cannot isolate long-term history. With frozen VGG,
`F_teacher(I_t)=F_student(I_t)`, so the corrected matrix adds a reset/current-
top-duplicate/no-history control.

The earlier unseeded corrected run reached MSE 0.094732 at epoch 8, but its
best weights were not saved. It remains exploratory evidence and is not used
in the seeded control comparison.

## Invalid predecessor stream result

Configuration:

- Stream state inheritance enabled.
- Dynamic error enabled at all five layers.
- `tau=0.5`, `gain=1.0`, `dt=0.1035` seconds.
- Ego-motion temporal target.
- EMA teacher with decay 0.99.
- Ten epochs, learning rate `1e-4`, batch size 1.

Key validation results:

- Best temporal cosine: 0.750144 at epoch 2.
- Epoch-10 temporal cosine: 0.651919.
- Epoch-10 weighted validation loss: 0.007630.

This run is not valid evidence of temporal prediction quality. Its temporal
context included errors formed with the future-frame teacher target, and the
temporal prediction loss weight defaulted to zero. The run only proves that the
continuous stateful execution path completed end to end. All prediction metrics
must be rerun after the causal-context and positive-loss-weight correction.

## Required next experiments

1. Preserve deterministic causal warp as the required baseline; its held-out
   advantage is positive but only 0.219%.
2. Improve post-warp residual generalization through broader training-drive
   coverage or a predeclared capacity/regularization diagnostic. Require it to
   beat warp-only on held-out ordered video before adding recurrent error state.
3. Preserve the completed same-drive corruption result as a mechanistic null:
   the state is used but does not improve systematic-bias response.
4. Keep seeds 1 and 2 and both Target Flow and Temporal Error `tau` sweeps
   paused while the predictor generalization gate fails.
5. Treat the existing 2-DoF matrix as a proxy-task diagnostic only. Additional
   motion seeds remain paused unless motion is later reintroduced as a
   secondary evaluation.

| Group | Cross-frame state | Error state | Extra current-top context | Purpose |
| --- | --- | --- | --- | --- |
| A | Inherit | Recursive EMA | From inherited state | Full method |
| B | Reset | EMA, no effective history | None | Stateless baseline |
| C | Reset | EMA, no effective history | Detached current-top duplicate | Extra-current-feature control |
| D | Inherit | Instant | From inherited state | Test recursive dynamic error |
| E | Inherit | Two-tap | From inherited state | Test EMA against finite two-tap memory |

All five groups use recursive target flow, frozen VGG, instantaneous local
loss, variance weight zero, temporal prediction weight one, identical dynamic
parameters, seed, drives, training-only normalization statistics, and best-
checkpoint selection by validation temporal loss. Formal runs set
`PREDIFY_FORMAL_SPLIT=1` and explicitly provide disjoint train and validation
drives.

The seed-0 runner fixes 10 epochs, learning rate `1e-4`, pretrained predictive-
VGG weights, train drive 0005, validation drive 0011, and the existing training-
only normalization procedure for all five conditions. It also explicitly sets
`PREDIFY_CURRENT_TEACHER_CONTEXT=0` so a stale login-shell value cannot alter a
run.

A positive variance weight is invalid while the backbone is frozen because
the top-feature variance has no gradient path to the trainable feedback or
temporal modules. Variance calibration is reserved for the separate
`PREDIFY_TRAIN_BACKBONE=1` adaptation experiment.

For the current complete training drive, horizon-1 normalization is based on
153 samples: forward mean/std `0.466140/0.107030 m`, yaw mean/std
`-0.001483/0.016870 rad`. Validation data is not used for these statistics.

Target Flow uses residual `r_t=F_t-T_t` and
`epsilon_t=(Ts_r/tau_r)r_t+(1-K_r*Ts_r/tau_r)epsilon_(t-1)`. Temporal
Prediction Error is separate: `e_t=Fhat_(t|t-1)-F_t` and
`epsilon_t=(Ts_e/tau_e)e_t+(1-K_e*Ts_e/tau_e)epsilon_(t-1)`. There is no
independent `d_t` term. Each recurrence must satisfy its own stability condition; the
current `Ts=0.1035, tau=0.5, K=1` gives memory coefficient 0.793 for both, but
their parameter families and physical meanings remain independent.

## Repository policy

Code, configuration, Markdown records, and lightweight metric summaries belong
in Git. KITTI data, pretrained weights, checkpoints, caches, and large pickle
outputs remain on the server and are excluded by `.gitignore`.

Completed code changes must be verified, committed, and pushed to
`myprivate/targetflow-arch` without waiting for a separate push request.
Completed formal training must likewise be audited, recorded in the tracked
research documents, committed, and pushed. Large training artifacts remain on
the server; their exact code revision and server path belong in the pushed
record. Use an ordinary fast-forward push and never rewrite remote history
unless the user explicitly requests it.
