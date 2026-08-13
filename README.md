# predify2021

Code for reproducing the results presented in the paper 'Predify:Augmenting deep neural networks with brain-inspired predictive coding dynamics' (https://arxiv.org/pdf/2106.02749.pdf)


## Dependencies
<pre>
- predify  (<a href="https://github.com/miladmozafari/predify">Repository</a>)
- torch 
- tensorboard
- loguru

For EfficientNetB0:
- timm     (<a href="https://github.com/rwightman/pytorch-image-models">Repository</a>)

For Adversarial Attacks:
- foolbox 3.x
</pre>


## Repository structure
-  `model_factory` provides predified versions of the models (VGG and EfficientNetB0) using `get_model` function. 
-  `adversarial_attacks` contains all scripts for performing and analysing adversarial attacks performed in the paper.
-  `mCE_scores` contains scripts for performing and calculating mCE scores on the predified networks.
-  `manifold_projection` contains scripts for calculating the correlation distances between clean and noisy representations (refer paper for more details).

## Weights of all the models

[Link to the PEfficientNetB0 weights](https://www.dropbox.com/s/0np1pzp3o3qhonv/weights_pefficientNetB0_imagenet.zip?dl=0) 

[Link to the PVGG_Weights](https://www.dropbox.com/s/8lzp6wfo6n3bymk/weights_pvgg16_imagenet.zip?dl=0)

## Research version boundary

The completed temporal-prediction research baseline is frozen at annotated
Git tag `predify-temporal-v1`, which points to commit
`4cc7a21280881813dbb972415a74dc400843fcc6`. That snapshot contains the Target
Flow and strict Temporal Prediction Error implementations, formal training and
controlled-corruption experiments, causal motion diagnostics, and the
versioned prediction-error separability result. It must remain unchanged so
the prediction direction can be reproduced or resumed independently.

New work on prediction-error-driven selective online adaptation starts from
that exact snapshot and belongs only on branch
`predify-selective-adaptation-v2`. Do not commit selective-adaptation changes
to the frozen tag or use `targetflow-arch` as the active development branch.

The local checkouts are also separated. The frozen baseline remains at
`/home/lin/predify2021_targetflow`, checked out on `targetflow-arch` at
`4cc7a21`. Selective-adaptation development uses the independent Git worktree
`/home/lin/predify2021_selective_adaptation`. Run new experiments and make all
new edits only from the latter directory.

The first experiment on the new branch is an inference-only persistence gate.
It compares long-dwell and temporally shuffled Gaussian blur while exactly
matching blur type, sigma multiset, occupancy, raw frames, and checkpoint.
Its implementation is launched by
`scripts/run_kitti_matched_blur_persistence.sh`; audited lightweight results
are under `results/matched_blur_persistence_99b7e21/`.

Gate 3 then froze that detector and tested whether its eight-frame persistence
score predicts clean-relative forward/yaw MAE degradation. For the existing
frozen 2-DoF model it did not: held-out forward correlation and positive-
degradation AUROC were near null, while yaw usually improved under both blur
organizations. Results are under
`results/gate3_persistence_task_degradation_80c4aee/`.

The next-frame feature study now includes an explicitly causal two-frame
fusion condition at VGG stage 5. It stores the previous top feature as a
detached drive-local memory and computes
`Z_t = F_t + T([F_(t-1), F_t])` before the existing future predictor. The first
frame of each drive uses `Z_t = F_t`; the pretrained VGG and Target Flow
feedback decoders remain frozen, while only the fusion module and future
predictor are optimized. The formal matrix is limited to Copy-current,
Current-only, and Temporal Fusion and is launched with:

```bash
scripts/run_kitti_seed0_temporal_fusion_matrix.sh
```

The runner trains on drive 0005, evaluates on held-out drive 0011, and writes
both `summary.json` and a per-frame CSV with feature MSE, cosine similarity,
normalized feature error, and fusion-state diagnostics.

The formal seed-0 result did not clear Copy-current on drive 0011: Temporal
Fusion MSE was `0.061926` versus `0.060080` for Copy-current. The complete
small artifacts and scope-limited interpretation are under
`results/temporal_fusion_matrix_10191a6/`; the approximately 4 GB of
checkpoints and logs remain server-side.

The aligned-history follow-up at revision `95aa61d` first maps `F_(t-1)` into
`F_t` coordinates with causal local matching, then passes
`[F_(t-1)_aligned, F_t]` through the unchanged residual fusion and future
predictor. The single-condition seed-0 run trained on drive 0005 and used the
previous Copy-current metrics as a fixed gate on drive 0011. It did not pass:
MSE was `0.06191402` versus `0.06008010`, cosine was `0.91520128` versus
`0.91990469`, and normalized error was `0.38581802` versus `0.37576611`.
Auditable small artifacts are under
`results/aligned_temporal_fusion_95aa61d/`; the checkpoint remains server-side.
The exact runner is:

```bash
scripts/run_kitti_seed0_aligned_temporal_fusion.sh
```

The aligned temporal-difference follow-up at revision `750d11f` leaves `F_t`
unchanged and feeds `[F_t, F_t - align(F_(t-1), F_t)]` directly to the existing
residual Future Predictor. The single-condition seed-0 run used drive 0005 for
training and the fixed Copy-current gate on drive 0011. It did not pass: MSE
was `0.06180378` versus `0.06008010`, cosine was `0.91604762` versus
`0.91990469`, and normalized error was `0.38434362` versus `0.37576611`.
Auditable small artifacts are under
`results/aligned_temporal_difference_750d11f/`; the checkpoint remains
server-side. The exact runner is:

```bash
scripts/run_kitti_seed0_aligned_temporal_difference.sh
```

Future-feature prediction is now independently selectable at VGG Stage 3, 4,
or 5 through `PREDIFY_FUTURE_FEATURE_STAGE`. This does not move the Predify
Target Flow boundary: its top target remains the next-frame Stage-5 feature,
while the future predictor receives and is supervised in the configured
prediction stage. The two target definitions, stages, channels, detached
prediction-stage memory, and feature-cell motion radius are recorded in every
new checkpoint.

The first Stage-4 follow-up keeps the aligned temporal-difference mechanism
and all training controls fixed. It uses radius 1 in Stage-4 feature cells and
compares its prediction only against a Stage-4 Copy-current replay over the
same frames. Stage-4 and Stage-5 absolute MSE values are different feature
spaces and must not be ranked against each other. Run the single condition
with:

```bash
scripts/run_kitti_seed0_stage4_aligned_temporal_difference.sh
```

The formal seed-0 run at revision `fdc4743` selected epoch 1 and did not clear
its same-stage held-out Copy-current gate. MSE was `0.74586091` versus
`0.74053836`, cosine was `0.85806100` versus `0.86544334`, and normalized
error was `0.49728997` versus `0.48848719`. Auditable lightweight outputs are
under `results/stage4_aligned_temporal_difference_fdc4743/`; checkpoint and
logs remain server-side.

The one-time Stage-4 same-drive diagnostic uses only drive 0005 and partitions
raw frames chronologically into first 60% Train, middle 20% Val, and final 20%
Test. A transition is admitted only when both raw frames lie inside one
partition, so no frame is shared across roles. Training and checkpoint
selection never read Test metrics; the selected checkpoint is replayed once on
the final segment. Run it with:

```bash
scripts/run_kitti_seed0_stage4_same_drive_diagnostic.sh
```

The one-time run at revision `69b24b1` selected epoch 10. On the untouched
last-20% segment, MSE improved by `9.964079%` and normalized error improved,
while cosine worsened by `0.01523336`. This is evidence of same-drive forward
generalization under the Euclidean feature objective, not an all-metric pass.
No further same-drive variants are planned. Auditable outputs are under
`results/stage4_same_drive_60_20_20_69b24b1/`.

The predeclared Stage-4 multi-drive experiment keeps the same aligned temporal
difference model and fixes Train to drives 0005, 0013, 0014, and 0036; Val to
0011 and 0039; and frozen Test to 0051 and 0056. Training and checkpoint
selection receive only Train/Val drive names. The independent evaluator claims
one atomic Test access only after the best Val checkpoint has been selected.
Run the single formal experiment with:

```bash
scripts/run_kitti_seed0_stage4_multidrive.sh
```

The formal run at revision `c887e94` selected epoch 7 using Val only. On the
once-read frozen Test drives 0051/0056, MSE was `0.77194043` versus same-stage
Copy-current `0.89291654`, a `13.548423%` improvement. Cosine improved from
`0.82239107` to `0.83322664`, and normalized error improved from `0.58067843`
to `0.54544640`; both Test drives also passed all three checks independently.
The result supports multi-drive generalization for this fixed Stage-4 model
and seed. Auditable outputs, including the one-time Test receipt, are under
`results/stage4_multidrive_c887e94/`. Do not rerun or tune against the frozen
Test drives.

To inspect or resume the frozen baseline without moving the new research
branch:

```bash
git switch --detach predify-temporal-v1
```

To return to selective online adaptation development:

```bash
git switch predify-selective-adaptation-v2
```

## Frozen stateful KITTI target-flow baseline

The frozen temporal baseline extends PVGG16 with five-layer target-flow state
and processes KITTI as an ordered video stream. Each frame executes the model
once; state is initialized on the first frame and inherited until the drive
boundary.
Stored cross-frame state is detached, so training uses stateful recurrence with
one-step gradients rather than BPTT. Primary mechanism experiments freeze the
pretrained VGG backbone and train the feedback decoders plus temporal head.
`PREDIFY_CURRENT_TOP_DUPLICATE=1` provides the dedicated reset/no-history
current-top duplicate control and requires `PREDIFY_RESET_EACH_FRAME=1`.
Formal cross-drive runs can set `PREDIFY_FORMAL_SPLIT=1` to require explicit,
disjoint `PREDIFY_TRAIN_DRIVES` and `PREDIFY_VAL_DRIVES`.

The reproducible seed-0 frozen-backbone matrix is launched with:

```bash
scripts/run_kitti_seed0_five_group_matrix.sh
```

The runner clears the inherited shell environment with `env -i`, explicitly
sets both the new and legacy current-context variables, executes A through E
sequentially, and stores the Git revision in every result. Individual groups
can be selected by passing their letters, for example `...sh A C`.

Current status and reproducibility records are maintained in:

- [`docs/PROJECT_STATE.md`](docs/PROJECT_STATE.md)
- [`docs/DECISIONS.md`](docs/DECISIONS.md)
- [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md)
