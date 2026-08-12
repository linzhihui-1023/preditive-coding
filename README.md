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
