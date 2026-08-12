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

The first experiment on the new branch is an inference-only persistence gate.
It compares long-dwell and temporally shuffled Gaussian blur while exactly
matching blur type, sigma multiset, occupancy, raw frames, and checkpoint.
Its implementation is launched by
`scripts/run_kitti_matched_blur_persistence.sh`; audited lightweight results
are under `results/matched_blur_persistence_99b7e21/`.

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
