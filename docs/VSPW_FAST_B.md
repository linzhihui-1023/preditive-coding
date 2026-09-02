# FAST-B → VSPW-480p

This path evaluates the validated FAST-B mainline on clean VSPW video. It
loads a 124-class `best_vspw_host.pt`, freezes the Host, and trains only the
FAST-B Semantic Branch plus the validated C4 output adapter and C4
Host-conditioned writeback. Backbone, decoder, auxiliary head, C1 path, and
dynamics are frozen.

## Mainline trace

The current FAST-B is the `fast_b_joint_c4_weak_z4` variant in
`train_kitti_step_semantic_v3.py`, not the legacy temporal correction script.
Its validated code/result chain is:

`6ac6e8f` (Joint-C4 A/B implementation) → `5754714` (remove rejected
context-gated Z4 head) → `9b68a68` (record FAST-A/B validation), with the
VSPW Host added at `b0477ad` and the current checkout at `95a904a`.

The runtime path is:

`Host backbone → z1/z4 128-channel adapters → frozen causal dynamics
prediction → e_t = observation_z4 - prediction_z4 → error encoder → semantic
state cell → Z4 restoration residual → C4 output adapter + Host-conditioned
writeback → frozen DeepLabV3+ decoder → 124 logits`.

The trainable modules are `semantic_error_encoder`, `semantic_state_cell`,
`semantic_restoration_head`, the C4 output adapter, and the C4 writeback.
The predictor input is the current latent observation and causal pending
prediction; its persistent state is the dynamics state, semantic hidden state,
and error temporal statistics. A FAST-B checkpoint stores the predictor state,
C4 adapter/writeback state, epoch, VSPW class count, clip/BPTT settings, and
parameter report.

The sequential loader keeps frames from one video/run together. `clip_length`
supports 16 and 32; `bptt` defaults to 16 and may be reduced only to 8. The
gradient graph is detached at BPTT windows, while Full temporal state is
carried until the next video (or a genuine frame-id gap).

VSPW uses clean RGB and ground-truth semantic cross entropy (`ignore_index=255`)
with the FAST-B feature objective. No corruption or teacher KL path is used.

The prior KITTI validation recorded FAST-B's best epoch-3 mean restored mIoU
as `0.436294` on its two diagnostic conditions, versus FAST-A `0.414644`; this
is provenance for choosing the current FAST-B variant, not a VSPW result.

## Commands after Host completion

```bash
export VSPW_HOST_CHECKPOINT=/home/lin/predify/experiments/vspw_static_deeplabv3plus_r50_v1/best_vspw_static_deeplabv3plus_r50.pt
scripts/run_vspw_fast_b_gate_v2a.sh
scripts/run_vspw_fast_b.sh
export FAST_B_CHECKPOINT=/home/lin/predify/experiments/vspw_fast_b/best_vspw_fast_b.pt
scripts/run_vspw_fast_b_eval.sh
```

The evaluator emits the unified Host / FAST-B Reset / FAST-B Full table with
mIoU, mVC8, and mVC16. It does not write per-frame prediction files.
