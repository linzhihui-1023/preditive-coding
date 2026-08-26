# Semantic Recurrent Predictor

Frozen Host/Adapter/Decoder evaluation of Persistence, Constant Velocity, the old CNN Predictor, and the new causal ConvGRU predictor on 9 KITTI-STEP validation sequences (2963 effective frames).

| Method | state mean MSE | NMSE | mIoU | wIoU | mVC8 | mVC16 |
|---|---:|---:|---:|---:|---:|---:|
| persistence | 0.0002596572 | 1.000000 | 0.1191174521 | 0.4125659927 | 0.5150084395 | 0.4749010563 |
| constant_velocity | 0.0007083061 | 2.727851 | 0.1282970488 | 0.4215565279 | 0.4589187559 | 0.4079076753 |
| old_cnn | 0.0001874364 | 0.721861 | 0.1159906888 | 0.4038862492 | 0.5075491230 | 0.4681939545 |
| new_predictor | 0.0077355818 | 29.791513 | 0.2975713712 | 0.7201568728 | 0.8842785285 | 0.8986084928 |

Oracle Current State Decode mIoU: `0.1246697003`.

New vs Constant Velocity MSE improvement: `-9.9212408683`. New vs strongest baseline mIoU improvement: `0.1692743224`.

DYNAMICS: **NO-GO**; SEMANTIC_PREDICTION: **GO**; NEW_PREDICTOR: **NO-GO**. The new predictor has semantic output above the baselines but fails the required dynamics criterion because its state MSE is much worse.
