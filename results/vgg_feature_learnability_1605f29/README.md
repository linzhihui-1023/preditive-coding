# VGG feature-task learnability matrix

This directory contains the complete small outputs from the forward-only KITTI
feature diagnostic at Git revision:

```text
1605f29c28b214b25f2f4c2df6c06adf45c8f554
```

The files were copied byte-for-byte from:

```text
/tmp/predify-storage/experiments/vgg_feature_learnability_1605f29/
```

`summary.json` contains the definitions, data/time-filter provenance, all 24
split/stage/horizon aggregate entries, distributions, and oracle shift
histograms. `per_frame.csv` contains 4,500 rows: one for each of 375 forecast
origins, three VGG stages, and four horizons. It records frame identities,
Copy-current MSE, causal constant-velocity MSE, noncausal oracle-translation
MSE, relative improvements, adjacent-delta cosine, and the selected oracle
shift. `SHA256SUMS` records the exact hashes of both outputs.

The oracle selects its translation using the future feature and is not a
causal prediction result. It searches integer `dy,dx in {-1,0,1}` feature
cells with zero fill and full-map MSE. All horizons share starts that are valid
from `t-1` through `t+5`.
