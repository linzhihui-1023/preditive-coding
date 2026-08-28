# Raw-State STCN Memory Correction

This rerun removes Value Projection, stores raw corrupted Z4 observations in the four-frame FIFO memory, and recomputes all memory keys with the current shared Key Projection at read time. The corrected memory reference ratio is accumulated only over corrupted frames as a global L1 distance ratio.

Training used the unchanged three-epoch persistent-blur protocol. Only the 8,192-parameter Key Projection and 129-parameter gate were trainable (`8,321` total); epoch 3 had the lowest validation total loss (`2.3632711024`).

On all 9 KITTI-STEP validation sequences, STCN reached mIoU `0.4797405872`, versus corrupted host `0.4701953800` and current d7 `0.5083952797`. Corrupted-phase normalized entropy was `0.9999746294` and corrected `R_memory` was `1.3752157463`. Both formal termination conditions are met, so the result is `STCN MEMORYREADER TERMINATED`.
