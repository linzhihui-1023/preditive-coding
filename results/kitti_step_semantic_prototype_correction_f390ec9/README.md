# Semantic Prototype Target Correction

The experiment added a frozen 19-class Z1/Z4 Prototype Bank from KITTI-STEP train clean features and labels. Error State `H` produced a semantic target-logit residual and a single-channel `tanh^2` gate; correction moved the observation toward the soft prototype mixture. Host, Adapter, Predictor, Writeback, Decoder, and the existing error encoder were frozen.

The best checkpoint was epoch 3 after 3 epochs of persistent Gaussian blur training. Full validation used 9 sequences and 2,963 effective frames. All required runtime gates passed, but `R_target` was `4.7353` (Z1) and `1.9489` (Z4), and new mIoU `0.4701567177` did not improve on corrupted Host `0.4701953800` or current reference `0.5083952797`. Final decision: `NO-GO`.
