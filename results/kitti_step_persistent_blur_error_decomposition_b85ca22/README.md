# Persistent Blur Error Decomposition

This experiment replaces iid Gaussian noise in the current Error Decomposition + Reliability line with the shared controlled persistent Gaussian blur generator: kernel 11, sigma levels 0.75, 1.5, 2.25, then persistent 3.0. Host, Adapter, Predictor, Writeback, and Decoder remain frozen; corrupted observations remain the Predictor history.

Under the same KITTI-STEP validation stream, Clean Host mIoU is 0.6552125562, Persistent-Blur Corrupted Host mIoU is 0.4701953800, Corrected B is 0.4929852801, and Error Decomposition + Reliability is 0.4963800845. The new model improves over Corrected B by 0.0033948043. All finite and identity gates passed.

Decision: `IMPROVED_OVER_CORRECTED_B`.
