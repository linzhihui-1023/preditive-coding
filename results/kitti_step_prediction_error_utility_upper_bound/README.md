# Prediction Error Utility Upper Bound

Inference-only Full9 diagnostic. The frozen Stage-C Epoch-3 checkpoint is unchanged; semantic state, correction head, and gate are bypassed. Existing frozen C4 writeback and decoder receive `Z4 + beta * (Z4 - Z4_prediction)`. The oracle selects beta per frame using ground-truth segmentation cross-entropy and is noncausal.
