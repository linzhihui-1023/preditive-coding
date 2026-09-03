# Z4-only Stage C-Balance
Stage C freezes the Stage-P Z4 predictor and trains only the semantic error encoder, semantic state cell and restoration head. RAFT is used only for training/evaluation temporal supervision. Lseg and Lsafe are computed on every frame. The C4 interface and DeepLabV3+ Host remain frozen. A passing dev3 Gate reloads best.pt and triggers one Full9 evaluation.
