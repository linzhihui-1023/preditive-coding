# Z4 Adaptive Gain

Only a frame-wise scalar `Z4AdaptiveGainHead` is trained. Host, C4 adapter/writeback, decoder, and Stage-P predictor remain frozen. Stage-P consumes raw `Z4` and `e=Z4-Z4_prediction`; the posterior is used only for the frozen segmentation path.
