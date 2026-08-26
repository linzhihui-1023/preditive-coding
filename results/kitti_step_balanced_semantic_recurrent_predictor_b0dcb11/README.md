# Balanced Semantic Recurrent Predictor

The per-pixel KL objective was calibrated once on 32 initialization clips with fixed `lambda_state=1593.4333`. A fresh Predictor was trained for 3 epochs; each epoch used 624 train clips and the first 64 validation clips.

The two-sequence quick Gate passed the semantic screen (`mIoU 0.26967 > old CNN 0.11143`) but failed the dynamics screen (`NMSE 1.95496 >= 1`). Per protocol, the complete 9-sequence validation was not run.

Decision: `STOP_AT_QUICK_GATE`.
