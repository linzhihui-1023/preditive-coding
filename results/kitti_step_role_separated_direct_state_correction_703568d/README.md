# Role-Separated Direct State Correction

Experiment A trains the fixed Role-Separated Prediction Error with Direct State Correction and state MSE. Experiment B uses the same zero-initialized correction with semantic cross-entropy plus the fixed relative state constraint. Host, Adapter, Predictor, and Writeback remain frozen; correction does not feed Predictor history.

The identity and finite gates passed on 2963 effective KITTI-STEP validation frames. A reached `0.3171036753` mIoU and did not exceed the historical direct open-loop result. B reached `0.3636786247`, improving A by `0.0465749494`; the semantic-state objective is `SEMANTIC_STATE_DIRECT_CORRECTION: STRONG GO`.
