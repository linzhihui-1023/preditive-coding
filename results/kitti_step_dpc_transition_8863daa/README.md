# DPC Dynamic Error Transition

This result uses code commit `8863daa` on the persistent Gaussian Blur KITTI-STEP protocol. The DPC model was trained for 15 epochs with the existing semantic plus distillation loss, BPTT 4, and frozen Host, Predictor, and writeback components. The best checkpoint was selected by validation mIoU.

The DPC Full model reached mIoU `0.4735867930` and mVC16 `0.9003727839`. The same-checkpoint Zero-Dynamic control reached mIoU `0.4735866170` and mVC16 `0.9003732956`, so Dynamic Error conditioning did not show an independent contribution. The formal decision is `DPC_TRANSITION: NO-GO`.

Formal FPS is recorded in `efficiency.json`: Host-only `89.4479708` FPS and DPC Full `49.0462987` FPS, with `54.8322%` FPS retention.
