# Semantic-State Direct Correction With Residual Writeback

Experiment B was retrained from zero initialization with the same residual writeback used by formal validation: `F_noisy + W(delta) - W(0)`. All gates passed, including exact training/validation logits equivalence and zero-delta identity.

The corrected B result is `0.3505486298` mIoU on 2963 effective validation frames. It is below the old B reference `0.3636786247`, but remains above the required `0.350` threshold: `SEMANTIC_STATE_RESIDUAL_WRITEBACK: GO`.
