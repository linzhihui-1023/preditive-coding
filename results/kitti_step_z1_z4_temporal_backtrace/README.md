# Z1/Z4 temporal backtrace

Inference-only evaluation of `results/kitti_step_convgru_decoupling_quick_0c83ad0/full.pt`.
Both recurrent correction states are updated for every variant; only the
correction output applied to the final posterior is masked.

Full9 mTC ordering:

- Legacy-None: 70.7814%
- Legacy-Z1: 70.7481%
- Legacy-Z4: 70.8036%
- Legacy-Full: 70.8388%

The isolated Z4 path provides the positive single-scale mTC gain over None;
Z1 alone is slightly below None. Full is highest, indicating a small
multi-scale synergy, but the dominant full9 temporal contribution is Z4.
