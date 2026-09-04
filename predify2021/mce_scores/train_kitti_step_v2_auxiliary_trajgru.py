"""Stage T with TrajGRU while reusing the existing training and evaluation protocol."""

from predify2021.mce_scores import train_kitti_step_v2_auxiliary as base
from predify2021.model_factory.deeplabv3plus_resnet50.trajgru_predictor import (
    AuxiliaryTemporalTrajPredictor,
)


TRAJ_OUTPUT_DEFAULT = "/home/lin/predify/experiments/kitti_step_v2_auxiliary_stage_t_trajgru"
TRAJ_RESULT_DEFAULT = "results/kitti_step_v2_auxiliary_stage_t_trajgru"


def main(argv=None):
    # Replace only the recurrent predictor. Encoder, loss, TBPTT, data,
    # checkpoint selection and formal metrics remain exactly the Stage-T baseline.
    base.AuxiliaryTemporalPredictor = AuxiliaryTemporalTrajPredictor
    base.OUTPUT_DEFAULT = TRAJ_OUTPUT_DEFAULT
    base.RESULT_DEFAULT = TRAJ_RESULT_DEFAULT
    return base.main(argv)


if __name__ == "__main__":
    main()
