"""Stage T with TrajGRU while reusing the existing training and evaluation protocol."""

import json
import sys
from pathlib import Path

from predify2021.mce_scores import train_kitti_step_v2_auxiliary as base
from predify2021.model_factory.deeplabv3plus_resnet50.trajgru_predictor import (
    AuxiliaryTemporalTrajPredictor,
    trajgru_sanity_checks,
)


TRAJ_OUTPUT_DEFAULT = "/home/lin/predify/experiments/kitti_step_v2_auxiliary_stage_t_trajgru"
TRAJ_RESULT_DEFAULT = "results/kitti_step_v2_auxiliary_stage_t_trajgru"
TRAJ_LINKS = 5


def _argument_value(argv, name, default):
    tokens = list(sys.argv[1:] if argv is None else argv)
    if name not in tokens:
        return default
    index = tokens.index(name)
    if index + 1 >= len(tokens):
        return default
    return tokens[index + 1]


def _write_trajgru_metadata(result_output):
    result_dir = Path(result_output)
    summary_path = result_dir / "summary.json"
    if not summary_path.exists():
        return

    summary = json.loads(summary_path.read_text())
    summary["experiment"] = "V2-Auxiliary Stage T TrajGRU"
    summary["predictor_type"] = "TrajGRU"
    summary["predictor_config"] = {
        "links": TRAJ_LINKS,
        "controlled_change": "location-variant recurrent spatial connection",
        "gate_convention": "matched_to_stage_t_convgru",
        "candidate_activation": "tanh",
    }
    for record in summary.get("history", []):
        record["predictor_type"] = "TrajGRU"
        record["trajgru_links"] = TRAJ_LINKS
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    (result_dir / "README.md").write_text(
        "# V2-Auxiliary Stage T TrajGRU\n"
        "Only the recurrent spatial transition is changed from the Stage-T ConvGRU baseline. "
        "Encoder, loss, TBPTT, data, checkpoint selection, gate convention, candidate activation "
        "and formal metrics remain unchanged. TrajGRU uses 5 learned trajectory links.\n"
    )


def main(argv=None):
    # Replace only the recurrent predictor. Encoder, loss, TBPTT, data,
    # checkpoint selection and formal metrics remain exactly the Stage-T baseline.
    base.AuxiliaryTemporalPredictor = AuxiliaryTemporalTrajPredictor
    base.OUTPUT_DEFAULT = TRAJ_OUTPUT_DEFAULT
    base.RESULT_DEFAULT = TRAJ_RESULT_DEFAULT

    # Extend the existing E0 check without changing its persistence contract.
    original_zero_step_check = base.zero_step_check

    def zero_step_with_trajgru_checks(model, encoder, predictor, sample):
        result = original_zero_step_check(model, encoder, predictor, sample)
        device = next(predictor.parameters()).device
        result["trajgru"] = trajgru_sanity_checks(device)
        return result

    base.zero_step_check = zero_step_with_trajgru_checks
    result_output = _argument_value(argv, "--result-output", TRAJ_RESULT_DEFAULT)
    base.main(argv)
    _write_trajgru_metadata(result_output)


if __name__ == "__main__":
    main()
