"""C-V3 Stable-Memory Stage A with readout Memory-usage diagnostics.

Training is unchanged from
``train_kitti_step_task_space_prior_c_v3_memory_supervision_stage_a_main``.
After training, this entrypoint reads each saved C-V3 checkpoint and measures
how strongly the first Adaptive-Readout projection weights its six input roles:

    Memory H_t | appearance F_t | P_host | prediction error e_t | T_t | E1 C_t

The diagnostic adds no forward/backward work to training.  It is a structural
proxy for Readout shortcutting: if Memory cosine improves while the Readout's
Memory-input weights shrink relative to non-Memory inputs, the temporal state
may be learned but weakly used by the final correction path.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch

from predify2021.mce_scores import (
    train_kitti_step_task_space_prior_c_v3_memory_supervision_stage_a_main as stable,
)


MEMORY_CHANNELS = 64
APPEARANCE_CHANNELS = 32
NUM_CLASSES = 19
T_CHANNELS = 1

BLOCKS = (
    ("memory", MEMORY_CHANNELS),
    ("appearance", APPEARANCE_CHANNELS),
    ("host_probability", NUM_CLASSES),
    ("prediction_error", NUM_CLASSES),
    ("transportability", T_CHANNELS),
    ("semantic_state", NUM_CLASSES),
)
EXPECTED_INPUT_CHANNELS = sum(channels for _, channels in BLOCKS)
PRE_WEIGHT_KEY = "readout.pre.0.weight"
EPS = 1e-12


def _readout_input_weight_diagnostics(refiner_state_dict):
    """Measure per-role weight use in the first 1x1 Readout projection.

    Raw L2 norm depends on the number of channels in a role, so RMS is also
    reported.  Energy share is the fraction of squared weight energy assigned
    to each role.  For equal per-weight scale, the expected Memory energy share
    is simply 64 / 154; the relative-energy statistic divides by that baseline.
    """
    if PRE_WEIGHT_KEY not in refiner_state_dict:
        raise RuntimeError(
            f"Missing {PRE_WEIGHT_KEY!r} in C-V3 refiner checkpoint"
        )
    weight = refiner_state_dict[PRE_WEIGHT_KEY].detach().float().cpu()
    if weight.ndim != 4 or tuple(weight.shape[-2:]) != (1, 1):
        raise RuntimeError(
            f"Unexpected Readout pre weight shape: {tuple(weight.shape)}"
        )
    if int(weight.shape[1]) != EXPECTED_INPUT_CHANNELS:
        raise RuntimeError(
            "Unexpected Readout input channels: "
            f"{int(weight.shape[1])}, expected {EXPECTED_INPUT_CHANNELS}"
        )

    total_energy = float(weight.square().sum().item())
    total_energy_safe = max(total_energy, EPS)
    block_stats = {}
    offset = 0
    for name, channels in BLOCKS:
        block = weight[:, offset:offset + channels]
        energy = float(block.square().sum().item())
        l2 = math.sqrt(max(energy, 0.0))
        rms = math.sqrt(max(float(block.square().mean().item()), 0.0))
        block_stats[name] = {
            "channels": int(channels),
            "l2": l2,
            "rms": rms,
            "energy_share": energy / total_energy_safe,
        }
        offset += channels

    memory = weight[:, :MEMORY_CHANNELS]
    nonmemory = weight[:, MEMORY_CHANNELS:]
    memory_rms = math.sqrt(max(float(memory.square().mean().item()), 0.0))
    nonmemory_rms = math.sqrt(max(float(nonmemory.square().mean().item()), 0.0))
    expected_memory_energy_share = MEMORY_CHANNELS / EXPECTED_INPUT_CHANNELS
    observed_memory_energy_share = block_stats["memory"]["energy_share"]

    return {
        "definition": (
            "Adaptive Readout first 1x1 projection weight statistics; proxy for "
            "Memory-vs-current-evidence usage, not a causal sensitivity measure"
        ),
        "pre_weight_shape": [int(value) for value in weight.shape],
        "blocks": block_stats,
        "memory_vs_nonmemory_rms_ratio": memory_rms / max(nonmemory_rms, EPS),
        "memory_energy_share": observed_memory_energy_share,
        "expected_memory_energy_share_equal_scale": expected_memory_energy_share,
        "memory_energy_share_relative_to_channel_fraction": (
            observed_memory_energy_share / max(expected_memory_energy_share, EPS)
        ),
    }


def _epoch_number(path):
    stem = path.stem
    try:
        return int(stem.split("_")[-1])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Cannot parse epoch from {path}") from exc


def _load_checkpoint_diagnostic(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("refiner_state_dict")
    if state is None:
        raise RuntimeError(f"Missing refiner_state_dict in {path}")
    diagnostic = _readout_input_weight_diagnostics(state)
    return payload, diagnostic


def _inject_readout_diagnostics(output, result_output):
    """Add weight-use diagnostics to epoch JSON, summary, and local checkpoints."""
    per_epoch = {}
    for checkpoint_path in sorted(output.glob("epoch_*.pt")):
        epoch = _epoch_number(checkpoint_path)
        payload, diagnostic = _load_checkpoint_diagnostic(checkpoint_path)
        per_epoch[epoch] = diagnostic
        payload["readout_memory_usage"] = diagnostic
        torch.save(payload, checkpoint_path)

        epoch_json = result_output / f"epoch_{epoch:03d}.json"
        if epoch_json.exists():
            row = json.loads(epoch_json.read_text())
            row["readout_memory_usage"] = diagnostic
            epoch_json.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")

    best_diagnostic = None
    best_path = output / "best.pt"
    if best_path.exists():
        best_payload, best_diagnostic = _load_checkpoint_diagnostic(best_path)
        best_payload["readout_memory_usage"] = best_diagnostic
        torch.save(best_payload, best_path)

    summary_path = result_output / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        for row in summary.get("history", []):
            epoch = int(row.get("epoch", -1))
            if epoch in per_epoch:
                row["readout_memory_usage"] = per_epoch[epoch]
        if best_diagnostic is not None:
            summary.setdefault("best", {})["readout_memory_usage"] = best_diagnostic
        summary["readout_memory_usage_diagnostic"] = {
            "purpose": (
                "Check whether a stable 64-D Memory is actually retained by the "
                "Adaptive Readout instead of being bypassed by F_t/P_host/e_t/T/C_t"
            ),
            "measure": (
                "first Readout 1x1 input-weight RMS and squared-energy shares by role"
            ),
            "interpretation": (
                "Memory weight statistics near zero relative to non-Memory inputs "
                "support a Readout-shortcut explanation; nonzero weights alone do "
                "not prove causal dependence because activation scales also matter"
            ),
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    return per_epoch, best_diagnostic


def _resolve_output_paths(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", default=stable.OUTPUT_DEFAULT)
    parser.add_argument("--result-output", default=stable.RESULT_DEFAULT)
    known, _ = parser.parse_known_args(argv)
    return Path(known.output), Path(known.result_output)


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    output, result_output = _resolve_output_paths(args)

    # Executes the unchanged Stable-Memory / Correct-Output training contract.
    stable.main(args)
    per_epoch, best_diagnostic = _inject_readout_diagnostics(output, result_output)

    print(
        json.dumps(
            {
                "readout_memory_usage": {
                    "epochs": sorted(per_epoch),
                    "best": best_diagnostic,
                },
                "result": str(result_output / "summary.json"),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
