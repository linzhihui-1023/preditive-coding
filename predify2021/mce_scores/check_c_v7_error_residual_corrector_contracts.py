"""Lightweight C-V7 structural contract checks.

中文：C-V7 轻量级结构契约检查。

This check does not load KITTI-STEP, checkpoints, Host or RAFT.
It verifies the new trainable module itself before any expensive experiment.
"""

import inspect

import torch

from predify2021.model_factory.deeplabv3plus_resnet50.task_space_multihypothesis_error_residual_corrector import (
    MultiHypothesisErrorResidualCorrector,
)


def main():
    torch.manual_seed(0)
    num_classes = 19
    history_length = 4
    height, width = 8, 12

    model = MultiHypothesisErrorResidualCorrector(
        num_classes=num_classes,
        history_length=history_length,
        hidden_channels=32,
        g_max=0.25,
        gate_bias=-2.0,
    )

    signature = inspect.signature(model.forward)
    forbidden = {
        "current_probability",
        "history_probability",
        "history_probabilities",
        "current_logits",
        "history_logits",
    }
    leaked = forbidden.intersection(signature.parameters)
    if leaked:
        raise RuntimeError(f"Raw semantic shortcut leaked into C-V7 forward API: {sorted(leaked)}")

    prediction_errors = [
        torch.randn(1, num_classes, height, width) * 0.05
        for _ in range(history_length)
    ]
    dynamics_error = torch.randn(1, num_classes, height, width) * 0.05
    current_margin = torch.rand(1, 1, height, width)
    history_margins = [torch.rand(1, 1, height, width) for _ in range(history_length)]
    transportability = torch.rand(1, 1, height, width)
    memory_reliability = torch.rand(1, 1, height, width)
    validities = [torch.ones(1, 1, height, width) for _ in range(history_length)]
    backward_motion = torch.zeros(1, 2, height, width)

    row0 = model(
        prediction_errors,
        dynamics_error,
        current_margin,
        history_margins,
        transportability,
        memory_reliability,
        validities,
        backward_motion,
        None,
    )

    if row0["delta_z_raw"].shape != (1, num_classes, height, width):
        raise RuntimeError("Correction Head must output 19-channel DeltaZ")
    if row0["gate"].shape != (1, 1, height, width):
        raise RuntimeError("Gate Head must remain single-channel")
    if float(row0["delta_z_raw"].detach().abs().max().item()) != 0.0:
        raise RuntimeError("Correction Head zero initialization failed")
    if float(row0["delta_z"].detach().abs().max().item()) != 0.0:
        raise RuntimeError("E0 bounded correction must be exactly zero")
    if float(row0["gate"].detach().min().item()) < 0.0:
        raise RuntimeError("Gate must be non-negative")
    if float(row0["gate"].detach().max().item()) > model.g_max + 1e-7:
        raise RuntimeError("Gate exceeded g_max")

    # A second step exercises motion-aligned recurrent history.
    row1 = model(
        prediction_errors,
        dynamics_error,
        current_margin,
        history_margins,
        transportability,
        memory_reliability,
        validities,
        backward_motion,
        row0["hidden"],
    )
    if row1["warped_hidden"].shape != row0["hidden"].shape:
        raise RuntimeError("Motion-aligned Error Memory shape mismatch")
    if not bool(torch.isfinite(row1["error_reliability"]).all()):
        raise RuntimeError("Non-finite Error Memory reliability")

    # Explicitly check frozen-current isolation and first-step trainability.
    z_cur = torch.randn(1, num_classes, height, width, requires_grad=True)
    final = z_cur.detach() + row0["gate"] * row0["delta_z"]
    loss = final.square().mean()
    model.zero_grad(set_to_none=True)
    loss.backward()
    if z_cur.grad is not None:
        raise RuntimeError("Gradient leaked into detached Z_cur")
    correction_grad = model.correction_head[-1].weight.grad
    if correction_grad is None or float(correction_grad.abs().sum().item()) <= 0.0:
        raise RuntimeError("Zero-initialized Correction Head did not receive usable gradient")

    print(
        {
            "passed": True,
            "raw_history_shortcut": False,
            "correction_channels": num_classes,
            "gate_channels": 1,
            "delta_z_e0_abs_max": float(row0["delta_z"].detach().abs().max().item()),
            "gate_max": float(row0["gate"].detach().max().item()),
            "g_max": model.g_max,
            "z_cur_gradient": None,
            "correction_head_gradient_abs_sum": float(correction_grad.abs().sum().item()),
            "motion_aligned_recurrent_state": True,
        }
    )


if __name__ == "__main__":
    main()
