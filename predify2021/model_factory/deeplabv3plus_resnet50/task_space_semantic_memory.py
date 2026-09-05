"""C-V3 semantic memory and adaptive correction readout.

This module deliberately does not replace the validated C-V2 E1 correction.
It treats the existing 19-D carried correction C_t as a frozen baseline and
learns only an additive vector refinement DeltaR_t.

The recurrent state H_t is a 64-D temporal semantic evidence state, not logits
and not a reconstruction target. Historical memory is motion-compensated and
suppressed when geometry/semantic agreement makes the warped state unreliable.
Current C1 appearance is re-injected every frame as a spatial anchor.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .semantic_recurrent_predictor import ConvGRUCell
from .task_space_motion_transport import low_flow_grid


class ReliabilityGatedSemanticMemory(nn.Module):
    """Motion-compensated 64-D semantic evidence with a current-frame anchor.

    Inputs intentionally avoid the linearly redundant tuple
    (P_host, P_prior, P_host-P_prior) in the recurrent seed. The seed uses
    P_host and prediction error; P_prior is used only to estimate historical
    agreement/reliability.
    """

    def __init__(
        self,
        c1_channels=256,
        num_classes=19,
        feature_channels=32,
        memory_channels=64,
    ):
        super().__init__()
        self.c1_channels = int(c1_channels)
        self.num_classes = int(num_classes)
        self.feature_channels = int(feature_channels)
        self.memory_channels = int(memory_channels)

        groups = 8 if self.feature_channels % 8 == 0 else 1
        self.feature_projector = nn.Sequential(
            nn.Conv2d(self.c1_channels, self.feature_channels, 1, bias=False),
            nn.GroupNorm(groups, self.feature_channels),
            nn.SiLU(),
        )

        # F_t + P_host + e_t + T_t -> current semantic seed.
        seed_channels = self.feature_channels + 2 * self.num_classes + 1
        self.current_seed = nn.Sequential(
            nn.Conv2d(seed_channels, self.memory_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8 if self.memory_channels % 8 == 0 else 1, self.memory_channels),
            nn.SiLU(),
        )
        self.recurrent = ConvGRUCell(self.memory_channels, self.memory_channels)

    @staticmethod
    def _warp_zero_invalid(previous_memory, backward_motion):
        if previous_memory.shape[-2:] != backward_motion.shape[-2:]:
            raise ValueError("Semantic memory and motion must share spatial size")
        grid, valid = low_flow_grid(backward_motion)
        warped = F.grid_sample(
            previous_memory.float(),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        warped = warped * valid.unsqueeze(1).to(warped.dtype)
        return warped.to(previous_memory.dtype), valid

    def forward(
        self,
        current_c1,
        host_logits_low,
        prior_logits_low,
        transportability_low,
        backward_motion_low,
        previous_memory=None,
    ):
        if current_c1.shape[-2:] != host_logits_low.shape[-2:]:
            raise ValueError("C1 and Host logits must share spatial size")
        if host_logits_low.shape != prior_logits_low.shape:
            raise ValueError("Host and prior logits must share shape")
        if transportability_low.shape[-2:] != host_logits_low.shape[-2:]:
            raise ValueError("Transportability and logits must share spatial size")
        if transportability_low.shape[1] != 1:
            raise ValueError("Transportability must be single-channel")

        host_probability = F.softmax(host_logits_low, dim=1)
        prior_probability = F.softmax(prior_logits_low, dim=1)
        prediction_error = host_probability - prior_probability
        appearance = self.feature_projector(current_c1)

        seed = self.current_seed(
            torch.cat(
                (
                    appearance,
                    host_probability,
                    prediction_error,
                    transportability_low,
                ),
                dim=1,
            )
        )

        if previous_memory is None:
            previous_memory = torch.zeros(
                seed.shape[0],
                self.memory_channels,
                seed.shape[2],
                seed.shape[3],
                device=seed.device,
                dtype=seed.dtype,
            )
        warped_memory, valid = self._warp_zero_invalid(
            previous_memory, backward_motion_low
        )

        # Probability L1 lies in [0,2], so 1 - 0.5*L1 is naturally bounded
        # in [0,1] and needs no tuned temperature or threshold.
        agreement = (
            1.0
            - 0.5
            * (host_probability - prior_probability).abs().sum(dim=1, keepdim=True)
        ).clamp(0.0, 1.0)
        reliability = (
            transportability_low
            * valid.unsqueeze(1).to(transportability_low.dtype)
            * agreement
        )
        gated_history = reliability * warped_memory
        memory = self.recurrent(seed, gated_history)

        return {
            "memory": memory,
            "warped_memory": warped_memory,
            "memory_reliability": reliability,
            "agreement": agreement,
            "appearance": appearance,
            "host_probability": host_probability,
            "prior_probability": prior_probability,
            "prediction_error": prediction_error,
        }


class MultiScaleAdaptiveCorrectionReadout(nn.Module):
    """Refine the frozen E1 semantic correction direction.

    The output is DeltaR_t only. It is never allowed to replace Host logits or
    the validated E1 C_t baseline directly. The final 1x1 layer is zero-initialized,
    making the complete C-V3 zero-step output exactly equal to E1 Base.
    """

    def __init__(
        self,
        num_classes=19,
        feature_channels=32,
        memory_channels=64,
        branch_channels=32,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.feature_channels = int(feature_channels)
        self.memory_channels = int(memory_channels)
        self.branch_channels = int(branch_channels)

        input_channels = (
            self.memory_channels
            + self.feature_channels
            + self.num_classes  # P_host
            + self.num_classes  # prediction error
            + 1                 # T
            + self.num_classes  # frozen E1 C_t
        )
        self.pre = nn.Sequential(
            nn.Conv2d(input_channels, self.branch_channels, 1, bias=False),
            nn.GroupNorm(8 if self.branch_channels % 8 == 0 else 1, self.branch_channels),
            nn.SiLU(),
        )
        self.local_branch = nn.Sequential(
            nn.Conv2d(
                self.branch_channels,
                self.branch_channels,
                3,
                padding=1,
                dilation=1,
                bias=False,
            ),
            nn.SiLU(),
        )
        self.context_branch = nn.Sequential(
            nn.Conv2d(
                self.branch_channels,
                self.branch_channels,
                3,
                padding=2,
                dilation=2,
                bias=False,
            ),
            nn.SiLU(),
        )
        self.delta_head = nn.Conv2d(
            2 * self.branch_channels, self.num_classes, 1, bias=True
        )
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward(
        self,
        memory,
        appearance,
        host_probability,
        prediction_error,
        transportability_low,
        semantic_state_low,
    ):
        evidence = torch.cat(
            (
                memory,
                appearance,
                host_probability,
                prediction_error,
                transportability_low,
                semantic_state_low,
            ),
            dim=1,
        )
        shared = self.pre(evidence)
        local = self.local_branch(shared)
        context = self.context_branch(shared)
        return self.delta_head(torch.cat((local, context), dim=1))


class MotionGatedSemanticMemoryRefiner(nn.Module):
    """C-V3 Stage-A trainable semantic-memory + readout module."""

    def __init__(
        self,
        c1_channels=256,
        num_classes=19,
        feature_channels=32,
        memory_channels=64,
        branch_channels=32,
    ):
        super().__init__()
        self.memory = ReliabilityGatedSemanticMemory(
            c1_channels=c1_channels,
            num_classes=num_classes,
            feature_channels=feature_channels,
            memory_channels=memory_channels,
        )
        self.readout = MultiScaleAdaptiveCorrectionReadout(
            num_classes=num_classes,
            feature_channels=feature_channels,
            memory_channels=memory_channels,
            branch_channels=branch_channels,
        )

    def forward(
        self,
        current_c1,
        host_logits_low,
        prior_logits_low,
        transportability_low,
        backward_motion_low,
        semantic_state_low,
        previous_memory=None,
    ):
        row = self.memory(
            current_c1,
            host_logits_low,
            prior_logits_low,
            transportability_low,
            backward_motion_low,
            previous_memory,
        )
        delta_refinement = self.readout(
            row["memory"],
            row["appearance"],
            row["host_probability"],
            row["prediction_error"],
            transportability_low,
            semantic_state_low,
        )
        row["delta_refinement"] = delta_refinement
        return row
