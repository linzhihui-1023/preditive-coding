"""C-V16-A: frozen C-V15 proposal with a fresh reliability gate.

The checkpoint load order is part of the experiment:

1. load the complete C-V15 corrector checkpoint;
2. freeze the proposal generator;
3. reset every trainable module that contributes to the deployed gate.

The C-V15 gate is not just ``reliability_head``.  Its final gate is the
combination of the old reliability prior and the proposal-conditioned
acceptance residual path, so all of those modules are reset here.
"""

import torch
from torch import nn

from .task_space_post_writeback_reliability_feature_correction_aligned import (
    AlignedPostWritebackReliabilityFeatureCorrector,
)
from .task_space_proposal_conditioned_soft_acceptance import (
    ProposalConditionedSoftAcceptanceCorrector,
)


PROPOSAL_MODULE_NAMES = (
    "semantic_encoder",
    "temporal_pre",
    "writeback",
)

# Every module whose output reaches the final deployed g_t and which can carry
# the C-V15 BCE history.  Keep this list explicit so a future gate expansion
# cannot silently make "fresh" an alias for resetting one layer only.
GATE_MODULE_NAMES = (
    "reliability_head",
    "proposal_encoder",
    "host_encoder",
    "acceptance_fusion",
    "acceptance_residual_head",
)


class CV16AFreshGateCorrector(ProposalConditionedSoftAcceptanceCorrector):
    """C-V15 proposal generator, with a newly initialized gate."""

    experiment_name = "c_v16_a_fresh_gate_lseg"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cv15_checkpoint_loaded = False
        self._proposal_frozen = False
        self._gate_reset_after_checkpoint = False
        self._gate_initialization = None

    @classmethod
    def proposal_module_names(cls):
        return PROPOSAL_MODULE_NAMES

    @classmethod
    def gate_module_names(cls):
        return GATE_MODULE_NAMES

    def load_c_v15_state_dict(self, state_dict):
        """Load C-V15 first, then freeze Proposal and reset the Gate."""
        self.load_state_dict(state_dict, strict=True)
        self._cv15_checkpoint_loaded = True
        self.freeze_proposal_generator()
        self.reset_fresh_gate()
        return self

    def freeze_proposal_generator(self):
        if not self._cv15_checkpoint_loaded:
            raise RuntimeError("Proposal cannot be frozen before C-V15 checkpoint loading")
        for name in PROPOSAL_MODULE_NAMES:
            getattr(self, name).requires_grad_(False)
            getattr(self, name).eval()
        self._proposal_frozen = True
        return self

    @staticmethod
    def _reset_conv(module, *, std=None):
        if std is None:
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        else:
            nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    def reset_fresh_gate(self):
        """Reset the deployed gate without making a multilayer net all-zero."""
        if not self._cv15_checkpoint_loaded:
            raise RuntimeError("Fresh Gate reset must happen after C-V15 checkpoint loading")

        # The legacy prior starts exactly neutral: sigmoid(0) = 0.5.
        nn.init.zeros_(self.reliability_head.weight)
        nn.init.zeros_(self.reliability_head.bias)

        # Hidden layers use normal Kaiming/identity GroupNorm initialization.
        # This preserves learning signals through every layer.
        for module in (
            self.proposal_encoder,
            self.host_encoder,
            self.acceptance_fusion,
        ):
            for child in module.modules():
                if isinstance(child, nn.Conv2d):
                    self._reset_conv(child)
                elif isinstance(child, nn.GroupNorm):
                    nn.init.ones_(child.weight)
                    nn.init.zeros_(child.bias)

        # A tiny non-zero residual head keeps upstream gate layers trainable on
        # the first L_seg backward, while bias=0 keeps its initial state neutral.
        self._reset_conv(self.acceptance_residual_head, std=1e-3)
        nn.init.zeros_(self.acceptance_residual_head.bias)

        for name in GATE_MODULE_NAMES:
            getattr(self, name).requires_grad_(True)
        self._gate_reset_after_checkpoint = True
        self._gate_initialization = {
            "reliability_head": "weight=0,bias=0; sigmoid prior=0.5",
            "proposal_encoder": "Kaiming Conv1x1 + configured GroupNorm + SiLU",
            "host_encoder": "Kaiming Conv1x1 + configured GroupNorm + SiLU",
            "acceptance_fusion": "Kaiming Conv3x3 + configured GroupNorm + SiLU",
            "acceptance_residual_head": "Normal(std=1e-3), bias=0",
            "all_zero_multilayer_reset": False,
        }
        return self

    def train(self, mode=True):
        super().train(mode)
        if self._proposal_frozen:
            for name in PROPOSAL_MODULE_NAMES:
                getattr(self, name).eval()
        return self

    def forward(self, *args, **kwargs):
        """Apply the fresh gate with L_seg gradients to all gate modules.

        C-V15 intentionally detached its inherited reliability prior because
        BCE was meant to train only the new residual path.  C-V16-A is a new
        gate, so that detachment would make the reset ``reliability_head`` a
        dead parameter.  Call the frozen Proposal core directly and combine
        its live prior logit with the fresh residual logit.
        """
        current_c4 = kwargs.get("current_c4")
        if current_c4 is None:
            raise ValueError("C-V16-A requires current_c4 as a keyword argument")

        row = AlignedPostWritebackReliabilityFeatureCorrector.forward(
            self, *args, **kwargs
        )
        base_reliability_c4 = row["reliability"]
        base_reliability_low = row["reliability_low"]
        acceptance = self._proposal_conditioned_residual_logit(
            semantic_latent=row["semantic_latent"],
            temporal_latent=row["temporal_latent"],
            bounded_semantic_delta_c4=row["bounded_semantic_delta_c4"],
            c4_channel_rms=row["c4_channel_rms"],
            current_c4=current_c4,
        )

        eps = 1e-6
        base_logit_c4 = torch.logit(
            base_reliability_c4.clamp(eps, 1.0 - eps)
        )
        reliability_logit_c4 = (
            base_logit_c4 + acceptance["acceptance_residual_logit"]
        )
        reliability_c4 = torch.sigmoid(reliability_logit_c4)
        current_c4 = current_c4.detach()
        delta_c4 = row["bounded_semantic_delta_c4"] * reliability_c4
        corrected_c4 = current_c4 + delta_c4

        row.update(
            {
                "base_reliability_low": base_reliability_low,
                "base_reliability_c4": base_reliability_c4,
                "base_reliability_logit_c4": base_logit_c4,
                "reliability_logit": reliability_logit_c4,
                "reliability_c4": reliability_c4,
                "reliability": reliability_c4,
                "delta_c4": delta_c4,
                "corrected_c4": corrected_c4,
                **acceptance,
            }
        )
        return row

    def contract_state(self):
        proposal_parameters = [
            parameter
            for name in PROPOSAL_MODULE_NAMES
            for parameter in getattr(self, name).parameters()
        ]
        gate_parameters = [
            parameter
            for name in GATE_MODULE_NAMES
            for parameter in getattr(self, name).parameters()
        ]
        return {
            "checkpoint_loaded_before_gate_reset": self._cv15_checkpoint_loaded,
            "proposal_generator_frozen": self._proposal_frozen
            and not any(parameter.requires_grad for parameter in proposal_parameters),
            "proposal_parameters_receive_gradient": any(
                parameter.requires_grad for parameter in proposal_parameters
            ),
            "gate_initialized_after_checkpoint_loading": self._gate_reset_after_checkpoint,
            "gate_initialization": self._gate_initialization,
            "gate_parameter_count": sum(parameter.numel() for parameter in gate_parameters),
            "proposal_parameter_count": sum(
                parameter.numel() for parameter in proposal_parameters
            ),
        }
