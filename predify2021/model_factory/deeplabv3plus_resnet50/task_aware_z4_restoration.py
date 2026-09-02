import torch
from torch import nn

from .adapters import UnifiedFeatures, UNIFIED_STATE_CHANNELS
from .semantic_recurrent_predictor import (
    ErrorRegulatedSemanticRestorationPredictor as BaseErrorRegulatedSemanticRestorationPredictor,
    update_error_temporal_statistics,
)


class ContextGatedZ4RestorationHead(nn.Module):
    """Task-oriented Z4 restoration from observation, semantic state and discrepancy.

    The legacy discrepancy projection is preserved as the initial correction path.
    A zero-initialized contextual residual branch learns richer task-relevant
    corrections, while a neutral-initialized magnitude gate controls how strongly
    the learned Z4 residual is applied at each spatial/channel location.
    """

    def __init__(self, channels=UNIFIED_STATE_CHANNELS):
        super().__init__()

        # Keep the legacy parameter name and initialization so legacy checkpoints
        # can still initialize the original discrepancy path exactly.
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        nn.init.dirac_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        with torch.no_grad():
            self.conv.weight.mul_(0.1)

        context_channels = 3 * channels
        self.context_projection = nn.Sequential(
            nn.Conv2d(context_channels, channels, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GELU(),
        )
        self.context_output = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        nn.init.zeros_(self.context_output.weight)

        self.magnitude_gate = nn.Conv2d(context_channels, channels, 1)
        nn.init.zeros_(self.magnitude_gate.weight)
        nn.init.zeros_(self.magnitude_gate.bias)

    def forward(self, observation_z4, semantic_hidden, semantic_discrepancy):
        context = torch.cat((observation_z4, semantic_hidden, semantic_discrepancy), dim=1)
        legacy_delta = self.conv(semantic_discrepancy)
        contextual_delta = self.context_output(self.context_projection(context))

        # 2*sigmoid(0)=1, so initialization is exactly neutral. The gate can
        # subsequently suppress or amplify the correction in [0, 2].
        correction_magnitude_gate = 2.0 * torch.sigmoid(self.magnitude_gate(context))
        restoration_delta_z4 = correction_magnitude_gate * (legacy_delta + contextual_delta)
        return restoration_delta_z4, correction_magnitude_gate, contextual_delta


class ErrorRegulatedSemanticRestorationPredictor(BaseErrorRegulatedSemanticRestorationPredictor):
    """Semantic V3 with a richer task-aware Z4 restoration readout.

    Dynamics, prediction-error encoding, temporal statistics and semantic-state
    update are unchanged. Only the final Z4 correction readout is upgraded.
    """

    def __init__(self, hidden_channels=128, use_error_temporal_stats=False):
        super().__init__(
            hidden_channels=hidden_channels,
            use_error_temporal_stats=use_error_temporal_stats,
        )
        self.semantic_restoration_head = ContextGatedZ4RestorationHead()

    def load_state_dict(self, state_dict, strict=True):
        """Load both new checkpoints and legacy V3 checkpoints safely.

        Legacy checkpoints only contain ``semantic_restoration_head.conv``.
        New contextual/gating parameters are neutral at initialization, so allowing
        only those missing keys preserves the legacy restoration function.
        """
        result = super().load_state_dict(state_dict, strict=False)
        if strict:
            allowed_missing_prefixes = (
                "semantic_restoration_head.context_projection.",
                "semantic_restoration_head.context_output.",
                "semantic_restoration_head.magnitude_gate.",
            )
            unexpected = list(result.unexpected_keys)
            disallowed_missing = [
                key for key in result.missing_keys
                if not key.startswith(allowed_missing_prefixes)
            ]
            if unexpected or disallowed_missing:
                raise RuntimeError(
                    "State dict mismatch: "
                    f"missing={disallowed_missing}, unexpected={unexpected}"
                )
        return result

    def restore_current(
        self,
        observation,
        current_prediction,
        semantic_hidden=None,
        error_temporal_state=None,
        zero_encoded_prediction_error=False,
        disable_error_temporal_stats=False,
        update_error_temporal_state=True,
    ):
        prediction_error_z4 = observation.z4 - current_prediction.z4
        if self.use_error_temporal_stats and update_error_temporal_state:
            error_temporal_state = update_error_temporal_statistics(
                error_temporal_state, prediction_error_z4
            )

        encoded_prediction_error = self.semantic_error_encoder(prediction_error_z4)
        if zero_encoded_prediction_error:
            encoded_prediction_error = torch.zeros_like(encoded_prediction_error)

        semantic_hidden, state_diagnostics = self.semantic_state_cell(
            observation.z4,
            encoded_prediction_error,
            semantic_hidden,
            error_history_statistics=(
                error_temporal_state.as_features()
                if self.use_error_temporal_stats and not disable_error_temporal_stats
                else None
            ),
            disable_error_temporal_stats=disable_error_temporal_stats,
        )

        restoration_delta_z4, correction_magnitude_gate, contextual_delta_z4 = (
            self.semantic_restoration_head(
                observation.z4,
                semantic_hidden,
                state_diagnostics["semantic_discrepancy"],
            )
        )

        restored = UnifiedFeatures(
            observation.z1,
            observation.z2,
            observation.z3,
            observation.z4 + restoration_delta_z4,
        )
        diagnostics = {
            "prediction_error_z4": prediction_error_z4,
            "encoded_prediction_error": encoded_prediction_error,
            "semantic_hidden": semantic_hidden,
            "error_temporal_state": error_temporal_state,
            "history_embedding": state_diagnostics.get("history_embedding"),
            "restoration_delta_z4": restoration_delta_z4,
            "contextual_delta_z4": contextual_delta_z4,
            "correction_magnitude_gate": correction_magnitude_gate,
            **state_diagnostics,
        }
        return restored, semantic_hidden, diagnostics
