import copy

import torch
import torch.nn as nn
from torch.nn import ConvTranspose2d, ReLU

from .targetflow import (
    RealFramePCoderLayerState,
    TargetFlowDynamicErrorConfig,
    TargetFlowFeedbackModule,
    TargetFlowLayerState,
    TemporalPredictionErrorConfig,
    align_source_to_target,
    build_targetflow_error,
    build_targetflow_instant_error,
    build_temporal_prediction_error_state,
    build_targetflow_learn_signal_from_error,
    build_targetflow_local_loss_from_error,
    compute_pcoder_c_sqrt,
    compute_module_grad_stats,
    estimate_local_displacement,
    forward_splat_discrete,
    project_dynamic_error_to_representation,
    run_backward_target_flow,
)


def _pool_spatial(tensor: torch.Tensor):
    if tensor is None:
        return None
    if tensor.dim() <= 2:
        return tensor
    return tensor.mean(dim=tuple(range(2, tensor.dim())))


def _pool_temporal_top_features(tensor: torch.Tensor):
    if tensor is None:
        return None
    if tensor.dim() == 5:
        return tensor.mean(dim=tuple(range(3, tensor.dim())))
    return _pool_spatial(tensor)


def _expand_per_layer_values(values, num_layers, name):
    if isinstance(values, (int, float)):
        return tuple(float(values) for _ in range(num_layers))
    expanded = tuple(float(value) for value in values)
    if len(expanded) != num_layers:
        raise ValueError(f"Expected {num_layers} {name} values, but got {len(expanded)}.")
    return expanded


def _make_feedback_modules():
    """
    Feedback/target modules aligned with the PVGG16 stage boundaries.

    These map:
    stage5 -> stage4
    stage4 -> stage3
    stage3 -> stage2
    stage2 -> stage1
    """
    return nn.ModuleList(
        [
            TargetFlowFeedbackModule(
                ConvTranspose2d(128, 64, kernel_size=(10, 10), stride=(2, 2), padding=(4, 4)),
                ReLU(inplace=True),
            ),
            TargetFlowFeedbackModule(
                ConvTranspose2d(256, 128, kernel_size=(14, 14), stride=(2, 2), padding=(6, 6)),
                ReLU(inplace=True),
            ),
            TargetFlowFeedbackModule(
                ConvTranspose2d(512, 256, kernel_size=(14, 14), stride=(2, 2), padding=(6, 6)),
                ReLU(inplace=True),
            ),
            TargetFlowFeedbackModule(
                ConvTranspose2d(512, 512, kernel_size=(14, 14), stride=(2, 2), padding=(6, 6)),
                ReLU(inplace=True),
            ),
        ]
    )


def _make_input_prediction_module():
    """Original PVGG16 PCoder-1 decoder: Stage 1 -> image space."""
    return TargetFlowFeedbackModule(
        ConvTranspose2d(64, 3, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    )


class ConvGRUErrorTransition(nn.Module):
    """One-cell state transition driven by a matched recurrent input and feedback."""

    def __init__(self, channels: int):
        super().__init__()
        combined_channels = 3 * channels
        self.gates = nn.Conv2d(combined_channels, 2 * channels, kernel_size=1)
        self.candidate = nn.Conv2d(combined_channels, channels, kernel_size=1)

        nn.init.zeros_(self.gates.weight)
        nn.init.zeros_(self.gates.bias)
        nn.init.constant_(self.gates.bias[channels:], -2.0)
        nn.init.zeros_(self.candidate.weight)
        nn.init.zeros_(self.candidate.bias)

    def forward(
        self,
        previous_representation: torch.Tensor,
        error_drive: torch.Tensor,
        feedback_drive: torch.Tensor,
        base_representation: torch.Tensor,
    ):
        reset_gate, update_gate = self.gates(
            torch.cat(
                (
                    previous_representation,
                    error_drive,
                    feedback_drive,
                ),
                dim=1,
            )
        ).chunk(2, dim=1)
        reset_gate = torch.sigmoid(reset_gate)
        update_gate = torch.sigmoid(update_gate)
        candidate_delta = torch.tanh(
            self.candidate(
                torch.cat(
                    (
                        error_drive,
                        feedback_drive,
                        reset_gate * previous_representation,
                    ),
                    dim=1,
                )
            )
        )
        return base_representation + update_gate * candidate_delta


class PVGG16TargetFlow(nn.Module):
    """
    PVGG16 with a real-frame predictive-coding recurrence.

    ``task='real_frame_pc'`` performs one state update per observed video frame
    and never resolves a future-frame target. Legacy motion/future-feature tasks
    remain loadable only so previously versioned experiments stay reproducible.
    """

    def __init__(
        self,
        backbone: nn.Module,
        target_flow_mode: str = "recursive",
        compute_local_param_grads: bool = False,
        temporal_target_mode: str = "next_top",
        temporal_horizons=(1,),
        dynamic_error: bool = True,
        error_state_mode: str = None,
        local_loss_error_source: str = "instant",
        error_sample_time: float = 0.1035,
        error_time_constant=0.5,
        error_gain=1.0,
        temporal_error_sample_time: float = 1.0,
        temporal_error_time_constant: float = 1.0,
        temporal_error_gain: float = 1.0,
        task: str = "real_frame_pc",
        pc_ff_multiplier=(0.2, 0.4, 0.4, 0.5, 0.6),
        pc_fb_multiplier=(0.05, 0.1, 0.1, 0.1, 0.0),
        pc_error_multiplier=(0.01, 0.01, 0.01, 0.01, 0.01),
        real_frame_transition_mode: str = "predify",
        real_frame_recurrent_error_input: str = "dynamic",
        future_feature_stage: int = 5,
        future_feature_history_mode: str = "none",
        future_feature_temporal_fusion_mode: str = "none",
        future_feature_predictor_kernel_size: int = 1,
        future_feature_prediction_form: str = "current_residual",
        future_motion_radius: int = 1,
        future_motion_patch_size: int = 3,
    ):
        super().__init__()
        self.backbone = copy.deepcopy(backbone)
        self.backbone.eval()

        for module in self.backbone.modules():
            if hasattr(module, "inplace"):
                module.inplace = False

        features = self.backbone.features
        self.forward_stages = nn.ModuleList(
            [
                nn.Sequential(*features[:4]),
                nn.Sequential(*features[4:9]),
                nn.Sequential(*features[9:16]),
                nn.Sequential(*features[16:23]),
                nn.Sequential(*features[23:30]),
            ]
        )
        self.forward_tail = nn.Sequential(*features[30:])
        self.avgpool = copy.deepcopy(self.backbone.avgpool)
        self.classifier = copy.deepcopy(self.backbone.classifier)

        # Ordered low->high so index i maps target[i+1] -> target[i].
        self.feedback_modules = _make_feedback_modules()
        self.number_of_layers = len(self.forward_stages)
        self.number_of_pcoders = self.number_of_layers
        self.target_flow_mode = target_flow_mode
        self.compute_local_param_grads = compute_local_param_grads
        self.error_state_mode = error_state_mode or ("ema" if dynamic_error else "instant")
        self.error_state_mode = {
            "lag1": "two_tap",
        }.get(self.error_state_mode, self.error_state_mode)
        if self.error_state_mode not in {"instant", "ema", "two_tap"}:
            raise ValueError(f"Unsupported error_state_mode: {self.error_state_mode}")
        if local_loss_error_source not in {"instant", "state"}:
            raise ValueError(
                f"Unsupported local_loss_error_source: {local_loss_error_source}"
            )
        self.local_loss_error_source = local_loss_error_source
        expanded_time_constants = _expand_per_layer_values(
            error_time_constant,
            self.number_of_layers,
            "error_time_constant",
        )
        expanded_error_gains = _expand_per_layer_values(
            error_gain,
            self.number_of_layers,
            "error_gain",
        )
        self.dynamic_error_config = TargetFlowDynamicErrorConfig(
            sample_time=float(error_sample_time),
            time_constant=expanded_time_constants,
            error_gain=expanded_error_gains,
            enabled=self.error_state_mode != "instant",
        )
        self.register_buffer(
            "error_time_constants",
            torch.tensor(expanded_time_constants, dtype=torch.float32),
        )
        self.register_buffer(
            "error_gains",
            torch.tensor(expanded_error_gains, dtype=torch.float32),
        )
        self.temporal_error_config = TemporalPredictionErrorConfig(
            sample_time=float(temporal_error_sample_time),
            time_constant=float(temporal_error_time_constant),
            error_gain=float(temporal_error_gain),
        )
        self.register_buffer(
            "temporal_error_time_constant",
            torch.tensor(float(temporal_error_time_constant), dtype=torch.float32),
        )
        self.register_buffer(
            "temporal_error_gain",
            torch.tensor(float(temporal_error_gain), dtype=torch.float32),
        )
        if temporal_target_mode not in {"next_top", "delta_top", "ego_motion"}:
            raise ValueError(f"Unsupported temporal_target_mode: {temporal_target_mode}")
        temporal_horizons = tuple(int(horizon) for horizon in temporal_horizons)
        if not temporal_horizons:
            raise ValueError("At least one temporal horizon is required.")
        if any(horizon <= 0 for horizon in temporal_horizons):
            raise ValueError(f"Temporal horizons must be positive: {temporal_horizons}")
        self.temporal_target_mode = temporal_target_mode
        self.temporal_horizons = temporal_horizons
        self.num_temporal_horizons = len(self.temporal_horizons)
        self.stage_channels = (64, 128, 256, 512, 512)
        if task not in {"real_frame_pc", "motion", "future_feature"}:
            raise ValueError(f"Unsupported task: {task}")
        future_feature_history_mode = {
            "instant": "latest",
            "lag1": "two_tap",
        }.get(future_feature_history_mode, future_feature_history_mode)
        if future_feature_history_mode not in {
            "none",
            "latest",
            "two_tap",
            "recursive",
            "temporal_error",
            "aligned_difference",
            "copy_current",
        }:
            raise ValueError(
                "Unsupported future_feature_history_mode: "
                f"{future_feature_history_mode}"
            )
        self.task = task
        if real_frame_transition_mode not in {"predify", "convgru_error"}:
            raise ValueError(
                "Unsupported real_frame_transition_mode: "
                f"{real_frame_transition_mode}"
            )
        if self.task != "real_frame_pc" and real_frame_transition_mode != "predify":
            raise ValueError(
                "real_frame_transition_mode is configurable only for real_frame_pc."
            )
        self.real_frame_transition_mode = real_frame_transition_mode
        if real_frame_recurrent_error_input not in {
            "dynamic",
            "observation",
            "zeroed",
        }:
            raise ValueError(
                "Unsupported real_frame_recurrent_error_input: "
                f"{real_frame_recurrent_error_input}"
            )
        if (
            real_frame_recurrent_error_input != "dynamic"
            and self.real_frame_transition_mode != "convgru_error"
        ):
            raise ValueError("Matched recurrent input modes require convgru_error mode.")
        self.real_frame_recurrent_error_input = real_frame_recurrent_error_input
        pc_ff_multipliers = _expand_per_layer_values(
            pc_ff_multiplier,
            self.number_of_layers,
            "pc_ff_multiplier",
        )
        pc_fb_multipliers = _expand_per_layer_values(
            pc_fb_multiplier,
            self.number_of_layers,
            "pc_fb_multiplier",
        )
        pc_error_multipliers = _expand_per_layer_values(
            pc_error_multiplier,
            self.number_of_layers,
            "pc_error_multiplier",
        )
        if any(value < 0.0 for value in pc_ff_multipliers):
            raise ValueError("pc_ff_multiplier values must be non-negative.")
        if any(value < 0.0 for value in pc_fb_multipliers):
            raise ValueError("pc_fb_multiplier values must be non-negative.")
        if any(value < 0.0 for value in pc_error_multipliers):
            raise ValueError("pc_error_multiplier values must be non-negative.")
        if any(
            ff_value + fb_value > 1.0
            for ff_value, fb_value in zip(pc_ff_multipliers, pc_fb_multipliers)
        ):
            raise ValueError(
                "Each pc_ff_multiplier + pc_fb_multiplier must be at most 1."
            )
        if pc_fb_multipliers[-1] != 0.0:
            raise ValueError("The highest PCoder layer cannot receive feedback.")
        self.register_buffer(
            "pc_ff_multipliers",
            torch.tensor(pc_ff_multipliers, dtype=torch.float32),
        )
        self.register_buffer(
            "pc_fb_multipliers",
            torch.tensor(pc_fb_multipliers, dtype=torch.float32),
        )
        self.register_buffer(
            "pc_error_multipliers",
            torch.tensor(pc_error_multipliers, dtype=torch.float32),
        )
        if self.task == "real_frame_pc":
            self.register_buffer(
                "pc_error_c_sqrt",
                torch.full((self.number_of_layers,), -1.0, dtype=torch.float32),
            )
        else:
            self.pc_error_c_sqrt = None
        if self.task == "real_frame_pc" and self.error_state_mode != "ema":
            raise ValueError(
                "real_frame_pc requires error_state_mode='ema' so there is one "
                "dynamic Target Flow error chain."
            )
        self.future_feature_stage = int(future_feature_stage)
        if self.future_feature_stage not in {3, 4, 5}:
            raise ValueError(
                "future_feature_stage must be one of 3, 4, or 5, got "
                f"{self.future_feature_stage}."
            )
        self.future_feature_stage_index = self.future_feature_stage - 1
        self.future_feature_channels = self.stage_channels[
            self.future_feature_stage_index
        ]
        self.future_feature_history_mode = future_feature_history_mode
        if future_feature_temporal_fusion_mode not in {
            "none",
            "two_frame_residual",
            "aligned_two_frame_residual",
        }:
            raise ValueError(
                "Unsupported future_feature_temporal_fusion_mode: "
                f"{future_feature_temporal_fusion_mode}"
            )
        if (
            future_feature_temporal_fusion_mode != "none"
            and future_feature_history_mode != "none"
        ):
            raise ValueError(
                "Two-frame temporal fusion cannot be combined with another "
                "future-feature history mode."
            )
        self.future_feature_temporal_fusion_mode = (
            future_feature_temporal_fusion_mode
        )
        if self.task == "real_frame_pc" and (
            self.future_feature_history_mode != "none"
            or self.future_feature_temporal_fusion_mode != "none"
        ):
            raise ValueError(
                "real_frame_pc does not use future-feature history or fusion modules."
            )
        if future_feature_predictor_kernel_size not in {1, 3}:
            raise ValueError(
                "future_feature_predictor_kernel_size must be 1 or 3, got "
                f"{future_feature_predictor_kernel_size}."
            )
        self.future_feature_predictor_kernel_size = (
            future_feature_predictor_kernel_size
        )
        if future_feature_prediction_form not in {
            "current_residual",
            "historical_warp",
            "historical_warp_residual",
        }:
            raise ValueError(
                "Unsupported future_feature_prediction_form: "
                f"{future_feature_prediction_form}"
            )
        if int(future_motion_radius) <= 0:
            raise ValueError("future_motion_radius must be positive.")
        if (
            int(future_motion_patch_size) <= 0
            or int(future_motion_patch_size) % 2 == 0
        ):
            raise ValueError("future_motion_patch_size must be a positive odd integer.")
        self.future_feature_prediction_form = future_feature_prediction_form
        if (
            self.future_feature_temporal_fusion_mode != "none"
            and self.future_feature_prediction_form != "current_residual"
        ):
            raise ValueError(
                "Two-frame temporal fusion requires current_residual prediction."
            )
        if (
            self.future_feature_history_mode == "aligned_difference"
            and self.future_feature_prediction_form != "current_residual"
        ):
            raise ValueError(
                "Aligned temporal difference requires current_residual prediction."
            )
        self.future_motion_radius = int(future_motion_radius)
        self.future_motion_patch_size = int(future_motion_patch_size)
        self.temporal_target_dim = 2 if self.temporal_target_mode == "ego_motion" else self.stage_channels[-1]
        self.input_prediction_module = None
        self.recurrent_transition_modules = None
        self.temporal_predictor = None
        self.future_feature_predictor = None
        if self.task == "real_frame_pc":
            self.input_prediction_module = _make_input_prediction_module()
            if self.real_frame_transition_mode == "convgru_error":
                self.recurrent_transition_modules = nn.ModuleList(
                    ConvGRUErrorTransition(channels)
                    for channels in self.stage_channels
                )
        else:
            temporal_context_dim = self.stage_channels[-1] + 2 * sum(self.stage_channels)
            self.temporal_predictor = nn.Sequential(
                nn.Linear(temporal_context_dim, 1024),
                nn.ReLU(inplace=False),
                nn.Linear(1024, self.temporal_target_dim * self.num_temporal_horizons),
            )
            self.future_feature_predictor = nn.Sequential(
                nn.Conv2d(
                    2 * self.future_feature_channels,
                    2 * self.future_feature_channels,
                    kernel_size=future_feature_predictor_kernel_size,
                    padding=future_feature_predictor_kernel_size // 2,
                ),
                nn.ReLU(inplace=False),
                nn.Conv2d(
                    2 * self.future_feature_channels,
                    self.future_feature_channels,
                    kernel_size=1,
                ),
            )
        self.temporal_fusion_module = None
        if self.future_feature_temporal_fusion_mode in {
            "two_frame_residual",
            "aligned_two_frame_residual",
        }:
            self.temporal_fusion_module = nn.Sequential(
                nn.Conv2d(
                    2 * self.future_feature_channels,
                    self.future_feature_channels,
                    1,
                ),
                nn.ReLU(inplace=False),
                nn.Conv2d(
                    self.future_feature_channels,
                    self.future_feature_channels,
                    1,
                ),
            )
            nn.init.zeros_(self.temporal_fusion_module[-1].weight)
            nn.init.zeros_(self.temporal_fusion_module[-1].bias)
        self.layer_states = []
        self.representation_state_memory = [
            None for _ in range(self.number_of_layers)
        ]
        self.error_state_memory = [None for _ in range(self.number_of_layers)]
        self.instant_error_state_memory = [None for _ in range(self.number_of_layers)]
        self.recursive_error_state_memory = [None for _ in range(self.number_of_layers)]
        self.two_tap_error_state_memory = [None for _ in range(self.number_of_layers)]
        self.temporal_prediction_error_memory = None
        self.temporal_error_state_memory = None
        self.temporal_error_state_update_count = 0
        self.future_feature_previous_prediction_stage_memory = None
        self.prediction_state_memory = [None for _ in range(self.number_of_layers)]
        self.temporal_context = None
        self.temporal_prediction = None
        self.temporal_target = None
        self.future_prediction_outputs = None
        self.real_frame_update_count = 0
        self.recurrence_outputs = None
        self.recurrent_transition_losses = []
        self.recurrent_transition_predictions = []

        if self.task == "real_frame_pc":
            for parameter in self.parameters():
                parameter.requires_grad_(False)
            if self.recurrent_transition_modules is not None:
                for parameter in self.recurrent_transition_modules.parameters():
                    parameter.requires_grad_(True)

    def reset(self):
        self.layer_states = []
        self.representation_state_memory = [
            None for _ in range(self.number_of_layers)
        ]
        self.error_state_memory = [None for _ in range(self.number_of_layers)]
        self.instant_error_state_memory = [None for _ in range(self.number_of_layers)]
        self.recursive_error_state_memory = [None for _ in range(self.number_of_layers)]
        self.two_tap_error_state_memory = [None for _ in range(self.number_of_layers)]
        self.temporal_prediction_error_memory = None
        self.temporal_error_state_memory = None
        self.temporal_error_state_update_count = 0
        self.future_feature_previous_prediction_stage_memory = None
        self.prediction_state_memory = [None for _ in range(self.number_of_layers)]
        self.temporal_context = None
        self.temporal_prediction = None
        self.temporal_target = None
        self.future_prediction_outputs = None
        self.real_frame_update_count = 0
        self.recurrence_outputs = None
        self.recurrent_transition_losses = []
        self.recurrent_transition_predictions = []

    @staticmethod
    def _resolve_memory(memory, reference_tensor: torch.Tensor):
        if memory is None or memory.shape != reference_tensor.shape:
            return None
        return memory.to(reference_tensor.device, reference_tensor.dtype)

    def _resolve_previous_error_state(self, layer_index: int, reference_tensor: torch.Tensor):
        return self._resolve_memory(self.error_state_memory[layer_index], reference_tensor)

    def _resolve_previous_instant_error_state(
        self,
        layer_index: int,
        reference_tensor: torch.Tensor,
    ):
        return self._resolve_memory(
            self.instant_error_state_memory[layer_index],
            reference_tensor,
        )

    def _resolve_previous_prediction_state(self, layer_index: int, reference_tensor: torch.Tensor):
        return self._resolve_memory(
            self.prediction_state_memory[layer_index],
            reference_tensor,
        )

    def _prediction_module_for_layer(self, layer_index: int):
        if layer_index == 0:
            return self.input_prediction_module
        return self.feedback_modules[layer_index - 1]

    def _real_frame_feature_targets(self, x: torch.Tensor):
        prediction_targets = [x]
        feedforward_drives = []
        current = x
        with torch.no_grad():
            for layer_index, stage in enumerate(self.forward_stages):
                current = stage(current)
                feedforward_drives.append(current)
                if layer_index + 1 < self.number_of_layers:
                    prediction_targets.append(current)
        return tuple(prediction_targets), tuple(feedforward_drives)

    def _step_real_frame_recurrence(self, x: torch.Tensor):
        """Advance every PCoder exactly once using only the observed frame."""
        if self.task != "real_frame_pc":
            raise RuntimeError("Real-frame recurrence requires task='real_frame_pc'.")

        previous_representations = tuple(self.representation_state_memory)
        previous_predictions = tuple(self.prediction_state_memory)
        previous_dynamic_errors = tuple(self.error_state_memory)
        frame_index = self.real_frame_update_count + 1
        current_input = x
        states = []
        transition_predictions = []
        learned_prediction_targets = None
        learned_feedforward_drives = None
        if self.real_frame_transition_mode == "convgru_error":
            learned_prediction_targets, learned_feedforward_drives = (
                self._real_frame_feature_targets(x)
            )

        for layer_index, stage in enumerate(self.forward_stages):
            if self.real_frame_transition_mode == "convgru_error":
                feedforward_drive = learned_feedforward_drives[layer_index]
                prediction_target = learned_prediction_targets[layer_index]
            else:
                with torch.no_grad():
                    feedforward_drive = stage(current_input)
                prediction_target = x if layer_index == 0 else states[-1].representation

            previous_representation = self._resolve_memory(
                previous_representations[layer_index],
                feedforward_drive,
            )
            previous_prediction = self._resolve_memory(
                previous_predictions[layer_index],
                prediction_target,
            )
            previous_dynamic_error = self._resolve_memory(
                previous_dynamic_errors[layer_index],
                prediction_target,
            )
            previous_feedback_prediction = None
            if layer_index + 1 < self.number_of_layers:
                previous_feedback_prediction = self._resolve_memory(
                    previous_predictions[layer_index + 1],
                    feedforward_drive,
                )

            prediction_module = self._prediction_module_for_layer(layer_index)
            error_correction = None
            error_scale = None
            c_sqrt = None
            if self.real_frame_transition_mode == "convgru_error":
                if previous_representation is None:
                    representation = feedforward_drive
                    instant_error = torch.zeros_like(prediction_target)
                    dynamic_error = torch.zeros_like(prediction_target)
                else:
                    fb_multiplier = self.pc_fb_multipliers[layer_index].to(
                        feedforward_drive
                    )
                    with torch.no_grad():
                        instant_error = build_targetflow_instant_error(
                            previous_prediction.detach(),
                            prediction_target.detach(),
                        )
                        dynamic_error = build_targetflow_error(
                            previous_prediction.detach(),
                            prediction_target.detach(),
                            previous_error=self._resolve_memory(
                                previous_dynamic_error,
                                instant_error,
                            ),
                            sample_time=self.dynamic_error_config.sample_time,
                            time_constant=float(
                                self.error_time_constants[layer_index].item()
                            ),
                            error_gain=float(self.error_gains[layer_index].item()),
                            mode="ema",
                        )
                    base_representation = previous_representation
                    feedback_drive = torch.zeros_like(previous_representation)
                    if previous_feedback_prediction is not None:
                        feedback_drive = previous_feedback_prediction
                        base_representation = base_representation + fb_multiplier * (
                            previous_feedback_prediction - previous_representation
                        )
                    with torch.no_grad():
                        if self.real_frame_recurrent_error_input == "observation":
                            recurrent_drive = feedforward_drive
                        else:
                            recurrent_drive = stage(dynamic_error)
                            if self.real_frame_recurrent_error_input == "zeroed":
                                recurrent_drive = torch.zeros_like(recurrent_drive)
                    representation = self.recurrent_transition_modules[layer_index](
                        previous_representation,
                        recurrent_drive,
                        feedback_drive,
                        base_representation,
                    )
                module_output = prediction_module(representation)
                prediction = (
                    module_output[-1]
                    if isinstance(module_output, tuple)
                    else module_output
                )
                if previous_representation is not None:
                    transition_predictions.append(prediction)
            else:
                c_sqrt = self.pc_error_c_sqrt[layer_index]
                error_correction = project_dynamic_error_to_representation(
                    prediction_module,
                    previous_representation,
                    previous_prediction,
                    previous_dynamic_error,
                    None if float(c_sqrt.item()) < 0.0 else c_sqrt,
                )

                if previous_representation is None:
                    representation = feedforward_drive
                else:
                    ff_multiplier = self.pc_ff_multipliers[layer_index].to(
                        feedforward_drive
                    )
                    fb_multiplier = self.pc_fb_multipliers[layer_index].to(
                        feedforward_drive
                    )
                    error_multiplier = self.pc_error_multipliers[layer_index].to(
                        feedforward_drive
                    )
                    representation = previous_representation + ff_multiplier * (
                        feedforward_drive - previous_representation
                    )
                    if previous_feedback_prediction is not None:
                        representation = representation + fb_multiplier * (
                            previous_feedback_prediction - previous_representation
                        )
                    if error_correction is not None:
                        representation = (
                            representation - error_multiplier * error_correction
                        )

                if float(c_sqrt.item()) < 0.0:
                    calibrated_c_sqrt = compute_pcoder_c_sqrt(
                        prediction_module,
                        representation,
                    )
                    self.pc_error_c_sqrt[layer_index].copy_(calibrated_c_sqrt)
                    c_sqrt = self.pc_error_c_sqrt[layer_index]
                error_scale = representation.new_tensor(
                    float(prediction_target.numel())
                ) / c_sqrt.to(representation)

                with torch.no_grad():
                    module_output = prediction_module(representation)
                    prediction = (
                        module_output[-1]
                        if isinstance(module_output, tuple)
                        else module_output
                    )

            if self.real_frame_transition_mode != "convgru_error":
                with torch.no_grad():
                    instant_error = build_targetflow_instant_error(
                        prediction.detach(),
                        prediction_target.detach(),
                    )
                    dynamic_error = build_targetflow_error(
                        prediction.detach(),
                        prediction_target.detach(),
                        previous_error=self._resolve_memory(
                            previous_dynamic_error,
                            instant_error,
                        ),
                        sample_time=self.dynamic_error_config.sample_time,
                        time_constant=float(
                            self.error_time_constants[layer_index].item()
                        ),
                        error_gain=float(self.error_gains[layer_index].item()),
                        mode="ema",
                    )

            state = RealFramePCoderLayerState(
                layer_index=layer_index + 1,
                frame_index=frame_index,
                feedforward_drive=feedforward_drive.detach(),
                previous_representation=(
                    None
                    if previous_representation is None
                    else previous_representation.detach()
                ),
                previous_prediction=(
                    None
                    if previous_prediction is None
                    else previous_prediction.detach()
                ),
                previous_feedback_prediction=(
                    None
                    if previous_feedback_prediction is None
                    else previous_feedback_prediction.detach()
                ),
                previous_dynamic_error=(
                    None
                    if previous_dynamic_error is None
                    else previous_dynamic_error.detach()
                ),
                error_correction=(
                    None if error_correction is None else error_correction.detach()
                ),
                error_scale=(None if error_scale is None else error_scale.detach()),
                c_sqrt=(None if c_sqrt is None else c_sqrt.detach().clone()),
                representation=representation.detach(),
                prediction_target=prediction_target.detach(),
                prediction=prediction.detach(),
                instant_error=instant_error.detach(),
                dynamic_error=dynamic_error.detach(),
            )
            states.append(state)
            current_input = state.representation

        self.layer_states = states
        self.recurrent_transition_losses = []
        self.recurrent_transition_predictions = transition_predictions
        for layer_index, state in enumerate(states):
            self.representation_state_memory[layer_index] = (
                state.representation.detach()
            )
            self.prediction_state_memory[layer_index] = state.prediction.detach()
            self.instant_error_state_memory[layer_index] = (
                state.instant_error.detach()
            )
            self.error_state_memory[layer_index] = state.dynamic_error.detach()

        self.real_frame_update_count = frame_index
        self.recurrence_outputs = {
            "frame_index": frame_index,
            "transition_mode": self.real_frame_transition_mode,
            "updates_per_layer": tuple(1 for _ in states),
            "used_future_frame": False,
            "parameter_update": False,
            "online_adaptation": False,
            "layer_states": tuple(states),
        }

        current = self.forward_tail(states[-1].representation)
        current = self.avgpool(current)
        current = torch.flatten(current, 1)
        with torch.no_grad():
            return self.classifier(current)

    def _resolve_future_feature_history(self, current_top: torch.Tensor):
        mode = self.future_feature_history_mode
        if mode in {"none", "copy_current"}:
            return torch.zeros_like(current_top)
        if mode == "temporal_error":
            history = self._resolve_memory(self.temporal_error_state_memory, current_top)
            return torch.zeros_like(current_top) if history is None else history.detach()
        memory_by_mode = {
            "latest": self.instant_error_state_memory,
            "two_tap": self.two_tap_error_state_memory,
            "recursive": self.recursive_error_state_memory,
        }
        history = self._resolve_memory(
            memory_by_mode[mode][self.future_feature_stage_index],
            current_top,
        )
        return torch.zeros_like(current_top) if history is None else history.detach()

    def _build_historical_warp_base(self, current_top: torch.Tensor):
        previous_top = self._resolve_memory(
            self.future_feature_previous_prediction_stage_memory,
            current_top,
        )
        if previous_top is None:
            return current_top, None, None, None
        with torch.no_grad():
            motion = estimate_local_displacement(
                previous_top.detach(),
                current_top.detach(),
                radius=self.future_motion_radius,
                patch_size=self.future_motion_patch_size,
            )
        splat = forward_splat_discrete(
            current_top,
            motion["dy"],
            motion["dx"],
            radius=self.future_motion_radius,
        )
        return (
            splat["warped"],
            motion["dy"].detach(),
            motion["dx"].detach(),
            splat,
        )

    def _build_aligned_difference_history(self, current_top: torch.Tensor):
        previous_top = self._resolve_memory(
            self.future_feature_previous_prediction_stage_memory,
            current_top,
        )
        if previous_top is None:
            return torch.zeros_like(current_top), None, None, None
        previous_top = previous_top.detach()
        with torch.no_grad():
            alignment_diagnostics = align_source_to_target(
                previous_top,
                current_top.detach(),
                radius=self.future_motion_radius,
                patch_size=self.future_motion_patch_size,
            )
            aligned_previous_top = alignment_diagnostics[
                "aligned_source"
            ].detach()
            temporal_difference = (
                current_top.detach() - aligned_previous_top
            ).detach()
        return (
            temporal_difference,
            previous_top,
            aligned_previous_top,
            alignment_diagnostics,
        )

    def _build_temporal_fusion_base(self, current_top: torch.Tensor):
        previous_top = self._resolve_memory(
            self.future_feature_previous_prediction_stage_memory,
            current_top,
        )
        if previous_top is None:
            return (
                current_top,
                None,
                torch.zeros_like(current_top),
                False,
                None,
                None,
                None,
            )
        previous_top = previous_top.detach()
        fusion_previous_top = previous_top
        aligned_previous_top = None
        alignment_diagnostics = None
        if (
            self.future_feature_temporal_fusion_mode
            == "aligned_two_frame_residual"
        ):
            with torch.no_grad():
                alignment_diagnostics = align_source_to_target(
                    previous_top,
                    current_top.detach(),
                    radius=self.future_motion_radius,
                    patch_size=self.future_motion_patch_size,
                )
            aligned_previous_top = alignment_diagnostics[
                "aligned_source"
            ].detach()
            fusion_previous_top = aligned_previous_top
        fusion_residual = self.temporal_fusion_module(
            torch.cat([fusion_previous_top, current_top], dim=1)
        )
        return (
            current_top + fusion_residual,
            fusion_previous_top,
            fusion_residual,
            True,
            previous_top,
            aligned_previous_top,
            alignment_diagnostics,
        )

    def _predict_future_feature(self, current_feature: torch.Tensor):
        current_top = current_feature
        fusion_previous_top = None
        raw_fusion_previous_top = None
        aligned_previous_top = None
        alignment_diagnostics = None
        aligned_difference_raw_previous_top = None
        aligned_temporal_difference_top = None
        if self.future_feature_history_mode == "aligned_difference":
            (
                history_top,
                aligned_difference_raw_previous_top,
                aligned_previous_top,
                alignment_diagnostics,
            ) = self._build_aligned_difference_history(current_top)
            aligned_temporal_difference_top = history_top
        else:
            history_top = self._resolve_future_feature_history(current_top)
        fusion_residual = torch.zeros_like(current_top)
        temporal_fusion_applied = False
        if self.future_feature_history_mode == "copy_current":
            prediction_base = current_top
            predicted_residual = torch.zeros_like(current_top)
            motion_dy = None
            motion_dx = None
            warp_diagnostics = None
        else:
            form = self.future_feature_prediction_form
            if self.future_feature_temporal_fusion_mode in {
                "two_frame_residual",
                "aligned_two_frame_residual",
            }:
                (
                    prediction_base,
                    fusion_previous_top,
                    fusion_residual,
                    temporal_fusion_applied,
                    raw_fusion_previous_top,
                    aligned_previous_top,
                    alignment_diagnostics,
                ) = self._build_temporal_fusion_base(current_top)
                motion_dy = (
                    None
                    if alignment_diagnostics is None
                    else alignment_diagnostics["dy"].detach()
                )
                motion_dx = (
                    None
                    if alignment_diagnostics is None
                    else alignment_diagnostics["dx"].detach()
                )
                warp_diagnostics = None
            elif form == "current_residual":
                prediction_base = current_top
                motion_dy = (
                    None
                    if alignment_diagnostics is None
                    else alignment_diagnostics["dy"].detach()
                )
                motion_dx = (
                    None
                    if alignment_diagnostics is None
                    else alignment_diagnostics["dx"].detach()
                )
                warp_diagnostics = None
            else:
                (
                    prediction_base,
                    motion_dy,
                    motion_dx,
                    warp_diagnostics,
                ) = self._build_historical_warp_base(current_top)
            if form == "historical_warp":
                predicted_residual = torch.zeros_like(current_top)
            else:
                predictor_input = torch.cat([prediction_base, history_top], dim=1)
                predicted_residual = self.future_feature_predictor(predictor_input)
        predicted_future = prediction_base + predicted_residual
        predicted_delta = predicted_future - current_top
        self.future_prediction_outputs = {
            "future_feature_stage": self.future_feature_stage,
            "target_flow_top_stage": self.number_of_layers,
            "current_prediction_feature": current_top,
            "history_prediction_feature": history_top,
            "prediction_base_feature": prediction_base,
            "predicted_residual_feature": predicted_residual,
            "predicted_delta_feature": predicted_delta,
            "predicted_future_feature": predicted_future,
            "previous_prediction_stage_feature": (
                aligned_difference_raw_previous_top
                if aligned_difference_raw_previous_top is not None
                else raw_fusion_previous_top
            ),
            "current_top": current_top,
            "history_top": history_top,
            "fusion_previous_top": fusion_previous_top,
            "raw_fusion_previous_top": raw_fusion_previous_top,
            "aligned_previous_top": aligned_previous_top,
            "alignment_applied": aligned_previous_top is not None,
            "aligned_difference_raw_previous_top": (
                aligned_difference_raw_previous_top
            ),
            "aligned_difference_previous_top": (
                aligned_previous_top
                if self.future_feature_history_mode == "aligned_difference"
                else None
            ),
            "aligned_temporal_difference_top": aligned_temporal_difference_top,
            "aligned_difference_applied": (
                aligned_temporal_difference_top is not None
                and aligned_previous_top is not None
            ),
            "alignment_patch_matching_cost": (
                None
                if alignment_diagnostics is None
                else alignment_diagnostics["patch_matching_cost"].detach()
            ),
            "fusion_residual_top": fusion_residual,
            "fused_top": prediction_base,
            "temporal_fusion_applied": temporal_fusion_applied,
            "prediction_base_top": prediction_base,
            "predicted_residual_top": predicted_residual,
            "predicted_delta_top": predicted_delta,
            "predicted_future_top": predicted_future,
            "motion_dy": motion_dy,
            "motion_dx": motion_dx,
            "warp_coverage_fraction": (
                None
                if warp_diagnostics is None
                else warp_diagnostics["coverage_fraction"].detach()
            ),
            "warp_collision_fraction": (
                None
                if warp_diagnostics is None
                else warp_diagnostics["collision_fraction"].detach()
            ),
            "future_top_target": None,
            "target_delta_top": None,
            "target_residual_top": None,
            "prediction_error_top": None,
            "target_flow_top_target": None,
            "future_prediction_target": None,
            "target_delta_prediction_feature": None,
            "target_residual_prediction_feature": None,
            "prediction_error_feature": None,
            "instantaneous_prediction_error_feature": None,
            "previous_dynamic_prediction_error_state_feature": None,
            "dynamic_prediction_error_state_feature": None,
            "temporal_error_state_update_index": None,
        }

    def _run_forward_stages(self, x: torch.Tensor):
        forward_inputs = []
        forward_outputs = []

        current = x
        for stage in self.forward_stages:
            forward_inputs.append(current)
            current = stage(current)
            forward_outputs.append(current)
        return forward_inputs, forward_outputs

    def extract_forward_feature(
        self,
        x: torch.Tensor,
        stage: int,
        detach: bool = True,
    ):
        stage = int(stage)
        if stage < 1 or stage > self.number_of_layers:
            raise ValueError(
                f"stage must be between 1 and {self.number_of_layers}, got {stage}."
            )
        if detach:
            with torch.no_grad():
                _, forward_outputs = self._run_forward_stages(x)
                return forward_outputs[stage - 1]

        _, forward_outputs = self._run_forward_stages(x)
        return forward_outputs[stage - 1]

    def extract_forward_features_at_stages(
        self,
        x: torch.Tensor,
        stages,
        detach: bool = True,
    ):
        stages = tuple(dict.fromkeys(int(stage) for stage in stages))
        if not stages:
            raise ValueError("At least one forward stage must be requested.")
        if any(stage < 1 or stage > self.number_of_layers for stage in stages):
            raise ValueError(
                f"stages must be between 1 and {self.number_of_layers}, got {stages}."
            )

        def extract():
            _, forward_outputs = self._run_forward_stages(x)
            return {stage: forward_outputs[stage - 1] for stage in stages}

        if detach:
            with torch.no_grad():
                return extract()
        return extract()

    def extract_prediction_stage_feature(
        self,
        x: torch.Tensor,
        detach: bool = True,
    ):
        return self.extract_forward_feature(
            x,
            stage=self.future_feature_stage,
            detach=detach,
        )

    def extract_top_forward_feature(self, x: torch.Tensor, detach: bool = True):
        return self.extract_forward_feature(
            x,
            stage=self.number_of_layers,
            detach=detach,
        )

    def extract_top_forward_features(self, x: torch.Tensor, detach: bool = True):
        if x.dim() == 4:
            return self.extract_top_forward_feature(x, detach=detach)
        if x.dim() != 5:
            raise ValueError(f"Expected future frames with 4 or 5 dims, but got {x.dim()}.")

        batch_size, num_horizons = x.shape[:2]
        flat_x = x.reshape(batch_size * num_horizons, *x.shape[2:])
        flat_top = self.extract_top_forward_feature(flat_x, detach=detach)
        return flat_top.reshape(batch_size, num_horizons, *flat_top.shape[1:])

    def extract_prediction_stage_features(
        self,
        x: torch.Tensor,
        detach: bool = True,
    ):
        if x.dim() == 4:
            return self.extract_prediction_stage_feature(x, detach=detach)
        if x.dim() != 5:
            raise ValueError(
                f"Expected future frames with 4 or 5 dims, but got {x.dim()}."
            )

        batch_size, num_horizons = x.shape[:2]
        flat_x = x.reshape(batch_size * num_horizons, *x.shape[2:])
        flat_feature = self.extract_prediction_stage_feature(flat_x, detach=detach)
        return flat_feature.reshape(
            batch_size,
            num_horizons,
            *flat_feature.shape[1:],
        )

    def _resolve_top_target(
        self,
        top_target: torch.Tensor = None,
        next_x: torch.Tensor = None,
        temporal_top_targets: torch.Tensor = None,
        future_x: torch.Tensor = None,
    ):
        provided_sources = sum(
            value is not None
            for value in (top_target, next_x, temporal_top_targets, future_x)
        )
        if provided_sources > 1 and top_target is not None:
            return top_target
        if temporal_top_targets is not None:
            return temporal_top_targets[:, 0] if temporal_top_targets.dim() == 5 else temporal_top_targets
        if future_x is not None:
            future_top_targets = self.extract_top_forward_features(future_x, detach=True)
            return future_top_targets[:, 0] if future_top_targets.dim() == 5 else future_top_targets
        if next_x is not None:
            return self.extract_top_forward_feature(next_x, detach=True)
        return top_target

    def _resolve_future_feature_target(
        self,
        future_feature_target: torch.Tensor = None,
        next_x: torch.Tensor = None,
        future_x: torch.Tensor = None,
        resolved_top_target: torch.Tensor = None,
    ):
        if future_feature_target is not None:
            return future_feature_target
        if future_x is not None:
            targets = self.extract_prediction_stage_features(future_x, detach=True)
            return targets[:, 0] if targets.dim() == 5 else targets
        if next_x is not None:
            return self.extract_prediction_stage_feature(next_x, detach=True)
        if self.future_feature_stage == self.number_of_layers:
            return resolved_top_target
        return None

    def _forward_impl(
        self,
        x: torch.Tensor,
        top_target: torch.Tensor = None,
        next_x: torch.Tensor = None,
        temporal_top_targets: torch.Tensor = None,
        future_x: torch.Tensor = None,
        temporal_target_override: torch.Tensor = None,
        duplicate_current_top_context: bool = False,
        top_target_provider=None,
        future_feature_target: torch.Tensor = None,
        future_feature_target_provider=None,
    ):
        if self.task == "real_frame_pc":
            forbidden_inputs = {
                "top_target": top_target,
                "next_x": next_x,
                "temporal_top_targets": temporal_top_targets,
                "future_x": future_x,
                "temporal_target_override": temporal_target_override,
                "top_target_provider": top_target_provider,
                "future_feature_target": future_feature_target,
                "future_feature_target_provider": future_feature_target_provider,
            }
            provided = [name for name, value in forbidden_inputs.items() if value is not None]
            if duplicate_current_top_context:
                provided.append("duplicate_current_top_context")
            if provided:
                raise ValueError(
                    "real_frame_pc accepts only the current observed frame; received "
                    + ", ".join(provided)
                    + "."
                )
            return self._step_real_frame_recurrence(x)

        if top_target_provider is not None and any(
            value is not None
            for value in (top_target, next_x, temporal_top_targets, future_x)
        ):
            raise ValueError(
                "top_target_provider cannot be combined with an eagerly resolved target source."
            )
        if (
            future_feature_target_provider is not None
            and future_feature_target is not None
        ):
            raise ValueError(
                "future_feature_target_provider cannot be combined with an eagerly "
                "resolved future_feature_target."
            )
        if future_feature_target_provider is not None and self.task != "future_feature":
            raise ValueError(
                "future_feature_target_provider is valid only for task='future_feature'."
            )
        forward_inputs, forward_outputs = self._run_forward_stages(x)

        self.layer_states = []
        for idx, (forward_input, forward_output) in enumerate(
            zip(forward_inputs, forward_outputs),
            start=1,
        ):
            self.layer_states.append(
                TargetFlowLayerState(
                    layer_index=idx,
                    forward_input=forward_input,
                    forward_output=forward_output,
                )
            )

        for zero_based_idx, state in enumerate(self.layer_states):
            state.previous_prediction = self._resolve_previous_prediction_state(
                zero_based_idx,
                state.forward_output,
            )
            state.previous_error = self._resolve_previous_error_state(
                zero_based_idx,
                state.forward_output,
            )
            state.previous_instant_error = self._resolve_previous_instant_error_state(
                zero_based_idx,
                state.forward_output,
            )

        # Predict before resolving any target derived from a future frame. At
        # time t the causal context may use F_t and state carried from t-1, but
        # it must not use the error that requires observing I_{t+1}.
        pooled_top_forward = _pool_spatial(forward_outputs[-1])
        if self.task == "future_feature":
            self.temporal_context = None
            self.temporal_prediction = None
            self.temporal_target = None
            self._predict_future_feature(
                forward_outputs[self.future_feature_stage_index]
            )
        else:
            self.future_prediction_outputs = None
            pooled_previous_errors = [
                _pool_spatial(
                    state.previous_error
                    if state.previous_error is not None
                    else torch.zeros_like(state.forward_output)
                )
                for state in self.layer_states
            ]
            pooled_previous_predictions = [
                _pool_spatial(
                    state.previous_prediction
                    if state.previous_prediction is not None
                    else torch.zeros_like(state.forward_output)
                )
                for state in self.layer_states
            ]
            if duplicate_current_top_context:
                pooled_previous_predictions[-1] = pooled_top_forward.detach()
            self.temporal_context = torch.cat(
                [pooled_top_forward] + pooled_previous_errors + pooled_previous_predictions,
                dim=1,
            )
            raw_temporal_prediction = self.temporal_predictor(self.temporal_context)
            self.temporal_prediction = raw_temporal_prediction.view(
                raw_temporal_prediction.shape[0],
                self.num_temporal_horizons,
                self.temporal_target_dim,
            )

        if top_target_provider is not None:
            top_target = top_target_provider()
        if future_feature_target_provider is not None:
            future_feature_target = future_feature_target_provider()

        resolved_top_target = self._resolve_top_target(
            top_target=top_target,
            next_x=next_x,
            temporal_top_targets=temporal_top_targets,
            future_x=future_x,
        )
        if resolved_top_target is None:
            raise ValueError("A top target is required to complete Target Flow state update.")
        if self.task == "future_feature":
            future_prediction_target = self._resolve_future_feature_target(
                future_feature_target=future_feature_target,
                next_x=next_x,
                future_x=future_x,
                resolved_top_target=resolved_top_target,
            )
            if future_prediction_target is None:
                raise ValueError(
                    "A separate future-feature target is required when "
                    f"future_feature_stage={self.future_feature_stage}; the Target "
                    "Flow top target remains Stage-5."
                )
            future_prediction_target = future_prediction_target.detach()
            current_feature = self.future_prediction_outputs[
                "current_prediction_feature"
            ]
            prediction_base = self.future_prediction_outputs[
                "prediction_base_feature"
            ]
            predicted_future = self.future_prediction_outputs[
                "predicted_future_feature"
            ]
            if future_prediction_target.shape != predicted_future.shape:
                raise ValueError(
                    "Future prediction target shape does not match the configured "
                    f"Stage-{self.future_feature_stage} predictor output: "
                    f"target={tuple(future_prediction_target.shape)}, "
                    f"prediction={tuple(predicted_future.shape)}."
                )
            self.future_prediction_outputs.update(
                {
                    "future_feature_stage": self.future_feature_stage,
                    "target_flow_top_stage": self.number_of_layers,
                    "target_flow_top_target": resolved_top_target.detach(),
                    "future_prediction_target": future_prediction_target,
                    "target_delta_prediction_feature": (
                        future_prediction_target - current_feature.detach()
                    ),
                    "target_residual_prediction_feature": future_prediction_target
                    - prediction_base.detach(),
                    "prediction_error_feature": (
                        predicted_future - future_prediction_target
                    ),
                    # Compatibility aliases for existing Stage-5 evaluators.
                    "future_top_target": future_prediction_target,
                    "target_delta_top": future_prediction_target
                    - current_feature.detach(),
                    "target_residual_top": future_prediction_target
                    - prediction_base.detach(),
                    "prediction_error_top": predicted_future
                    - future_prediction_target,
                }
            )
            previous_temporal_error_state = self._resolve_memory(
                self.temporal_error_state_memory,
                predicted_future,
            )
            temporal_prediction_error = self.future_prediction_outputs[
                "prediction_error_feature"
            ]
            temporal_error_state = build_temporal_prediction_error_state(
                temporal_prediction_error,
                previous_temporal_error_state,
                sample_time=self.temporal_error_config.sample_time,
                time_constant=float(self.temporal_error_time_constant.item()),
                error_gain=float(self.temporal_error_gain.item()),
            )
            self.temporal_error_state_update_count += 1
            self.temporal_prediction_error_memory = (
                temporal_prediction_error.detach()
            )
            self.temporal_error_state_memory = temporal_error_state.detach()
            self.future_prediction_outputs.update(
                {
                    "instantaneous_prediction_error_feature": (
                        temporal_prediction_error.detach()
                    ),
                    "previous_dynamic_prediction_error_state_feature": (
                        None
                        if previous_temporal_error_state is None
                        else previous_temporal_error_state.detach()
                    ),
                    "dynamic_prediction_error_state_feature": (
                        temporal_error_state.detach()
                    ),
                    "temporal_error_state_update_index": (
                        self.temporal_error_state_update_count
                    ),
                }
            )
        run_backward_target_flow(
            self.layer_states,
            self.feedback_modules,
            top_target=resolved_top_target,
            mode=self.target_flow_mode,
        )
        for zero_based_idx, (state, stage) in enumerate(zip(self.layer_states, self.forward_stages)):
            previous_recursive_error = self._resolve_memory(
                self.recursive_error_state_memory[zero_based_idx],
                state.forward_output,
            )
            state.instant_error = build_targetflow_instant_error(state.target_output, state.forward_output)
            state.error = build_targetflow_error(
                state.target_output,
                state.forward_output,
                previous_error=state.previous_error,
                sample_time=self.dynamic_error_config.sample_time,
                time_constant=float(self.error_time_constants[zero_based_idx].item()),
                error_gain=float(self.error_gains[zero_based_idx].item()),
                mode=self.error_state_mode,
                previous_instant_error=state.previous_instant_error,
            )
            state.loss_error = (
                state.instant_error
                if self.local_loss_error_source == "instant"
                else state.error
            )
            state.learn_signal = build_targetflow_learn_signal_from_error(state.loss_error)
            state.local_loss = build_targetflow_local_loss_from_error(state.loss_error)
            if self.compute_local_param_grads and torch.is_grad_enabled():
                state.parameter_grad_stats = compute_module_grad_stats(stage, state.local_loss)
            else:
                state.parameter_grad_stats = None
            recursive_error = build_targetflow_error(
                state.target_output,
                state.forward_output,
                previous_error=previous_recursive_error,
                sample_time=self.dynamic_error_config.sample_time,
                time_constant=float(self.error_time_constants[zero_based_idx].item()),
                error_gain=float(self.error_gains[zero_based_idx].item()),
                mode="ema",
                previous_instant_error=state.previous_instant_error,
            )
            two_tap_error = build_targetflow_error(
                state.target_output,
                state.forward_output,
                previous_error=None,
                sample_time=self.dynamic_error_config.sample_time,
                time_constant=float(self.error_time_constants[zero_based_idx].item()),
                error_gain=float(self.error_gains[zero_based_idx].item()),
                mode="two_tap",
                previous_instant_error=state.previous_instant_error,
            )
            self.error_state_memory[zero_based_idx] = state.error.detach()
            self.instant_error_state_memory[zero_based_idx] = state.instant_error.detach()
            self.recursive_error_state_memory[zero_based_idx] = recursive_error.detach()
            self.two_tap_error_state_memory[zero_based_idx] = two_tap_error.detach()
            self.prediction_state_memory[zero_based_idx] = state.target_output.detach()

        if self.task == "future_feature":
            self.future_feature_previous_prediction_stage_memory = forward_outputs[
                self.future_feature_stage_index
            ].detach()

        if self.task == "future_feature":
            self.temporal_target = None
        elif temporal_target_override is not None:
            resolved_temporal_targets = temporal_target_override
            if resolved_temporal_targets.dim() == 2:
                resolved_temporal_targets = resolved_temporal_targets.unsqueeze(1)
            if resolved_temporal_targets.shape[1] != self.num_temporal_horizons:
                raise ValueError(
                    f"Expected {self.num_temporal_horizons} override targets, "
                    f"but got {resolved_temporal_targets.shape[1]}."
                )
            self.temporal_target = resolved_temporal_targets
        else:
            resolved_temporal_targets = temporal_top_targets
            if resolved_temporal_targets is None and future_x is not None:
                resolved_temporal_targets = self.extract_top_forward_features(future_x, detach=True)
            if resolved_temporal_targets is None and resolved_top_target is not None:
                resolved_temporal_targets = resolved_top_target.unsqueeze(1)
            if resolved_temporal_targets is None:
                raise ValueError("Temporal targets could not be resolved.")
            if resolved_temporal_targets.dim() == 4:
                resolved_temporal_targets = resolved_temporal_targets.unsqueeze(1)
            if resolved_temporal_targets.shape[1] != self.num_temporal_horizons:
                raise ValueError(
                    f"Expected {self.num_temporal_horizons} temporal targets, "
                    f"but got {resolved_temporal_targets.shape[1]}."
                )

            pooled_top_target = _pool_temporal_top_features(resolved_temporal_targets)
            if self.temporal_target_mode == "next_top":
                self.temporal_target = pooled_top_target
            elif self.temporal_target_mode == "delta_top":
                self.temporal_target = pooled_top_target - pooled_top_forward.unsqueeze(1)
            elif self.temporal_target_mode == "ego_motion":
                raise ValueError(
                    "temporal_target_mode='ego_motion' requires temporal_target_override."
                )
            else:
                raise RuntimeError(f"Unexpected temporal_target_mode: {self.temporal_target_mode}")

        current = self.forward_tail(forward_outputs[-1])
        current = self.avgpool(current)
        current = torch.flatten(current, 1)
        return self.classifier(current)

    def collect_learn_flow_losses(self):
        if self.task == "real_frame_pc":
            return [None for _ in self.layer_states], None
        per_layer = []
        total = None
        for state in self.layer_states:
            if state.local_loss is None:
                per_layer.append(None)
                continue
            per_layer.append(state.local_loss)
            total = state.local_loss if total is None else total + state.local_loss
        return per_layer, total

    def collect_recurrent_transition_loss(self, future_frame=None):
        if self.real_frame_transition_mode != "convgru_error":
            return None
        if not self.recurrent_transition_predictions:
            return None
        if future_frame is None:
            raise ValueError("Next-frame targets are required for recurrent training.")
        future_targets, _ = self._real_frame_feature_targets(future_frame)
        self.recurrent_transition_losses = [
            nn.functional.mse_loss(prediction, target.detach())
            for prediction, target in zip(
                self.recurrent_transition_predictions,
                future_targets,
            )
        ]
        return torch.stack(self.recurrent_transition_losses).mean()

    def collect_temporal_prediction_loss(self):
        if self.temporal_prediction is None or self.temporal_target is None:
            return None
        return torch.mean((self.temporal_prediction - self.temporal_target.detach()) ** 2)

    def collect_future_feature_prediction_losses(self):
        outputs = self.future_prediction_outputs
        if outputs is None or outputs["future_prediction_target"] is None:
            return None
        future_target = outputs["future_prediction_target"].detach()
        target_delta = outputs["target_delta_prediction_feature"].detach()
        target_residual = outputs["target_residual_prediction_feature"].detach()
        return {
            "future_mse": torch.mean(
                (outputs["predicted_future_feature"] - future_target) ** 2
            ),
            "delta_mse": torch.mean(
                (outputs["predicted_delta_feature"] - target_delta) ** 2
            ),
            "residual_mse": torch.mean(
                (outputs["predicted_residual_feature"] - target_residual) ** 2
            ),
        }

    def forward(
        self,
        x: torch.Tensor,
        top_target: torch.Tensor = None,
        next_x: torch.Tensor = None,
        temporal_top_targets: torch.Tensor = None,
        future_x: torch.Tensor = None,
        temporal_target_override: torch.Tensor = None,
        duplicate_current_top_context: bool = False,
        top_target_provider=None,
        future_feature_target: torch.Tensor = None,
        future_feature_target_provider=None,
    ):
        self.reset()
        return self._forward_impl(
            x,
            top_target=top_target,
            next_x=next_x,
            temporal_top_targets=temporal_top_targets,
            future_x=future_x,
            temporal_target_override=temporal_target_override,
            duplicate_current_top_context=duplicate_current_top_context,
            top_target_provider=top_target_provider,
            future_feature_target=future_feature_target,
            future_feature_target_provider=future_feature_target_provider,
        )

    def forward_with_next_target(self, x: torch.Tensor, next_x: torch.Tensor):
        self.reset()
        return self._forward_impl(x, next_x=next_x, future_x=next_x.unsqueeze(1))

    def step_frame(
        self,
        x: torch.Tensor,
        top_target: torch.Tensor = None,
        next_x: torch.Tensor = None,
        temporal_top_targets: torch.Tensor = None,
        future_x: torch.Tensor = None,
        temporal_target_override: torch.Tensor = None,
        duplicate_current_top_context: bool = False,
        top_target_provider=None,
        future_feature_target: torch.Tensor = None,
        future_feature_target_provider=None,
    ):
        return self._forward_impl(
            x,
            top_target=top_target,
            next_x=next_x,
            temporal_top_targets=temporal_top_targets,
            future_x=future_x,
            temporal_target_override=temporal_target_override,
            duplicate_current_top_context=duplicate_current_top_context,
            top_target_provider=top_target_provider,
            future_feature_target=future_feature_target,
            future_feature_target_provider=future_feature_target_provider,
        )

    def step_pair(self, x: torch.Tensor, next_x: torch.Tensor):
        return self._forward_impl(x, next_x=next_x, future_x=next_x.unsqueeze(1))
