import copy

import torch
import torch.nn as nn
from torch.nn import ConvTranspose2d, ReLU

from .targetflow import (
    TargetFlowDynamicErrorConfig,
    TargetFlowFeedbackModule,
    TargetFlowLayerState,
    build_targetflow_error,
    build_targetflow_instant_error,
    build_targetflow_learn_signal_from_error,
    build_targetflow_local_loss_from_error,
    compute_module_grad_stats,
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


class PVGG16TargetFlow(nn.Module):
    """
    Target-flow PVGG16 skeleton with explicit temporal top targets.

    The current frame x_t is processed normally. The next frame x_{t+1} can be
    passed in explicitly, and its top-layer forward feature is used as a
    stop-gradient top target for x_t.
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
        error_sample_time: float = 1.0,
        error_time_constant=1.0,
        error_gain=1.0,
        task: str = "motion",
        future_feature_history_mode: str = "none",
        future_feature_predictor_kernel_size: int = 1,
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
        if task not in {"motion", "future_feature"}:
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
            "copy_current",
        }:
            raise ValueError(
                "Unsupported future_feature_history_mode: "
                f"{future_feature_history_mode}"
            )
        self.task = task
        self.future_feature_history_mode = future_feature_history_mode
        if future_feature_predictor_kernel_size not in {1, 3}:
            raise ValueError(
                "future_feature_predictor_kernel_size must be 1 or 3, got "
                f"{future_feature_predictor_kernel_size}."
            )
        self.future_feature_predictor_kernel_size = (
            future_feature_predictor_kernel_size
        )
        self.temporal_target_dim = 2 if self.temporal_target_mode == "ego_motion" else self.stage_channels[-1]
        temporal_context_dim = self.stage_channels[-1] + 2 * sum(self.stage_channels)
        self.temporal_predictor = nn.Sequential(
            nn.Linear(temporal_context_dim, 1024),
            nn.ReLU(inplace=False),
            nn.Linear(1024, self.temporal_target_dim * self.num_temporal_horizons),
        )
        self.future_feature_predictor = nn.Sequential(
            nn.Conv2d(
                2 * self.stage_channels[-1],
                1024,
                kernel_size=future_feature_predictor_kernel_size,
                padding=future_feature_predictor_kernel_size // 2,
            ),
            nn.ReLU(inplace=False),
            nn.Conv2d(1024, self.stage_channels[-1], kernel_size=1),
        )
        self.layer_states = []
        self.error_state_memory = [None for _ in range(self.number_of_layers)]
        self.instant_error_state_memory = [None for _ in range(self.number_of_layers)]
        self.recursive_error_state_memory = [None for _ in range(self.number_of_layers)]
        self.two_tap_error_state_memory = [None for _ in range(self.number_of_layers)]
        self.prediction_state_memory = [None for _ in range(self.number_of_layers)]
        self.temporal_context = None
        self.temporal_prediction = None
        self.temporal_target = None
        self.future_prediction_outputs = None

    def reset(self):
        self.layer_states = []
        self.error_state_memory = [None for _ in range(self.number_of_layers)]
        self.instant_error_state_memory = [None for _ in range(self.number_of_layers)]
        self.recursive_error_state_memory = [None for _ in range(self.number_of_layers)]
        self.two_tap_error_state_memory = [None for _ in range(self.number_of_layers)]
        self.prediction_state_memory = [None for _ in range(self.number_of_layers)]
        self.temporal_context = None
        self.temporal_prediction = None
        self.temporal_target = None
        self.future_prediction_outputs = None

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

    def _resolve_future_feature_history(self, current_top: torch.Tensor):
        mode = self.future_feature_history_mode
        if mode in {"none", "copy_current"}:
            return torch.zeros_like(current_top)
        memory_by_mode = {
            "latest": self.instant_error_state_memory,
            "two_tap": self.two_tap_error_state_memory,
            "recursive": self.recursive_error_state_memory,
        }
        history = self._resolve_memory(memory_by_mode[mode][-1], current_top)
        return torch.zeros_like(current_top) if history is None else history.detach()

    def _predict_future_top_feature(self, current_top: torch.Tensor):
        history_top = self._resolve_future_feature_history(current_top)
        if self.future_feature_history_mode == "copy_current":
            predicted_delta = torch.zeros_like(current_top)
        else:
            predictor_input = torch.cat([current_top, history_top], dim=1)
            predicted_delta = self.future_feature_predictor(predictor_input)
        predicted_future = current_top + predicted_delta
        self.future_prediction_outputs = {
            "current_top": current_top,
            "history_top": history_top,
            "predicted_delta_top": predicted_delta,
            "predicted_future_top": predicted_future,
            "future_top_target": None,
            "target_delta_top": None,
            "prediction_error_top": None,
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

    def extract_top_forward_feature(self, x: torch.Tensor, detach: bool = True):
        if detach:
            with torch.no_grad():
                _, forward_outputs = self._run_forward_stages(x)
                return forward_outputs[-1]

        _, forward_outputs = self._run_forward_stages(x)
        return forward_outputs[-1]

    def extract_top_forward_features(self, x: torch.Tensor, detach: bool = True):
        if x.dim() == 4:
            return self.extract_top_forward_feature(x, detach=detach)
        if x.dim() != 5:
            raise ValueError(f"Expected future frames with 4 or 5 dims, but got {x.dim()}.")

        batch_size, num_horizons = x.shape[:2]
        flat_x = x.reshape(batch_size * num_horizons, *x.shape[2:])
        flat_top = self.extract_top_forward_feature(flat_x, detach=detach)
        return flat_top.reshape(batch_size, num_horizons, *flat_top.shape[1:])

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
    ):
        if top_target_provider is not None and any(
            value is not None
            for value in (top_target, next_x, temporal_top_targets, future_x)
        ):
            raise ValueError(
                "top_target_provider cannot be combined with an eagerly resolved target source."
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
            self._predict_future_top_feature(forward_outputs[-1])
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

        resolved_top_target = self._resolve_top_target(
            top_target=top_target,
            next_x=next_x,
            temporal_top_targets=temporal_top_targets,
            future_x=future_x,
        )
        if resolved_top_target is None:
            raise ValueError("A top target is required to complete Target Flow state update.")
        if self.task == "future_feature":
            future_top_target = resolved_top_target.detach()
            current_top = self.future_prediction_outputs["current_top"]
            predicted_future = self.future_prediction_outputs["predicted_future_top"]
            self.future_prediction_outputs.update(
                {
                    "future_top_target": future_top_target,
                    "target_delta_top": future_top_target - current_top.detach(),
                    "prediction_error_top": future_top_target - predicted_future,
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
        per_layer = []
        total = None
        for state in self.layer_states:
            if state.local_loss is None:
                per_layer.append(None)
                continue
            per_layer.append(state.local_loss)
            total = state.local_loss if total is None else total + state.local_loss
        return per_layer, total

    def collect_temporal_prediction_loss(self):
        if self.temporal_prediction is None or self.temporal_target is None:
            return None
        return torch.mean((self.temporal_prediction - self.temporal_target.detach()) ** 2)

    def collect_future_feature_prediction_losses(self):
        outputs = self.future_prediction_outputs
        if outputs is None or outputs["future_top_target"] is None:
            return None
        future_target = outputs["future_top_target"].detach()
        target_delta = outputs["target_delta_top"].detach()
        return {
            "future_mse": torch.mean(
                (outputs["predicted_future_top"] - future_target) ** 2
            ),
            "delta_mse": torch.mean(
                (outputs["predicted_delta_top"] - target_delta) ** 2
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
        )

    def step_pair(self, x: torch.Tensor, next_x: torch.Tensor):
        return self._forward_impl(x, next_x=next_x, future_x=next_x.unsqueeze(1))
