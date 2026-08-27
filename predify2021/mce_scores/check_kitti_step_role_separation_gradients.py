import json
import os
import random
from pathlib import Path

import torch

from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import sequence_groups
from predify2021.mce_scores.evaluate_kitti_step_oracle_upper_bound import load_writeback_checkpoint
from predify2021.mce_scores.train_kitti_step_fixed_adapter_predictor import ADAPTER_CHECKPOINT_DEFAULT, STATIC_CHECKPOINT_DEFAULT
from predify2021.mce_scores.train_kitti_step_role_separated_predictor import (
    WRITEBACK_CHECKPOINT,
    role_separated_clip_loss,
)
from predify2021.mce_scores.train_kitti_step_semantic_recurrent_predictor import clips
from predify2021.mce_scores.train_kitti_step_state_predictor import load_static_kitti_checkpoint
from predify2021.model_factory.deeplabv3plus_resnet50 import RoleSeparatedRecurrentPredictor, build_deeplabv3plus_resnet50_host


def gradient_norm(loss, parameters):
    gradients = torch.autograd.grad(loss, tuple(parameters), retain_graph=True, allow_unused=True)
    total = loss.new_zeros(())
    for gradient in gradients:
        if gradient is not None:
            total = total + (gradient.detach() ** 2).sum()
    return total.sqrt().item()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Role-separation gradient check requires CUDA")
    random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    root = Path(os.environ.get("PREDIFY_KITTI_STEP_ROOT", "/home/lin/predify/kitti_step"))
    output = Path(os.environ.get("PREDIFY_ROLE_SEPARATION_GRADIENT_OUTPUT", "results/kitti_step_role_separation_gradient_gate.json"))
    model = build_deeplabv3plus_resnet50_host().cuda()
    load_static_kitti_checkpoint(model, Path(STATIC_CHECKPOINT_DEFAULT))
    adapter = torch.load(ADAPTER_CHECKPOINT_DEFAULT, map_location="cpu", weights_only=False)
    model.multi_layer_adapter.load_state_dict(adapter["adapter_state_dict"], strict=True)
    load_writeback_checkpoint(model, Path(WRITEBACK_CHECKPOINT))
    model.requires_grad_(False); model.eval()
    predictor = RoleSeparatedRecurrentPredictor().cuda()
    dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(root, "train")
    samples = clips(sequence_groups(dataset))[0]
    semantic_loss, dynamics_loss = role_separated_clip_loss(model, predictor, samples)
    dynamics_parameters = tuple(predictor.z4_dyn_recurrent.parameters()) + tuple(predictor.z1_dyn_recurrent.parameters()) + tuple(predictor.z4_dyn_delta.parameters()) + tuple(predictor.z1_dyn_delta.parameters())
    semantic_parameters = tuple(predictor.z4_sem_recurrent.parameters()) + tuple(predictor.z1_sem_recurrent.parameters()) + tuple(predictor.z4_sem_delta.parameters()) + tuple(predictor.z1_sem_delta.parameters())
    result = {
        "semantic_to_dynamics_grad_norm": gradient_norm(semantic_loss, dynamics_parameters),
        "semantic_to_semantic_grad_norm": gradient_norm(semantic_loss, semantic_parameters),
        "dynamics_to_semantic_grad_norm": gradient_norm(dynamics_loss, semantic_parameters),
        "dynamics_to_dynamics_grad_norm": gradient_norm(dynamics_loss, dynamics_parameters),
    }
    result["passed"] = result["semantic_to_dynamics_grad_norm"] == 0.0 and result["semantic_to_semantic_grad_norm"] > 0.0 and result["dynamics_to_semantic_grad_norm"] == 0.0 and result["dynamics_to_dynamics_grad_norm"] > 0.0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
