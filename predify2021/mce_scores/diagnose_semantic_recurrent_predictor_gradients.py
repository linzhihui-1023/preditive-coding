import json
import os
import statistics
from pathlib import Path

import torch

from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import sequence_groups
from predify2021.mce_scores.evaluate_kitti_step_oracle_upper_bound import load_writeback_checkpoint
from predify2021.mce_scores.train_kitti_step_fixed_adapter_predictor import ADAPTER_CHECKPOINT_DEFAULT, STATIC_CHECKPOINT_DEFAULT
from predify2021.mce_scores.train_kitti_step_semantic_recurrent_predictor import WRITEBACK_CHECKPOINT, clip_loss, clips
from predify2021.mce_scores.train_kitti_step_state_predictor import load_static_kitti_checkpoint
from predify2021.model_factory.deeplabv3plus_resnet50 import SemanticRecurrentPredictor, build_deeplabv3plus_resnet50_host


def gradient_norm(loss, parameters):
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return torch.sqrt(sum((gradient.detach() ** 2).sum() for gradient in gradients if gradient is not None)).item()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Gradient diagnosis requires CUDA")
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    root = Path(os.environ.get("PREDIFY_KITTI_STEP_ROOT", "/home/lin/predify/kitti_step"))
    checkpoint = os.environ.get("PREDIFY_SEMANTIC_RECURRENT_CHECKPOINT")
    model = build_deeplabv3plus_resnet50_host().cuda(); load_static_kitti_checkpoint(model, Path(STATIC_CHECKPOINT_DEFAULT))
    adapter = torch.load(ADAPTER_CHECKPOINT_DEFAULT, map_location="cpu", weights_only=False); model.multi_layer_adapter.load_state_dict(adapter["adapter_state_dict"], strict=True); load_writeback_checkpoint(model, Path(WRITEBACK_CHECKPOINT)); model.requires_grad_(False); model.eval()
    predictor = SemanticRecurrentPredictor().cuda()
    if checkpoint:
        predictor.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["predictor_state_dict"], strict=True)
    predictor.train(); parameters = tuple(predictor.parameters()); dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(root, "train")
    clip_list = clips(sequence_groups(dataset))[:32]; rows=[]
    for samples in clip_list:
        semantic_loss, state_loss = clip_loss(model, predictor, samples, True, semantic_normalized=False)
        g_sem = gradient_norm(semantic_loss, parameters); g_state = gradient_norm(state_loss, parameters)
        rows.append({"semantic_loss":semantic_loss.item(),"state_loss":state_loss.item(),"g_sem":g_sem,"g_state":g_state,"ten_g_state":10*g_state,"R":g_sem/(10*g_state)})
    result={"clip_count":len(rows),"checkpoint":checkpoint or "initialization","mean":{key:statistics.mean(row[key] for row in rows) for key in rows[0]},"median":{key:statistics.median(row[key] for row in rows) for key in rows[0]},"rows":rows}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
