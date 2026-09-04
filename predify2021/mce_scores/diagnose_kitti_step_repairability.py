"""Frozen C-vs-B repairability diagnostic on KITTI-STEP Full9.

For causal frames, class C is ``Host wrong / temporal proxy correct`` and
class B is ``Host correct / temporal proxy wrong``.  This script measures how
well signals already available at inference time separate C from B.  No
weights are changed and GT is used only to form the offline C/B labels.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata
from torch.nn import functional as F

from predify2021.datasets.kitti_step import (
    KITTISTEPSegmentationDataset,
    semantic_mask_from_panoptic_png,
)
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import (
    load_image,
    sequence_groups,
)
from predify2021.mce_scores.role_separated_dynamic_error_correction import (
    ADAPTER_CHECKPOINT_DEFAULT,
    ROLE_PREDICTOR_CHECKPOINT_DEFAULT,
    STATIC_CHECKPOINT_DEFAULT,
    WRITEBACK_CHECKPOINT_DEFAULT,
    load_components,
    zero_state,
)
from predify2021.mce_scores.train_kitti_step_frozen_error_utility_probe import (
    FAST_B_CHECKPOINT_DEFAULT,
    STAGE_T_CHECKPOINT_DEFAULT,
)
from predify2021.mce_scores.train_kitti_step_temporal_joint import (
    FrozenRAFT,
)
from predify2021.mce_scores.train_kitti_step_v2_auxiliary import probe_logits
from predify2021.model_factory.deeplabv3plus_resnet50 import (
    AuxiliaryTemporalPredictor,
    AuxiliaryTemporalStateEncoder,
    HostFeature,
)


FULL9 = ("0002", "0006", "0007", "0008", "0010", "0013", "0014", "0016", "0018")
NUM_CLASSES = 19
IGNORE_LABEL = 255
RESULT_DEFAULT = "results/kitti_step_repairability_diagnostic.json"
SAMPLE_PER_CLASS_PER_FRAME = 256


def _load_model(fast_b_checkpoint):
    model, role_predictor = load_components(
        STATIC_CHECKPOINT_DEFAULT,
        ADAPTER_CHECKPOINT_DEFAULT,
        ROLE_PREDICTOR_CHECKPOINT_DEFAULT,
        WRITEBACK_CHECKPOINT_DEFAULT,
    )
    payload = torch.load(fast_b_checkpoint, map_location="cpu", weights_only=False)
    model.multi_layer_adapter.output_adapters[3].load_state_dict(
        payload["c4_output_adapter_state_dict"], strict=True
    )
    model.host_conditioned_writebacks["3"].load_state_dict(
        payload["c4_writeback_state_dict"], strict=True
    )
    model.requires_grad_(False).eval()
    role_predictor.requires_grad_(False).eval()
    return model, role_predictor


def _entropy(logits):
    p = F.softmax(logits, dim=1)
    return (-(p * p.clamp_min(1e-12).log()).sum(1) / math.log(NUM_CLASSES)).squeeze(0)


def _confidence(logits):
    return F.softmax(logits, dim=1).amax(1).squeeze(0)


def _upsample(value, output_size):
    if value.ndim == 2:
        value = value[None, None]
    elif value.ndim == 3:
        value = value[None]
    return F.interpolate(value.float(), size=output_size, mode="bilinear", align_corners=False).squeeze()


class SignalSamples:
    def __init__(self):
        self.pos = {"c": []}
        self.neg = {"b": []}
        self.names = []
        self.counts = {"C": 0, "B": 0}

    def add(self, signals, c_mask, b_mask):
        c_idx = torch.where(c_mask.flatten())[0]
        b_idx = torch.where(b_mask.flatten())[0]
        if c_idx.numel() > SAMPLE_PER_CLASS_PER_FRAME:
            c_idx = c_idx[:: max(1, c_idx.numel() // SAMPLE_PER_CLASS_PER_FRAME)][:SAMPLE_PER_CLASS_PER_FRAME]
        if b_idx.numel() > SAMPLE_PER_CLASS_PER_FRAME:
            b_idx = b_idx[:: max(1, b_idx.numel() // SAMPLE_PER_CLASS_PER_FRAME)][:SAMPLE_PER_CLASS_PER_FRAME]
        c_idx = c_idx.cpu()
        b_idx = b_idx.cpu()
        self.counts["C"] += int(c_mask.sum().item())
        self.counts["B"] += int(b_mask.sum().item())
        if not self.names:
            self.names = list(signals)
            self.pos = {name: [] for name in self.names}
            self.neg = {name: [] for name in self.names}
        for name, value in signals.items():
            flat = value.flatten().detach().float().cpu()
            if c_idx.numel():
                self.pos[name].append(flat[c_idx].numpy())
            if b_idx.numel():
                self.neg[name].append(flat[b_idx].numpy())

    def summary(self):
        rows = {}
        for name in self.names:
            pos = np.concatenate(self.pos[name]) if self.pos[name] else np.empty(0)
            neg = np.concatenate(self.neg[name]) if self.neg[name] else np.empty(0)
            if pos.size and neg.size:
                values = np.concatenate((pos, neg))
                labels = np.concatenate((np.ones(pos.size), np.zeros(neg.size)))
                ranks = rankdata(values, method="average")
                auc = float((ranks[labels == 1].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))
            else:
                auc = float("nan")
            rows[name] = {
                "auroc_C_vs_B": auc,
                "C_sample_count": int(pos.size),
                "B_sample_count": int(neg.size),
                "C_mean": float(pos.mean()) if pos.size else float("nan"),
                "C_std": float(pos.std()) if pos.size else float("nan"),
                "B_mean": float(neg.mean()) if neg.size else float("nan"),
                "B_std": float(neg.std()) if neg.size else float("nan"),
            }
        return rows


@torch.inference_mode()
def evaluate(model, role_predictor, encoder, predictor, groups):
    samples = SignalSamples()
    per_sequence = {}
    for sequence in FULL9:
        seq_samples = SignalSamples()
        pending_t = None
        hidden_t = None
        pending_role = None
        h4_role = None
        h1_role = None
        history_mean = None
        history_mag = None
        history_second = None
        history_change = None
        previous_error = None
        for index, sample in enumerate(groups[sequence]):
            image = load_image(sample)
            raw = model.extract_backbone_features(image)
            observation = model.encode_backbone_features(raw)
            output_size = tuple(image.shape[-2:])
            state = encoder(observation.z4)
            if index == 0:
                pending_t, hidden_t = predictor.predict_next(state, torch.zeros_like(state), None)
                pending_role, _, h4_role, _, h1_role, _ = role_predictor.step(
                    observation, zero_state(observation), None, None
                )
            predicted_t = state if index == 0 else pending_t
            host_logits = model.decode_from_host_feature(HostFeature(raw.c4, raw.c1, output_size))
            temporal_logits = probe_logits(model, raw, observation, predicted_t, output_size)
            gt = semantic_mask_from_panoptic_png(sample["mask_path"]).cuda(non_blocking=True)
            valid = gt != IGNORE_LABEL
            host_pred = host_logits.argmax(1).squeeze(0)
            temporal_pred = temporal_logits.argmax(1).squeeze(0)
            if index > 0:
                host_correct = host_pred == gt
                temporal_correct = temporal_pred == gt
                c_mask = valid & ~host_correct & temporal_correct
                b_mask = valid & host_correct & ~temporal_correct
                error_t = state - pending_t
                if history_mean is None:
                    history_mean = error_t
                    history_mag = error_t.abs()
                    history_second = error_t.square()
                    history_change = torch.zeros_like(error_t)
                else:
                    beta_l = math.exp(-0.1035 / 1.5)
                    beta_s = math.exp(-0.1035 / 0.5)
                    history_mean = beta_l * history_mean + (1 - beta_l) * error_t
                    history_mag = beta_l * history_mag + (1 - beta_l) * error_t.abs()
                    history_second = beta_l * history_second + (1 - beta_l) * error_t.square()
                    history_change = beta_s * history_change + (1 - beta_s) * (error_t - previous_error).abs()
                variance = (history_second - history_mean.square()).clamp_min(0.0)
                host_entropy = _entropy(host_logits)
                temporal_entropy = _entropy(temporal_logits)
                host_conf = _confidence(host_logits)
                temporal_conf = _confidence(temporal_logits)
                signals = {
                    "host_entropy": host_entropy,
                    "temporal_entropy": temporal_entropy,
                    "host_confidence_minus_temporal_confidence": host_conf - temporal_conf,
                    "host_temporal_disagreement": (host_pred != temporal_pred).float(),
                    "abs_e_z1": _upsample((observation.z1 - pending_role.z1).abs().mean(1), output_size),
                    "abs_e_z4": _upsample(error_t.abs().mean(1), output_size),
                    "dynamics_state_norm": _upsample(h4_role.square().mean(1).sqrt(), output_size),
                    "error_history_mean": _upsample(history_mag.mean(1), output_size),
                    "error_history_variance": _upsample(variance.mean(1), output_size),
                    "error_history_change": _upsample(history_change.mean(1), output_size),
                }
                samples.add(signals, c_mask, b_mask)
                seq_samples.add(signals, c_mask, b_mask)
                previous_error = error_t
                pending_t, hidden_t = predictor.predict_next(state, error_t, hidden_t)
                role_error = observation - pending_role
                pending_role, _, h4_role, _, h1_role, _ = role_predictor.step(
                    observation, role_error, h4_role, h1_role
                )
            elif previous_error is None:
                previous_error = torch.zeros_like(state)
        per_sequence[sequence] = {
            "C_pixels": seq_samples.counts["C"],
            "B_pixels": seq_samples.counts["B"],
            "signals": seq_samples.summary(),
        }
    return {
        "C_pixels": samples.counts["C"],
        "B_pixels": samples.counts["B"],
        "signals": samples.summary(),
        "per_sequence": per_sequence,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/home/lin/predify/kitti_step")
    parser.add_argument("--fast-b-checkpoint", default=FAST_B_CHECKPOINT_DEFAULT)
    parser.add_argument("--stage-t-checkpoint", default=STAGE_T_CHECKPOINT_DEFAULT)
    parser.add_argument("--result-output", default=RESULT_DEFAULT)
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    model, role_predictor = _load_model(args.fast_b_checkpoint)
    payload = torch.load(args.stage_t_checkpoint, map_location="cpu", weights_only=False)
    encoder = AuxiliaryTemporalStateEncoder().cuda()
    predictor = AuxiliaryTemporalPredictor().cuda()
    encoder.load_state_dict(payload["encoder_state_dict"], strict=True)
    predictor.load_state_dict(payload["predictor_state_dict"], strict=True)
    encoder.requires_grad_(False).eval()
    predictor.requires_grad_(False).eval()
    dataset = KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "val")
    all_groups = sequence_groups(dataset)
    missing = [sequence for sequence in FULL9 if sequence not in all_groups]
    if missing:
        raise RuntimeError(f"Missing Full9 sequences: {missing}")
    groups = {sequence: all_groups[sequence] for sequence in FULL9}
    result = {
        "experiment": "Frozen C-vs-B Repairability Diagnostic",
        "full9": list(FULL9),
        "causal_frames": sum(max(len(groups[s]) - 1, 0) for s in FULL9),
        "class_definition": {
            "C": "Host wrong / temporal proxy correct",
            "B": "Host correct / temporal proxy wrong",
        },
        "signals": [
            "host_entropy", "temporal_entropy",
            "host_confidence_minus_temporal_confidence",
            "host_temporal_disagreement", "abs_e_z1", "abs_e_z4",
            "dynamics_state_norm", "error_history_mean",
            "error_history_variance", "error_history_change",
        ],
        "inference_only": True,
        "training": False,
        "fast_b_checkpoint": args.fast_b_checkpoint,
        "stage_t_checkpoint": args.stage_t_checkpoint,
        "results": evaluate(model, role_predictor, encoder, predictor, groups),
    }
    output = Path(args.result_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
