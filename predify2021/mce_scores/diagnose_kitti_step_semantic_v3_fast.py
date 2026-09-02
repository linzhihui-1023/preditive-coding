"""Fast decoder-free diagnostic for a trained Semantic V3 predictor."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F

from predify2021.datasets.kitti_step import KITTISTEPSegmentationDataset
from predify2021.mce_scores.evaluate_kitti_step_dynamic_error_correction import load_image, sequence_groups
from predify2021.mce_scores.kitti_step_persistent_blur import BLUR_KERNEL_SIZE, warmup_frame_count
from predify2021.mce_scores.role_separated_dynamic_error_correction import ADAPTER_CHECKPOINT_DEFAULT, STATIC_CHECKPOINT_DEFAULT, WRITEBACK_CHECKPOINT_DEFAULT, error_state, load_components, zero_state
from predify2021.mce_scores.train_kitti_step_convgru_decoupling_quick import diagnostic_blur
from predify2021.model_factory.deeplabv3plus_resnet50 import ErrorRegulatedSemanticRestorationPredictor, UnifiedFeatures

CONDITIONS = ("Blur-Mid", "Blur-Max")
EPS = 1e-12
DEFAULT_CHECKPOINT = "/home/lin/predify/experiments/kitti_step_semantic_v3/v3_structure/best.pt"


def slice_state(state, index):
    return UnifiedFeatures(*(value[index:index + 1] for value in state.as_tuple()))


def rms(x):
    return float(x.detach().float().square().mean().sqrt().item())


def cosine(a, b):
    return float(F.cosine_similarity(a.detach().float().flatten(1), b.detach().float().flatten(1), dim=1, eps=EPS).mean().item())


def alpha(a, b):
    a, b = a.detach().float().flatten(1), b.detach().float().flatten(1)
    return float(((a * b).sum(1) / b.square().sum(1).clamp_min(EPS)).mean().item())


def batch_observations(model, sample, frame, length):
    clean = load_image(sample)
    mid = diagnostic_blur(clean, frame, length, "Blur-Mid")
    maximum = diagnostic_blur(clean, frame, length, "Blur-Max")
    with torch.inference_mode():
        states = model.encode_backbone_features(model.extract_backbone_features(torch.cat((clean, mid, maximum), dim=0)))
    return tuple(slice_state(states, i) for i in range(3))


def empty():
    return {"frames": 0, "obs_sse": 0., "variant_sse": defaultdict(float), "state_sse": 0., "count": 0, "values": defaultdict(list), "first": None, "middle": None, "last": None}


def add(summary, name, value):
    summary["values"][name].append(float(value))


def run(model, predictor, groups, limit):
    summaries = {c: empty() for c in CONDITIONS}; traces = []
    for sequence, samples in groups.items():
        onset = warmup_frame_count(len(samples)); _, mid0, max0 = batch_observations(model, samples[0], 0, len(samples))
        observations = {"Blur-Mid": mid0, "Blur-Max": max0}
        pending = {c: (*predictor.predict_next(o, zero_state(o), None, None), None) for c, o in observations.items()}
        hidden = {c: predictor.initial_semantic_state(observations[c]) for c in CONDITIONS}
        hidden_zero = {c: predictor.initial_semantic_state(observations[c]) for c in CONDITIONS}
        count = 0
        for frame in range(1, len(samples)):
            if count >= limit: break
            clean, mid, maximum = batch_observations(model, samples[frame], frame, len(samples)); observations = {"Blur-Mid": mid, "Blur-Max": maximum}
            if frame < onset:
                for c in CONDITIONS:
                    o = observations[c]; p, h4, h1, _ = pending[c]; e = error_state(o, p)
                    with torch.inference_mode():
                        _, hidden[c], _ = predictor.restore_current(o, p, hidden[c]); _, hidden_zero[c], _ = predictor.restore_current(o, p, hidden_zero[c], zero_encoded_prediction_error=True); pending[c] = (*predictor.predict_next(o, e, h4, h1), None)
                continue
            for c in CONDITIONS:
                o = observations[c]; p, h4, h1, _ = pending[c]; e = error_state(o, p); s = summaries[c]
                with torch.inference_mode():
                    restored, hidden[c], d = predictor.restore_current(o, p, hidden[c])
                    nohist, _, nd = predictor.restore_current(o, p, o.z4.detach())
                    zero, hidden_zero[c], zd = predictor.restore_current(o, p, hidden_zero[c], zero_encoded_prediction_error=True)
                oracle = clean.z4 - o.z4; delta = d["restoration_delta_z4"]; discrepancy = d["semantic_discrepancy"]
                obs_mse = float(F.mse_loss(o.z4, clean.z4).item()); s["obs_sse"] += obs_mse; s["count"] += 1; s["frames"] += 1
                for name, value in (("continuous", restored), ("nohistory", nohist), ("zeroerror", zero)):
                    s["variant_sse"][name] += float(F.mse_loss(value.z4, clean.z4).item())
                    add(s, f"{name}_direction", cosine(value.z4 - o.z4, oracle)); add(s, f"{name}_alpha", alpha(value.z4 - o.z4, oracle)); add(s, f"{name}_amplitude", rms(value.z4 - o.z4) / max(rms(oracle), EPS))
                state_mse = float(F.mse_loss(hidden[c], clean.z4).item()); s["state_sse"] += state_mse
                row = {"sequence": sequence, "condition": c, "frame": frame, "prediction_error_rms": rms(d["prediction_error_z4"]), "encoded_error_rms": rms(d["encoded_prediction_error"]), "semantic_candidate_rms": rms(d["semantic_candidate"]), "semantic_update_gain_mean": float(d["semantic_update_gain"].mean().item()), "semantic_update_gain_std": float(d["semantic_update_gain"].std(unbiased=False).item()), "semantic_hidden_rms": rms(hidden[c]), "semantic_state_mse_to_clean": state_mse, "state_recovery": 1. - state_mse / max(obs_mse, EPS), "state_direction_cosine": cosine(hidden[c] - o.z4, oracle), "semantic_discrepancy_rms": rms(discrepancy), "restoration_delta_rms": rms(delta), "oracle_residual_rms": rms(oracle), "restoration_direction_cosine": cosine(delta, oracle), "projection_scale_alpha": alpha(delta, oracle), "amplitude_ratio": rms(delta) / max(rms(oracle), EPS), "feature_recovery": 1. - float(F.mse_loss(restored.z4, clean.z4).item()) / max(obs_mse, EPS), "continuous_feature_recovery": 1. - float(F.mse_loss(restored.z4, clean.z4).item()) / max(obs_mse, EPS), "nohistory_feature_recovery": 1. - float(F.mse_loss(nohist.z4, clean.z4).item()) / max(obs_mse, EPS), "zeroerror_feature_recovery": 1. - float(F.mse_loss(zero.z4, clean.z4).item()) / max(obs_mse, EPS)}
                row["temporal_feature_gain"] = row["continuous_feature_recovery"] - row["nohistory_feature_recovery"]; row["error_contribution"] = row["continuous_feature_recovery"] - row["zeroerror_feature_recovery"]
                traces.append(row); s["values"]["continuous_feature_recovery"].append(row["continuous_feature_recovery"]); s["values"]["nohistory_feature_recovery"].append(row["nohistory_feature_recovery"]); s["values"]["zeroerror_feature_recovery"].append(row["zeroerror_feature_recovery"]); s["values"]["temporal_feature_gain"].append(row["temporal_feature_gain"]); s["values"]["error_contribution"].append(row["error_contribution"]); s["first"] = s["first"] or row["semantic_hidden_rms"]; s["middle"] = row["semantic_hidden_rms"] if frame - onset >= max(1, limit // 2) else s["middle"]; s["last"] = row["semantic_hidden_rms"]
                with torch.inference_mode(): pending[c] = (*predictor.predict_next(o, e, h4, h1), None)
            count += 1
    results = {}
    for c, s in summaries.items():
        means = {k: sum(v) / max(len(v), 1) for k, v in s["values"].items()}; obs = s["obs_sse"] / max(s["count"], 1); results[c] = {"effective_frame_count": s["frames"], "prediction_recovery": means.get("prediction_recovery", 0.), "observation_mse": obs, "continuous_feature_recovery": 1. - s["variant_sse"]["continuous"] / max(s["count"], 1) / max(obs, EPS), "nohistory_feature_recovery": 1. - s["variant_sse"]["nohistory"] / max(s["count"], 1) / max(obs, EPS), "zeroerror_feature_recovery": 1. - s["variant_sse"]["zeroerror"] / max(s["count"], 1) / max(obs, EPS), "temporal_feature_gain": means.get("temporal_feature_gain", 0.), "error_contribution": means.get("error_contribution", 0.), "continuous": {k.removeprefix("continuous_"): v for k, v in means.items() if k.startswith("continuous_")}, "no_history": {k.removeprefix("nohistory_"): v for k, v in means.items() if k.startswith("nohistory_")}, "zero_error": {k.removeprefix("zeroerror_"): v for k, v in means.items() if k.startswith("zeroerror_")}, "signal_means": {k: v for k, v in means.items() if k in ("prediction_error_rms", "encoded_error_rms")}, "semantic_state": {"early_hidden_rms": s["first"], "middle_hidden_rms": s["middle"], "final_hidden_rms": s["last"], "growth_ratio": s["last"] / max(s["first"], EPS), "state_recovery": 1. - s["state_sse"] / max(s["count"], 1) / max(obs, EPS), "state_direction_cosine": means.get("state_direction_cosine", 0.)}}
    return results, traces


def gradients(model, predictor, groups, steps):
    sequence, samples = next(iter(groups.items())); onset = warmup_frame_count(len(samples)); predictor.train(); predictor.zero_grad(set_to_none=True)
    _, mid, maximum = batch_observations(model, samples[0], 0, len(samples)); pending = {c: (*predictor.predict_next(o, zero_state(o), None, None), None) for c, o in (("Blur-Mid", mid), ("Blur-Max", maximum))}; hidden = {c: predictor.initial_semantic_state(o) for c, o in (("Blur-Mid", mid), ("Blur-Max", maximum))}
    for frame in range(1, onset):
        _, mid, maximum = batch_observations(model, samples[frame], frame, len(samples))
        for c, o in (("Blur-Mid", mid), ("Blur-Max", maximum)):
            p, h4, h1, _ = pending[c]; e = error_state(o, p)
            with torch.no_grad(): _, hidden[c], _ = predictor.restore_current(o, p, hidden[c]); pending[c] = (*predictor.predict_next(o, e, h4, h1), None)
    losses = []
    for frame in range(onset, onset + steps):
        clean, mid, maximum = batch_observations(model, samples[frame], frame, len(samples))
        for c, o in (("Blur-Mid", mid), ("Blur-Max", maximum)):
            p, h4, h1, _ = pending[c]; p = UnifiedFeatures(*(v.clone() for v in p.as_tuple())); restored, hidden[c], _ = predictor.restore_current(o, p, hidden[c]); losses.append(F.smooth_l1_loss(restored.z4, clean.z4.detach()))
            with torch.no_grad(): pending[c] = (*predictor.predict_next(o, error_state(o, p), h4, h1), None)
    torch.stack(losses).mean().backward(); modules = {"semantic_error_encoder": predictor.semantic_error_encoder, "semantic_state_cell.candidate": predictor.semantic_state_cell.candidate, "semantic_state_cell.update_gain": predictor.semantic_state_cell.update_gain, "semantic_restoration_head": predictor.semantic_restoration_head}; result = {"sequence": sequence, "start_frame": onset, "step_count": steps, "optimizer_step_performed": False, "gradient_norms": {name: math.sqrt(sum(float(p.grad.detach().float().square().sum().item()) for p in module.parameters() if p.grad is not None)) for name, module in modules.items()}}; predictor.zero_grad(set_to_none=True); predictor.eval(); return result


def write_csv(path, rows):
    if not rows: return
    with path.open("w", newline="") as f: writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--root", default="/home/lin/predify/kitti_step"); p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT); p.add_argument("--output", default="results/kitti_step_semantic_v3_fast"); p.add_argument("--max-frames-per-sequence", type=int, default=250); p.add_argument("--gradient-steps", type=int, default=8); p.add_argument("--skip-gradient-check", action="store_true"); p.add_argument("--smoke", action="store_true"); args = p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False); model, _ = load_components(STATIC_CHECKPOINT_DEFAULT, ADAPTER_CHECKPOINT_DEFAULT, payload["source_dynamics_checkpoint"], WRITEBACK_CHECKPOINT_DEFAULT); predictor = ErrorRegulatedSemanticRestorationPredictor().cuda(); predictor.load_state_dict(payload["model_state_dict"], strict=True); predictor.freeze_dynamics(); model.requires_grad_(False); model.eval(); predictor.eval()
    groups = sequence_groups(KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "val")); ids = sorted(groups); chosen = [ids[0], ids[len(ids) // 2], ids[-1]]; chosen = chosen[:1] if args.smoke else chosen; groups = {k: groups[k] for k in chosen}; limit = 8 if args.smoke else args.max_frames_per_sequence
    results, trace = run(model, predictor, groups, limit); gradient = None if args.skip_gradient_check else gradients(model, predictor, sequence_groups(KITTISTEPSegmentationDataset.from_kitti_step_root(Path(args.root), "train")), args.gradient_steps); output = Path(args.output) / "smoke" if args.smoke else Path(args.output); output.mkdir(parents=True, exist_ok=True)
    summary = {"experiment": "kitti_step_semantic_v3_fast", "diagnostic_only": True, "parameters_updated": False, "checkpoint": args.checkpoint, "selected_sequences": chosen, "max_frames_per_sequence": limit, "protocol": {"batch": ["clean", "Blur-Mid", "Blur-Max"], "sigma": {"Blur-Mid": 2.25, "Blur-Max": 3.0}, "kernel": [BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE], "warmup": "existing 10%", "states_start": "frame 0"}, "results": results, "gradient_norms": gradient, "stage_a": {c: "GO" if results[c]["continuous_feature_recovery"] > 0 and results[c]["state_recovery"] > 0 and results[c]["continuous"]["direction"] > 0 and results[c]["continuous"]["amplitude"] >= 0.08 and results[c]["semantic_state"]["growth_ratio"] < 3 else "NO-GO" for c in CONDITIONS}}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n"); write_csv(output / "condition_summary.csv", [{"condition": c, "stage_a": summary["stage_a"][c], **{k: v for k, v in results[c].items() if not isinstance(v, (dict, list))}, **{f"continuous_{k}": v for k, v in results[c]["continuous"].items()}, **{f"no_history_{k}": v for k, v in results[c]["no_history"].items()}, **{f"zero_error_{k}": v for k, v in results[c]["zero_error"].items()}, **{f"semantic_{k}": v for k, v in results[c]["semantic_state"].items()}} for c in CONDITIONS]); write_csv(output / "temporal_trace.csv", trace); (output / "gradient_norms.json").write_text(json.dumps(gradient, indent=2, sort_keys=True) + "\n" if gradient else "null\n"); print(json.dumps({"stage_a": summary["stage_a"]}, indent=2), flush=True)


if __name__ == "__main__": main()
