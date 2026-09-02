"""Full-Val evaluator for VSPW ISS Host, FAST-B Reset, and FAST-B Full."""

import argparse
import json
from pathlib import Path

import torch

from predify2021.datasets.vspw import VSPWSegmentationDataset
from predify2021.mce_scores.vspw_fast_b_common import (
    build_fast_b_predictor,
    configure_fast_b_host,
    load_joint_c4_payload,
    load_vspw_host,
    parameter_report,
)
from predify2021.mce_scores.vspw_fast_b_evaluator import (
    build_validation_loader,
    evaluate_vspw_fast_b,
)


DEFAULT_DYNAMICS_CHECKPOINT = (
    "/home/lin/predify/experiments/"
    "kitti_step_role_separated_predictor_3370f78/"
    "best_role_separated_predictor.pt"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VSPW FAST-B with Full/Reset state modes.")
    parser.add_argument("--data-root", default="/home/lin/datasets/VSPW_480p")
    parser.add_argument("--host-checkpoint", required=True)
    parser.add_argument("--fast-b-checkpoint", required=True)
    parser.add_argument("--dynamics-checkpoint", default=DEFAULT_DYNAMICS_CHECKPOINT)
    parser.add_argument("--output-dir", default="/home/lin/predify/experiments/vspw_fast_b/full_val")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--temporal-mode", choices=("full", "reset", "both"), default="both")
    parser.add_argument("--max-videos", type=int, default=0, help="Smoke limit; 0 means all Val videos.")
    return parser.parse_args()


def _limit_dataset(dataset, max_videos):
    if not max_videos:
        return dataset
    video_ids = tuple(dataset.video_ids[:max_videos])
    allowed = set(video_ids)
    dataset.video_ids = video_ids
    dataset.samples = tuple(sample for sample in dataset.samples if sample["video_id"] in allowed)
    return dataset


def _print_table(results):
    host = results["host"]
    full = results.get("full", {}).get("fast_b", host)
    reset = results.get("reset", {}).get("fast_b", host)
    print("| Model | mIoU | mVC8 | mVC16 |")
    print("| --- | ---: | ---: | ---: |")
    print(f"| ISS Host | {host['mIoU']:.6f} | {host['mVC8']:.6f} | {host['mVC16']:.6f} |")
    if "reset" in results:
        print(f"| FAST-B Reset | {reset['mIoU']:.6f} | {reset['mVC8']:.6f} | {reset['mVC16']:.6f} |")
    if "full" in results:
        print(f"| FAST-B Full | {full['mIoU']:.6f} | {full['mVC8']:.6f} | {full['mVC16']:.6f} |")
        print(f"| Full - Host | {full['mIoU']-host['mIoU']:+.6f} | {full['mVC8']-host['mVC8']:+.6f} | {full['mVC16']-host['mVC16']:+.6f} |")
    if "reset" in results and "full" in results:
        print(f"| Full - Reset | {full['mIoU']-reset['mIoU']:+.6f} | {full['mVC8']-reset['mVC8']:+.6f} | {full['mVC16']-reset['mVC16']:+.6f} |")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Formal VSPW FAST-B evaluation requires CUDA.")
    host, host_payload = load_vspw_host(args.host_checkpoint, device)
    predictor, _, fast_b_payload = build_fast_b_predictor(
        args.dynamics_checkpoint, args.fast_b_checkpoint, device
    )
    load_joint_c4_payload(host, fast_b_payload)
    configure_fast_b_host(host, trainable=False)
    report = parameter_report(host, predictor)
    dataset = _limit_dataset(
        VSPWSegmentationDataset(args.data_root, "val"), args.max_videos
    )
    loader = build_validation_loader(dataset, args.num_workers)
    modes = ("full", "reset") if args.temporal_mode == "both" else (args.temporal_mode,)
    results = {}
    for mode in modes:
        result = evaluate_vspw_fast_b(host, predictor, loader, device, mode)
        results[mode] = result
        results["host"] = result["host"]
    summary = {
        "experiment": "vspw_fast_b_full_reset_evaluation",
        "data_root": str(args.data_root),
        "host_checkpoint": str(args.host_checkpoint),
        "fast_b_checkpoint": str(args.fast_b_checkpoint),
        "dynamics_checkpoint": str(args.dynamics_checkpoint),
        "num_classes": 124,
        "video_count": results[modes[0]]["video_count"],
        "frame_count": results[modes[0]]["frame_count"],
        "metrics": results,
        "parameter_report": report,
        "host_epoch": host_payload.get("epoch"),
        "fast_b_epoch": fast_b_payload.get("epoch"),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _print_table(results)
    print(json.dumps({"summary": str(output_dir / "summary.json"), "parameter_report": report}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
