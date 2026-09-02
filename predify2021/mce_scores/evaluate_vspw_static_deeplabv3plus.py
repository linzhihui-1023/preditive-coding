import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from predify2021.datasets.vspw import VSPW_NUM_CLASSES, VSPWSegmentationDataset
from predify2021.mce_scores.vspw_metrics import compute_vspw_miou, compute_vspw_vc
from predify2021.mce_scores.train_vspw_static_deeplabv3plus import collate_val, segmentation_logits
from predify2021.model_factory.deeplabv3plus_resnet50 import build_deeplabv3plus_resnet50_host


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a static VSPW DeepLabV3+ host on sequential Val videos.")
    parser.add_argument("--data-root", default="/home/lin/datasets/VSPW_480p")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="/home/lin/predify/experiments/vspw_static_deeplabv3plus_r50/full_val")
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("VSPW evaluation requires a CUDA device.")
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    prediction_root = output_dir / "predictions"
    prediction_root.mkdir(parents=True, exist_ok=True)
    dataset = VSPWSegmentationDataset(data_root, "val")
    workers = max(0, args.num_workers)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        collate_fn=collate_val,
    )
    model = build_deeplabv3plus_resnet50_host(num_classes=VSPW_NUM_CLASSES, load_cityscapes_checkpoint=False).to(device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    with torch.inference_mode():
        for images, _, metadata in loader:
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                logits, _ = segmentation_logits(model, images)
            prediction = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            target_dir = prediction_root / metadata["video_id"][0]
            target_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(prediction, mode="L").save(target_dir / metadata["frame_name"][0])
    miou = compute_vspw_miou(data_root, prediction_root, "val")
    vc = compute_vspw_vc(data_root, prediction_root, "val")
    summary = {
        "experiment": "vspw_static_deeplabv3plus_resnet50_iss_host_full_val",
        "data_root": str(data_root),
        "checkpoint": str(args.checkpoint),
        "split": "val",
        "video_count": len(dataset.video_ids),
        "frame_count": len(dataset),
        "mIoU": miou["mIoU"],
        "mVC8": vc[8]["score"],
        "mVC16": vc[16]["score"],
        "mIoU_per_class": miou["per_class_iou"],
        "confusion_matrix": miou["confusion_matrix"],
        "vc_window_counts": {"8": vc[8]["window_count"], "16": vc[16]["window_count"]},
        "best_epoch": payload.get("epoch"),
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    readme = output_dir / "README.md"
    readme.write_text(
        "# VSPW-480p Static Host Full-Val\n\n"
        "DeepLabV3+-ResNet50 ISS Host, 124 classes, sequential validation over all 343 Val videos.\n\n"
        f"- Best epoch: `{summary['best_epoch']}`\n"
        f"- mIoU: `{summary['mIoU']:.8f}`\n"
        f"- mVC8: `{summary['mVC8']:.8f}`\n"
        f"- mVC16: `{summary['mVC16']:.8f}`\n"
        f"- Frames: `{summary['frame_count']}`\n"
        f"- VC windows: `{summary['vc_window_counts']}`\n\n"
        "mVC uses the VSPW reference window convention (`range(T - C)`).\n"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
