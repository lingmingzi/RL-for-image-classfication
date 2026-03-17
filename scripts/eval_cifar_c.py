import argparse
import json
import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.augmentations import get_eval_transform
from src.data import CIFARCDataset
from src.engine import evaluate
from src.models import build_model
from src.utils import get_device


CORRUPTIONS = [
    "gaussian_noise.npy",
    "shot_noise.npy",
    "impulse_noise.npy",
    "defocus_blur.npy",
    "glass_blur.npy",
    "motion_blur.npy",
    "zoom_blur.npy",
    "snow.npy",
    "frost.npy",
    "fog.npy",
    "brightness.npy",
    "contrast.npy",
    "elastic_transform.npy",
    "pixelate.npy",
    "jpeg_compression.npy",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model robustness on CIFAR-C")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--model", choices=["resnet18", "resnet50"], default="resnet18")
    parser.add_argument("--cifar_c_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)

    model = build_model(args.model, num_classes=10).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)

    criterion = nn.CrossEntropyLoss()
    transform = get_eval_transform()
    labels_file = os.path.join(args.cifar_c_dir, "labels.npy")

    results = {}
    for corruption in CORRUPTIONS:
        c_path = os.path.join(args.cifar_c_dir, corruption)
        if not os.path.exists(c_path):
            continue
        ds = CIFARCDataset(corruption_file=c_path, labels_file=labels_file, transform=transform)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        stats = evaluate(model, loader, criterion, device, num_classes=10)
        results[corruption.replace('.npy', '')] = {
            "acc": stats["acc"],
            "error_rate": 1.0 - stats["acc"],
            "ece": stats["ece"],
        }

    avg_error = sum(v["error_rate"] for v in results.values()) / max(len(results), 1)
    summary = {
        "mean_corruption_error": avg_error,
        "num_corruptions": len(results),
        "details": results,
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
