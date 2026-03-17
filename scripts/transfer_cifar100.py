import argparse
import csv
import json
import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.augmentations import get_eval_transform, get_subpolicy_space
from src.data import build_cifar100, build_loaders
from src.engine import evaluate, train_one_epoch
from src.models import build_model
from src.utils import ensure_dir, get_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Transfer learned CIFAR-10 policy to CIFAR-100")
    parser.add_argument("--policy_summary", type=str, required=True, help="summary.json from RL run")
    parser.add_argument("--model", choices=["resnet18", "resnet50"], default="resnet18")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./outputs/transfer")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--val_size", type=int, default=5000)
    parser.add_argument("--run_name", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    with open(args.policy_summary, "r", encoding="utf-8") as f:
        rl_summary = json.load(f)
    best_policy = rl_summary["best_policy"]

    policy_names, policy_builders, _ = get_subpolicy_space()
    if best_policy not in policy_names:
        raise ValueError(f"Policy '{best_policy}' is not found in current policy space")

    run_name = args.run_name.strip() or f"cifar100_{args.model}_transfer_{best_policy}_seed{args.seed}"
    out_dir = os.path.join(args.output_dir, run_name)
    ensure_dir(out_dir)

    bundle = build_cifar100(
        data_dir=args.data_dir,
        seed=args.seed,
        val_size=args.val_size,
        train_transform=policy_builders[best_policy](),
        eval_transform=get_eval_transform(),
    )
    train_loader, val_loader, test_loader = build_loaders(bundle, args.batch_size, args.num_workers)

    model = build_model(args.model, num_classes=100).to(device)
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_val = -1.0
    best_path = os.path.join(out_dir, "best_model.pt")
    history_path = os.path.join(out_dir, "history.csv")

    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_top5", "val_ece", "val_min_class_acc"])
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            train_stats = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_stats = evaluate(model, val_loader, criterion, device, num_classes=100)
            scheduler.step()

            row = {
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "train_acc": train_stats["acc"],
                "val_loss": val_stats["loss"],
                "val_acc": val_stats["acc"],
                "val_top5": val_stats["top5"],
                "val_ece": val_stats["ece"],
                "val_min_class_acc": val_stats["min_class_acc"],
            }
            writer.writerow(row)
            print(json.dumps(row, ensure_ascii=False))

            if val_stats["acc"] > best_val:
                best_val = val_stats["acc"]
                torch.save({"model": model.state_dict(), "epoch": epoch, "val_acc": best_val, "policy": best_policy}, best_path)

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_stats = evaluate(model, test_loader, criterion, device, num_classes=100)

    summary = {
        "source_policy": best_policy,
        "best_val_acc": best_val,
        "test_acc": test_stats["acc"],
        "test_top5": test_stats["top5"],
        "test_ece": test_stats["ece"],
        "test_min_class_acc": test_stats["min_class_acc"],
        "checkpoint": best_path,
        "history": history_path,
    }

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
