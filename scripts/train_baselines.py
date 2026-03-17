import argparse
import csv
import json
import os

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.augmentations import (
    get_eval_transform,
    get_manual_transform,
    get_noaug_transform,
    get_randaugment_transform,
)
from src.data import build_cifar10, build_cifar100, build_loaders
from src.engine import evaluate, train_one_epoch
from src.models import build_model
from src.utils import ensure_dir, get_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train baseline augmentation strategies on CIFAR")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    parser.add_argument("--model", choices=["resnet18", "resnet50"], default="resnet18")
    parser.add_argument("--policy", choices=["noaug", "manual", "randaugment"], default="randaugment")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./outputs/baselines")
    parser.add_argument("--epochs", type=int, default=100)
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


def build_train_transform(policy: str):
    if policy == "noaug":
        return get_noaug_transform()
    if policy == "manual":
        return get_manual_transform()
    if policy == "randaugment":
        return get_randaugment_transform()
    raise ValueError(f"Unknown policy: {policy}")


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    run_name = args.run_name.strip() or f"{args.dataset}_{args.model}_{args.policy}_seed{args.seed}"
    out_dir = os.path.join(args.output_dir, run_name)
    ensure_dir(out_dir)

    eval_transform = get_eval_transform()
    train_transform = build_train_transform(args.policy)

    if args.dataset == "cifar10":
        bundle = build_cifar10(args.data_dir, args.seed, val_size=args.val_size, train_transform=train_transform, eval_transform=eval_transform)
        num_classes = 10
    else:
        bundle = build_cifar100(args.data_dir, args.seed, val_size=args.val_size, train_transform=train_transform, eval_transform=eval_transform)
        num_classes = 100

    train_loader, val_loader, test_loader = build_loaders(bundle, args.batch_size, args.num_workers)

    model = build_model(args.model, num_classes=num_classes).to(device)
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_val = -1.0
    best_path = os.path.join(out_dir, "best_model.pt")
    log_path = os.path.join(out_dir, "history.csv")

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_top5", "val_ece", "val_min_class_acc"])
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            train_stats = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_stats = evaluate(model, val_loader, criterion, device, num_classes=num_classes)
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

            if val_stats["acc"] > best_val:
                best_val = val_stats["acc"]
                torch.save({"model": model.state_dict(), "epoch": epoch, "val_acc": best_val, "args": vars(args)}, best_path)

            print(json.dumps({"epoch": epoch, **row}, ensure_ascii=False))

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_stats = evaluate(model, test_loader, criterion, device, num_classes=num_classes)

    summary = {
        "run_name": run_name,
        "best_val_acc": best_val,
        "test_acc": test_stats["acc"],
        "test_top5": test_stats["top5"],
        "test_ece": test_stats["ece"],
        "test_min_class_acc": test_stats["min_class_acc"],
        "best_model": best_path,
        "history": log_path,
    }

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
