import argparse
import csv
import json
import math
import os
import sys
from collections import Counter


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.augmentations import get_eval_transform, get_subpolicy_space
from src.config import RLConfig
from src.controller import PPOController, Transition
from src.data import CIFARCDataset, build_cifar10, build_loaders
from src.engine import evaluate, get_confidence_summary, train_one_epoch
from src.models import build_model
from src.utils import ensure_dir, get_device, moving_average, normalize_delta, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="RL-driven augmentation training on CIFAR-10")
    parser.add_argument("--model", choices=["resnet18", "resnet50"], default="resnet18")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./outputs/rl")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--episode_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--controller_lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--val_size", type=int, default=5000)
    parser.add_argument("--run_name", type=str, default="")

    parser.add_argument("--reward_w_acc", type=float, default=1.0)
    parser.add_argument("--reward_w_loss", type=float, default=0.5)
    parser.add_argument("--reward_w_complexity", type=float, default=0.1)
    parser.add_argument("--reward_w_robust", type=float, default=0.0)
    parser.add_argument("--reward_w_class_balance", type=float, default=0.0)
    parser.add_argument("--baseline_decay", type=float, default=0.99)

    parser.add_argument("--ppo_update_every", type=int, default=8)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--ppo_batch_size", type=int, default=32)

    parser.add_argument("--cifar_c_dir", type=str, default="", help="Directory that contains corruption .npy files and labels.npy")
    parser.add_argument("--cifar_c_corruption", type=str, default="gaussian_noise.npy")
    parser.add_argument("--cifar_c_eval_freq", type=int, default=5)

    return parser.parse_args()


def eval_cifar_c(model, data_dir: str, corruption_file: str, batch_size: int, num_workers: int, device, transform) -> float:
    labels_file = os.path.join(data_dir, "labels.npy")
    corruption_path = os.path.join(data_dir, corruption_file)
    if not os.path.exists(labels_file) or not os.path.exists(corruption_path):
        return 0.0
    ds = CIFARCDataset(corruption_file=corruption_path, labels_file=labels_file, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            pred = logits.argmax(dim=1)
            correct += int((pred == targets).sum().item())
            total += targets.size(0)
    return correct / max(total, 1)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    run_name = args.run_name.strip() or f"cifar10_{args.model}_rl_seed{args.seed}"
    out_dir = os.path.join(args.output_dir, run_name)
    ensure_dir(out_dir)

    eval_transform = get_eval_transform()
    policy_names, policy_builders, complexity_map = get_subpolicy_space()
    action_dim = len(policy_names)

    default_policy = policy_names[0]
    train_transform = policy_builders[default_policy]()
    bundle = build_cifar10(args.data_dir, args.seed, val_size=args.val_size, train_transform=train_transform, eval_transform=eval_transform)
    train_loader, val_loader, test_loader = build_loaders(bundle, args.batch_size, args.num_workers)

    model = build_model(args.model, num_classes=10).to(device)
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    rl_cfg = RLConfig(
        state_dim=6,
        hidden_dim=128,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        ppo_epochs=args.ppo_epochs,
        mini_batch_size=args.ppo_batch_size,
        update_every=args.ppo_update_every,
        episode_epochs=args.episode_epochs,
        reward_baseline_decay=args.baseline_decay,
        reward_w_acc=args.reward_w_acc,
        reward_w_loss=args.reward_w_loss,
        reward_w_complexity=args.reward_w_complexity,
    )

    controller = PPOController(
        state_dim=rl_cfg.state_dim,
        hidden_dim=rl_cfg.hidden_dim,
        action_dim=action_dim,
        lr=args.controller_lr,
        gamma=rl_cfg.gamma,
        gae_lambda=rl_cfg.gae_lambda,
        clip_eps=rl_cfg.clip_eps,
        value_coef=rl_cfg.value_coef,
        entropy_coef=rl_cfg.entropy_coef,
        ppo_epochs=rl_cfg.ppo_epochs,
        mini_batch_size=rl_cfg.mini_batch_size,
        device=device,
    )

    total_episodes = max(1, math.ceil(args.epochs / max(args.episode_epochs, 1)))
    action_counter = Counter()

    train_loss_ema = 1.0
    val_acc_ema = 0.0
    robust_acc_ema = 0.0
    min_cls_ema = 0.0
    reward_baseline = 0.0
    last_action_norm = 0.0

    best_val = -1.0
    best_ckpt = os.path.join(out_dir, "best_model.pt")
    history_path = os.path.join(out_dir, "rl_history.csv")

    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "action",
                "train_loss",
                "train_acc",
                "val_loss",
                "val_acc",
                "val_top5",
                "reward",
                "r_acc",
                "r_loss",
                "r_complex",
                "r_robust",
                "r_class_balance",
                "ppo_policy_loss",
                "ppo_value_loss",
                "ppo_entropy",
            ],
        )
        writer.writeheader()

        global_epoch = 0
        for episode in range(1, total_episodes + 1):
            progress = global_epoch / max(args.epochs, 1)
            conf_mean, entropy_mean = get_confidence_summary(model, val_loader, device)

            state_vec = torch.tensor(
                [
                    progress,
                    train_loss_ema,
                    val_acc_ema,
                    conf_mean,
                    entropy_mean,
                    last_action_norm,
                ],
                dtype=torch.float32,
            )

            action_idx, logprob, value = controller.select_action(state_vec)
            action_name = policy_names[action_idx]
            action_counter[action_name] += 1
            last_action_norm = action_idx / max(action_dim - 1, 1)

            bundle.train_set.set_transform(policy_builders[action_name]())

            train_stats_episode = []
            for _ in range(args.episode_epochs):
                if global_epoch >= args.epochs:
                    break
                train_stats = train_one_epoch(model, train_loader, optimizer, criterion, device)
                scheduler.step()
                train_stats_episode.append(train_stats)
                global_epoch += 1

            if not train_stats_episode:
                break

            train_loss = float(np.mean([x["loss"] for x in train_stats_episode]))
            train_acc = float(np.mean([x["acc"] for x in train_stats_episode]))

            val_stats = evaluate(model, val_loader, criterion, device, num_classes=10)

            prev_train_loss_ema = train_loss_ema
            prev_val_acc_ema = val_acc_ema
            prev_robust_ema = robust_acc_ema
            prev_min_cls_ema = min_cls_ema

            train_loss_ema = moving_average(train_loss_ema, train_loss, decay=0.9)
            val_acc_ema = moving_average(val_acc_ema, val_stats["acc"], decay=0.9)
            min_cls_ema = moving_average(min_cls_ema, val_stats["min_class_acc"], decay=0.9)

            r_acc = normalize_delta(val_acc_ema - prev_val_acc_ema, scale=0.01)
            r_loss = -normalize_delta(train_loss_ema - prev_train_loss_ema, scale=0.05)
            r_complex = -float(complexity_map[action_name])

            r_robust = 0.0
            if args.cifar_c_dir and (episode % max(args.cifar_c_eval_freq, 1) == 0):
                robust_acc = eval_cifar_c(
                    model,
                    data_dir=args.cifar_c_dir,
                    corruption_file=args.cifar_c_corruption,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    device=device,
                    transform=eval_transform,
                )
                robust_acc_ema = moving_average(robust_acc_ema, robust_acc, decay=0.9)
                r_robust = normalize_delta(robust_acc_ema - prev_robust_ema, scale=0.01)

            r_class_balance = normalize_delta(min_cls_ema - prev_min_cls_ema, scale=0.01)

            reward = (
                args.reward_w_acc * r_acc
                + args.reward_w_loss * r_loss
                + args.reward_w_complexity * r_complex
                + args.reward_w_robust * r_robust
                + args.reward_w_class_balance * r_class_balance
            )

            reward_baseline = args.baseline_decay * reward_baseline + (1.0 - args.baseline_decay) * reward
            centered_reward = reward - reward_baseline

            controller.push(
                Transition(
                    state=state_vec,
                    action=torch.tensor(action_idx),
                    logprob=torch.tensor(logprob),
                    reward=centered_reward,
                    value=torch.tensor(value),
                    done=(episode == total_episodes),
                )
            )

            ppo_stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
            if episode % args.ppo_update_every == 0 or episode == total_episodes:
                ppo_stats = controller.update()

            row = {
                "episode": episode,
                "action": action_name,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_stats["loss"],
                "val_acc": val_stats["acc"],
                "val_top5": val_stats["top5"],
                "reward": reward,
                "r_acc": r_acc,
                "r_loss": r_loss,
                "r_complex": r_complex,
                "r_robust": r_robust,
                "r_class_balance": r_class_balance,
                "ppo_policy_loss": ppo_stats["policy_loss"],
                "ppo_value_loss": ppo_stats["value_loss"],
                "ppo_entropy": ppo_stats["entropy"],
            }
            writer.writerow(row)
            print(json.dumps(row, ensure_ascii=False))

            if val_stats["acc"] > best_val:
                best_val = val_stats["acc"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "controller": controller.net.state_dict(),
                        "episode": episode,
                        "val_acc": best_val,
                        "action": action_name,
                        "args": vars(args),
                    },
                    best_ckpt,
                )

    checkpoint = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(checkpoint["model"])
    test_stats = evaluate(model, test_loader, criterion, device, num_classes=10)

    action_dist = {k: v / max(sum(action_counter.values()), 1) for k, v in action_counter.items()}
    best_policy = action_counter.most_common(1)[0][0] if action_counter else default_policy

    with open(os.path.join(out_dir, "action_distribution.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["action", "count", "ratio"])
        writer.writeheader()
        total_actions = max(sum(action_counter.values()), 1)
        for action, count in action_counter.most_common():
            writer.writerow({"action": action, "count": count, "ratio": count / total_actions})

    summary = {
        "run_name": run_name,
        "best_val_acc": best_val,
        "test_acc": test_stats["acc"],
        "test_top5": test_stats["top5"],
        "test_ece": test_stats["ece"],
        "test_min_class_acc": test_stats["min_class_acc"],
        "best_policy": best_policy,
        "action_distribution": action_dist,
        "history": history_path,
        "checkpoint": best_ckpt,
    }

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
