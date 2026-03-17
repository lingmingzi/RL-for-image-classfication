from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.utils import ece_score, topk_accuracy


def train_one_epoch(model, loader, optimizer, criterion, device) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    total = 0

    for images, targets in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = targets.size(0)
        total += bs
        running_loss += float(loss.item()) * bs
        running_acc += float((logits.argmax(1) == targets).float().sum().item())

    return {
        "loss": running_loss / max(total, 1),
        "acc": running_acc / max(total, 1),
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes: int) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    running_top5 = 0.0
    total = 0

    all_conf, all_pred, all_label = [], [], []
    class_correct = np.zeros(num_classes, dtype=np.float64)
    class_total = np.zeros(num_classes, dtype=np.float64)

    for images, targets in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)
        prob = F.softmax(logits, dim=1)
        conf, pred = prob.max(dim=1)

        bs = targets.size(0)
        total += bs
        running_loss += float(loss.item()) * bs
        running_acc += float((pred == targets).float().sum().item())
        running_top5 += float(topk_accuracy(logits, targets, k=min(5, logits.size(1))) * bs)

        all_conf.append(conf.cpu().numpy())
        all_pred.append(pred.cpu().numpy())
        all_label.append(targets.cpu().numpy())

        for i in range(bs):
            label = int(targets[i].item())
            class_total[label] += 1
            class_correct[label] += float(pred[i].item() == label)

    conf = np.concatenate(all_conf) if all_conf else np.array([])
    pred = np.concatenate(all_pred) if all_pred else np.array([])
    label = np.concatenate(all_label) if all_label else np.array([])
    ece = ece_score(conf, pred, label) if len(label) > 0 else 0.0

    class_acc = np.divide(class_correct, np.maximum(class_total, 1.0))
    min_class_acc = float(class_acc.min()) if class_acc.size > 0 else 0.0

    return {
        "loss": running_loss / max(total, 1),
        "acc": running_acc / max(total, 1),
        "top5": running_top5 / max(total, 1),
        "ece": ece,
        "min_class_acc": min_class_acc,
    }


def get_confidence_summary(model, loader, device, max_batches: int = 4) -> Tuple[float, float]:
    model.eval()
    all_conf = []
    all_entropy = []
    batches = 0

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            prob = torch.softmax(logits, dim=1)
            conf, _ = prob.max(dim=1)
            entropy = -torch.sum(prob * torch.log(prob + 1e-8), dim=1)
            all_conf.append(conf.mean().item())
            all_entropy.append(entropy.mean().item())
            batches += 1
            if batches >= max_batches:
                break

    if not all_conf:
        return 0.0, 0.0
    return float(np.mean(all_conf)), float(np.mean(all_entropy))
