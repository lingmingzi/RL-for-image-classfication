import os
import random
from typing import Dict

import numpy as np
import torch


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_hint: str) -> torch.device:
    if device_hint == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
    topk = logits.topk(k, dim=1).indices
    matched = topk.eq(targets.view(-1, 1)).any(dim=1)
    return matched.float().mean().item()


def ece_score(confidences: np.ndarray, predictions: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        left, right = bins[i], bins[i + 1]
        mask = (confidences > left) & (confidences <= right)
        if mask.sum() == 0:
            continue
        acc = (predictions[mask] == labels[mask]).mean()
        conf = confidences[mask].mean()
        ece += (mask.sum() / len(labels)) * abs(acc - conf)
    return float(ece)


def moving_average(current: float, new_value: float, decay: float) -> float:
    return decay * current + (1.0 - decay) * new_value


def normalize_delta(delta: float, scale: float = 0.05) -> float:
    if scale <= 0:
        return delta
    value = delta / scale
    return float(max(-1.0, min(1.0, value)))


def dict_mean(items: Dict[str, float]) -> float:
    if not items:
        return 0.0
    return float(sum(items.values()) / len(items))
