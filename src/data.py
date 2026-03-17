from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10, CIFAR100


@dataclass
class DataBundle:
    train_set: torch.utils.data.Dataset
    val_set: torch.utils.data.Dataset
    test_set: torch.utils.data.Dataset


class TransformProxyDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset: torch.utils.data.Dataset, transform=None):
        self.base_dataset = base_dataset
        self.transform = transform

    def set_transform(self, transform) -> None:
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def build_cifar10(data_dir: str, seed: int, val_size: int, train_transform, eval_transform) -> DataBundle:
    base_train = CIFAR10(root=data_dir, train=True, download=True, transform=None)
    test_set = CIFAR10(root=data_dir, train=False, download=True, transform=eval_transform)

    train_len = len(base_train) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_idx_set, val_idx_set = random_split(base_train, [train_len, val_size], generator=generator)

    train_set = TransformProxyDataset(train_idx_set, transform=train_transform)
    val_set = TransformProxyDataset(val_idx_set, transform=eval_transform)
    return DataBundle(train_set=train_set, val_set=val_set, test_set=test_set)


def build_cifar100(data_dir: str, seed: int, val_size: int, train_transform, eval_transform) -> DataBundle:
    base_train = CIFAR100(root=data_dir, train=True, download=True, transform=None)
    test_set = CIFAR100(root=data_dir, train=False, download=True, transform=eval_transform)

    train_len = len(base_train) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_idx_set, val_idx_set = random_split(base_train, [train_len, val_size], generator=generator)

    train_set = TransformProxyDataset(train_idx_set, transform=train_transform)
    val_set = TransformProxyDataset(val_idx_set, transform=eval_transform)
    return DataBundle(train_set=train_set, val_set=val_set, test_set=test_set)


def build_loaders(bundle: DataBundle, batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(bundle.train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(bundle.val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(bundle.test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


class CIFARCDataset(torch.utils.data.Dataset):
    def __init__(self, corruption_file: str, labels_file: str, transform=None):
        self.images = np.load(corruption_file)
        self.labels = np.load(labels_file)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = int(self.labels[idx])
        if self.transform is not None:
            image = self.transform(image)
        return image, label
