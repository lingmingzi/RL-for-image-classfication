from typing import Callable, Dict, List, Tuple

from torchvision import transforms


MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)


def _base_tail() -> List[transforms.Transform]:
    return [
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]


def get_eval_transform() -> transforms.Compose:
    return transforms.Compose(_base_tail())


def get_noaug_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        *_base_tail(),
    ])


def get_manual_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        *_base_tail(),
    ])


def get_randaugment_transform(num_ops: int = 2, magnitude: int = 9) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=num_ops, magnitude=magnitude),
        *_base_tail(),
    ])


def get_subpolicy_space() -> Tuple[List[str], Dict[str, Callable[[], transforms.Compose]], Dict[str, float]]:
    policy_names = [
        "crop_flip",
        "crop_flip_color",
        "crop_flip_rotate",
        "crop_flip_affine",
        "crop_flip_posterize",
        "crop_flip_solarize",
        "crop_flip_autocontrast",
        "crop_flip_equalize",
        "crop_flip_sharpness",
        "crop_flip_invert",
        "crop_flip_cutout",
        "crop_flip_trivialaug",
    ]

    builders: Dict[str, Callable[[], transforms.Compose]] = {
        "crop_flip": lambda: transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            *_base_tail(),
        ]),
        "crop_flip_color": lambda: transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.06),
            *_base_tail(),
        ]),
        "crop_flip_rotate": lambda: transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=20),
            *_base_tail(),
        ]),
        "crop_flip_affine": lambda: transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            *_base_tail(),
        ]),
        "crop_flip_posterize": lambda: transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomPosterize(bits=4, p=0.5),
            *_base_tail(),
        ]),
        "crop_flip_solarize": lambda: transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomSolarize(threshold=128, p=0.5),
            *_base_tail(),
        ]),
        "crop_flip_autocontrast": lambda: transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAutocontrast(p=0.7),
            *_base_tail(),
        ]),
        "crop_flip_equalize": lambda: transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomEqualize(p=0.7),
            *_base_tail(),
        ]),
        "crop_flip_sharpness": lambda: transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAdjustSharpness(sharpness_factor=2.0, p=0.6),
            *_base_tail(),
        ]),
        "crop_flip_invert": lambda: transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomInvert(p=0.4),
            *_base_tail(),
        ]),
        "crop_flip_cutout": lambda: transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.04, 0.2), ratio=(0.3, 3.3), value="random"),
            transforms.Normalize(MEAN, STD),
        ]),
        "crop_flip_trivialaug": lambda: transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.TrivialAugmentWide(),
            *_base_tail(),
        ]),
    }

    complexity = {
        "crop_flip": 0.2,
        "crop_flip_color": 0.4,
        "crop_flip_rotate": 0.45,
        "crop_flip_affine": 0.55,
        "crop_flip_posterize": 0.5,
        "crop_flip_solarize": 0.5,
        "crop_flip_autocontrast": 0.35,
        "crop_flip_equalize": 0.35,
        "crop_flip_sharpness": 0.35,
        "crop_flip_invert": 0.35,
        "crop_flip_cutout": 0.6,
        "crop_flip_trivialaug": 0.75,
    }

    return policy_names, builders, complexity
