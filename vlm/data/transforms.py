"""Image transforms."""
from __future__ import annotations

from typing import Any

from torchvision import transforms as T


def build_transforms(cfg: dict) -> Any:
    """Return torchvision transforms."""
    image_size = int(cfg.get("image_size", 224))
    is_train = bool(cfg.get("is_train", True))
    mean = cfg.get("mean", [0.48145466, 0.4578275, 0.40821073])
    std = cfg.get("std", [0.26862954, 0.26130258, 0.27577711])

    if is_train:
        aug = [
            T.RandomResizedCrop(image_size, scale=(0.7, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
        ]
    else:
        aug = [
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
        ]

    return T.Compose(
        [
            *aug,
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )
