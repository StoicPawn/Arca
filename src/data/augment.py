"""Augmentation utilities for vision and audio modalities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

import numpy as np


@dataclass
class VisionAugmentConfig:
    crop: float = 0.8
    flip: bool = True
    color_jitter: float = 0.2
    blur: float = 0.3


def build_vision_augmentations(cfg: Dict[str, float | bool]) -> Callable[[torch.Tensor], torch.Tensor]:
    transforms = []
    crop_scale = cfg.get("crop", 0.8)
    transforms.append(
        T.RandomResizedCrop(
            size=96,
            scale=(crop_scale, 1.0),
            ratio=(0.9, 1.1),
            interpolation=InterpolationMode.BICUBIC,
        )
    )
    if cfg.get("flip", True):
        transforms.append(T.RandomHorizontalFlip())
    jitter_strength = cfg.get("color_jitter", 0.0)
    if jitter_strength > 0:
        transforms.append(
            T.ColorJitter(
                brightness=jitter_strength,
                contrast=jitter_strength,
                saturation=jitter_strength,
                hue=min(0.5, jitter_strength / 2),
            )
        )
    p_blur = float(cfg.get("blur", 0.0))
    if p_blur > 0:
        transforms.append(
            T.RandomApply(
                [T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))],
                p=p_blur,
            )
        )

    transforms.extend([T.ToTensor()])
    return T.Compose(transforms)


@dataclass
class AudioAugmentConfig:
    time_shift: float = 0.2
    mask_time: float = 0.1
    noise: float = 0.01


def build_audio_augmentations(cfg: Dict[str, float]) -> Callable[[torch.Tensor], torch.Tensor]:
    time_shift = float(cfg.get("time_shift", 0.0))
    mask_time = float(cfg.get("mask_time", 0.0))
    noise = float(cfg.get("noise", 0.0))

    def _augment(spec: torch.Tensor) -> torch.Tensor:
        augmented = spec.clone()
        if time_shift > 0:
            shift = int(np.random.uniform(-time_shift, time_shift) * augmented.size(-1))
            augmented = torch.roll(augmented, shifts=shift, dims=-1)
        if mask_time > 0:
            length = augmented.size(-1)
            mask_width = int(mask_time * length)
            if mask_width > 0:
                start = np.random.randint(0, max(1, length - mask_width))
                augmented[..., start : start + mask_width] = 0
        if noise > 0:
            augmented = augmented + noise * torch.randn_like(augmented)
        return augmented

    return _augment


__all__ = [
    "VisionAugmentConfig",
    "AudioAugmentConfig",
    "build_vision_augmentations",
    "build_audio_augmentations",
]
