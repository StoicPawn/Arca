from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torchvision.transforms as T


@dataclass
class VisionAugmentConfig:
    crop: float = 0.8
    flip: bool = True
    color_jitter: float = 0.0
    blur: float = 0.0


@dataclass
class AudioAugmentConfig:
    time_shift: float = 0.0
    mask_time: float = 0.0
    noise: float = 0.0


def build_vision_augment(
    config: VisionAugmentConfig,
    *,
    mean: tuple[float, float, float] | None = None,
    std: tuple[float, float, float] | None = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    transforms = [T.RandomResizedCrop(size=96, scale=(config.crop, 1.0))]
    if config.flip:
        transforms.append(T.RandomHorizontalFlip())
    if config.color_jitter > 0:
        transforms.append(
            T.ColorJitter(
                brightness=config.color_jitter,
                contrast=config.color_jitter,
                saturation=config.color_jitter,
                hue=min(0.5, config.color_jitter / 2),
            )
        )
    if config.blur > 0:
        transforms.append(T.GaussianBlur(kernel_size=3, sigma=(0.1, config.blur)))
    if mean is None:
        mean = (0.5, 0.5, 0.5)
    if std is None:
        std = (0.5, 0.5, 0.5)
    transforms.append(T.Normalize(mean=mean, std=std))
    return T.Compose(transforms)


def build_audio_augment(config: AudioAugmentConfig) -> Callable[[torch.Tensor], torch.Tensor]:
    def augment(waveform: torch.Tensor) -> torch.Tensor:
        augmented = waveform.clone()
        if config.time_shift > 0:
            shift = int(augmented.size(-1) * config.time_shift)
            augmented = torch.roll(augmented, shifts=shift, dims=-1)
        if config.mask_time > 0:
            length = augmented.size(-1)
            mask = int(length * config.mask_time)
            start = torch.randint(0, max(1, length - mask), (1,)).item()
            augmented[..., start : start + mask] = 0
        if config.noise > 0:
            noise = torch.randn_like(augmented) * config.noise
            augmented = augmented + noise
        return augmented

    return augment


__all__ = [
    "VisionAugmentConfig",
    "AudioAugmentConfig",
    "build_vision_augment",
    "build_audio_augment",
]
