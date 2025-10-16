"""Vision adapter composed of lightweight convolutional blocks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """LayerNorm operating on the channel dimension of 2D feature maps."""

    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 4:
            raise ValueError("LayerNorm2d expects input with shape (batch, channels, height, width)")
        batch, channels, height, width = x.shape
        reshaped = x.permute(0, 2, 3, 1).reshape(-1, channels)
        normalised = self.norm(reshaped)
        return normalised.reshape(batch, height, width, channels).permute(0, 3, 1, 2)


@dataclass
class VisionAdapterConfig:
    """Configuration for :class:`VisionAdapter`."""

    in_channels: int = 3
    hidden_dims: tuple[int, int] = (64, 128)
    embedding_dim: int = 512


class VisionAdapter(nn.Module):
    """Simple convolutional encoder returning L2-normalised embeddings."""

    def __init__(self, config: VisionAdapterConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = VisionAdapterConfig()

        self.config = config

        layers: list[nn.Module] = []
        in_channels = config.in_channels
        for idx, hidden in enumerate(config.hidden_dims):
            stride = 1 if idx == 0 else 2
            layers.append(nn.Conv2d(in_channels, hidden, kernel_size=3, stride=stride, padding=1))
            layers.append(nn.GELU())
            layers.append(LayerNorm2d(hidden))
            in_channels = hidden

        self.feature_extractor = nn.Sequential(*layers)
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
        )

    def forward(self, images: Tensor) -> Tensor:
        if images.dim() != 4:
            raise ValueError("images must have shape (batch, channels, height, width)")

        features = self.feature_extractor(images)
        embedding = self.projection(features)
        return F.normalize(embedding, dim=-1)


__all__ = ["VisionAdapter", "VisionAdapterConfig"]

