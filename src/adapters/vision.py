"""Vision adapter that produces object-centric slots on a fixed spatial grid."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from torch import Tensor, nn


class LayerNorm2d(nn.Module):
    """LayerNorm operating on the channel dimension of 2D feature maps."""

    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 4:
            raise ValueError(
                "LayerNorm2d expects input with shape (batch, channels, height, width)"
            )
        batch, channels, height, width = x.shape
        reshaped = x.permute(0, 2, 3, 1).reshape(-1, channels)
        normalised = self.norm(reshaped)
        return normalised.reshape(batch, height, width, channels).permute(0, 3, 1, 2)


def _make_group_norm(num_channels: int) -> nn.GroupNorm:
    """Return a lightweight GroupNorm with a reasonable number of groups."""

    if num_channels < 8:
        return nn.GroupNorm(1, num_channels)
    # Use powers of two when possible to keep the groups balanced.
    for groups in (8, 4, 2):
        if num_channels % groups == 0:
            return nn.GroupNorm(groups, num_channels)
    return nn.GroupNorm(1, num_channels)


@dataclass
class VisionAdapterConfig:
    """Configuration for :class:`VisionAdapter`."""

    in_channels: int = 3
    hidden_dims: Sequence[int] = (48, 96, 128)
    grid_size: int = 4
    slot_dim: int = 48
    use_group_norm: bool = True
    aggregator: str = "mlp"
    aggregator_hidden_dim: int | None = None


class VisionAdapter(nn.Module):
    """Shallow CNN that maps images to a grid of object-centric slots."""

    def __init__(self, config: VisionAdapterConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = VisionAdapterConfig()

        self.config = config

        layers: list[nn.Module] = []
        in_channels = config.in_channels
        for hidden in config.hidden_dims:
            layers.append(
                nn.Conv2d(
                    in_channels,
                    hidden,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            if config.use_group_norm:
                layers.append(_make_group_norm(hidden))
            else:
                layers.append(LayerNorm2d(hidden))
            layers.append(nn.GELU())
            in_channels = hidden

        self.backbone = nn.Sequential(*layers)
        self.grid_pool = nn.AdaptiveAvgPool2d((config.grid_size, config.grid_size))
        self.slot_norm = nn.LayerNorm(in_channels)
        self.slot_projection = nn.Linear(in_channels, config.slot_dim)
        self.output_norm = nn.LayerNorm(config.slot_dim)

        num_slots = config.grid_size * config.grid_size
        if config.aggregator == "mlp":
            hidden = config.aggregator_hidden_dim or (num_slots * config.slot_dim)
            self.slot_mixer = nn.Sequential(
                nn.LayerNorm(num_slots * config.slot_dim),
                nn.Linear(num_slots * config.slot_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, config.slot_dim),
            )
        else:
            self.slot_mixer = None
        self.final_norm = nn.LayerNorm(config.slot_dim)

    def forward(self, images: Tensor, *, return_grid: bool = False) -> Tensor:
        if images.dim() != 4:
            raise ValueError("images must have shape (batch, channels, height, width)")

        features = self.backbone(images)
        grid = self.grid_pool(features)
        batch_size, channels, height, width = grid.shape
        slots = grid.view(batch_size, channels, height * width).transpose(1, 2)
        slots = self.slot_norm(slots)
        projected = self.slot_projection(slots)
        slots = self.output_norm(projected)
        if return_grid:
            return slots
        aggregated = self._aggregate_slots(slots)
        return self.final_norm(aggregated)

    def _aggregate_slots(self, slots: Tensor) -> Tensor:
        if slots.dim() != 3:
            raise ValueError("slots must have shape (batch, num_slots, slot_dim)")

        if self.config.aggregator == "mlp":
            if self.slot_mixer is None:
                raise RuntimeError("slot_mixer is not initialised for MLP aggregation")
            flattened = slots.reshape(slots.size(0), -1)
            return self.slot_mixer(flattened)
        if self.config.aggregator == "mean":
            return slots.mean(dim=1)
        if self.config.aggregator == "max":
            return slots.max(dim=1).values
        raise ValueError(f"Unknown aggregator '{self.config.aggregator}'")


__all__ = ["VisionAdapter", "VisionAdapterConfig"]

