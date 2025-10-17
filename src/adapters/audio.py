"""Audio adapter that extracts temporally windowed slots from log-mel inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from torch import Tensor, nn
import torch.nn.functional as F


def _make_group_norm(num_channels: int) -> nn.GroupNorm:
    if num_channels < 8:
        return nn.GroupNorm(1, num_channels)
    for groups in (8, 4, 2):
        if num_channels % groups == 0:
            return nn.GroupNorm(groups, num_channels)
    return nn.GroupNorm(1, num_channels)


@dataclass
class AudioAdapterConfig:
    """Configuration for the :class:`AudioAdapter`."""

    n_mels: int = 64
    hidden_dims: Sequence[int] = (96, 160, 192)
    window_size: int = 8
    window_stride: int = 4
    slot_dim: int = 48
    dropout: float = 0.0


class TemporalWindowing(nn.Module):
    """Extract sliding temporal windows and aggregate them into slots."""

    def __init__(self, window_size: int, stride: int) -> None:
        super().__init__()
        if window_size <= 0 or stride <= 0:
            raise ValueError("window_size and stride must be positive integers")
        self.window_size = int(window_size)
        self.stride = int(stride)

    def forward(self, features: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        """Return pooled windows and the corresponding validity mask."""

        if features.dim() != 3:
            raise ValueError("features must have shape (batch, channels, time)")

        batch, channels, time = features.shape
        if mask is not None:
            if mask.dim() != 2:
                raise ValueError("mask must have shape (batch, time)")
            mask = mask.to(features.dtype)
            if mask.size(1) != time:
                # Downsample or upsample the mask to match the feature sequence length.
                mask = F.interpolate(
                    mask.unsqueeze(1), size=time, mode="nearest"
                ).squeeze(1)
        if time < self.window_size:
            padding = self.window_size - time
            features = F.pad(features, (0, padding))
            if mask is not None:
                mask = F.pad(mask, (0, padding))
            time = features.size(-1)

        windows = features.unfold(dimension=-1, size=self.window_size, step=self.stride)
        pooled = windows.mean(dim=-1)
        pooled = pooled.transpose(1, 2)  # (batch, num_windows, channels)

        window_mask: Tensor | None = None
        if mask is not None:
            if mask.size(1) < self.window_size:
                padding = self.window_size - mask.size(1)
                mask = F.pad(mask, (0, padding))
            mask_windows = mask.unfold(dimension=-1, size=self.window_size, step=self.stride)
            # Consider a window valid if at least half of the frames are valid.
            threshold = 0.5 * float(self.window_size)
            window_mask = mask_windows.sum(dim=-1) >= threshold

        return pooled, window_mask


class AudioAdapter(nn.Module):
    """Convolutional frontend followed by temporal windowing into slots."""

    def __init__(self, config: AudioAdapterConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = AudioAdapterConfig()

        self.config = config

        layers: list[nn.Module] = []
        in_channels = config.n_mels
        for hidden in config.hidden_dims:
            layers.append(
                nn.Conv1d(
                    in_channels,
                    hidden,
                    kernel_size=5,
                    stride=2,
                    padding=2,
                    bias=False,
                )
            )
            layers.append(_make_group_norm(hidden))
            layers.append(nn.GELU())
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
            in_channels = hidden

        self.feature_extractor = nn.Sequential(*layers)
        self.windowing = TemporalWindowing(config.window_size, config.window_stride)
        self.slot_norm = nn.LayerNorm(in_channels)
        self.slot_projection = nn.Linear(in_channels, config.slot_dim)
        self.output_norm = nn.LayerNorm(config.slot_dim)

    def forward(
        self,
        spectrogram: Tensor,
        mask: Tensor | None = None,
        *,
        return_mask: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor | None]:
        """Encode log-mel spectrograms into a sequence of latent slots.

        Parameters
        ----------
        spectrogram:
            Input tensor with shape ``(batch, n_mels, time)``.
        mask:
            Optional boolean tensor marking valid time steps.
        return_mask:
            When ``True`` also return the validity mask for each temporal slot.
        """

        if spectrogram.dim() != 3:
            raise ValueError("spectrogram must have shape (batch, n_mels, time)")

        features = self.feature_extractor(spectrogram)
        slots, slot_mask = self.windowing(features, mask)
        slots = self.slot_norm(slots)
        projected = self.slot_projection(slots)
        projected = self.output_norm(projected)
        if return_mask:
            return projected, slot_mask
        return projected
