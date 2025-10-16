"""Audio adapter with masked pooling for variable-length inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class AudioAdapterConfig:
    """Configuration for the :class:`AudioAdapter`."""

    n_mels: int = 96
    hidden_dims: Sequence[int] = (128, 256)
    embedding_dim: int = 512
    dropout: float = 0.0


class MaskedMeanPooling(nn.Module):
    """Mean pooling that ignores padded positions using a boolean mask."""

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        if mask is None:
            return x.mean(dim=-1)

        if mask.dim() != 2:
            raise ValueError("mask must have shape (batch, time)")

        if x.size(-1) != mask.size(1):
            raise ValueError(
                "Mask length does not match the temporal dimension of the input"
            )

        mask = mask.unsqueeze(1).to(dtype=x.dtype)
        total = mask.sum(dim=-1).clamp(min=1.0)
        pooled = (x * mask).sum(dim=-1) / total
        return pooled


class AudioAdapter(nn.Module):
    """Temporal convolutional encoder with masked pooling for audio spectrograms."""

    def __init__(self, config: AudioAdapterConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = AudioAdapterConfig()

        self.config = config

        layers: list[nn.Module] = []
        in_channels = config.n_mels
        for hidden in config.hidden_dims:
            layers.append(nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1))
            layers.append(nn.GELU())
            layers.append(nn.GroupNorm(1, hidden))
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            in_channels = hidden

        self.feature_extractor = nn.Sequential(*layers)
        self.pool = MaskedMeanPooling()
        self.projection = nn.Sequential(
            nn.Linear(in_channels, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
        )

    def forward(self, spectrogram: Tensor, mask: Tensor | None = None) -> Tensor:
        """Encode a batch of log-mel spectrograms into fixed-size embeddings.

        Parameters
        ----------
        spectrogram:
            Tensor with shape ``(batch, n_mels, time)`` containing log-mel features.
        mask:
            Optional boolean tensor with shape ``(batch, time)`` indicating valid
            (non-padded) positions for each sample.
        """

        if spectrogram.dim() != 3:
            raise ValueError("spectrogram must have shape (batch, n_mels, time)")

        features = self.feature_extractor(spectrogram)
        pooled = self.pool(features, mask)
        embedding = self.projection(pooled)
        return F.normalize(embedding, dim=-1)
