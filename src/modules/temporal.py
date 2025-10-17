"""Temporal association module with multi-scale Hebbian traces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import Tensor, nn


@dataclass
class TemporalAssociationConfig:
    """Configuration for :class:`TemporalAssociationModule`."""

    num_codes: int = 128
    scales: Sequence[float] = (0.3, 1.0, 3.0, 8.0)
    scale_weights: Sequence[float] | None = None
    lambda_visual: Sequence[float] = (0.97, 0.95, 0.90, 0.85)
    lambda_audio: Sequence[float] = (0.97, 0.95, 0.90, 0.85)
    max_lag: float = 8.0
    causal: bool = True
    learning_rate: float = 1e-2
    weight_decay: float = 1e-5
    update_clip: float = 1e-2


class TemporalAssociationModule(nn.Module):
    """Maintain cross-modal associations via multi-scale Hebbian traces."""

    def __init__(self, config: TemporalAssociationConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = TemporalAssociationConfig()

        if config.num_codes <= 0:
            raise ValueError("num_codes must be positive")
        if len(config.scales) == 0:
            raise ValueError("At least one temporal scale must be provided")
        if len(config.lambda_visual) != len(config.scales):
            raise ValueError("lambda_visual must match the number of scales")
        if len(config.lambda_audio) != len(config.scales):
            raise ValueError("lambda_audio must match the number of scales")

        self.config = config
        self.register_buffer("scales", torch.tensor(config.scales, dtype=torch.float32))
        default_weights = config.scale_weights or [1.0] * len(config.scales)
        weights = self._normalise_weights(default_weights)
        self.register_buffer("scale_weights", weights)

        self.register_buffer("association", torch.zeros(config.num_codes, config.num_codes))
        self.register_buffer("cooccurrence", torch.zeros(config.num_codes, config.num_codes))
        self.register_buffer(
            "visual_traces", torch.zeros(len(config.scales), config.num_codes)
        )
        self.register_buffer(
            "audio_traces", torch.zeros(len(config.scales), config.num_codes)
        )

    def reset_state(self) -> None:
        """Reset running traces for a new sequence."""

        self.visual_traces.zero_()
        self.audio_traces.zero_()

    def compute_kernel(self, delta_t: Tensor) -> Tensor:
        """Compute temporal kernel weights for the provided time differences."""

        if delta_t.dim() == 0:
            delta_t = delta_t.unsqueeze(0)
        abs_delta = delta_t.abs()
        mask = (abs_delta <= self.config.max_lag).to(self.association.dtype)
        if self.config.causal:
            mask = mask * (delta_t >= 0).to(self.association.dtype)

        weights = torch.zeros_like(delta_t, dtype=self.association.dtype)
        for scale, weight in zip(self.scales, self.scale_weights):
            weights = weights + weight * torch.exp(-abs_delta / scale)

        return weights * mask

    def accumulate_cooccurrence(
        self,
        visual_codes: Tensor,
        audio_codes: Tensor,
        delta_t: Tensor,
    ) -> None:
        """Update the empirical co-occurrence matrix with temporal weighting."""

        if visual_codes.numel() == 0 or audio_codes.numel() == 0:
            return

        visual_codes = self._ensure_long(visual_codes, "visual_codes")
        audio_codes = self._ensure_long(audio_codes, "audio_codes")
        kernel = self.compute_kernel(delta_t)
        self.cooccurrence.index_put_(
            (visual_codes.unsqueeze(-1), audio_codes.unsqueeze(0)), kernel, accumulate=True
        )

    def step(self, visual_codes: Tensor, audio_codes: Tensor) -> Tensor:
        """Perform a Hebbian update of the association matrix."""

        device = self.association.device
        visual_counts = self._counts(self._ensure_long(visual_codes, "visual_codes"), device)
        audio_counts = self._counts(self._ensure_long(audio_codes, "audio_codes"), device)

        weighted = torch.zeros_like(self.association)
        for scale_idx, weight in enumerate(self.scale_weights):
            self.visual_traces[scale_idx].mul_(self.config.lambda_visual[scale_idx]).add_(
                visual_counts
            )
            self.audio_traces[scale_idx].mul_(self.config.lambda_audio[scale_idx]).add_(
                audio_counts
            )
            outer = torch.outer(self.visual_traces[scale_idx], self.audio_traces[scale_idx])
            weighted.add_(weight * outer)

        update = weighted - self.config.weight_decay * self.association
        if self.config.update_clip > 0:
            update = torch.clamp(update, -self.config.update_clip, self.config.update_clip)

        self.association.add_(self.config.learning_rate * update)
        return update

    def _counts(self, indices: Tensor, device: torch.device) -> Tensor:
        if indices.numel() == 0:
            return torch.zeros(self.config.num_codes, device=device)
        counts = torch.bincount(indices, minlength=self.config.num_codes)
        return counts.to(self.association.dtype)

    def _ensure_long(self, indices: Tensor, name: str) -> Tensor:
        if indices.dtype == torch.long:
            return indices
        if not torch.is_floating_point(indices):
            return indices.to(torch.long)
        raise ValueError(f"{name} must be integer indices, got floating point tensor")

    def _normalise_weights(self, values: Iterable[float]) -> Tensor:
        weights = torch.tensor(list(values), dtype=torch.float32)
        if (weights <= 0).all():
            raise ValueError("scale weights must contain positive entries")
        weights = torch.relu(weights)
        weights_sum = weights.sum().clamp_min(1e-6)
        return weights / weights_sum


__all__ = ["TemporalAssociationConfig", "TemporalAssociationModule"]

