"""Shared discrete bottleneck implemented via EMA vector quantisation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class VectorQuantizerEMAConfig:
    """Configuration for :class:`VectorQuantizerEMA`."""

    num_codes: int = 128
    code_dim: int = 48
    ema_decay: float = 0.99
    commitment_weight: float = 0.25
    epsilon: float = 1e-5
    usage_ema: float = 0.99
    usage_entropy_target: tuple[float, float] = (0.5, 0.9)
    entropy_weight: float = 0.1
    variance_floor: float = 0.05
    variance_weight: float = 1e-3
    decorrelation_weight: float = 1e-3


class VectorQuantizerEMA(nn.Module):
    """Vector quantiser with EMA updates and entropy regularisation."""

    def __init__(self, config: VectorQuantizerEMAConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = VectorQuantizerEMAConfig()

        if config.num_codes <= 0:
            raise ValueError("num_codes must be positive")
        if config.code_dim <= 0:
            raise ValueError("code_dim must be positive")

        self.config = config
        self.embedding = nn.Embedding(config.num_codes, config.code_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / config.num_codes, 1.0 / config.num_codes)

        self.register_buffer("ema_cluster_size", torch.zeros(config.num_codes))
        self.register_buffer("ema_weight_sum", torch.zeros(config.num_codes, config.code_dim))
        self.register_buffer("usage_counts", torch.zeros(config.num_codes))

    @property
    def codebook(self) -> Tensor:
        """Return the current codebook weights."""

        return self.embedding.weight

    def forward(self, inputs: Tensor) -> dict[str, Tensor | float]:
        """Quantise ``inputs`` and return auxiliary losses and statistics."""

        if inputs.size(-1) != self.config.code_dim:
            raise ValueError(
                f"Expected last dimension {self.config.code_dim}, got {inputs.size(-1)}"
            )

        flat_inputs = inputs.reshape(-1, self.config.code_dim)
        if flat_inputs.numel() == 0:
            dummy = torch.zeros_like(inputs)
            return {
                "quantized": dummy,
                "indices": torch.empty(0, dtype=torch.long, device=inputs.device),
                "loss": torch.zeros((), device=inputs.device),
                "commitment_loss": torch.zeros((), device=inputs.device),
                "entropy_loss": torch.zeros((), device=inputs.device),
                "variance_loss": torch.zeros((), device=inputs.device),
                "decorrelation_loss": torch.zeros((), device=inputs.device),
                "usage_entropy": torch.zeros((), device=inputs.device),
            }

        distances = (
            flat_inputs.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat_inputs @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(dim=1)
        )
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.config.num_codes).to(flat_inputs.dtype)

        if self.training:
            ema_decay = self.config.ema_decay
            cluster_size = encodings.sum(dim=0)
            embed_sum = encodings.t() @ flat_inputs

            self.ema_cluster_size.mul_(ema_decay).add_(cluster_size, alpha=1.0 - ema_decay)
            self.ema_weight_sum.mul_(ema_decay).add_(embed_sum, alpha=1.0 - ema_decay)

            usage_decay = self.config.usage_ema
            self.usage_counts.mul_(usage_decay).add_(cluster_size, alpha=1.0 - usage_decay)

            n = self.ema_cluster_size.sum()
            cluster_size = (
                (self.ema_cluster_size + self.config.epsilon)
                / (n + self.config.num_codes * self.config.epsilon)
                * n
            )
            denom = cluster_size.unsqueeze(1).clamp_min(self.config.epsilon)
            embed_normalised = self.ema_weight_sum / denom
            self.embedding.weight.data.copy_(embed_normalised)

        quantized = F.embedding(encoding_indices, self.embedding.weight)
        quantized = quantized.view_as(inputs)
        commitment_loss = self.config.commitment_weight * F.mse_loss(quantized.detach(), inputs)
        quantized = inputs + (quantized - inputs).detach()

        prob = self._usage_probabilities()
        usage_entropy = self._normalised_entropy(prob)
        entropy_loss = self._entropy_penalty(usage_entropy)
        variance_loss = self._variance_penalty()
        decorrelation_loss = self._decorrelation_penalty()

        total_loss = commitment_loss + entropy_loss + variance_loss + decorrelation_loss

        return {
            "quantized": quantized,
            "indices": encoding_indices.view(*inputs.shape[:-1]),
            "loss": total_loss,
            "commitment_loss": commitment_loss,
            "entropy_loss": entropy_loss,
            "variance_loss": variance_loss,
            "decorrelation_loss": decorrelation_loss,
            "usage_entropy": usage_entropy,
        }

    def _usage_probabilities(self) -> Tensor:
        counts = self.usage_counts
        probs = counts + self.config.epsilon
        probs = probs / probs.sum().clamp_min(self.config.epsilon * self.config.num_codes)
        return probs

    def _normalised_entropy(self, probs: Tensor) -> Tensor:
        entropy = -torch.sum(probs * torch.log(probs + self.config.epsilon))
        normaliser = math.log(float(self.config.num_codes))
        if normaliser <= 0.0:
            return torch.zeros((), device=probs.device)
        return entropy / normaliser

    def _entropy_penalty(self, entropy: Tensor) -> Tensor:
        lower, upper = self.config.usage_entropy_target
        if entropy < lower:
            gap = lower - entropy
        elif entropy > upper:
            gap = entropy - upper
        else:
            return torch.zeros((), device=entropy.device)
        return self.config.entropy_weight * gap.pow(2)

    def _variance_penalty(self) -> Tensor:
        variance = self.embedding.weight.var(dim=0, unbiased=False)
        shortfall = torch.relu(self.config.variance_floor - variance)
        return self.config.variance_weight * shortfall.mean()

    def _decorrelation_penalty(self) -> Tensor:
        weights = self.embedding.weight - self.embedding.weight.mean(dim=0, keepdim=True)
        cov = weights.t() @ weights / float(self.config.num_codes)
        off_diag = cov - torch.diag(torch.diag(cov))
        return self.config.decorrelation_weight * off_diag.pow(2).mean()


__all__ = ["VectorQuantizerEMA", "VectorQuantizerEMAConfig"]

