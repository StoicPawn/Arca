"""Temporal co-occurrence trainer for modality adapters."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..adapters import (
    AudioAdapter,
    AudioAdapterConfig,
    VisionAdapter,
    VisionAdapterConfig,
)
from ..utils.config import Config, build_arg_parser, load_config
from ..utils.logging import configure_logging, get_logger
from ..utils.seed import create_generator, seed_worker, set_global_seed


LOGGER = get_logger(__name__)


class CooccurrenceDataset(Dataset):
    """Dataset exposing object-centric images and temporally aligned audio events."""

    def __init__(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset file '{path}' not found. Run the data pipeline before training."
            )

        with np.load(path, allow_pickle=False) as npz:
            self.vision = torch.from_numpy(npz["vision"]).float()
            self.vision_mask = torch.from_numpy(npz["vision_mask"]).bool()
            self.vision_times = torch.from_numpy(npz["vision_times"]).float()
            self.audio = torch.from_numpy(npz["audio"]).float()
            self.audio_mask = torch.from_numpy(npz["audio_mask"]).bool()
            self.audio_event_mask = torch.from_numpy(npz["audio_event_mask"]).bool()
            self.audio_times = torch.from_numpy(npz["audio_times"]).float()
            self.temporal_kernel = torch.from_numpy(npz["temporal_kernel"]).float()
            self.clip_length = torch.from_numpy(npz["clip_length"]).float()
            self.audio_mean = torch.from_numpy(npz["audio_mean"]).float()
            self.audio_std = torch.from_numpy(npz["audio_std"]).float().clamp_min(1e-6)
            self.temporal_scales = torch.from_numpy(npz["temporal_scales"]).float()

        self.max_objects = int(self.vision.size(1))
        self.max_events = int(self.audio.size(1))

    def __len__(self) -> int:
        return int(self.vision.size(0))

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        vision = self.vision[idx]
        vision_mask = self.vision_mask[idx]
        audio = self.audio[idx]
        audio = (audio - self.audio_mean.view(1, -1, 1)) / self.audio_std.view(1, -1, 1)
        sample = {
            "vision": vision,
            "vision_mask": vision_mask,
            "vision_times": self.vision_times[idx],
            "audio": audio,
            "audio_mask": self.audio_mask[idx],
            "audio_event_mask": self.audio_event_mask[idx],
            "audio_times": self.audio_times[idx],
            "temporal_kernel": self.temporal_kernel[idx],
            "clip_length": self.clip_length[idx],
        }
        return sample


@dataclass
class TrainerConfig:
    epochs: int
    lr: float
    weight_decay: float
    variance_floor: float


class AssociativeTrainer:
    """Learn visual/audio adapters purely from temporal co-occurrence."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_cfg = config.raw.get("model", {})
        self.trainer_cfg = TrainerConfig(
            epochs=int(model_cfg.get("epochs", 100)),
            lr=float(model_cfg.get("lr", 5e-4)),
            weight_decay=float(model_cfg.get("weight_decay", 0.0)),
            variance_floor=float(model_cfg.get("variance_floor", 0.05)),
        )

        self._prepare_logging()
        seed = int(config.raw.get("seed", 42))
        set_global_seed(seed)

        self.dataset = self._build_dataset()
        self.loader = self._build_loader(self.dataset)

        embedding_dim = int(model_cfg.get("d", 512))
        self.vision_model = VisionAdapter(
            VisionAdapterConfig(slot_dim=embedding_dim)
        ).to(self.device)
        n_mels = int(self.dataset.audio.size(2))
        self.audio_model = AudioAdapter(
            AudioAdapterConfig(slot_dim=embedding_dim, n_mels=n_mels)
        ).to(self.device)

        params = list(self.vision_model.parameters()) + list(self.audio_model.parameters())
        self.optimizer = torch.optim.AdamW(
            params,
            lr=self.trainer_cfg.lr,
            weight_decay=self.trainer_cfg.weight_decay,
        )

        self.history: List[Dict[str, float]] = []

    def _prepare_logging(self) -> None:
        log_dir = self.config.logging_dir / "logs"
        configure_logging(log_dir)
        LOGGER.info("Logging initialised at %s", log_dir)

    def _dataset_path(self) -> Path:
        processed_dir = self.config.data_root / "processed"
        return processed_dir / "cooccurrence_train.npz"

    def _build_dataset(self) -> CooccurrenceDataset:
        dataset_path = self._dataset_path()
        LOGGER.info("Loading training dataset from %s", dataset_path)
        return CooccurrenceDataset(dataset_path)

    def _build_loader(self, dataset: Dataset) -> DataLoader:
        batch_size = int(self.config.raw.get("data", {}).get("batch_size", 32))
        num_workers = int(self.config.raw.get("data", {}).get("num_workers", 0))
        generator = create_generator(int(self.config.raw.get("seed", 42)))
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=generator,
            drop_last=False,
        )

    def train(self) -> None:
        LOGGER.info(
            "Starting co-occurrence training for %d epochs (batch=%d)",
            self.trainer_cfg.epochs,
            self.loader.batch_size,
        )
        for epoch in range(1, self.trainer_cfg.epochs + 1):
            epoch_loss = 0.0
            total_steps = 0
            progress = tqdm(self.loader, desc=f"Epoch {epoch}", leave=False)
            for batch in progress:
                loss = self._train_step(batch)
                epoch_loss += loss
                total_steps += 1
                progress.set_postfix(loss=loss)

            mean_loss = epoch_loss / max(total_steps, 1)
            LOGGER.info("Epoch %d | mean loss: %.5f", epoch, mean_loss)
            self.history.append({"epoch": epoch, "loss": float(mean_loss)})

        self._save_checkpoint()

    def _train_step(self, batch: Dict[str, Tensor]) -> float:
        self.optimizer.zero_grad()

        vision = batch["vision"].to(self.device)
        vision_mask = batch["vision_mask"].to(self.device)
        audio = batch["audio"].to(self.device)
        audio_mask = batch["audio_mask"].to(self.device)
        audio_event_mask = batch["audio_event_mask"].to(self.device)
        kernel_target = batch["temporal_kernel"].to(self.device)

        vision_embeddings = self._encode_vision(vision, vision_mask)
        audio_embeddings = self._encode_audio(audio, audio_mask, audio_event_mask)

        similarity = torch.einsum("bod,bed->boe", vision_embeddings, audio_embeddings)
        similarity = (similarity + 1.0) * 0.5  # map cosine similarity to [0, 1]

        mask_matrix = vision_mask.unsqueeze(-1) & audio_event_mask.unsqueeze(1)
        mask = mask_matrix.to(similarity.dtype)
        masked_similarity = similarity * mask

        temporal_target = kernel_target * mask
        row_sum = temporal_target.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        col_sum = temporal_target.sum(dim=-2, keepdim=True).clamp_min(1e-6)
        row_norm = temporal_target / row_sum
        col_norm = temporal_target / col_sum
        target = 0.5 * (row_norm + col_norm)
        target = target * mask

        error = (masked_similarity - target) ** 2
        temporal_loss = error.sum() / mask.sum().clamp_min(1.0)

        variance_loss = self._variance_penalty(vision_embeddings, vision_mask)
        variance_loss += self._variance_penalty(audio_embeddings, audio_event_mask)

        balance_loss = self._balance_penalty(masked_similarity, mask)

        loss = temporal_loss + 0.1 * variance_loss + 0.05 * balance_loss
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def _encode_vision(self, vision: Tensor, mask: Tensor) -> Tensor:
        batch, num_objects, channels, height, width = vision.shape
        flattened = vision.view(batch * num_objects, channels, height, width)
        mask_flat = mask.view(batch * num_objects)
        embeddings = torch.zeros(
            batch * num_objects,
            self.vision_model.config.slot_dim,
            device=self.device,
        )
        if mask_flat.any():
            valid = flattened[mask_flat]
            valid = valid * 2.0 - 1.0
            encoded = self.vision_model(valid)
            embeddings[mask_flat] = encoded
        return embeddings.view(batch, num_objects, -1)

    def _encode_audio(self, audio: Tensor, mask: Tensor, event_mask: Tensor) -> Tensor:
        batch, num_events, n_mels, time = audio.shape
        flattened = audio.view(batch * num_events, n_mels, time)
        mask_flat = mask.view(batch * num_events, time)
        event_mask_flat = event_mask.view(batch * num_events)

        embeddings = torch.zeros(
            batch * num_events,
            self.audio_model.config.slot_dim,
            device=self.device,
        )
        if event_mask_flat.any():
            valid = flattened[event_mask_flat]
            valid_mask = mask_flat[event_mask_flat]
            encoded = self.audio_model(valid, valid_mask)
            embeddings[event_mask_flat] = encoded
        return embeddings.view(batch, num_events, -1)

    def _variance_penalty(self, embeddings: Tensor, mask: Tensor) -> Tensor:
        valid = embeddings[mask]
        if valid.size(0) <= 1:
            return torch.zeros((), device=self.device)
        variance = valid.var(dim=0, unbiased=False)
        penalty = torch.relu(self.trainer_cfg.variance_floor - variance)
        return penalty.mean()

    @staticmethod
    def _balance_penalty(similarity: Tensor, mask: Tensor) -> Tensor:
        row_counts = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        col_counts = mask.sum(dim=-2, keepdim=True).clamp_min(1.0)
        row_mean = (similarity.sum(dim=-1, keepdim=True) / row_counts)
        col_mean = (similarity.sum(dim=-2, keepdim=True) / col_counts)
        deviation = (row_mean - 0.5).pow(2).mean() + (col_mean - 0.5).pow(2).mean()
        return deviation

    def _save_checkpoint(self) -> None:
        ckpt_dir = self.config.logging_dir / "ckpts"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        history_path = ckpt_dir / "training_history.json"
        weights_path = ckpt_dir / "associative_adapters.pt"

        torch.save(
            {
                "vision_adapter": self.vision_model.state_dict(),
                "audio_adapter": self.audio_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "history": self.history,
                "config": self.config.raw,
            },
            weights_path,
        )

        history_path.write_text(json.dumps(self.history, indent=2), encoding="utf-8")
        LOGGER.info("Saved checkpoint to %s", weights_path)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    trainer = AssociativeTrainer(config)
    trainer.train()


def run(config: Config) -> None:
    """Convenience entrypoint used by :mod:`src.trainers`."""
    trainer = AssociativeTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()

