"""Self-supervised training entrypoint for modality adapters."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..adapters import (
    AudioAdapter,
    AudioAdapterConfig,
    VisionAdapter,
    VisionAdapterConfig,
)
from ..data.augment import (
    AudioAugmentConfig,
    VisionAugmentConfig,
    build_audio_augment,
    build_vision_augment,
)
from ..utils.config import Config, build_arg_parser, load_config
from ..utils.logging import configure_logging, get_logger
from ..utils.seed import create_generator, seed_worker, set_global_seed


LOGGER = get_logger(__name__)


class VisionDataset(Dataset):
    """Dataset returning two augmented views per sample."""

    def __init__(
        self,
        path: Path,
        augment_config: VisionAugmentConfig,
    ) -> None:
        with np.load(path) as npz:
            self.data = torch.from_numpy(npz["data"]).float()
            mean_arr = npz.get("mean", np.array([0.5, 0.5, 0.5], dtype=np.float32))
            std_arr = npz.get("std", np.array([0.5, 0.5, 0.5], dtype=np.float32))
            mean = tuple(float(x) for x in mean_arr)
            std = tuple(float(x) for x in std_arr)
        self.augment = build_vision_augment(augment_config, mean=mean, std=std)

    def __len__(self) -> int:  # pragma: no cover - container execution
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        sample = self.data[idx]
        view1 = self.augment(sample.clone())
        view2 = self.augment(sample.clone())
        return view1, view2


class AudioDataset(Dataset):
    """Dataset returning log-mel spectrograms with optional masks."""

    def __init__(self, path: Path, augment) -> None:
        with np.load(path) as npz:
            self.data = torch.from_numpy(npz["data"]).float()
            self.mask = (
                torch.from_numpy(npz["mask"]).bool() if "mask" in npz.files else None
            )
            self.lengths = (
                torch.from_numpy(npz["lengths"]).long() if "lengths" in npz.files else None
            )
            mean = torch.from_numpy(
                npz.get("mean", np.zeros(self.data.size(1), dtype=np.float32))
            ).view(-1, 1)
            std = torch.from_numpy(
                npz.get("std", np.ones(self.data.size(1), dtype=np.float32))
            ).view(-1, 1)
        self.mean = mean.float()
        self.std = std.float().clamp_min(1e-6)
        self.augment = augment

    def __len__(self) -> int:  # pragma: no cover - container execution
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor | None]:
        sample = (self.data[idx] - self.mean) / self.std
        mask = self.mask[idx] if self.mask is not None else None
        view1 = self.augment(sample.clone())
        view2 = self.augment(sample.clone())
        return view1, view2, mask


def simclr_loss(z1: Tensor, z2: Tensor, temperature: float) -> Tensor:
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    batch_size = z1.size(0)
    representations = torch.cat([z1, z2], dim=0)
    similarity_matrix = F.cosine_similarity(
        representations.unsqueeze(1), representations.unsqueeze(0), dim=-1
    )

    labels = torch.arange(batch_size, device=z1.device)
    labels = torch.cat([labels + batch_size, labels], dim=0)

    mask = torch.eye(2 * batch_size, device=z1.device, dtype=torch.bool)
    similarity_matrix = similarity_matrix / temperature
    similarity_matrix.masked_fill_(mask, float("-inf"))

    loss = F.cross_entropy(similarity_matrix, labels)
    return loss


def _variance_loss(z: Tensor, eps: float = 1e-4) -> Tensor:
    std = torch.sqrt(z.var(dim=0) + eps)
    penalty = torch.relu(1.0 - std)
    return penalty.mean()


def _off_diagonal_elements(matrix: Tensor) -> Tensor:
    n, m = matrix.shape
    assert n == m
    return matrix.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vicreg_loss(z1: Tensor, z2: Tensor) -> Tensor:
    sim_loss = F.mse_loss(z1, z2)
    var_loss = _variance_loss(z1) + _variance_loss(z2)
    z1_c = z1 - z1.mean(dim=0)
    z2_c = z2 - z2.mean(dim=0)
    cov_z1 = (z1_c.T @ z1_c) / (z1_c.size(0) - 1)
    cov_z2 = (z2_c.T @ z2_c) / (z2_c.size(0) - 1)
    cov_loss = (_off_diagonal_elements(cov_z1).pow(2).mean() + _off_diagonal_elements(cov_z2).pow(2).mean())
    return 25.0 * sim_loss + 25.0 * var_loss + cov_loss


class DINOHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class DINOLoss(nn.Module):
    def __init__(
        self,
        out_dim: int,
        teacher_temp: float = 0.04,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ) -> None:
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_outputs: List[Tensor], teacher_outputs: List[Tensor]) -> Tensor:
        total_loss = 0.0
        n_terms = 0
        for student, teacher in zip(student_outputs, teacher_outputs):
            student = student / self.student_temp
            teacher = F.softmax((teacher - self.center) / self.teacher_temp, dim=-1)
            total_loss += torch.sum(-teacher * F.log_softmax(student, dim=-1), dim=-1).mean()
            n_terms += 1

        if n_terms == 0:
            raise ValueError("DINOLoss received no outputs to compare")

        mean_teacher = torch.cat(teacher_outputs, dim=0).mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + mean_teacher * (1 - self.center_momentum)
        return total_loss / n_terms


@dataclass
class TrainerConfig:
    epochs: int
    lr: float
    weight_decay: float
    temperature: float
    trainer_type: str


class SSLTrainer:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.modalities: Dict[str, Dict[str, object]] = {}

        model_cfg = config.raw.get("model", {})
        self.trainer_cfg = TrainerConfig(
            epochs=int(model_cfg.get("epochs", 100)),
            lr=float(model_cfg.get("lr", 1e-3)),
            weight_decay=float(model_cfg.get("weight_decay", 0.0)),
            temperature=float(model_cfg.get("temperature", 0.1)),
            trainer_type=str(model_cfg.get("trainer", "vicreg")).lower(),
        )

        self._prepare_logging()
        set_global_seed(int(config.raw.get("seed", 42)))

        self._prepare_modalities()
        params = chain.from_iterable(module["model"].parameters() for module in self.modalities.values())
        self.optimizer = torch.optim.AdamW(params, lr=self.trainer_cfg.lr, weight_decay=self.trainer_cfg.weight_decay)

        if self.trainer_cfg.trainer_type == "dino":
            self._init_dino()

    def _prepare_logging(self) -> None:
        log_dir = self.config.logging_dir / "logs"
        configure_logging(log_dir)
        LOGGER.info("Logging initialised at %s", log_dir)

    def _dataset_paths(self, modality: str) -> Tuple[Path, Path]:
        datasets_cfg = self.config.raw.get("data", {}).get("datasets", {})
        modality_cfg = datasets_cfg.get(modality, {})
        default_names = {"vision": "cifar10", "audio": "yesno"}
        name = str(modality_cfg.get("name", default_names.get(modality, modality))).lower()
        processed_dir = self.config.data_root / "processed"
        train = processed_dir / f"{modality}_{name}_train.npz"
        val = processed_dir / f"{modality}_{name}_val.npz"
        return train, val

    @staticmethod
    def _ensure_exists(path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file '{path}' not found. Run the data pipeline before training."
            )

    def _build_vision_module(self) -> Dict[str, object]:
        train_path, _ = self._dataset_paths("vision")
        self._ensure_exists(train_path)
        augment_cfg = self.config.raw.get("model", {}).get("augment", {}).get("vision", {})
        dataset = VisionDataset(train_path, VisionAugmentConfig(**augment_cfg))
        batch_size = int(self.config.raw.get("data", {}).get("batch_size", {}).get("vision", 256))
        loader = self._build_loader(dataset, batch_size)
        embedding_dim = int(self.config.raw.get("model", {}).get("d", 512))
        model = VisionAdapter(VisionAdapterConfig(embedding_dim=embedding_dim)).to(self.device)
        return {"dataset": dataset, "loader": loader, "model": model}

    def _build_audio_module(self) -> Dict[str, object]:
        train_path, _ = self._dataset_paths("audio")
        self._ensure_exists(train_path)
        augment_cfg = self.config.raw.get("model", {}).get("augment", {}).get("audio", {})
        dataset = AudioDataset(train_path, build_audio_augment(AudioAugmentConfig(**augment_cfg)))
        batch_size = int(self.config.raw.get("data", {}).get("batch_size", {}).get("audio", 128))
        loader = self._build_loader(dataset, batch_size)
        embedding_dim = int(self.config.raw.get("model", {}).get("d", 512))
        model = AudioAdapter(AudioAdapterConfig(embedding_dim=embedding_dim)).to(self.device)
        return {"dataset": dataset, "loader": loader, "model": model}

    def _prepare_modalities(self) -> None:
        modalities_cfg = self.config.raw.get("modalities", {})
        if modalities_cfg.get("vision", True):
            self.modalities["vision"] = self._build_vision_module()
        if modalities_cfg.get("audio", True):
            self.modalities["audio"] = self._build_audio_module()
        if not self.modalities:
            raise RuntimeError("At least one modality must be enabled for training")

    def _build_loader(self, dataset: Dataset, batch_size: int) -> DataLoader:
        num_workers = int(self.config.raw.get("data", {}).get("num_workers", 0))
        generator = create_generator(int(self.config.raw.get("seed", 42)))
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=generator,
            drop_last=True,
        )

    def _init_dino(self) -> None:
        self.dino_state: Dict[str, Dict[str, object]] = {}
        for name, module in self.modalities.items():
            model: nn.Module = module["model"]
            teacher = copy.deepcopy(model).to(self.device)
            for param in teacher.parameters():
                param.requires_grad_(False)

            embedding_dim = model.config.embedding_dim if hasattr(model, "config") else 512
            student_head = DINOHead(embedding_dim).to(self.device)
            teacher_head = copy.deepcopy(student_head).to(self.device)
            for param in teacher_head.parameters():
                param.requires_grad_(False)

            self.dino_state[name] = {
                "teacher": teacher,
                "student_head": student_head,
                "teacher_head": teacher_head,
                "loss": DINOLoss(out_dim=256).to(self.device),
            }

        base_params = chain.from_iterable(pg["params"] for pg in self.optimizer.param_groups)
        params = chain(
            base_params,
            *(state["student_head"].parameters() for state in self.dino_state.values()),
        )
        self.optimizer = torch.optim.AdamW(params, lr=self.trainer_cfg.lr, weight_decay=self.trainer_cfg.weight_decay)
        self.dino_momentum = 0.996

    def train(self) -> None:
        LOGGER.info("Starting %s training for %d epochs", self.trainer_cfg.trainer_type, self.trainer_cfg.epochs)
        history: List[Dict[str, float]] = []
        for epoch in range(1, self.trainer_cfg.epochs + 1):
            epoch_loss = 0.0
            total_steps = 0
            for modality, module in self.modalities.items():
                loader: DataLoader = module["loader"]
                model: nn.Module = module["model"]
                model.train()
                progress = tqdm(loader, desc=f"Epoch {epoch} [{modality}]", leave=False)
                for batch in progress:
                    loss = self._train_step(modality, model, batch)
                    epoch_loss += loss
                    total_steps += 1
                    progress.set_postfix(loss=loss)

            mean_loss = epoch_loss / max(total_steps, 1)
            LOGGER.info("Epoch %d | mean loss: %.4f", epoch, mean_loss)
            history.append({"epoch": epoch, "loss": float(mean_loss)})

        self._save_checkpoint(history)

    def _train_step(self, modality: str, model: nn.Module, batch) -> float:
        self.optimizer.zero_grad()
        if modality == "audio":
            view1, view2, mask = batch
            mask = mask.to(self.device) if mask is not None else None
            view1 = view1.to(self.device)
            view2 = view2.to(self.device)
            z1 = model(view1, mask)
            z2 = model(view2, mask)
        else:
            view1, view2 = batch
            view1 = view1.to(self.device)
            view2 = view2.to(self.device)
            z1 = model(view1)
            z2 = model(view2)

        if self.trainer_cfg.trainer_type == "simclr":
            loss = simclr_loss(z1, z2, self.trainer_cfg.temperature)
        elif self.trainer_cfg.trainer_type == "vicreg":
            loss = vicreg_loss(z1, z2)
        elif self.trainer_cfg.trainer_type == "dino":
            loss, updater = self._dino_step(modality, model, z1, z2, view1, view2, batch)
        else:
            raise ValueError(f"Unknown trainer type: {self.trainer_cfg.trainer_type}")

        loss.backward()
        self.optimizer.step()
        if self.trainer_cfg.trainer_type == "dino":
            updater()
        return float(loss.item())

    def _dino_step(
        self,
        modality: str,
        model: nn.Module,
        z1: Tensor,
        z2: Tensor,
        view1: Tensor,
        view2: Tensor,
        batch,
    ) -> Tuple[Tensor, Callable[[], None]]:
        state = self.dino_state[modality]
        teacher: nn.Module = state["teacher"]
        student_head: DINOHead = state["student_head"]
        teacher_head: DINOHead = state["teacher_head"]
        loss_fn: DINOLoss = state["loss"]

        with torch.no_grad():
            if modality == "audio":
                _, _, mask = batch
                mask = mask.to(self.device) if mask is not None else None
                teacher_z1 = teacher(view1.to(self.device), mask)
                teacher_z2 = teacher(view2.to(self.device), mask)
            else:
                teacher_z1 = teacher(view1.to(self.device))
                teacher_z2 = teacher(view2.to(self.device))

        student_outputs = [student_head(z1), student_head(z2)]
        teacher_outputs = [teacher_head(teacher_z1), teacher_head(teacher_z2)]
        loss = loss_fn(student_outputs, teacher_outputs)

        def update_teacher() -> None:
            with torch.no_grad():
                for student_param, teacher_param in zip(model.parameters(), teacher.parameters()):
                    teacher_param.data.mul_(self.dino_momentum).add_(
                        student_param.data * (1 - self.dino_momentum)
                    )
                for student_param, teacher_param in zip(
                    student_head.parameters(), teacher_head.parameters()
                ):
                    teacher_param.data.mul_(self.dino_momentum).add_(
                        student_param.data * (1 - self.dino_momentum)
                    )

        return loss, update_teacher

    def _save_checkpoint(self, history: List[Dict[str, float]]) -> None:
        ckpt_dir = self.config.logging_dir / "ckpts"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / "last.pt"

        state = {
            "config": self.config.raw,
            "trainer": self.trainer_cfg.trainer_type,
            "modalities": {name: module["model"].state_dict() for name, module in self.modalities.items()},
            "history": history,
        }
        torch.save(state, ckpt_path)
        LOGGER.info("Saved checkpoint to %s", ckpt_path)

        log_dir = self.config.logging_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        history_path = log_dir / "training_history.json"
        with history_path.open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)
        LOGGER.info("Saved training history to %s", history_path)


def run(config: Config) -> None:
    trainer = SSLTrainer(config)
    trainer.train()


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    run(config)


if __name__ == "__main__":  # pragma: no cover - entrypoint
    main()

