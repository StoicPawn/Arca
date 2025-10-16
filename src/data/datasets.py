"""Dataset download, preprocessing, and DataLoader utilities."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from torchvision import datasets as tv_datasets
import torchvision.transforms as T

import torchaudio
from torchaudio import transforms as ta_transforms

from src.utils.config import Config, load_from_args
from src.utils.seed import set_seed

# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DatasetPaths:
    root: Path
    raw: Path
    processed: Path

    @classmethod
    def from_root(cls, root: Path) -> "DatasetPaths":
        raw = root / "raw"
        processed = root / "processed"
        raw.mkdir(parents=True, exist_ok=True)
        processed.mkdir(parents=True, exist_ok=True)
        return cls(root=root, raw=raw, processed=processed)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _md5(path: Path) -> str:
    hash_md5 = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def verify_checksum(directory: Path, expected: str) -> bool:
    """Verify that any file inside ``directory`` matches the expected md5 hash."""

    for path in directory.iterdir():
        if path.is_file() and _md5(path) == expected:
            return True
    return False


# ---------------------------------------------------------------------------
# Vision preprocessing
# ---------------------------------------------------------------------------


def _prepare_stl10(paths: DatasetPaths, checksum: str) -> Dict[str, torch.Tensor]:
    dataset = tv_datasets.STL10(
        root=str(paths.raw),
        split="unlabeled",
        download=True,
    )

    # STL10 ships with 96x96x3 numpy arrays
    images = torch.from_numpy(dataset.data).permute(0, 3, 1, 2).float() / 255.0

    if checksum and not verify_checksum(paths.raw, checksum):
        raise RuntimeError("Checksum verification failed for STL10 dataset.")

    return {"features": images}


def _prepare_dtd(paths: DatasetPaths, checksum: str) -> Dict[str, torch.Tensor]:
    dataset = tv_datasets.DTD(root=str(paths.raw), split="train", download=True)
    if checksum and not verify_checksum(paths.raw, checksum):
        raise RuntimeError("Checksum verification failed for DTD dataset.")

    resize = T.Resize((96, 96))
    to_tensor = T.ToTensor()

    tensors = []
    for img, _ in tqdm(dataset, desc="Processing DTD", leave=False):
        tensors.append(resize(img))
    data = torch.stack([to_tensor(x) for x in tensors])
    return {"features": data}


VISION_DATASETS = {
    "STL10": _prepare_stl10,
    "DTD": _prepare_dtd,
}


# ---------------------------------------------------------------------------
# Audio preprocessing
# ---------------------------------------------------------------------------

def _compute_log_mel(
    waveform: torch.Tensor,
    sample_rate: int,
    target_rate: int = 16_000,
    n_mels: int = 96,
    win_length_ms: float = 25.0,
    hop_length_ms: float = 10.0,
) -> torch.Tensor:
    if sample_rate != target_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_rate)
    win_length = int(target_rate * win_length_ms / 1000)
    hop_length = int(target_rate * hop_length_ms / 1000)
    mel = ta_transforms.MelSpectrogram(
        sample_rate=target_rate,
        n_fft=2048,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )(waveform)
    log_mel = torch.log(mel + 1e-6)
    return log_mel


def _prepare_urbansound8k(paths: DatasetPaths, checksum: str) -> Dict[str, torch.Tensor]:
    dataset = torchaudio.datasets.URBANSOUND8K(
        root=str(paths.raw),
        download=True,
    )
    if checksum and not verify_checksum(paths.raw, checksum):
        raise RuntimeError("Checksum verification failed for UrbanSound8K dataset.")

    specs: List[torch.Tensor] = []
    lengths: List[int] = []
    for waveform, sample_rate, _, _, _, _, _ in tqdm(dataset, desc="Processing UrbanSound8K", leave=False):
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        log_mel = _compute_log_mel(waveform, sample_rate)
        specs.append(log_mel)
        lengths.append(log_mel.size(-1))

    padded = torch.nn.utils.rnn.pad_sequence(
        [spec.squeeze(0).transpose(0, 1) for spec in specs],
        batch_first=True,
    )
    # Reorder dimensions to [B, F, T]
    padded = padded.permute(0, 2, 1)
    return {"features": padded, "lengths": torch.tensor(lengths, dtype=torch.long)}


def _prepare_esc50(paths: DatasetPaths, checksum: str) -> Dict[str, torch.Tensor]:
    dataset = torchaudio.datasets.ESC50(root=str(paths.raw), download=True)
    if checksum and not verify_checksum(paths.raw, checksum):
        raise RuntimeError("Checksum verification failed for ESC-50 dataset.")

    specs: List[torch.Tensor] = []
    lengths: List[int] = []
    for waveform, sample_rate, _, _, _ in tqdm(dataset, desc="Processing ESC-50", leave=False):
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        log_mel = _compute_log_mel(waveform, sample_rate)
        specs.append(log_mel)
        lengths.append(log_mel.size(-1))

    padded = torch.nn.utils.rnn.pad_sequence(
        [spec.squeeze(0).transpose(0, 1) for spec in specs],
        batch_first=True,
    )
    padded = padded.permute(0, 2, 1)
    return {"features": padded, "lengths": torch.tensor(lengths, dtype=torch.long)}


AUDIO_DATASETS = {
    "UrbanSound8K": _prepare_urbansound8k,
    "ESC50": _prepare_esc50,
}


# ---------------------------------------------------------------------------
# Dataset packaging utilities
# ---------------------------------------------------------------------------


class VisionTensorDataset(Dataset):
    """Simple dataset wrapper for image tensors."""

    def __init__(self, features: torch.Tensor):
        self.features = features

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.features.size(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {"image": self.features[idx]}


class AudioTensorDataset(Dataset):
    """Dataset wrapper for padded log-mel spectrograms."""

    def __init__(self, features: torch.Tensor, lengths: torch.Tensor):
        self.features = features
        self.lengths = lengths

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.features.size(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {"audio": self.features[idx], "length": self.lengths[idx]}


def _split_payload(
    payload: Dict[str, torch.Tensor], split: float, seed: int
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    features = payload["features"]
    num_samples = features.size(0)
    indices = torch.randperm(num_samples, generator=torch.Generator().manual_seed(seed))
    split_idx = int(num_samples * split)
    val_idx = indices[:split_idx]
    train_idx = indices[split_idx:]

    def _subset(idxs: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {key: tensor[idxs] for key, tensor in payload.items()}

    return _subset(train_idx), _subset(val_idx)


def _compute_statistics(tensor: torch.Tensor) -> Dict[str, List[float]]:
    if tensor.dim() == 4:
        # Vision: [N, C, H, W]
        mean = tensor.mean(dim=(0, 2, 3))
        std = tensor.std(dim=(0, 2, 3))
    elif tensor.dim() == 3:
        # Audio: [N, F, T]
        mean = tensor.mean(dim=(0, 2))
        std = tensor.std(dim=(0, 2))
    else:
        raise ValueError("Unsupported tensor shape for statistics computation")
    return {"mean": mean.tolist(), "std": std.tolist()}


def _save_processed(
    tensor_dict: Dict[str, torch.Tensor],
    split_name: Literal["train", "val"],
    modality: Literal["vision", "audio"],
    dataset_name: str,
    paths: DatasetPaths,
) -> Path:
    filename = f"{dataset_name.lower()}_{modality}_{split_name}.pt"
    output_path = paths.processed / filename
    torch.save(tensor_dict, output_path)
    return output_path


def _save_statistics(stats: Dict[str, List[float]], modality: str, dataset_name: str, paths: DatasetPaths) -> Path:
    output_path = paths.processed / f"{dataset_name.lower()}_{modality}_stats.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return output_path


def _build_dataloader(
    payload: Dict[str, torch.Tensor],
    modality: Literal["vision", "audio"],
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    if modality == "vision":
        dataset: Dataset = VisionTensorDataset(payload["features"])
    elif modality == "audio":
        dataset = AudioTensorDataset(payload["features"], payload["lengths"])
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported modality '{modality}'.")

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def prepare_datasets(cfg: Config) -> Dict[str, Dict[str, DataLoader]]:
    set_seed(cfg.seed)

    base_path = DatasetPaths.from_root(Path(cfg.data["root"]))
    dataloaders: Dict[str, Dict[str, DataLoader]] = {}
    modalities = cfg.modalities or {"vision": True, "audio": True}

    if modalities.get("vision", False):
        dataset_name = cfg.data.get("vision_dataset", "STL10")
        preprocess_fn = VISION_DATASETS.get(dataset_name)
        if preprocess_fn is None:
            raise ValueError(f"Unsupported vision dataset '{dataset_name}'.")
        payload = preprocess_fn(base_path, cfg.data.get("checksum", {}).get(dataset_name, ""))
        train_payload, val_payload = _split_payload(payload, split=0.05, seed=cfg.seed)
        stats = _compute_statistics(train_payload["features"])
        _save_statistics(stats, "vision", dataset_name, base_path)
        _save_processed(train_payload, "train", "vision", dataset_name, base_path)
        _save_processed(val_payload, "val", "vision", dataset_name, base_path)
        dataloaders["vision"] = {
            "train": _build_dataloader(
                train_payload,
                modality="vision",
                batch_size=cfg.data["batch_size"]["vision"],
                num_workers=cfg.data["num_workers"],
                shuffle=True,
            ),
            "val": _build_dataloader(
                val_payload,
                modality="vision",
                batch_size=cfg.data["batch_size"]["vision"],
                num_workers=cfg.data["num_workers"],
                shuffle=False,
            ),
        }

    if modalities.get("audio", False):
        dataset_name = cfg.data.get("audio_dataset", "UrbanSound8K")
        preprocess_fn = AUDIO_DATASETS.get(dataset_name)
        if preprocess_fn is None:
            raise ValueError(f"Unsupported audio dataset '{dataset_name}'.")
        payload = preprocess_fn(base_path, cfg.data.get("checksum", {}).get(dataset_name, ""))
        train_payload, val_payload = _split_payload(payload, split=0.05, seed=cfg.seed)
        stats = _compute_statistics(train_payload["features"])
        _save_statistics(stats, "audio", dataset_name, base_path)
        _save_processed(train_payload, "train", "audio", dataset_name, base_path)
        _save_processed(val_payload, "val", "audio", dataset_name, base_path)
        dataloaders["audio"] = {
            "train": _build_dataloader(
                train_payload,
                modality="audio",
                batch_size=cfg.data["batch_size"]["audio"],
                num_workers=cfg.data["num_workers"],
                shuffle=True,
            ),
            "val": _build_dataloader(
                val_payload,
                modality="audio",
                batch_size=cfg.data["batch_size"]["audio"],
                num_workers=cfg.data["num_workers"],
                shuffle=False,
            ),
        }

    return dataloaders


# ---------------------------------------------------------------------------
# Command line interface
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = load_from_args()
    loaders = prepare_datasets(cfg)
    for modality, splits in loaders.items():
        for split_name, loader in splits.items():
            print(
                f"Prepared {modality} {split_name} loader "
                f"with {len(loader.dataset)} samples and batch size {loader.batch_size}"
            )


if __name__ == "__main__":
    main()


__all__ = [
    "DatasetPaths",
    "VisionTensorDataset",
    "AudioTensorDataset",
    "prepare_datasets",
    "verify_checksum",
]
