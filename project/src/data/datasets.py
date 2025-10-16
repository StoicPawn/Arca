from __future__ import annotations

import json
import pathlib
import urllib.error
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchvision
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

import librosa
import torchaudio

from ..utils.config import Config, build_arg_parser, load_config
from ..utils.logging import configure_logging, get_logger
from ..utils.seed import set_global_seed


LOGGER = get_logger(__name__)
TARGET_SR = 16_000
N_MELS = 96
WIN_LENGTH = 0.025
HOP_LENGTH = 0.01


@dataclass
class SplitResult:
    train_path: pathlib.Path
    val_path: pathlib.Path
    stats_path: pathlib.Path


VISION_DATASETS: Dict[str, type[Dataset]] = {
    "CIFAR10": torchvision.datasets.CIFAR10,
    "STL10": torchvision.datasets.STL10,
    "DTD": torchvision.datasets.DTD,
}

AUDIO_DATASETS: Dict[str, type] = {
    "YESNO": torchaudio.datasets.YESNO,
}


def ensure_dirs(config: Config) -> Tuple[pathlib.Path, pathlib.Path]:
    data_root = config.data_root
    raw_dir = data_root / "raw"
    processed_dir = data_root / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, processed_dir


def _extract_limit(options: Dict[str, object]) -> Tuple[Dict[str, object], int | None]:
    opts = dict(options)
    limit = opts.pop("limit", None)
    if limit is not None:
        limit = int(limit)
    return opts, limit


def _split_indices(num_samples: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)
    split = int(num_samples * 0.95)
    train_idx = indices[:split]
    val_idx = indices[split:]
    return train_idx, val_idx


def _save_npz(
    path: pathlib.Path,
    data: np.ndarray,
    stats: Dict[str, np.ndarray],
    mask: np.ndarray | None = None,
    lengths: np.ndarray | None = None,
) -> None:
    arrays: Dict[str, np.ndarray] = {"data": data}
    if mask is not None:
        arrays["mask"] = mask
    if lengths is not None:
        arrays["lengths"] = lengths
    arrays.update({k: v.astype(np.float32) for k, v in stats.items()})
    np.savez_compressed(path, **arrays)


def _save_stats(path: pathlib.Path, stats: Dict[str, np.ndarray]) -> None:
    serialisable = {k: v.tolist() for k, v in stats.items()}
    with path.open("w", encoding="utf-8") as handle:
        json.dump(serialisable, handle, indent=2)


def process_vision_dataset(
    name: str,
    raw_dir: pathlib.Path,
    processed_dir: pathlib.Path,
    options: Dict[str, object],
    seed: int,
) -> SplitResult:
    if name not in VISION_DATASETS:
        raise ValueError(f"Unsupported vision dataset: {name}")

    dataset_class = VISION_DATASETS[name]
    dataset_dir = raw_dir / "vision" / name.lower()
    dataset_dir.mkdir(parents=True, exist_ok=True)

    opts, limit = _extract_limit(options)
    try:
        dataset = dataset_class(root=str(dataset_dir), download=True, **opts)
    except (urllib.error.URLError, RuntimeError) as exc:
        raise RuntimeError(
            f"Failed to download {name}. Check your network connection or manually place the files in {dataset_dir}."
        ) from exc

    transform = transforms.Compose(
        [
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
        ]
    )

    samples: List[np.ndarray] = []
    for idx in tqdm(range(len(dataset)), desc=f"Processing {name}"):
        image, _ = dataset[idx]
        tensor = transform(image)
        samples.append(tensor.numpy())
        if limit is not None and len(samples) >= limit:
            break

    data = np.stack(samples)
    mean = data.mean(axis=(0, 2, 3))
    std = data.std(axis=(0, 2, 3)) + 1e-6

    train_idx, val_idx = _split_indices(len(data), seed)
    train_data = data[train_idx]
    val_data = data[val_idx]

    stats = {"mean": mean, "std": std}

    train_path = processed_dir / f"vision_{name.lower()}_train.npz"
    val_path = processed_dir / f"vision_{name.lower()}_val.npz"
    stats_path = processed_dir / f"vision_{name.lower()}_stats.json"

    _save_npz(train_path, train_data, stats)
    _save_npz(val_path, val_data, stats)
    _save_stats(stats_path, stats)

    LOGGER.info("Saved vision dataset '%s' to %s", name, processed_dir)
    return SplitResult(train_path, val_path, stats_path)


def _resample_waveform(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    if sample_rate == TARGET_SR:
        return waveform
    return torchaudio.functional.resample(waveform, sample_rate, TARGET_SR)


def _to_mono(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.ndim == 1:
        return waveform
    if waveform.size(0) == 1:
        return waveform.squeeze(0)
    return waveform.mean(dim=0)


def _compute_log_mel(audio: np.ndarray) -> np.ndarray:
    win_length = int(WIN_LENGTH * TARGET_SR)
    hop_length = int(HOP_LENGTH * TARGET_SR)
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=TARGET_SR,
        n_fft=win_length * 2,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=N_MELS,
        power=2.0,
        center=True,
    )
    log_mel = librosa.power_to_db(mel + 1e-10)
    return log_mel.astype(np.float32)


def process_audio_dataset(
    name: str,
    raw_dir: pathlib.Path,
    processed_dir: pathlib.Path,
    options: Dict[str, object],
    seed: int,
) -> SplitResult:
    if name not in AUDIO_DATASETS:
        raise ValueError(
            f"Unsupported audio dataset: {name}. Available: {', '.join(AUDIO_DATASETS.keys())}"
        )

    dataset_class = AUDIO_DATASETS[name]
    dataset_dir = raw_dir / "audio" / name.lower()
    dataset_dir.mkdir(parents=True, exist_ok=True)

    opts, limit = _extract_limit(options)
    try:
        dataset = dataset_class(root=str(dataset_dir), download=True, **opts)
    except (urllib.error.URLError, RuntimeError) as exc:
        raise RuntimeError(
            f"Failed to download {name}. Check your network connection or manually place the files in {dataset_dir}."
        ) from exc

    features: List[torch.Tensor] = []
    lengths: List[int] = []
    for idx in tqdm(range(len(dataset)), desc=f"Processing {name}"):
        waveform, sample_rate, *_ = dataset[idx]
        waveform = _to_mono(waveform)
        waveform = _resample_waveform(waveform, sample_rate)
        audio = waveform.numpy()
        log_mel = _compute_log_mel(audio)
        mel_tensor = torch.from_numpy(log_mel.T)
        features.append(mel_tensor)
        lengths.append(int(mel_tensor.shape[0]))
        if limit is not None and len(features) >= limit:
            break

    if not features:
        raise RuntimeError(f"No audio features were extracted from dataset '{name}'")

    padded = pad_sequence(features, batch_first=True)
    lengths_tensor = torch.tensor(lengths, dtype=torch.int64)
    max_length = int(lengths_tensor.max().item())
    time_indices = torch.arange(max_length, dtype=torch.int64)
    mask = time_indices.unsqueeze(0) < lengths_tensor.unsqueeze(1)

    valid = mask.unsqueeze(-1).to(padded.dtype)
    weighted = padded * valid
    sum_mel = weighted.sum(dim=(0, 1))
    sum_sq_mel = (padded.pow(2) * valid).sum(dim=(0, 1))
    total_weight = valid.sum().clamp(min=1.0)
    mean = sum_mel / total_weight
    var = (sum_sq_mel / total_weight) - mean.pow(2)
    std = torch.sqrt(torch.clamp(var, min=1e-6))

    data = padded.transpose(1, 2).numpy()
    mask_np = mask.numpy()
    lengths_np = lengths_tensor.numpy()
    mean_np = mean.numpy()
    std_np = std.numpy()

    train_idx, val_idx = _split_indices(len(data), seed)
    train_data = data[train_idx]
    val_data = data[val_idx]
    train_mask = mask_np[train_idx]
    val_mask = mask_np[val_idx]
    train_lengths = lengths_np[train_idx]
    val_lengths = lengths_np[val_idx]

    stats = {"mean": mean_np, "std": std_np}

    train_path = processed_dir / f"audio_{name.lower()}_train.npz"
    val_path = processed_dir / f"audio_{name.lower()}_val.npz"
    stats_path = processed_dir / f"audio_{name.lower()}_stats.json"

    _save_npz(train_path, train_data, stats, mask=train_mask, lengths=train_lengths)
    _save_npz(val_path, val_data, stats, mask=val_mask, lengths=val_lengths)
    _save_stats(stats_path, stats)

    LOGGER.info("Saved audio dataset '%s' to %s", name, processed_dir)
    return SplitResult(train_path, val_path, stats_path)


def _prepare_logging(config: Config) -> None:
    log_dir = config.logging_dir / "logs"
    configure_logging(log_dir)
    LOGGER.info("Logging initialised at %s", log_dir)


def run_pipeline(config: Config) -> Dict[str, SplitResult]:
    _prepare_logging(config)

    seed = int(config.raw.get("seed", 42))
    set_global_seed(seed)

    raw_dir, processed_dir = ensure_dirs(config)
    LOGGER.info("Raw data directory: %s", raw_dir)
    LOGGER.info("Processed data directory: %s", processed_dir)

    datasets_cfg = config.raw.get("data", {}).get("datasets", {})
    results: Dict[str, SplitResult] = {}

    if config.raw.get("modalities", {}).get("vision", True) and "vision" in datasets_cfg:
        vision_cfg = datasets_cfg["vision"]
        name = vision_cfg.get("name", "CIFAR10")
        options = vision_cfg.get("options", {})
        results["vision"] = process_vision_dataset(name, raw_dir, processed_dir, options, seed)

    if config.raw.get("modalities", {}).get("audio", True) and "audio" in datasets_cfg:
        audio_cfg = datasets_cfg["audio"]
        name = audio_cfg.get("name", "YESNO")
        options = audio_cfg.get("options", {})
        results["audio"] = process_audio_dataset(name, raw_dir, processed_dir, options, seed)

    return results


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    run_pipeline(config)


if __name__ == "__main__":
    main()
